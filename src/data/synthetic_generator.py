"""Synthetic ICU dataset generator (NOT MIMIC-IV; no real patient records).

Pipeline per episode: scenario -> existing ICU census -> NHPP cohort arrivals in a 30-min window ->
latent severity z -> raw clinical variables -> SOFA (separate module) -> hidden LOS / deterioration parameters.
Then predictors are fitted on training episodes only and predictions are attached.
Command: python -m src.data.synthetic_generator
"""
import json
import numpy as np
import pandas as pd

from ..common import load_config, DATA, CKPT
from .sofa_calculator import sofa_components, COMPONENTS
from ..prediction.los_predictor import oof_predictions, RAW_COLS


def raw_from_latent(z, age, nz, u):
    """Deterministic map (latent severity z, patient noise nz[14], uniforms u[3]) -> raw variables.

    Using fixed per-patient noise lets a deterioration (z -> z + dz) update every raw measurement coherently."""
    z = np.asarray(z, float)
    pf = np.clip(460 - 400 * z + 45 * nz[0], 55, 520)
    fio2 = np.clip(0.21 + 0.75 * z ** 1.2 + 0.07 * nz[1], 0.21, 1.0)
    vent = (((pf < 250) & (u[0] < 0.9)) | (u[0] > 1 - 0.12 * z)).astype(int)
    spo2 = np.clip(99 - 14 * z ** 1.5 + 1.5 * nz[2], 70, 100)
    plt = np.clip(260 * (1 - 0.75 * z) * np.exp(0.3 * nz[3]), 5, 600)
    bili = np.clip(0.7 * np.exp(2.6 * z + 0.5 * nz[4]), 0.2, 30)
    mapv = np.clip(90 - 38 * z + 7 * nz[5], 40, 120)
    on = ((mapv < 68) & (u[1] < 0.9)) | ((z > 0.65) & (u[1] < 0.5))
    drug = np.where(on, np.where(u[2] < 0.8, 2, 1), 0)
    dose = np.where(drug == 2, np.clip(0.05 * np.exp(2.0 * (z - 0.5) + 0.5 * nz[6]), 0.01, 1.0),
                    np.where(drug == 1, np.clip(6 * np.exp(2.0 * (z - 0.5) + 0.4 * nz[6]), 2, 25), 0.0))
    gcs = np.round(np.clip(15 - 11 * z ** 1.3 + 1.6 * nz[7], 3, 15)).astype(int)
    creat = np.clip(0.9 * np.exp(1.8 * z + 0.4 * nz[8]), 0.3, 12)
    uo = np.clip(2000 * (1 - 0.85 * z) * np.exp(0.3 * nz[9]), 20, 4000)
    hr = np.clip(78 + 35 * z + 10 * nz[10], 40, 190)
    rr = np.clip(15 + 16 * z + 3 * nz[11], 6, 50)
    sbp = np.clip(mapv * 1.45 - 4 + 6 * nz[12], 60, 220)
    dbp = np.clip((3 * mapv - sbp) / 2, 30, 130)
    temp = 36.8 + 0.5 * z + 0.6 * nz[13]
    return dict(pao2=pf * fio2, fio2=fio2, pf_ratio=pf, mech_vent=vent, platelets=plt, bilirubin=bili, map=mapv,
                vaso_drug=drug, vaso_dose=dose, gcs=gcs, creatinine=creat, urine_output=uo, heart_rate=hr,
                resp_rate=rr, spo2=spo2, sbp=sbp, dbp=dbp, temperature=temp)


def true_hazard(z, age, spo2, frailty, c):
    """Hidden deterioration hazard per hour for a waiting patient."""
    d = c["data"]["deterioration"]
    lg = np.log(d["base_rate_h"]) + d["z_coef"] * z + d["age_coef"] * (age - 62) + d["hypox_coef"] * (spo2 < 92) + frailty
    return np.exp(lg)


def sample_age(rng, n):
    return np.clip(rng.normal(62, 15, n), 18, 95)


def make_patients(rng, n, c, role, ep, id0):
    """Draw n patients; returns DataFrame with raw variables + hidden columns (latent z, noise, LOS, hazard)."""
    L = c["data"]["los"]; D = c["data"]["deterioration"]
    z = rng.beta(2, 3, n) * (0.85 if role == "occupant" else 1.0)  # stabilised occupants slightly milder
    age = sample_age(rng, n)
    nz = rng.standard_normal((14, n)); u = rng.random((3, n))
    raw = raw_from_latent(z, age, nz, u)
    frailty = rng.normal(0, D["frailty_sd"], n)
    haz = true_hazard(z, age, raw["spo2"], frailty, c)
    mu = np.log(L["median_arrival_h"]) + L["z_coef"] * (z - 0.4) + L["age_coef"] * (age - 62) + L["vent_coef"] * (raw["mech_vent"] - 0.3)
    los = np.exp(mu + L["sigma"] * rng.standard_normal(n)) * (L["occupant_mult"] if role == "occupant" else 1.0)
    df = pd.DataFrame(raw)
    df.insert(0, "patient_id", [f"E{ep:04d}_{role[0].upper()}{id0 + i:02d}" for i in range(n)])
    df.insert(1, "episode_id", ep); df.insert(2, "role", role)
    df["age"] = age; df["z_latent"] = z; df["actual_los_total_h"] = los; df["true_hazard"] = haz; df["frailty"] = frailty
    for k in range(14): df[f"noise_{k}"] = nz[k]
    for k in range(3): df[f"unif_{k}"] = u[k]
    # hidden deterioration schedule: unit-exponential budgets + severity jumps
    for k in range(D["max_events"]):
        df[f"det_E{k}"] = rng.exponential(1.0, n)
        df[f"det_dz{k}"] = rng.uniform(*D["dz_range"], n)
    return df


def generate(cfg=None, out=DATA):
    c = cfg or load_config()
    seed = c["seed"]; C = c["hospital"]["icu_capacity"]; win_h = c["window"]["duration_min"] / 60.0
    slope = c["data"]["arrival_intensity_slope"]; lo, hi = c["data"]["occupant_elapsed_frac"]
    names = list(c["data"]["scenarios"]); probs = np.array([c["data"]["scenarios"][s]["prob"] for s in names])
    nE = c["data"]["n_episodes"]; sp = c["data"]["split"]
    split = np.array(["train"] * sp["train"] + ["val"] * sp["val"] + ["test"] * sp["test"])
    assert len(split) == nE
    meta, pats = [], []
    master = np.random.default_rng(seed)
    perm = master.permutation(nE); split = split[np.argsort(perm)]  # random assignment of episodes to splits
    for ep in range(nE):
        rng = np.random.default_rng([seed, ep])
        slot = ep % c["window"]["slots_per_day"]; w0 = slot * c["window"]["duration_min"]    # minutes after midnight
        hour = (w0 + c["window"]["duration_min"] / 2) / 60.0
        d = np.cos(2 * np.pi * (hour - 15.0) / 24.0)                                           # +1 at 15:00, -1 at 03:00
        pr = probs * np.array([1 - c["window"]["diurnal_congestion_shift"] * d if n == "low" else 1 + c["window"]["diurnal_congestion_shift"] * d if n == "high" else 1.0 for n in names])
        sc = names[rng.choice(len(names), p=pr / pr.sum())]; S = c["data"]["scenarios"][sc]
        free = int(rng.integers(S["free_beds"][0], S["free_beds"][1] + 1)); n_occ = C - free
        n_arr = max(1, int(rng.poisson(S["arrival_mean"] * (1 + c["window"]["diurnal_amplitude"] * d))))
        # NHPP with intensity ~ 1 + slope*t/T by rejection sampling of n_arr order-statistic times
        ts = []
        while len(ts) < n_arr:
            t = rng.uniform(0, win_h)
            if rng.random() < (1 + slope * t / win_h) / (1 + slope): ts.append(t)
        ts = np.sort(ts)
        occ = make_patients(rng, n_occ, c, "occupant", ep, 0)
        arr = make_patients(rng, n_arr, c, "arrival", ep, 0)
        frac = rng.uniform(lo, hi, n_occ)
        occ["elapsed_icu_h"] = occ["actual_los_total_h"] * frac
        occ["actual_remaining_h"] = np.maximum(occ["actual_los_total_h"] * (1 - frac), 0.5)
        occ["icu_admission_time_h"] = -occ["elapsed_icu_h"]
        arr["elapsed_icu_h"] = 0.0; arr["actual_remaining_h"] = arr["actual_los_total_h"]
        arr["icu_request_time_h"] = ts; arr["inter_arrival_h"] = np.diff(ts, prepend=0.0)
        arr["window_start_h"] = 0.0; arr["window_end_h"] = win_h; arr["window_start_clock"] = f"{w0 // 60:02d}:{w0 % 60:02d}"
        meta.append(dict(episode_id=ep, scenario_id=f"day{ep:04d}", congestion=sc, split=split[ep],
                         slot=slot, window_start=f"{w0 // 60:02d}:{w0 % 60:02d}", window_end=f"{(w0 + 30) // 60 % 24:02d}:{(w0 + 30) % 60:02d}", window_duration_min=c["window"]["duration_min"],
                         icu_capacity=C, initial_occupancy=n_occ, initial_free_beds=free, n_cohort=n_arr, seed=f"{seed}-{ep}"))
        pats += [occ, arr]
    df = pd.concat(pats, ignore_index=True)
    df["split"] = split[df["episode_id"].values]
    # SOFA from raw (separate module)
    comp = sofa_components(df)
    for col in COMPONENTS + ["sofa_total"]: df[col] = comp[col].values
    # predictors (train-only fit, OOF on train)
    pt, pr, ph, P = oof_predictions(df, (df["split"] == "train").values, seed=seed)
    df["pred_duration_h"] = np.where(df["role"] == "occupant", pr, pt)
    df["pred_total_los_h"] = pt; df["pred_hazard"] = ph
    P.save(CKPT / "predictors.pkl")
    # ---- write observable / hidden files
    ob, hd = out / "observable", out / "hidden"; ob.mkdir(parents=True, exist_ok=True); hd.mkdir(parents=True, exist_ok=True)
    key = ["patient_id", "episode_id", "role", "split"]
    snap_cols = key + ["age"] + RAW_COLS + COMPONENTS + ["sofa_total"]
    occ_df, arr_df = df[df.role == "occupant"], df[df.role == "arrival"]
    occ_df[key + ["elapsed_icu_h", "icu_admission_time_h"]].to_csv(ob / "initial_icu_census.csv", index=False)
    arr_df[key + ["icu_request_time_h", "inter_arrival_h", "window_start_h", "window_end_h", "window_start_clock"]].to_csv(ob / "cohort_arrivals.csv", index=False)
    df[snap_cols].to_csv(ob / "patient_clinical_snapshots.csv", index=False)
    df[key + ["pred_duration_h", "pred_hazard"]].to_csv(ob / "predicted_features.csv", index=False)
    df[key + ["actual_los_total_h", "actual_remaining_h"]].to_csv(hd / "actual_icu_durations.csv", index=False)
    hcols = key + ["z_latent", "frailty", "true_hazard"] + [f"noise_{k}" for k in range(14)] + [f"unif_{k}" for k in range(3)]
    df[hcols].to_csv(hd / "true_deterioration_parameters.csv", index=False)
    evt = [f"det_E{k}" for k in range(c["data"]["deterioration"]["max_events"])] + [f"det_dz{k}" for k in range(c["data"]["deterioration"]["max_events"])]
    df[key + evt + ["icu_request_time_h", "actual_remaining_h"]].to_csv(hd / "event_schedule.csv", index=False)
    df.to_pickle(DATA.parent.parent / "data" / "processed" / "full_patients.pkl")
    pd.DataFrame(meta).to_csv(out / "episode_metadata.csv", index=False)
    info = dict(note="SYNTHETIC data; not MIMIC-IV; no real patient records", config=c, n_patients=len(df), n_episodes=nE,
                features=dict(raw=RAW_COLS, state_features=["sofa", "expected_icu_duration_h", "waiting_time_h", "pred_hazard_per_h", "status"]),
                hidden_columns=["actual_*", "true_hazard", "z_latent", "noise_*", "det_*"])
    (out / "dataset_metadata.json").write_text(json.dumps(info, indent=2, default=str))
    return df, pd.DataFrame(meta)


if __name__ == "__main__":
    df, meta = generate()
    print(meta.groupby(["split", "congestion"]).size().unstack()); print(len(df), "patients")
    print(df.groupby("role")[["sofa_total", "actual_los_total_h", "pred_duration_h", "true_hazard", "pred_hazard"]].describe().T.to_string())
