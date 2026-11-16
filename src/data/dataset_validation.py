"""Dataset validation + data-leakage audit. Command: python -m src.data.dataset_validation"""
import numpy as np
import pandas as pd

from ..common import DATA, load_config
from .sofa_calculator import sofa_components, COMPONENTS
from .synthetic_generator import generate
from ..prediction.los_predictor import feature_frame
from sklearn.metrics import r2_score


def validate(verbose=True):
    c = load_config(); df = pd.read_pickle(DATA.parent / "processed" / "full_patients.pkl")
    meta = pd.read_csv(DATA / "episode_metadata.csv"); C = c["hospital"]["icu_capacity"]; win = c["window"]["duration_min"] / 60
    chk = {}
    comp = sofa_components(df)
    chk["sofa_total_in_0_24"] = bool(df.sofa_total.between(0, 24).all())
    chk["sofa_components_in_range"] = bool(all(df[k].between(0, 4).all() for k in COMPONENTS))
    chk["sofa_matches_raw_recalc"] = bool((comp.sofa_total.values == df.sofa_total.values).all())
    chk["sofa_is_sum_of_components"] = bool((df[COMPONENTS].sum(axis=1) == df.sofa_total).all())
    ranges = dict(map=(40, 120), gcs=(3, 15), spo2=(70, 100), heart_rate=(40, 190), temperature=(33, 41), pf_ratio=(55, 520),
                  platelets=(5, 600), creatinine=(0.3, 12), bilirubin=(0.2, 30))
    chk["raw_clinically_plausible"] = bool(all(df[k].between(*v).all() for k, v in ranges.items()))
    chk["los_positive"] = bool((df.actual_los_total_h > 0).all() and (df.actual_remaining_h > 0).all())
    arr = df[df.role == "arrival"]; occ = df[df.role == "occupant"]
    chk["arrivals_inside_window"] = bool(((arr.icu_request_time_h >= 0) & (arr.icu_request_time_h < win)).all())
    chk["arrivals_sorted_per_episode"] = bool(arr.groupby("episode_id").icu_request_time_h.apply(lambda s: s.is_monotonic_increasing).all())
    chk["occupants_admitted_before_start"] = bool((occ.icu_admission_time_h < 0).all())
    chk["occupancy_leq_capacity"] = bool((meta.initial_occupancy <= C).all())
    chk["free_beds_nonnegative"] = bool((meta.initial_free_beds >= 0).all())
    chk["cohort_count_matches_meta"] = bool((arr.groupby("episode_id").size().values == meta.n_cohort.values).all())
    chk["cohort_count_variable"] = bool(meta.n_cohort.nunique() > 3)
    chk["all_48_daily_windows_present"] = bool(set(meta.slot) == set(range(c["window"]["slots_per_day"])))
    st = pd.to_datetime(meta.window_start, format="%H:%M"); en = pd.to_datetime(meta.window_end, format="%H:%M")
    chk["every_window_is_30_min"] = bool((((en - st).dt.total_seconds() % 86400) == 1800).all())
    chk["ids_unique"] = bool(df.patient_id.is_unique)
    obs_files = [DATA / "observable" / f for f in ["initial_icu_census.csv", "cohort_arrivals.csv", "patient_clinical_snapshots.csv", "predicted_features.csv"]]
    hidden = ("actual_", "true_hazard", "z_latent", "noise_", "det_", "frailty", "unif_")
    chk["observable_files_hide_actual_los_and_latents"] = bool(all(not any(col.startswith(hidden) for col in pd.read_csv(f, nrows=1).columns) for f in obs_files))
    # leakage audit: predictions must be imperfect and not equal to actual
    chk["pred_not_equal_actual"] = bool((np.abs(df.pred_total_los_h - df.actual_los_total_h) > 1e-6).all())
    te = df[(df.split == "test") & (df.role == "arrival")]          # agent-relevant: total stay of waiting candidates
    r2 = r2_score(np.log(te.actual_los_total_h), np.log(te.pred_total_los_h))
    r2h = r2_score(np.log(te.true_hazard), np.log(te.pred_hazard))
    to = df[(df.split == "test") & (df.role == "occupant")]
    r2o = r2_score(np.log(to.actual_remaining_h), np.log(to.pred_duration_h))
    chk["pred_has_signal_but_not_perfect"] = bool(0.1 < r2 < 0.95)
    chk["splits_disjoint_by_episode"] = bool(df.groupby("episode_id").split.nunique().max() == 1)
    stats = dict(test_r2_log_los_arrivals=float(r2), test_r2_log_remaining_census=float(r2o), test_r2_log_hazard_arrivals=float(r2h),
                 test_corr_sofa_loglos=float(np.corrcoef(te.sofa_total, np.log(te.actual_los_total_h))[0, 1]),
                 n_patients=len(df), n_episodes=len(meta), mean_sofa=float(df.sofa_total.mean()))
    # reproducibility: regenerate a few episodes' latent draws with fixed seed
    from .synthetic_generator import make_patients
    a = make_patients(np.random.default_rng([c["seed"], 5]), 5, c, "arrival", 5, 0); b = make_patients(np.random.default_rng([c["seed"], 5]), 5, c, "arrival", 5, 0)
    chk["fixed_seed_reproducible"] = bool(a.equals(b))
    if verbose:
        for k, v in chk.items(): print(("PASS " if v else "FAIL ") + k)
        print(stats)
    return chk, stats


if __name__ == "__main__":
    chk, stats = validate()
    import json
    (DATA / "validation_report.json").write_text(json.dumps(dict(checks=chk, stats=stats), indent=2))
    raise SystemExit(0 if all(chk.values()) else 1)
