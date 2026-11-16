"""Dataset figures (1-10). Command: python -m src.visualization.dataset_plots"""
import numpy as np
import pandas as pd
from .style import *
from ..common import DATA, load_config
from ..data.sofa_calculator import COMPONENTS


def main():
    df = pd.read_pickle(DATA.parent / "processed" / "full_patients.pkl"); meta = pd.read_csv(DATA / "episode_metadata.csv")
    arr, occ = df[df.role == "arrival"], df[df.role == "occupant"]
    # 1 raw clinical
    fig, ax = plt.subplots(2, 3, figsize=(WIDE, 3.6))
    for a, (c, t, b) in zip(ax.ravel(), [("pf_ratio", "PaO$_2$/FiO$_2$ (mmHg)", 40), ("platelets", "Platelets (10$^3$/$\\mu$L)", 40), ("bilirubin", "Bilirubin (mg/dL)", 40),
                                          ("map", "MAP (mmHg)", 40), ("gcs", "GCS", 13), ("creatinine", "Creatinine (mg/dL)", 40)]):
        a.hist(df[c], bins=b, color=PAL["blue"], alpha=.85); a.set_xlabel(t); a.set_ylabel("Patients")
    fig.tight_layout(); save(fig, "raw_clinical_distributions.png")
    # 2 SOFA
    fig, a = plt.subplots(figsize=(COL, 2.4)); bins = np.arange(-.5, 25.5)
    a.hist([arr.sofa_total, occ.sofa_total], bins=bins, density=True, label=["Cohort arrivals", "Initial ICU census"], color=[PAL["orange"], PAL["blue"]])
    a.set_xlabel("Total SOFA (computed from raw variables)"); a.set_ylabel("Density"); a.legend(); save(fig, "sofa_distribution.png")
    # 3 components
    fig, ax = plt.subplots(2, 3, figsize=(WIDE, 3.3), sharey=True)
    for a, c in zip(ax.ravel(), COMPONENTS):
        v = df[c].value_counts(normalize=True).reindex(range(5), fill_value=0); a.bar(v.index, v.values, color=PAL["green"])
        a.set_title(c.replace("sofa_", "").capitalize()); a.set_xlabel("Component score")
    ax[0, 0].set_ylabel("Fraction"); ax[1, 0].set_ylabel("Fraction"); fig.tight_layout(); save(fig, "sofa_components.png")
    # 4 LOS
    fig, a = plt.subplots(figsize=(COL, 2.4)); b = np.logspace(0, 2.7, 45)
    a.hist([arr.actual_los_total_h, occ.actual_remaining_h], bins=b, density=True, label=["Arrivals: total ICU stay", "Census: remaining stay"], color=[PAL["orange"], PAL["blue"]])
    a.set_xscale("log"); a.set_xlabel("Actual ICU duration (h, hidden from agent)"); a.set_ylabel("Density"); a.legend(); save(fig, "los_distribution.png")
    # 5 predicted vs actual
    te = df[(df.split == "test") & (df.role == "arrival")]; fig, a = plt.subplots(figsize=(COL, 2.8))
    a.scatter(te.actual_los_total_h, te.pred_total_los_h, s=3, alpha=.25, color=PAL["blue"]); lim = [1, 400]
    a.plot(lim, lim, "k--", lw=.8, label="y = x"); a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lim); a.set_ylim(3, 300)
    from scipy.stats import spearmanr
    rho = spearmanr(te.actual_los_total_h, te.pred_total_los_h)[0]; bias = np.exp(np.median(np.log(te.pred_total_los_h / te.actual_los_total_h)))
    a.set_xlabel("Actual total ICU stay (h)"); a.set_ylabel("Predicted total ICU stay (h)"); a.set_title(f"Test arrivals: Spearman {rho:.2f}; median over-estimate {bias:.1f}x", fontsize=7.5); a.legend(); save(fig, "los_pred_vs_actual.png")
    # 6 arrival times
    fig, a = plt.subplots(figsize=(COL, 2.3)); a.hist(arr.icu_request_time_h * 60, bins=15, color=PAL["orange"]); a.set_xlim(0, 30)
    a.set_xlabel("ICU request time within window (min after window start)"); a.set_ylabel("Arrivals"); save(fig, "patient_arrival_distribution.png")
    # 7 cohort size
    fig, a = plt.subplots(figsize=(COL, 2.4)); mx = meta.n_cohort.max()
    cols = dict(low=PAL["green"], medium=PAL["orange"], high=PAL["red"]); bot = np.zeros(mx + 1)
    for k in ["low", "medium", "high"]:
        h = np.bincount(meta[meta.congestion == k].n_cohort, minlength=mx + 1); a.bar(range(mx + 1), h, bottom=bot, label=k, color=cols[k]); bot += h
    a.set_xlabel("Cohort arrivals per 30-min window"); a.set_ylabel("Episodes"); a.legend(title="Scenario"); save(fig, "cohort_arrivals_per_episode.png")
    # 8/9 occupancy and free beds
    fig, ax = plt.subplots(1, 2, figsize=(WIDE, 2.3))
    ax[0].hist(meta.initial_occupancy, bins=np.arange(13.5, 21.5), color=PAL["blue"]); ax[0].set_xlabel("Initially occupied beds (capacity 20)"); ax[0].set_ylabel("Episodes")
    ax[1].hist(meta.initial_free_beds, bins=np.arange(-.5, 7.5), color=PAL["green"]); ax[1].set_xlabel("Initially available beds"); fig.tight_layout(); save(fig, "initial_icu_occupancy.png")
    # 10 hazard
    fig, a = plt.subplots(figsize=(COL, 2.4)); b = np.linspace(0, .5, 40)
    a.hist([arr.true_hazard.clip(upper=.5), arr.pred_hazard.clip(upper=.5)], bins=b, density=True, label=["True (hidden)", "Predicted (observed)"], color=[PAL["red"], PAL["blue"]])
    a.set_xlabel("Deterioration hazard while waiting (per hour)"); a.set_ylabel("Density"); a.legend(); save(fig, "hazard_distribution.png")

    # 11 windows through the day
    g = meta.groupby("slot")[["n_cohort", "initial_free_beds"]].mean(); fig, a = plt.subplots(figsize=(COL * 1.15, 2.5)); hrs = g.index * 0.5
    a.plot(hrs, g.n_cohort, "o-", ms=3, color=PAL["orange"], label="Cohort arrivals / window"); a.set_xlabel("Window start (hour of day)"); a.set_ylabel("Mean cohort arrivals")
    b = a.twinx(); b.plot(hrs, g.initial_free_beds, "s-", ms=3, color=PAL["blue"], label="Initially free beds"); b.set_ylabel("Mean free beds"); b.grid(False); b.spines["right"].set_visible(True)
    a.set_xticks(range(0, 25, 4)); h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0), frameon=False); save(fig, "windows_through_day.png")


if __name__ == "__main__":
    main()
