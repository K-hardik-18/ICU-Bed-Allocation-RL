"""RL experiment figures (20-30). Command: python -m src.visualization.training_plots"""
import json
import numpy as np
import pandas as pd
from .style import *
from ..common import RES, load_config
from ..evaluation.analyze_exp1 import load_logs
from ..evaluation.metrics import METRICS, per_seed

ALG = ["dqn", "ddqn", "ppo"]


def convergence(ref):
    L = load_logs("r1")
    one = {}
    for a in ALG:
        fig, ax = plt.subplots(figsize=(COL, 2.5)); g = L[L.algo == a]
        for s, gs in g.groupby("seed"): ax.plot(gs.episode, gs.val_return, color=ALGO_COL[a], alpha=.25, lw=.8)
        m = g.groupby("episode").val_return.agg(["mean", "std"])
        ax.plot(m.index, m["mean"], color=ALGO_COL[a], lw=1.6, label=f"{ALGO_LBL[a]} (mean of 5 seeds)")
        ax.fill_between(m.index, m["mean"] - m["std"], m["mean"] + m["std"], color=ALGO_COL[a], alpha=.15)
        ax.axhline(ref, color=PAL["grey"], ls="--", lw=1, label="Random (reference)")
        ax.set_ylim(ref - 1.5, ref + 0.9); ax.set_xlabel("Training episodes"); ax.set_ylabel("Validation return (Reward 1)"); ax.legend(loc="lower right")
        save(fig, f"convergence_{a}.png")
    fig, ax = plt.subplots(figsize=(COL, 2.7))
    for a in ALG:
        m = L[L.algo == a].groupby("episode").val_return.agg(["mean", "std"])
        ax.plot(m.index, m["mean"], color=ALGO_COL[a], lw=1.5, label=ALGO_LBL[a]); ax.fill_between(m.index, m["mean"] - m["std"], m["mean"] + m["std"], color=ALGO_COL[a], alpha=.15)
    ax.axhline(ref, color=PAL["grey"], ls="--", lw=1, label="Random (reference)")
    ax.set_ylim(ref - 1.5, ref + 0.9); ax.set_xlim(0, L.episode.max())
    ax.set_xlabel("Training episodes"); ax.set_ylabel("Validation return (Reward 1)"); ax.legend(loc="lower right"); save(fig, "convergence_comparison.png")


def bars(test, col, fname, ylabel, algos=("random", "spf_heuristic", "sofa_heuristic", "dqn", "ddqn", "ppo"), pct=False):
    ps = per_seed(test, [col]); g = ps.groupby("algo")[col]
    fig, ax = plt.subplots(figsize=(COL, 2.5)); xs = np.arange(len(algos))
    mu = [g.mean()[a] * (100 if pct else 1) for a in algos]; sd = [np.nan_to_num(g.std(ddof=1).get(a, 0)) * (100 if pct else 1) for a in algos]
    ax.bar(xs, mu, yerr=sd, color=[ALGO_COL[a] for a in algos], capsize=2, error_kw=dict(lw=.8))
    ax.set_xticks(xs); ax.set_xticklabels([ALGO_LBL[a].replace("Shortest-pred.-first", "SPF").replace("Highest-SOFA-first", "SOFA-first") for a in algos], rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if col == "utilization": ax.set_ylim(90, 100)
    if col == "unresolved":
        ax.set_ylim(0, 1); ax.text(0.5, 0.5, "0 unresolved patients for every policy", transform=ax.transAxes, ha="center", fontsize=8)
    save(fig, fname)


def reward_fig(algo="ddqn"):
    mu = pd.read_csv(RES / "tables" / "exp3_means.csv", index_col=0); sd = pd.read_csv(RES / "tables" / "exp3_sds.csv", index_col=0)
    cols = [("total_wait", "Total wait\n(patient-h)"), ("max_wait", "Max wait\n(h)"), ("high_sofa_wait", "High-SOFA\nwait (h)"), ("sev_wait_cost", "Severity-wtd\ncost")]
    fig, ax = plt.subplots(1, 4, figsize=(WIDE, 2.3)); x = np.arange(2)
    for a, (c, t) in zip(ax, cols):
        a.bar(x, [mu.loc["trained_R1", c], mu.loc["trained_R2", c]], yerr=[sd.loc["trained_R1", c], sd.loc["trained_R2", c]], color=[PAL["blue"], PAL["red"]], capsize=2)
        a.set_xticks(x); a.set_xticklabels(["Reward 1", "Reward 2"]); a.set_title(t, fontsize=8)
    fig.tight_layout(); save(fig, "reward_comparison.png")
    S = pd.read_csv(RES / "csv" / f"reward_sensitivity_{algo}.csv"); g = S.groupby(["alpha", "beta"])[["total_wait", "high_sofa_wait"]].mean().reset_index()
    fig, a = plt.subplots(figsize=(COL, 2.6))
    a.scatter(g.total_wait, g.high_sofa_wait, color=PAL["red"], s=25, zorder=3, label="Reward 2 ($\\alpha,\\beta$)")
    for _, r in g.iterrows():
        off = (-8, -11) if (r.alpha, r.beta) == (0.25, 1.0) else (3, 3)   # keep clear of the Reward 1 marker
        a.annotate(f"({r.alpha:g},{r.beta:g})", (r.total_wait, r.high_sofa_wait), fontsize=6, xytext=off, textcoords="offset points")
    t = pd.read_csv(RES / "csv" / "test_eval_r1.csv"); ps = t[t.algo == algo].groupby("seed")[["total_wait", "high_sofa_wait"]].mean()
    a.scatter(ps.total_wait.mean(), ps.high_sofa_wait.mean(), color=PAL["blue"], s=25, marker="s", label="Reward 1")
    a.scatter(mu.loc["trained_R2", "total_wait"], mu.loc["trained_R2", "high_sofa_wait"], color=PAL["red"], s=25, zorder=3)
    a.annotate("(1,1) default", (mu.loc["trained_R2", "total_wait"], mu.loc["trained_R2", "high_sofa_wait"]), fontsize=6, xytext=(3, 3), textcoords="offset points"); a.legend()
    a.set_xlabel("Total waiting (patient-h)"); a.set_ylabel("High-SOFA mean wait (h)"); save(fig, "reward_sensitivity.png")


def main():
    ref = json.load(open(RES / "tables" / "random_val_ref.json"))["random_val_ref"]
    convergence(ref)
    t = pd.read_csv(RES / "csv" / "test_eval_r1.csv")
    bars(t, "total_wait", "cmp_total_wait.png", "Total waiting time (patient-h)")
    bars(t, "mean_wait", "cmp_mean_wait.png", "Mean waiting time (h)")
    bars(t, "high_sofa_wait", "cmp_high_sofa_wait.png", "High-SOFA mean wait (h)")
    bars(t, "completion", "cmp_completion.png", "Admission completion (%)", pct=True)
    bars(t, "unresolved", "cmp_unresolved.png", "Unresolved patients per episode")
    reward_fig()
    bars(t, "utilization", "cmp_utilization.png", "ICU utilisation (%)", pct=True)


if __name__ == "__main__":
    main()
