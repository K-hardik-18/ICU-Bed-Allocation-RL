"""Experiment 3: selected algorithm trained with R1 vs R2 (+ alpha/beta sensitivity), identical test episodes.
Command: python -m src.evaluation.reward_comparison --algo ppo [--sensitivity]"""
import argparse
import pandas as pd

from ..common import load_config, RES
from .evaluate_agents import evaluate
from .metrics import METRICS, summarize

SENS = [(1.0, 0.25), (0.25, 1.0), (1.0, 0.0), (0.0, 1.0)]   # (alpha, beta); default (1,1) is the main R2 run


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--algo", required=True); ap.add_argument("--sensitivity", action="store_true"); a = ap.parse_args()
    cfg = load_config(); S = cfg["train"]["seeds"]; parts = []
    for tr in ("r1", "r2"):
        for ev in ("r1", "r2"):
            d = evaluate(ev, algos=(a.algo,), train_reward=tr, include_baselines=(tr == "r1"))
            d["train_reward"] = d["train_reward"].fillna("baseline"); d["eval_reward"] = ev; parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    # keep one copy of baselines (they do not depend on training reward) per eval reward
    df.to_csv(RES / "csv" / f"reward_comparison_{a.algo}.csv", index=False)
    cols = list(METRICS)
    pol = df[df.algo == a.algo].copy(); pol["algo"] = pol["train_reward"].map({"r1": "trained_R1", "r2": "trained_R2"})
    pol = pol[pol.eval_reward == "r1"]  # task metrics identical across eval rewards; returns reported below
    mu, sd = summarize(pol, cols); mu.to_csv(RES / "tables" / "exp3_means.csv"); sd.to_csv(RES / "tables" / "exp3_sds.csv")
    print(mu.round(3).T.to_string()); print(sd.round(3).T.to_string())
    ret = df[df.algo == a.algo].groupby(["train_reward", "eval_reward"]).ep_return.mean().unstack(); ret.to_csv(RES / "tables" / "exp3_cross_returns.csv"); print(ret)
    if a.sensitivity:
        rows = []
        for al, be in SENS:
            tags = {(a.algo, s): f"sens_{a.algo}_a{al}_b{be}_s{s}" for s in S[:3]}
            d = evaluate("r2", algos=(a.algo,), train_reward="r2", seeds=S[:3], tags=tags, include_baselines=False)
            d["alpha"] = al; d["beta"] = be; rows.append(d)
        D = pd.concat(rows); D.to_csv(RES / "csv" / f"reward_sensitivity_{a.algo}.csv", index=False)
        print(D.groupby(["alpha", "beta"])[["total_wait", "high_sofa_wait", "sev_wait_cost", "n_det"]].mean().round(3))


if __name__ == "__main__":
    main()
