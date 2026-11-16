"""Experiment 1/2 analysis: convergence stats, test table, selection criteria. Command: python -m src.evaluation.analyze_exp1"""
import json
import numpy as np
import pandas as pd

from ..common import RES, load_config
from ..environment.icu_environment import ICUEnv, load_episodes
from ..agents.heuristics import RandomAgent
from ..training.common_trainer import run_policy
from .metrics import *


def load_logs(reward="r1", algos=("dqn", "ddqn", "ppo"), seeds=None):
    cfg = load_config(); seeds = seeds or cfg["train"]["seeds"]
    out = []
    for a in algos:
        for s in seeds:
            l = pd.read_csv(RES / "logs" / f"{a}_{reward}_s{s}.csv"); l["algo"] = a; l["seed"] = s; out.append(l)
    return pd.concat(out, ignore_index=True)


def random_reference(reward="r1"):
    cfg = load_config(); meta, g = load_episodes("val"); ids = list(meta.episode_id)[:cfg["train"]["eval_episodes"]]
    env = ICUEnv(g, meta, cfg, reward)
    m = pd.concat([run_policy(RandomAgent(s).act, env, ids) for s in cfg["eval"]["random_seeds"]])
    return m.ep_return.mean(), m.ep_return.std(), m.total_wait.mean()


def main():
    cfg = load_config(); L = load_logs("r1"); ref, ref_sd, ref_tw = random_reference("r1")
    test = pd.read_csv(RES / "csv" / "test_eval_r1.csv")
    rows = []
    for a in ("dqn", "ddqn", "ppo"):
        conv = [convergence_episode(L[(L.algo == a) & (L.seed == s)], ref) for s in cfg["train"]["seeds"]]
        tt = [L[(L.algo == a) & (L.seed == s)].train_time_s.iloc[-1] for s in cfg["train"]["seeds"]]
        ts = per_seed(test[test.algo == a], ["total_wait"]).total_wait.values
        fin = [L[(L.algo == a) & (L.seed == s)].val_return.tail(5).mean() for s in cfg["train"]["seeds"]]
        rows.append(dict(algo=a, conv_ep_mean=np.nanmean(conv), conv_ep_sd=np.nanstd(conv, ddof=1), conv_n=int(np.sum(~np.isnan(conv))),
                         train_time_mean=np.mean(tt), train_time_sd=np.std(tt, ddof=1), test_total_wait_mean=ts.mean(), test_total_wait_sd=ts.std(ddof=1),
                         val_plateau_mean=np.mean(fin), val_plateau_sd=np.std(fin, ddof=1)))
    S = pd.DataFrame(rows); S.to_csv(RES / "tables" / "exp1_selection_criteria.csv", index=False)
    print("random val ref return", ref, "+-", ref_sd, "total_wait", ref_tw); print(S.round(3).to_string())
    cols = list(METRICS)
    mu, sd = summarize(test, cols); mu.to_csv(RES / "tables" / "exp1_test_means.csv"); sd.to_csv(RES / "tables" / "exp1_test_sds.csv")
    print(mu[cols[:13]].round(3).to_string()); print(sd[["total_wait", "mean_wait", "ep_return"]].round(3).to_string())
    pr = []
    for a in ("dqn", "ddqn", "ppo", "spf_heuristic", "sofa_heuristic"):
        for col in ("total_wait", "high_sofa_wait", "sev_wait_cost"):
            m, lo, hi = paired_diff(test, a, "random", col); pr.append(dict(algo=a, metric=col, diff_vs_random=m, ci_lo=lo, ci_hi=hi))
    P = pd.DataFrame(pr); P.to_csv(RES / "tables" / "exp1_paired_vs_random.csv", index=False); print(P.round(3).to_string())
    json.dump(dict(random_val_ref=ref), open(RES / "tables" / "random_val_ref.json", "w"))


if __name__ == "__main__":
    main()
