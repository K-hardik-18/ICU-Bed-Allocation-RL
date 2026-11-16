"""Evaluate all policies on the SAME held-out test episodes. Command: python -m src.evaluation.evaluate_agents --reward r1"""
import argparse
import pandas as pd
import torch

from ..common import load_config, CKPT, RES
from ..environment.icu_environment import ICUEnv, load_episodes
from ..agents.heuristics import RandomAgent, ShortestPredictedFirst, HighestSofaFirst
from ..training.common_trainer import run_policy, make_agent, greedy_fn


def evaluate(reward_env, algos=("dqn", "ddqn", "ppo"), train_reward=None, seeds=None, tags=None, split="test", include_baselines=True, cfg=None):
    cfg = cfg or load_config(); train_reward = train_reward or reward_env
    meta, g = load_episodes(split); ids = list(meta.episode_id); out = []
    env = ICUEnv(g, meta, cfg, reward_env)
    seeds = seeds if seeds is not None else cfg["train"]["seeds"]
    if include_baselines:
        for s in cfg["eval"]["random_seeds"]:
            m = run_policy(RandomAgent(s).act, env, ids); m["algo"] = "random"; m["seed"] = s; out.append(m)
        for ag in (ShortestPredictedFirst(), HighestSofaFirst()):
            m = run_policy(ag.act, env, ids); m["algo"] = ag.name; m["seed"] = 0; out.append(m)
    for a in algos:
        for s in seeds:
            tag = (tags or {}).get((a, s)) or f"{a}_{train_reward}_s{s}"
            ag = make_agent(a, cfg, s)
            ag.load(CKPT / f"{tag}.pt")
            m = run_policy(greedy_fn(ag), env, ids); m["algo"] = a; m["seed"] = s; m["train_reward"] = train_reward; out.append(m)
    df = pd.concat(out, ignore_index=True); df["eval_reward"] = reward_env
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reward", default="r1"); a = ap.parse_args()
    df = evaluate(a.reward)
    (RES / "csv").mkdir(exist_ok=True, parents=True); df.to_csv(RES / "csv" / f"test_eval_{a.reward}.csv", index=False)
    cols = ["total_wait", "mean_wait", "max_wait", "high_sofa_wait", "sev_wait_cost", "completion", "unresolved", "n_det", "utilization", "ep_return", "violations"]
    print(df.groupby("algo")[cols].mean().round(3).to_string())
