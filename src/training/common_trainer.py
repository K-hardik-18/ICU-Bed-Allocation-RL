"""Common training / evaluation loops. Same environment, episodes, budget and evaluation protocol for all learners."""
import argparse
import time
import numpy as np
import pandas as pd
import torch

from ..common import load_config, CKPT, RES
from ..environment.icu_environment import ICUEnv, load_episodes
from ..agents.dqn_agent import DQNAgent
from ..agents.ppo_agent import PPOAgent

torch.set_num_threads(1)


def run_policy(act_fn, env, ids):
    rows = []
    for e in ids:
        o = env.reset(e); d = False
        while not d:
            o, r, d, info = env.step(act_fn(o))
        rows.append(info["metrics"])
    return pd.DataFrame(rows)


def make_agent(algo, cfg, seed):
    if algo in ("dqn", "ddqn"): return DQNAgent(cfg, double=(algo == "ddqn"), seed=seed)
    return PPOAgent(cfg, seed=seed)


def greedy_fn(agent):
    return (lambda o: agent.act(o, 0.0)) if isinstance(agent, DQNAgent) else (lambda o: agent.act(o, False))


def train(algo, reward, seed, cfg=None, episodes=None, tag=None, reward_params=None):
    cfg = cfg or load_config(); T = cfg["train"]; episodes = episodes or T["episodes"]
    tag = tag or f"{algo}_{reward}_s{seed}"
    meta_tr, g_tr = load_episodes("train"); meta_va, g_va = load_episodes("val")
    env = ICUEnv(g_tr, meta_tr, cfg, reward, reward_params=reward_params)
    venv = ICUEnv(g_va, meta_va, cfg, reward, reward_params=reward_params)
    train_ids = list(meta_tr.episode_id); val_ids = list(meta_va.episode_id)[:T["eval_episodes"]]
    rng = np.random.default_rng(1000 + seed); agent = make_agent(algo, cfg, seed)
    log = []; steps = 0; t_train = 0.0; t_start = time.time(); ppo_buf = []; d = T["dqn"]
    decay = d["eps_decay_frac"] * episodes
    best = [-1e18]
    def evaluate(ep):
        m = run_policy(greedy_fn(agent), venv, val_ids)
        if ep > 0 and m.ep_return.mean() > best[0]:   # model selection on validation episodes only
            best[0] = m.ep_return.mean(); agent.save(CKPT / f"{tag}.pt")
        log.append(dict(episode=ep, env_steps=steps, train_time_s=t_train, val_return=m.ep_return.mean(),
                        val_total_wait=m.total_wait.mean(), val_sev_cost=m.sev_wait_cost.mean(), val_high_wait=m.high_sofa_wait.mean()))
    evaluate(0)
    for ep in range(1, episodes + 1):
        t0 = time.time(); e = int(rng.choice(train_ids)); o = env.reset(e); done = False; epi = []
        eps = max(d["eps_end"], d["eps_start"] - (d["eps_start"] - d["eps_end"]) * ep / decay)
        while not done:
            if isinstance(agent, DQNAgent):
                a = agent.act(o, eps)
                o2, r, done, info = env.step(a); agent.store(o, a, r, o2, float(done), info["dt"]); agent.update(); o = o2
            else:
                a = agent.act(o, True); lp, v = agent.last()
                o2, r, done, info = env.step(a)
                epi.append(dict(o=o, a=a, lp=lp, v=v, r=r, dt=info["dt"], done=done)); o = o2
            steps += 1
        if not isinstance(agent, DQNAgent):
            ppo_buf.append(epi)
            if len(ppo_buf) >= T["ppo"]["episodes_per_iter"]: agent.update(ppo_buf); ppo_buf = []
        t_train += time.time() - t0
        if ep % T["eval_every"] == 0: evaluate(ep)
    agent.save(CKPT / f"{tag}_final.pt")
    (RES / "logs").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(RES / "logs" / f"{tag}.csv", index=False)
    return agent, pd.DataFrame(log)


def main(algo):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", default="r1"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=None); ap.add_argument("--tag", default=None)
    ap.add_argument("--alpha", type=float, default=None); ap.add_argument("--beta", type=float, default=None)
    a = ap.parse_args()
    rp = {k: v for k, v in dict(alpha=a.alpha, beta=a.beta).items() if v is not None}
    _, log = train(algo, a.reward, a.seed, episodes=a.episodes, tag=a.tag, reward_params=rp or None)
    print(log.tail(3).to_string())
