"""Environment figures (11-19) from real simulated episodes. Command: python -m src.visualization.environment_visualization --algo ppo --seed 0"""
import argparse
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from .style import *
from ..common import load_config
from ..environment.icu_environment import ICUEnv, load_episodes
from ..agents.heuristics import RandomAgent
from ..training.common_trainer import make_agent, greedy_fn
from ..common import CKPT


def box(ax, x, y, w, h, text, fc="#EAF2F8", ec=PAL["black"], fs=7.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, p, q, text=None, off=(0, 0.03), color=PAL["black"], fs=7, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.1, color=color))
    if text: ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha=ha, va="bottom", fontsize=fs, color=color)


def mdp_diagram():
    W, Hh = 12.0, 6.2
    fig, ax = plt.subplots(figsize=(WIDE, WIDE * Hh / W)); ax.set_xlim(0, W); ax.set_ylim(0, Hh); ax.axis("off"); ax.grid(False)
    box(ax, 0.1, 2.0, 2.7, 2.4, "RL AGENT\nDQN / DDQN / PPO\n\nshared patient encoder\nmasked pooling\nscore per patient\n+ STOP score", fc="#FDEBD0", fs=7)
    ax.add_patch(Rectangle((5.0, 0.15), 6.85, 5.9, fc="#F4F6F7", ec=PAL["black"], lw=1.2))
    ax.text(8.425, 5.7, "ICU ENVIRONMENT  (one ICU, C = 20 beds)", ha="center", va="center", fontsize=7.5, fontweight="bold")
    box(ax, 5.2, 3.85, 3.2, 1.45, "30-min arrival window\n(any of 48 per day)\nvariable cohort size\nfuture arrivals hidden", fc="#D6EAF8", fs=6.6)
    box(ax, 8.55, 3.85, 3.1, 1.45, "Waiting queue $Q_t$\n(status 1)\nSOFA recomputed\non deterioration", fc="#FCF3CF", fs=6.6)
    box(ax, 5.2, 1.95, 3.2, 1.5, "Event scheduler\ndischarge < arrival <\ndeterioration < window end", fc="#E8DAEF", fs=6.6)
    box(ax, 8.55, 1.95, 3.1, 1.5, "ICU beds (status 2)\nhidden actual discharge\n$O_t \\leq C$ always", fc="#D5F5E3", fs=6.6)
    box(ax, 5.2, 0.35, 6.45, 1.2, "Follow-up after the window closes until every cohort\npatient is admitted (72 h horizon: unresolved + penalty)", fc="#FADBD8", fs=6.6)
    arrow(ax, (2.8, 3.9), (5.0, 4.0)); ax.text(3.9, 4.15, "action\nADMIT($i$) / STOP", ha="center", va="bottom", fontsize=6.5)
    arrow(ax, (5.0, 3.2), (2.8, 3.2)); ax.text(3.9, 3.27, "state $X_t$, $B_t$, mask", ha="center", va="bottom", fontsize=6.5)
    arrow(ax, (5.0, 2.4), (2.8, 2.4), color=PAL["red"]); ax.text(3.9, 2.47, "reward, $\\Delta t$", ha="center", va="bottom", fontsize=6.5, color=PAL["red"])
    arrow(ax, (6.8, 3.85), (6.8, 3.45)); arrow(ax, (10.1, 3.85), (10.1, 3.45)); arrow(ax, (8.4, 2.7), (8.55, 2.7))
    ax.text(0.1, 0.95, "Semi-Markov: $\\Delta t = t_{next}-t$ is set by\nevent times; discount $\\gamma^{\\Delta t/\\tau_0}$.\nHidden from the agent: actual stays,\ndischarge times, future arrivals,\ntrue hazards.", fontsize=6.3, va="center")
    save(fig, "mdp_diagram.png")


def window_diagram(ep_obj):
    env = ep_obj; fig, ax = plt.subplots(figsize=(WIDE, 2.4)); ax.grid(False)
    co = [p for p in env.pts if p.role == "arrival"]; oc = [p for p in env.pts if p.role == "occupant"]
    T = max(env.t, 1.0) * 1.05
    ax.axvspan(0, 0.5, color=PAL["sky"], alpha=.35); ax.text(0.25, 1.01, "arrival\nwindow", ha="center", va="bottom", fontsize=6.5, transform=ax.get_xaxis_transform())
    ax.axvspan(0.5, T, color=PAL["orange"], alpha=.10); ax.text((0.5 + T) / 2, 1.01, "follow-up: no new cohort arrivals; discharges and admissions continue", ha="center", va="bottom", fontsize=7, transform=ax.get_xaxis_transform())
    for p in co: ax.plot(p.t_req, 1, "v", color=PAL["blue"], ms=6)
    ax.plot([], [], "v", color=PAL["blue"], label="Cohort arrival (request)")
    for p in oc:
        if p.t_dis is not None and p.t_dis <= T: ax.plot(p.t_dis, 0, "s", color=PAL["grey"], ms=5)
    ax.plot([], [], "s", color=PAL["grey"], label="Census patient discharged (bed released)")
    for p in co:
        if p.t_adm is not None: ax.plot(p.t_adm, 2, "o", color=PAL["green"], ms=6)
    ax.plot([], [], "o", color=PAL["green"], label="Cohort admission")
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["Bed releases", "Arrivals", "Admissions"]); ax.set_ylim(-.6, 2.6); ax.set_xlim(0, T)
    ax.set_xlabel("Hours since window start"); ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.22), frameon=False); save(fig, "arrival_window_diagram.png")


def event_timeline(env):
    fig, ax = plt.subplots(figsize=(WIDE, 2.2)); ax.grid(False)
    ev = []
    for p in env.pts:
        if p.role == "arrival": ev.append((p.t_req, "arrival", p.pid))
        if p.t_adm is not None and p.role == "arrival": ev.append((p.t_adm, "admission", p.pid))
        if p.t_dis is not None and p.t_dis <= env.t: ev.append((p.t_dis, "discharge", p.pid))
    ev.sort(); cols = dict(arrival=PAL["blue"], admission=PAL["green"], discharge=PAL["grey"])
    for k, (t, kind, pid) in enumerate(ev[:26]):
        ax.vlines(t, 0, 1, color=cols[kind], lw=1.4); ax.text(t, 1.03 + .09 * (k % 3), pid[-3:], rotation=90, fontsize=5.5, ha="center", va="bottom", color=cols[kind])
    for kind, c in cols.items(): ax.plot([], [], "|", color=c, label=kind.capitalize(), ms=8)
    ax.axvline(0.5, color=PAL["red"], ls="--", lw=1); ax.text(0.5, -.12, "window closes", color=PAL["red"], fontsize=7, ha="left", va="top")
    ax.set_xlim(-.1, min(max(e[0] for e in ev[:26]) * 1.05, env.t)); ax.set_ylim(0, 1.5); ax.set_yticks([]); ax.set_xlabel("Event time (h since window start); inter-event gaps (decision intervals) vary"); ax.legend(ncol=3, loc="upper right")
    save(fig, "event_timeline.png")


def state_matrix(env_cls, groups, meta, ep, fname="state_example.png"):
    """Snapshot of X_t at the first decision epoch after the window closes (or the last decision before)."""
    env = env_cls; o = env.reset(ep); best = None; d = False; ag = RandomAgent(0)
    while not d:
        if o["n"] >= 12 and o["mask"][:-1].sum() >= 2 and best is None and env.t > .2: best = (o, env.t, [(p.pid, p.status, p.t_req) for p in env.active()])
        o, r, d, i = env.step(ag.act(o))
    if best is None:
        o = env.reset(ep); best = (o, env.t, [(p.pid, p.status, p.t_req) for p in env.active()])
    o, t, rows = best
    X = o["x"]; cfg = load_config()["env"]
    tab = []
    pairs = list(zip(rows, X)); icu = [q for q in pairs if q[0][1] == 2]; que = [q for q in pairs if q[0][1] == 1]
    hidden_icu = max(0, len(icu) - 6); pairs = icu[:6] + que
    for (pid, st, treq), x in pairs:
        w = (t - treq) if st == 1 else 0.0
        tab.append([pid[-3:], int(round(x[0] * 24)), f"{x[1] * cfg['dur_scale_h']:.1f}", f"{w:.2f}", f"{x[3] * cfg['hazard_scale']:.3f}", "Queue" if st == 1 else "ICU"])
    if hidden_icu: tab.insert(6, ["...", "", f"+{hidden_icu} more", "ICU rows", "", "ICU"])
    fig, ax = plt.subplots(figsize=(COL * 1.15, 0.2 * len(tab) + 0.6)); ax.axis("off")
    tb = ax.table(cellText=tab, colLabels=["Patient", "SOFA", "Stay (h)", "Wait (h)", "Hazard/h", "Status"], loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(6.5); tb.scale(1, 1.15)
    for (r_, c_), cell in tb.get_celld().items():
        if r_ == 0: cell.set_facecolor("#D6EAF8"); cell.set_text_props(fontweight="bold")
        elif tab[r_ - 1][-1] == "Queue": cell.set_facecolor("#FCF3CF")
    fig.suptitle(f"$X_t$ at t = {t:.2f} h:  $N_t$ = {len(rows)} rows, free beds $B_t$ = {int(round(o['free'] * 20))}", fontsize=8, y=0.97)
    save(fig, fname)


def trajectories(envs, labels, colors, fname_occ="icu_occupancy_over_time.png", fname_q="queue_size_over_time.png"):
    for idx, (fname, ylab) in enumerate([(fname_occ, "Occupied ICU beds $O_t$"), (fname_q, "Waiting patients $|Q_t|$")]):
        fig, ax = plt.subplots(figsize=(COL, 2.4))
        for k, (env, lab, c) in reversed(list(enumerate(zip(envs, labels, colors)))):   # draw the RL policy last, on top
            t, o, q = zip(*env.timeline); y = (o, q)[idx]
            ax.step(t, y, where="post", color=c, lw=1.6 if k == 0 else 1.2, ls="-" if k == 0 else "--", label=lab, zorder=3 if k == 0 else 2)
        if idx == 0: ax.axhline(20, color=PAL["red"], ls="--", lw=1, label="Capacity C = 20"); ax.set_ylim(min(min(e.timeline, key=lambda r: r[1])[1] for e in envs) - 1.5, 21.5)
        ax.axvline(.5, color=PAL["grey"], ls=":", lw=1); ax.set_xlabel("Hours since window start"); ax.set_ylabel(ylab); ax.legend(); save(fig, fname)


def gantt(env, fname="admission_decisions.png"):
    co = sorted([p for p in env.pts if p.role == "arrival"], key=lambda p: p.t_req)
    fig, ax = plt.subplots(figsize=(COL * 1.1, 0.33 * len(co) + 1)); cm = plt.cm.YlOrRd
    for k, p in enumerate(co):
        ax.barh(k, (p.t_adm or env.t) - p.t_req, left=p.t_req, color=cm(p.sofa0 / 24), edgecolor="k", lw=.4)
        if p.t_adm is not None: ax.plot(p.t_adm, k, ">", color=PAL["green"], ms=5)
        ax.text((p.t_adm or env.t) + 0.08, k, f"SOFA {int(p.sofa0)}, pred. {p.pred_dur:.0f} h", va="center", fontsize=5.5)
    ax.set_xlim(0, max((p.t_adm or env.t) for p in co) * 1.45 + 0.5); ax.axvline(.5, color=PAL["grey"], ls=":"); ax.set_yticks(range(len(co))); ax.set_yticklabels([p.pid[-3:] for p in co]); ax.invert_yaxis()
    ax.set_xlabel("Hours since window start (bar = waiting; green arrow = admitted)"); ax.set_ylabel("Cohort patient"); save(fig, fname)


def bed_release(env, fname="bed_release_admission.png"):
    fig, ax = plt.subplots(figsize=(COL * 1.1, 2.6))
    t, o, q = zip(*env.timeline); ax.step(t, o, where="post", color=PAL["blue"], lw=1.3, label="Occupied beds")
    adm = [(p.t_adm, p.pid) for p in env.pts if p.role == "arrival" and p.t_adm is not None]
    rel = [p.t_dis for p in env.pts if p.t_dis is not None and p.t_dis <= env.t and (p.role == "occupant" or p.t_adm is not None)]
    tt = np.array(t); oo = np.array(o)
    for a, _ in adm: ax.plot(a, oo[tt <= a][-1] if (tt <= a).any() else 0, "^", color=PAL["green"], ms=6)
    for r in rel: ax.plot(r, oo[tt <= r][-1] + .3 if (tt <= r).any() else 0, "v", color=PAL["red"], ms=5)
    ax.plot([], [], "^", color=PAL["green"], label="Cohort admission"); ax.plot([], [], "v", color=PAL["red"], label="Discharge / bed release")
    ax.axhline(20, color=PAL["grey"], ls="--", lw=.8); ax.set_ylim(min(o) - 1.5, 21.5); ax.set_xlabel("Hours since window start"); ax.set_ylabel("Occupied beds"); ax.legend(loc="lower right"); save(fig, fname)


def run_episode(env, ep, act):
    o = env.reset(ep); d = False
    while not d: o, r, d, i = env.step(act(o))
    return env


def main(algo="ppo", seed=0, reward="r1"):
    cfg = load_config(); meta, g = load_episodes("test")
    cand = meta[(meta.congestion == "medium") & (meta.n_cohort.between(6, 9)) & (meta.initial_free_beds >= 2)]
    ep = int(cand.episode_id.iloc[0])
    ag = make_agent(algo, cfg, seed); ag.load(CKPT / f"{algo}_{reward}_s{seed}.pt")
    e_rl = run_episode(ICUEnv(g, meta, cfg, reward), ep, greedy_fn(ag)); e_rn = run_episode(ICUEnv(g, meta, cfg, reward), ep, RandomAgent(0).act)
    mdp_diagram(); window_diagram(e_rl); event_timeline(e_rl)
    state_matrix(ICUEnv(g, meta, cfg, reward), g, meta, ep)
    trajectories([e_rl, e_rn], [algo.upper(), "Random"], [ALGO_COL[algo], ALGO_COL["random"]])
    gantt(e_rl); bed_release(e_rl)
    print("example episode", ep, meta.set_index("episode_id").loc[ep, "window_start"], "cohort", len([p for p in e_rl.pts if p.role == 'arrival']), "total wait RL", e_rl.metrics()["total_wait"], "random", e_rn.metrics()["total_wait"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--algo", default="ppo"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--reward", default="r1"); a = ap.parse_args()
    main(a.algo, a.seed, a.reward)
