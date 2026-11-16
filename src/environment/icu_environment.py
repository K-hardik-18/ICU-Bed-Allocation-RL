"""Event-driven single-hospital, single-ICU bed-allocation environment.

One episode = one fixed 30-minute arrival window (cohort) + follow-up until the cohort is admitted
(or max horizon). Decisions occur only at event timestamps when >=1 bed is free and >=1 patient is waiting.
Event order at equal timestamps: discharge(0) < arrival(1) < deterioration(2) < window-end(3).
Statuses: 0 not arrived, 1 waiting, 2 in ICU, 3 discharged.
Action = row index of a waiting patient (ADMIT) or N (= STOP); sequential masked selection.
"""
import heapq
import numpy as np
import pandas as pd

from ..common import load_config, DATA, CKPT
from ..data.sofa_calculator import sofa_components
from ..data.synthetic_generator import raw_from_latent, true_hazard
from ..prediction.los_predictor import Predictors, feature_frame, RAW_COLS
from .reward_functions import interval_cost, terminal_cost

NOT_ARRIVED, WAITING, ICU, DONE = 0, 1, 2, 3
P_DISCH, P_ARR, P_DET, P_END = 0, 1, 2, 3


class Pt:
    __slots__ = ("pid", "role", "status", "age", "z", "nz", "u", "frailty", "sofa", "pred_dur", "pred_haz", "t_req", "t_adm",
                 "t_dis", "los", "E", "dz", "ndet", "ver", "haz_true", "sofa0", "raw")


def load_episodes(split=None, data_pickle=None):
    df = pd.read_pickle(data_pickle or DATA.parent / "processed" / "full_patients.pkl")
    meta = pd.read_csv(DATA / "episode_metadata.csv")
    if split is not None:
        meta = meta[meta.split == split]
    groups = {e: g for e, g in df.groupby("episode_id") if e in set(meta.episode_id)}
    return meta.reset_index(drop=True), groups


class ICUEnv:
    def __init__(self, groups, meta, cfg=None, reward="r1", predictors=None, reward_params=None):
        self.c = cfg or load_config(); self.groups = groups
        self.meta = meta.set_index("episode_id")
        self.C = self.c["hospital"]["icu_capacity"]
        self.win = self.c["window"]["duration_min"] / 60.0
        self.horizon = self.c["window"]["max_horizon_h"]
        e = self.c["env"]; self.wref = e["w_ref_h"]; self.dscale = e["dur_scale_h"]; self.hscale = e["hazard_scale"]
        self.gamma = e["gamma"]; self.tau0 = e["tau0_h"]
        self.kind = reward
        rp = dict(self.c["reward"][reward]); self.reward_scale = rp.pop("scale"); self.lam_u = rp.pop("lambda_u")
        if reward_params: rp.update(reward_params)
        rp["wref"] = self.wref; self.rp = rp
        self.pred = predictors or Predictors.load(CKPT / "predictors.pkl")
        self.high_thr = e["high_sofa_threshold"]
        self.cap_violations = 0

    # ------------------------------------------------------------------ setup
    def reset(self, episode_id):
        g = self.groups[episode_id]; self.ep = episode_id
        self.t = 0.0; self.pts = []; self.heap = []; self.seq = 0; self.skip = False; self.done = False
        self.occ_area = 0.0; self.cost_total = 0.0; self.ret = 0.0; self.n_det = 0; self.unresolved = 0
        self.history = []; self.timeline = []; self.cap_violations = 0
        for r in g.itertuples(index=False):
            p = Pt(); p.pid = r.patient_id; p.role = r.role; p.age = r.age; p.z = r.z_latent
            p.nz = np.array([getattr(r, f"noise_{k}") for k in range(14)]); p.u = np.array([getattr(r, f"unif_{k}") for k in range(3)])
            p.frailty = r.frailty; p.los = r.actual_los_total_h; p.ndet = 0; p.ver = 0
            p.E = [getattr(r, f"det_E{k}") for k in range(self.c["data"]["deterioration"]["max_events"])]
            p.dz = [getattr(r, f"det_dz{k}") for k in range(self.c["data"]["deterioration"]["max_events"])]
            p.sofa = float(r.sofa_total); p.sofa0 = p.sofa; p.pred_dur = float(r.pred_duration_h); p.pred_haz = float(r.pred_hazard)
            p.haz_true = float(r.true_hazard); p.t_adm = None; p.t_dis = None; p.raw = None
            if r.role == "occupant":
                p.status = ICU; p.t_req = None; p.t_adm = -r.elapsed_icu_h; p.t_dis = r.actual_remaining_h
                self._push(p.t_dis, P_DISCH, p, 0)
            else:
                p.status = NOT_ARRIVED; p.t_req = r.icu_request_time_h
                self._push(p.t_req, P_ARR, p, 0)
            self.pts.append(p)
        self.n_cohort = sum(p.role == "arrival" for p in self.pts)
        self.window_open = True
        self.timeline.append((0.0, self.occupied(), len(self.waiting())))  # state at window start
        heapq.heappush(self.heap, (self.win, P_END, self.seq, None, 0)); self.seq += 1
        _, self._pending = self._advance(initial=True)   # policy-independent pre-decision cost
        return self.observe()

    def _push(self, t, prio, p, ver):
        heapq.heappush(self.heap, (t, prio, self.seq, p, ver)); self.seq += 1

    # ------------------------------------------------------------------ properties
    def occupied(self): return sum(p.status == ICU for p in self.pts)
    def free_beds(self): return self.C - self.occupied()
    def waiting(self): return [p for p in self.pts if p.status == WAITING]

    # ------------------------------------------------------------------ observation
    def active(self):
        """Active rows: ICU occupants then waiting patients (stable order by patient id)."""
        return [p for p in self.pts if p.status in (ICU, WAITING)]

    def observe(self):
        rows = self.active(); n = len(rows)
        x = np.zeros((n, 4), np.float32); st = np.zeros(n, np.int64); mask = np.zeros(n + 1, bool)
        beds = self.free_beds()
        for i, p in enumerate(rows):
            w = (self.t - p.t_req) if p.status == WAITING else 0.0
            x[i] = (p.sofa / 24.0, min(p.pred_dur / self.dscale, 5.0), min(w / self.wref, 3.0),
                    (min(p.pred_haz / self.hscale, 3.0) if p.status == WAITING else 0.0))
            st[i] = p.status
            if p.status == WAITING and beds > 0: mask[i] = True
        mask[n] = bool(mask[:n].any())  # STOP allowed whenever a decision is offered
        return dict(x=x, status=st - 1, mask=mask, free=np.float32(beds / self.C), n=n, ids=[p.pid for p in rows],
                    t=self.t, done=self.done)

    # ------------------------------------------------------------------ dynamics
    def _recompute(self, p):
        raw = raw_from_latent(np.array([p.z]), np.array([p.age]), p.nz[:, None], p.u[:, None])
        df = pd.DataFrame({k: np.asarray(v) for k, v in raw.items()}); df["age"] = p.age
        F = feature_frame(df)
        p.sofa = float(F["sofa_total"].iloc[0]); p.raw = df
        p.pred_dur = float(self.pred.predict_total(F)[0]); p.pred_haz = float(self.pred.predict_hazard(F)[0])
        p.haz_true = float(true_hazard(p.z, p.age, df["spo2"].iloc[0], p.frailty, self.c))

    def _schedule_det(self, p):
        if p.ndet < len(p.E):
            self._push(self.t + p.E[p.ndet] / p.haz_true, P_DET, p, p.ver)

    def _cost(self, dt):
        q = self.waiting()
        s = np.array([p.sofa for p in q]); w0 = np.array([self.t - p.t_req for p in q])
        return interval_cost(self.kind, s, w0, dt, self.rp)

    def _process_batch(self, t_next):
        self.t = t_next
        while self.heap and self.heap[0][0] <= t_next + 1e-12:
            _, prio, _, p, ver = heapq.heappop(self.heap)
            if prio == P_DISCH:
                p.status = DONE; p.t_dis = self.t
            elif prio == P_ARR:
                p.status = WAITING
                self._schedule_det(p)
            elif prio == P_DET:
                if p.status != WAITING or ver != p.ver: continue
                p.z = min(p.z + p.dz[p.ndet], 1.0); p.ndet += 1; p.ver += 1; self.n_det += 1
                self._recompute(p); self._schedule_det(p)
            else:
                self.window_open = False
        if self.occupied() > self.C: self.cap_violations += 1

    def _terminal(self):
        return (not self.window_open) and not self.waiting() and all(p.status != NOT_ARRIVED for p in self.pts)

    def _advance(self, initial=False):
        """Run events until the next decision epoch or termination. Returns (elapsed, cost)."""
        t0 = self.t; cost = 0.0
        while True:
            if self._terminal(): self.done = True; break
            if self.free_beds() > 0 and self.waiting() and not self.skip: break
            nxt = self.heap[0][0] if self.heap else self.horizon + 1
            if nxt > self.horizon:
                dt = self.horizon - self.t
                cost += self._cost(dt); self.occ_area += self.occupied() * dt; self.t = self.horizon
                self.unresolved = len(self.waiting()); cost += terminal_cost(self.unresolved, self.lam_u)
                self.done = True; break
            dt = nxt - self.t
            cost += self._cost(dt); self.occ_area += self.occupied() * dt
            self._process_batch(nxt); self.skip = False
            self.timeline.append((self.t, self.occupied(), len(self.waiting())))
        return self.t - t0, cost

    def step(self, action):
        assert not self.done
        rows = self.active(); n = len(rows)
        obs = self.observe()
        if not obs["mask"][action]: raise ValueError("invalid action")
        if action == n:  # STOP
            self.skip = True
            dt, cost = self._advance()
            self.history.append((self.t, "STOP", None))
        else:
            p = rows[action]
            assert p.status == WAITING and self.free_beds() > 0
            p.status = ICU; p.t_adm = self.t; p.t_dis = self.t + p.los; p.ver += 1
            self._push(p.t_dis, P_DISCH, p, 0)
            self.history.append((self.t, "ADMIT", p.pid))
            self.timeline.append((self.t, self.occupied(), len(self.waiting())))
            if self.occupied() > self.C: self.cap_violations += 1
            if self.free_beds() > 0 and self.waiting():   # same epoch, dt = 0
                dt, cost = 0.0, 0.0
            else:
                dt, cost = self._advance()
        cost += self._pending; self._pending = 0.0
        self.cost_total += cost
        r = -cost * self.reward_scale; self.ret += r
        info = dict(dt=dt, cost=cost)
        if self.done: info["metrics"] = self.metrics()
        return self.observe(), r, self.done, info

    # ------------------------------------------------------------------ metrics
    def metrics(self):
        co = [p for p in self.pts if p.role == "arrival"]
        end = self.t
        W, Wh, adm = [], [], 0
        for p in co:
            if p.t_adm is not None: w = p.t_adm - p.t_req; adm += 1
            else: w = self.horizon - p.t_req
            W.append(w)
            if p.sofa0 >= self.high_thr: Wh.append(w)
        W = np.array(W)
        # severity-weighted waiting cost evaluated post-hoc with SOFA at arrival (policy-independent definition)
        swc = float(sum((p.sofa0 / 24.0 * self.rp.get("alpha", 1.0) + self.rp.get("beta", 1.0) * min(w / self.wref, 1.0)) * w
                        for p, w in zip(co, W)))
        return dict(episode=self.ep, n_cohort=len(co), total_wait=float(W.sum()), mean_wait=float(W.mean()),
                    median_wait=float(np.median(W)), max_wait=float(W.max()), p90_wait=float(np.percentile(W, 90)),
                    p95_wait=float(np.percentile(W, 95)), high_sofa_wait=float(np.mean(Wh)) if Wh else np.nan,
                    n_high=len(Wh), sev_wait_cost=swc, completion=adm / len(co), unresolved=len(co) - adm,
                    n_det=self.n_det, utilization=self.occ_area / (self.C * max(end, 1e-9)),
                    ep_return=self.ret, ep_cost=self.cost_total, end_time=end, violations=self.cap_violations)
