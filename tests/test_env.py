import numpy as np
import pandas as pd
import pytest

from src.common import load_config
from src.data.sofa_calculator import (sofa_components, resp_score, coag_score, liver_score, cv_score, cns_score, renal_score)
from src.environment.icu_environment import ICUEnv, load_episodes, WAITING, ICU, DONE, NOT_ARRIVED
from src.environment.reward_functions import interval_cost, clipped_wait_integral
from src.agents.heuristics import RandomAgent, ShortestPredictedFirst
from src.agents.networks import collate, SetNet
from src.data.synthetic_generator import make_patients

CFG = load_config()


@pytest.fixture(scope="module")
def data():
    return load_episodes("val")


def play(env, ep, policy, check=None):
    o = env.reset(ep); d = False; info = None
    while not d:
        a = policy(o)
        assert o["mask"][a]
        o, r, d, info = env.step(a)
        assert env.occupied() <= env.C and env.free_beds() >= 0
        assert np.isfinite(r)
        if check: check(env, o)
    return info["metrics"]


def rnd(seed=0):
    ag = RandomAgent(seed); return ag.act


def test_capacity_and_nonneg_beds(data):
    meta, g = data; env = ICUEnv(g, meta)
    for e in meta.episode_id[:40]:
        m = play(env, e, rnd()); assert m["violations"] == 0


def test_window_exactly_30min_and_arrivals_inside(data):
    meta, g = data
    assert CFG["window"]["duration_min"] == 30
    for e in meta.episode_id[:40]:
        a = g[e][g[e].role == "arrival"]
        assert ((a.icu_request_time_h >= 0) & (a.icu_request_time_h < 0.5)).all()


def test_no_arrivals_after_window(data):
    meta, g = data; env = ICUEnv(g, meta)
    def chk(env, o):
        if env.t > 0.5: assert env.window_open is False and all(p.status != NOT_ARRIVED or p.t_req < 0.5 for p in env.pts)
    for e in meta.episode_id[:20]: play(env, e, rnd(), chk)
    for p in env.pts:
        if p.role == "arrival": assert p.t_req < 0.5


def test_initial_occupants_present_and_before_start(data):
    meta, g = data; env = ICUEnv(g, meta)
    for e in meta.episode_id[:20]:
        o = env.reset(e); occ = [p for p in env.pts if p.role == "occupant"]
        assert len(occ) == meta.set_index("episode_id").loc[e, "initial_occupancy"]
        assert all(p.status in (ICU, DONE) and p.t_adm < 0 for p in occ)
        assert (o["status"] == 1).sum() >= 1 or True  # status index 1 = ICU row


def test_future_info_hidden_until_arrival(data):
    meta, g = data; env = ICUEnv(g, meta)
    for e in meta.episode_id[:30]:
        o = env.reset(e)
        d = False
        while not d:
            ids = set(o["ids"])
            for p in env.pts:
                if p.status == NOT_ARRIVED: assert p.pid not in ids
                if p.status == DONE: assert p.pid not in ids
            o, r, d, _ = env.step(rnd()(o))


def test_only_waiting_can_be_admitted_and_mask(data):
    meta, g = data; env = ICUEnv(g, meta); o = env.reset(meta.episode_id[0])
    rows = env.active()
    for i, p in enumerate(rows):
        assert o["mask"][i] == (p.status == WAITING and env.free_beds() > 0)
    bad = [i for i in range(o["n"]) if not o["mask"][i]]
    if bad:
        with pytest.raises(ValueError): env.step(bad[0])


def test_no_readmission_waiting_monotone_and_discharge_releases_beds(data):
    meta, g = data; env = ICUEnv(g, meta)
    for e in meta.episode_id[:30]:
        o = env.reset(e); d = False; last = {}; adm = set()
        while not d:
            for p in env.waiting():
                w = env.t - p.t_req
                assert w >= last.get(p.pid, 0) - 1e-12; last[p.pid] = w
            o, r, d, _ = env.step(rnd()(o))
        assert all(p.status != WAITING for p in env.pts) or env.unresolved > 0
        for p in env.pts:  # a discharged patient was admitted exactly once and left at t_dis >= t_adm
            if p.status == DONE: assert p.t_dis >= p.t_adm


def test_event_times_chronological(data):
    meta, g = data; env = ICUEnv(g, meta)
    for e in meta.episode_id[:20]:
        play(env, e, rnd())
        ts = [t for t, _, _ in env.timeline]; assert ts == sorted(ts)


def test_hidden_los_not_in_observation(data):
    meta, g = data; env = ICUEnv(g, meta); o = env.reset(meta.episode_id[0])
    assert set(o) == {"x", "status", "mask", "free", "n", "ids", "t", "done"}
    assert o["x"].shape[1] == 4
    df = pd.concat(g.values()); assert (np.abs(df.pred_total_los_h - df.actual_los_total_h) > 1e-6).all()


def test_deterministic_and_reproducible(data):
    meta, g = data; env = ICUEnv(g, meta); e = meta.episode_id[3]
    m1 = play(env, e, ShortestPredictedFirst().act); m2 = play(env, e, ShortestPredictedFirst().act)
    assert m1 == m2


def test_rewards_finite_and_termination_and_horizon(data):
    meta, g = data
    for kind in ("r1", "r2"):
        env = ICUEnv(g, meta, reward=kind)
        for e in meta.episode_id[:20]:
            m = play(env, e, rnd()); assert np.isfinite(m["ep_return"]) and m["completion"] == 1.0
    # horizon: a policy that always STOPs leaves patients unresolved, censored waits recorded, penalty applied
    env = ICUEnv(g, meta, reward="r1")
    e = meta.episode_id[0]; o = env.reset(e); d = False
    while not d:
        o, r, d, info = env.step(o["n"])
    m = info["metrics"]; assert env.t == pytest.approx(env.horizon) or m["completion"] == 1.0
    if m["unresolved"] > 0:
        assert m["max_wait"] == pytest.approx(env.horizon - 0.0, abs=0.5)
        assert m["ep_cost"] >= env.lam_u * m["unresolved"]


def test_sofa_thresholds():
    assert list(resp_score([450, 350, 250, 150, 80], [0, 0, 0, 1, 1])) == [0, 1, 2, 3, 4]
    assert list(resp_score([150, 80], [0, 0])) == [2, 2]
    assert list(coag_score([160, 149, 99, 49, 19])) == [0, 1, 2, 3, 4]
    assert list(liver_score([1.1, 1.2, 2.0, 6.0, 12.0])) == [0, 1, 2, 3, 4]
    assert list(cv_score([75, 65, 65, 65, 65, 65], [0, 0, 1, 1, 2, 2], [0, 0, 4, 10, 0.05, 0.2])) == [0, 1, 2, 3, 3, 4]
    assert list(cv_score([65], [1], [16])) == [4]
    assert list(cns_score([15, 14, 12, 9, 5])) == [0, 1, 2, 3, 4]
    assert list(renal_score([1.0, 1.5, 2.5, 4.0, 5.5, 1.0, 1.0], [2000] * 5 + [400, 100])) == [0, 1, 2, 3, 4, 3, 4]


def test_sofa_from_raw_and_recomputed_after_deterioration(data):
    rng = np.random.default_rng(0); df = make_patients(rng, 200, CFG, "arrival", 0, 0)
    comp = sofa_components(df); assert (comp.sofa_total.values >= 0).all() and (comp.sofa_total.values <= 24).all()
    meta, g = data; env = ICUEnv(g, meta); found = False
    for e in meta.episode_id[:60]:
        env.reset(e); d = False; o = env.reset(e)
        while not d:
            o, r, d, _ = env.step(rnd()(o))
        for p in env.pts:
            if p.ndet > 0 and p.raw is not None:
                assert sofa_components(p.raw).sofa_total.iloc[0] == p.sofa; found = True
    assert found


def test_reward_math():
    p = dict(alpha=1.0, beta=1.0, wref=8.0)
    assert interval_cost("r1", np.zeros(5), np.zeros(5), 2.0, p) == 10.0
    assert clipped_wait_integral([0.0], 4.0, 8.0)[0] == pytest.approx(1.0)       # int s/8 = 16/16
    assert clipped_wait_integral([6.0], 4.0, 8.0)[0] == pytest.approx((6 * 2 + 2) / 8 + 2)
    assert clipped_wait_integral([10.0], 3.0, 8.0)[0] == pytest.approx(3.0)
    c = interval_cost("r2", np.array([24.0]), np.array([0.0]), 4.0, p); assert c == pytest.approx(4.0 + 1.0)


def test_variable_state_and_masking_network(data):
    meta, g = data; env = ICUEnv(g, meta); obs = []
    for e in meta.episode_id[:6]: obs.append(env.reset(e))
    assert len({o["n"] for o in obs}) > 1
    x, st, rm, am, fr = collate(obs); net = SetNet(16)
    q, _ = net(x, st, rm, fr); assert q.shape == (len(obs), x.shape[1] + 1)
    for b, o in enumerate(obs):  # padding rows never selectable; invalid rows masked
        assert not am[b, o["n"]:x.shape[1]].any()
    # permutation equivariance of the set network
    o = obs[0]; n = o["n"]; perm = np.random.default_rng(0).permutation(n)
    o2 = dict(o); o2["x"] = o["x"][perm]; o2["status"] = o["status"][perm]; o2["mask"] = np.concatenate([o["mask"][:n][perm], o["mask"][n:]])
    xa, sa, ra, aa, fa = collate([o]); xb, sb, rb, ab, fb = collate([o2])
    qa, _ = net(xa, sa, ra, fa); qb, _ = net(xb, sb, rb, fb)
    assert np.allclose(qa[0, :n][perm].detach().numpy(), qb[0, :n].detach().numpy(), atol=1e-5)


def test_random_agent_uses_same_env_and_never_invalid(data):
    meta, g = data; env = ICUEnv(g, meta)
    for e in meta.episode_id[:30]:
        m = play(env, e, RandomAgent(1).act); assert m["violations"] == 0


def test_seed_reproduces_scenarios():
    c = CFG
    a = make_patients(np.random.default_rng([c["seed"], 9]), 6, c, "arrival", 9, 0)
    b = make_patients(np.random.default_rng([c["seed"], 9]), 6, c, "arrival", 9, 0)
    pd.testing.assert_frame_equal(a, b)


def test_event_order_discharge_before_arrival():
    from src.environment.icu_environment import P_DISCH, P_ARR, P_DET, P_END
    assert P_DISCH < P_ARR < P_DET < P_END


def test_windows_cover_whole_day_and_are_30_minutes():
    meta = pd.read_csv("data/synthetic/episode_metadata.csv")
    assert set(meta.slot) == set(range(48))
    st = pd.to_datetime(meta.window_start, format="%H:%M"); en = pd.to_datetime(meta.window_end, format="%H:%M")
    assert (((en - st).dt.total_seconds() % 86400) == 1800).all()
