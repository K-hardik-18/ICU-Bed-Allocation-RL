"""Reward 1 (total waiting) and Reward 2 (severity-weighted waiting) integrated exactly over an event interval.

Between events the queue is constant and SOFA is constant, so
  R1 interval cost = |Q| * dt
  R2 interval cost = sum_i [ alpha*S_i*dt + beta * Int_0^dt min((W_i+s)/Wref, 1) ds ]   (closed form)
Rewards returned are the NEGATIVE of the cost (unscaled); callers multiply by the configured scale.
"""
import numpy as np


def clipped_wait_integral(w0, dt, wref):
    """Int_0^dt min((w0+s)/wref, 1) ds, vectorised over w0."""
    w0 = np.asarray(w0, float)
    tau = np.maximum(wref - w0, 0.0)            # time until the normalised wait saturates at 1
    ramp_t = np.minimum(dt, tau)
    ramp = (w0 * ramp_t + 0.5 * ramp_t ** 2) / wref
    return ramp + (dt - ramp_t)


def interval_cost(kind, sofa, wait0, dt, p):
    """sofa, wait0: arrays over waiting patients at interval start."""
    if dt <= 0 or len(sofa) == 0:
        return 0.0
    if kind == "r1":
        return float(len(sofa) * dt)
    S = np.asarray(sofa, float) / 24.0
    return float(p["alpha"] * S.sum() * dt + p["beta"] * clipped_wait_integral(wait0, dt, p["wref"]).sum())


def terminal_cost(n_unresolved, lam_u):
    return float(lam_u * n_unresolved)
