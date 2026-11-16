"""Metric definitions (units / direction) and aggregation helpers."""
import numpy as np
import pandas as pd

METRICS = {  # name: (label, unit, direction)
    "total_wait": ("Total cumulative waiting time", "patient-h", "lower"),
    "mean_wait": ("Mean waiting time", "h", "lower"),
    "median_wait": ("Median waiting time", "h", "lower"),
    "max_wait": ("Maximum waiting time", "h", "lower"),
    "p90_wait": ("90th percentile waiting time", "h", "lower"),
    "p95_wait": ("95th percentile waiting time", "h", "lower"),
    "high_sofa_wait": ("High-SOFA (>=11) mean waiting time", "h", "lower"),
    "sev_wait_cost": ("Severity-weighted waiting cost", "cost units", "lower"),
    "completion": ("Admission completion rate", "fraction", "higher"),
    "unresolved": ("Unresolved patients at horizon", "count", "lower"),
    "n_det": ("Deterioration events while waiting", "count", "lower"),
    "utilization": ("Time-weighted ICU utilisation (descriptive)", "fraction", "descriptive"),
    "ep_return": ("Episode return (scaled reward)", "reward", "higher"),
    "violations": ("Capacity violations", "count", "exactly 0"),
}


def per_seed(df, cols):
    """Mean over test episodes for each (algo, seed)."""
    return df.groupby(["algo", "seed"])[cols].mean().reset_index()


def summarize(df, cols):
    ps = per_seed(df, cols)
    g = ps.groupby("algo")[cols]
    return g.mean(), g.std(ddof=1).fillna(0.0)


def paired_diff(df, a, b, col, n_boot=4000, seed=0):
    """Mean per-episode difference (a - b) on identical test episodes, bootstrap 95% CI over episodes."""
    ea = df[df.algo == a].groupby("episode")[col].mean(); eb = df[df.algo == b].groupby("episode")[col].mean()
    d = (ea - eb.reindex(ea.index)).dropna().values
    rng = np.random.default_rng(seed)
    bs = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def convergence_episode(log, ref_return, frac=0.95, tail=5):
    """First evaluation episode where val_return closes `frac` of the gap between the random reference and the
    plateau (mean of the last `tail` evaluations). NaN if the plateau does not exceed the reference."""
    plateau = log.val_return.tail(tail).mean()
    if plateau <= ref_return: return np.nan
    thr = ref_return + frac * (plateau - ref_return)
    hit = log[log.val_return >= thr]
    return float(hit.episode.iloc[0]) if len(hit) else np.nan
