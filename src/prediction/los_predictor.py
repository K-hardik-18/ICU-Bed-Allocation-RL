"""Observable-feature predictors for ICU duration and deterioration hazard.

Trained ONLY on training-split patients; training rows receive out-of-fold predictions (grouped by episode)
so that the agent never sees predictions fitted on the record's own outcome.
Inputs are the raw clinical snapshot + age (+ elapsed ICU time for the remaining-LOS model).
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from ..data.sofa_calculator import sofa_components, COMPONENTS

RAW_COLS = ["pao2", "fio2", "pf_ratio", "mech_vent", "platelets", "bilirubin", "map", "vaso_drug", "vaso_dose",
            "gcs", "creatinine", "urine_output", "heart_rate", "resp_rate", "spo2", "sbp", "dbp", "temperature"]
FEATS = RAW_COLS + ["age"] + COMPONENTS + ["sofa_total"]


def feature_frame(df):
    """Attach SOFA components (derived from raw) and return the predictor design matrix columns."""
    out = df[RAW_COLS + ["age"]].copy()
    comp = sofa_components(out)
    for c in COMPONENTS + ["sofa_total"]:
        out[c] = comp[c].values
    return out[FEATS]


def _gbr(seed):
    return GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.06, subsample=0.8, random_state=seed)


class Predictors:
    """log total-LOS (for waiting patients), log remaining-LOS (occupants), log hazard (waiting patients)."""

    def __init__(self, seed=0):
        self.seed = seed
        self.m_total = _gbr(seed); self.m_rem = _gbr(seed); self.m_haz = _gbr(seed)

    @staticmethod
    def _rem_X(F, elapsed):
        X = F.copy(); X["elapsed"] = np.asarray(elapsed, float); return X

    def fit(self, df_train):
        F = feature_frame(df_train)
        self.m_total.fit(F, np.log(df_train["actual_los_total_h"]))
        self.m_haz.fit(F, np.log(df_train["true_hazard"]))
        occ = df_train["role"] == "occupant"
        self.m_rem.fit(self._rem_X(F[occ], df_train.loc[occ, "elapsed_icu_h"]), np.log(df_train.loc[occ, "actual_remaining_h"]))
        return self

    def predict_total(self, F): return np.exp(self.m_total.predict(F))
    def predict_hazard(self, F): return np.clip(np.exp(self.m_haz.predict(F)), 0.005, 1.0)
    def predict_remaining(self, F, elapsed): return np.exp(self.m_rem.predict(self._rem_X(F, elapsed)))

    def save(self, path):
        with open(path, "wb") as f: pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f: return pickle.load(f)


def oof_predictions(df, train_mask, seed=0, k=5):
    """Return (pred_total, pred_remaining, pred_hazard, fitted Predictors).

    Training rows: K-fold (grouped by episode) out-of-fold. Val/test rows: model fitted on all training rows."""
    F = feature_frame(df)
    n = len(df)
    pt = np.zeros(n); pr = np.zeros(n); ph = np.zeros(n)
    tr_idx = np.where(train_mask)[0]
    gkf = GroupKFold(n_splits=k)
    for a, b in gkf.split(tr_idx, groups=df["episode_id"].values[tr_idx]):
        ia, ib = tr_idx[a], tr_idx[b]
        P = Predictors(seed).fit(df.iloc[ia])
        pt[ib] = P.predict_total(F.iloc[ib]); ph[ib] = P.predict_hazard(F.iloc[ib])
        pr[ib] = P.predict_remaining(F.iloc[ib], df["elapsed_icu_h"].values[ib])
    P = Predictors(seed).fit(df.iloc[tr_idx])
    ho = np.where(~train_mask)[0]
    pt[ho] = P.predict_total(F.iloc[ho]); ph[ho] = P.predict_hazard(F.iloc[ho])
    pr[ho] = P.predict_remaining(F.iloc[ho], df["elapsed_icu_h"].values[ho])
    return pt, pr, ph, P
