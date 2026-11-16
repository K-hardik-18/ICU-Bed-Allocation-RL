"""SOFA from raw clinical variables (Vincent et al. 1996; Singer et al. 2016 Sepsis-3 table).

Vectorised over numpy arrays / pandas columns. Total SOFA = sum of six organ components (0-24).
Vasopressor encoding: vaso_drug 0 = none, 1 = dopamine, 2 = norepinephrine; vaso_dose in mcg/kg/min.
Urine output is mL/day.
"""
import numpy as np
import pandas as pd

COMPONENTS = ["sofa_resp", "sofa_coag", "sofa_liver", "sofa_cv", "sofa_cns", "sofa_renal"]


def resp_score(pf, vent):
    pf = np.asarray(pf, float); vent = np.asarray(vent).astype(bool)
    s = np.zeros(pf.shape, int)
    s = np.where(pf < 400, 1, s)
    s = np.where(pf < 300, 2, s)
    s = np.where((pf < 200) & vent, 3, s)
    s = np.where((pf < 100) & vent, 4, s)
    return s


def coag_score(plt):
    plt = np.asarray(plt, float)
    return np.select([plt < 20, plt < 50, plt < 100, plt < 150], [4, 3, 2, 1], 0)


def liver_score(bili):
    b = np.asarray(bili, float)
    return np.select([b >= 12.0, b >= 6.0, b >= 2.0, b >= 1.2], [4, 3, 2, 1], 0)


def cv_score(map_, drug, dose):
    map_ = np.asarray(map_, float); drug = np.asarray(drug); dose = np.asarray(dose, float)
    s = np.where(map_ < 70, 1, 0)
    dopa = drug == 1
    s = np.where(dopa, np.select([dose > 15, dose > 5], [4, 3], 2), s)
    ne = drug == 2
    s = np.where(ne, np.where(dose > 0.1, 4, 3), s)
    return s


def cns_score(gcs):
    g = np.asarray(gcs, float)
    return np.select([g < 6, g < 10, g < 13, g < 15], [4, 3, 2, 1], 0)


def renal_score(creat, uo):
    c = np.asarray(creat, float); u = np.asarray(uo, float)
    sc = np.select([c >= 5.0, c >= 3.5, c >= 2.0, c >= 1.2], [4, 3, 2, 1], 0)
    su = np.select([u < 200, u < 500], [4, 3], 0)
    return np.maximum(sc, su)


def sofa_components(df):
    """df has columns pf_ratio, mech_vent, platelets, bilirubin, map, vaso_drug, vaso_dose, gcs, creatinine, urine_output."""
    out = pd.DataFrame(index=df.index)
    out["sofa_resp"] = resp_score(df["pf_ratio"], df["mech_vent"])
    out["sofa_coag"] = coag_score(df["platelets"])
    out["sofa_liver"] = liver_score(df["bilirubin"])
    out["sofa_cv"] = cv_score(df["map"], df["vaso_drug"], df["vaso_dose"])
    out["sofa_cns"] = cns_score(df["gcs"])
    out["sofa_renal"] = renal_score(df["creatinine"], df["urine_output"])
    out["sofa_total"] = out[COMPONENTS].sum(axis=1)
    return out


if __name__ == "__main__":  # command: python -m src.data.sofa_calculator  (self-check on tiny worked example)
    ex = pd.DataFrame([dict(pf_ratio=150, mech_vent=1, platelets=45, bilirubin=2.5, map=60, vaso_drug=2,
                            vaso_dose=0.05, gcs=11, creatinine=2.4, urine_output=900)])
    print(sofa_components(ex).iloc[0].to_dict())  # expected: 3,3,2,3,2,2 = 15
