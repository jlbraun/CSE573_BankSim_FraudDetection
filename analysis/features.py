"""Shared helpers for analysis/tuning scripts (Mansi Ganga Nayak).

build_features(): runs every detector on a frame. Label-based detectors
(merchant_risk, category_risk, category_amount) can be computed out-of-fold on
the training split so the ensemble never trains on scores that saw their own
labels.
realistic_test(): every row of the full BankSim CSV that is NOT in the
downsampled train split -> keeps the real 1.21% fraud rate.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from detectors import (amount_anomaly, category_amount, category_novelty,  # noqa: E402
                       category_risk, merchant_novelty, merchant_risk, time_gap,
                       velocity)

FEATURES = ["amount_anomaly", "merchant_risk", "merchant_novelty", "velocity",
            "time_gap", "category_amount", "category_risk", "category_novelty"]
LABEL_BASED = {"merchant_risk", "category_risk", "category_amount"}


def load():
    def clean(df):
        for c in df.select_dtypes(include=["object", "string"]).columns:
            df[c] = df[c].astype(str).str.strip("'")
        return df
    train = clean(pd.read_csv(os.path.join(ROOT, "data/banksim_downsampled_train.csv")))
    test = clean(pd.read_csv(os.path.join(ROOT, "data/banksim_downsampled_test.csv")))
    full = clean(pd.read_csv(os.path.join(ROOT, "data/bs140513_032310.csv")))
    return train, test, full


def realistic_test(full, train):
    """Full dataset minus the rows used for training (keeps the real fraud rate)."""
    keys = list(full.columns)
    def key(df):
        k = df[keys].copy()
        for c in keys:
            k[c] = df[c].astype(str) if c != "amount" else df[c].round(2).astype(str)
        k["_n"] = k.groupby(keys).cumcount()
        return k
    kf, kt = key(full), key(train).assign(_in_train=1)
    m = kf.merge(kt, on=keys + ["_n"], how="left")
    keep = m["_in_train"].isna().to_numpy()
    return full[keep].reset_index(drop=True)


def _label_feature(name, df, ref, p):
    if name == "merchant_risk":
        return merchant_risk.score(df, ref, smoothing=p.get("merchant_smoothing", 20.0))
    if name == "category_risk":
        return category_risk.score(df, ref, smoothing=p.get("category_smoothing", 20.0))
    if name == "category_amount":
        return category_amount.score(df, ref)
    raise KeyError(name)


def build_features(df, train, history, params=None, oof=False, folds=5, seed=42):
    """Return DataFrame of detector scores for df.

    oof=True (only valid when df is train): label-based detectors are computed
    out-of-fold, so each training row is scored by a model fit without it.
    """
    p = params or {}
    X = pd.DataFrame(index=df.index)
    X["amount_anomaly"] = np.asarray(amount_anomaly.score(df))
    X["velocity"] = np.asarray(velocity.score(df, window_steps=p.get("velocity_window", 1),
                                              scale=p.get("velocity_scale", 5.0)))
    X["merchant_novelty"] = np.asarray(merchant_novelty.score(df, history,
                                                              scale=p.get("mnov_scale", 1.0)))
    X["time_gap"] = np.asarray(time_gap.score(df, history, scale=p.get("tgap_scale", 2.0),
                                              first_score=p.get("tgap_first", 0.5)))
    X["category_novelty"] = np.asarray(category_novelty.score(df, history, k=p.get("cnov_k", 2.0)))
    for name in LABEL_BASED:
        if oof:
            col = np.zeros(len(df))
            skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
            for fit_idx, pred_idx in skf.split(df, df["fraud"]):
                col[pred_idx] = np.asarray(_label_feature(name, df.iloc[pred_idx], df.iloc[fit_idx], p))
            X[name] = col
        else:
            X[name] = np.asarray(_label_feature(name, df, train, p))
    return X[FEATURES]
