# Merchant Risk Detector - Pranav
# Smoothed fraud rate of each merchant, learned from train.

import pandas as pd


def rates(train, smoothing=20.0):
    required = {"merchant", "fraud"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if smoothing < 0:
        raise ValueError("smoothing must be >= 0")

    merchants = train["merchant"].astype(str).str.strip("'")
    fraud = train["fraud"].astype(float)
    overall = float(fraud.mean())

    stats = fraud.groupby(merchants).agg(["sum", "count"])
    per_merchant = (stats["sum"] + smoothing * overall) / (stats["count"] + smoothing)
    return per_merchant, overall


def score(df, train=None, smoothing=20.0):
    if "merchant" not in df.columns:
        raise ValueError("Missing columns: ['merchant']")
    if train is None:
        train = df

    per_merchant, overall = rates(train, smoothing)
    merchants = df["merchant"].astype(str).str.strip("'")
    return merchants.map(per_merchant).fillna(overall).astype(float).rename("score_merchant_risk")
