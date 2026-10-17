# Merchant Risk Detector
# Author: Pranav
#
# Fraud in BankSim is concentrated at a handful of merchants, even more sharply than by category
# (every merchant belongs to exactly one category, so this is a finer-grained version of category_risk).
# Each transaction gets the smoothed fraud rate of its merchant, learned from labeled TRAINING data.
#
#     score = (fraud_at_merchant + smoothing * overall_fraud_rate) / (rows_at_merchant + smoothing)
#
# Smoothing pulls merchants with only a few rows toward the overall rate so they can't jump to 0 or 1.
# Merchants never seen in training get the overall rate.
# The score is a real fraud rate, so a score of 0.9 means "~90% of past payments to this merchant were fraud".

import pandas as pd


def rates(train, smoothing=20.0):
    """Return (per-merchant fraud rate Series, overall fraud rate) learned from labeled data."""
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
    """
    Return a merchant risk score between 0 and 1 for every row of df.

    train: labeled data to learn merchant fraud rates from. ALWAYS pass the training split when
           scoring test data, e.g. merchant_risk.score(test, train). If omitted, rates are learned
           from df itself (only OK when df is the training data).
    """
    if "merchant" not in df.columns:
        raise ValueError("Missing columns: ['merchant']")
    if train is None:
        train = df

    per_merchant, overall = rates(train, smoothing)
    merchants = df["merchant"].astype(str).str.strip("'")
    return merchants.map(per_merchant).fillna(overall).astype(float).rename("score_merchant_risk")
