# Category Risk Detector
# Author: Mansi Ganga Nayak
#
# Some merchant categories are much riskier than others in BankSim
# (e.g. es_leisure / es_travel are mostly fraud, es_transportation / es_food are ~never fraud).
# Each transaction gets the smoothed fraud rate of its category, learned from labeled TRAINING data.
#
#     score = (fraud_in_category + smoothing * overall_fraud_rate) / (rows_in_category + smoothing)
#
# Smoothing pulls small categories toward the overall rate so a category with only a few rows
# can't jump to 0 or 1. Categories never seen in training get the overall rate.
# The score is a real fraud rate, so a score of 0.9 means "~90% of past payments like this were fraud".

import pandas as pd


def rates(train, smoothing=20.0):
    """Return (per-category fraud rate Series, overall fraud rate) learned from labeled data."""
    required = {"category", "fraud"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if smoothing < 0:
        raise ValueError("smoothing must be >= 0")

    cats = train["category"].astype(str).str.strip("'")
    fraud = train["fraud"].astype(float)
    overall = float(fraud.mean())

    stats = fraud.groupby(cats).agg(["sum", "count"])
    per_category = (stats["sum"] + smoothing * overall) / (stats["count"] + smoothing)
    return per_category, overall


def score(df, train=None, smoothing=20.0):
    """
    Return a category risk score between 0 and 1 for every row of df.

    train: labeled data to learn category fraud rates from. ALWAYS pass the training split when
           scoring test data, e.g. category_risk.score(test, train). If omitted, rates are learned
           from df itself (only OK when df is the training data).
    """
    if "category" not in df.columns:
        raise ValueError("Missing columns: ['category']")
    if train is None:
        train = df

    per_category, overall = rates(train, smoothing)
    cats = df["category"].astype(str).str.strip("'")
    return cats.map(per_category).fillna(overall).astype(float).rename("score_category_risk")
