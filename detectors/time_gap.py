# Consecutive Transaction Time Gap Detector
# Author: Mansi Ganga Nayak
#
# Measures how long it has been since the customer's previous transaction.
# BankSim only records the day (step), so the gap is in days. Customers in BankSim normally
# pay every day; a customer coming back after several quiet days is far more likely to be fraud
# (dormant-account reactivation): gap 1 day ~0.8% fraud, gap 3+ days ~15-23% fraud.
#
#     extra_days = max(gap - 1, 0)
#     score      = extra_days / (extra_days + scale)
#
# gap 1 -> 0.00, gap 2 -> 0.33, gap 3 -> 0.50, gap 5 -> 0.67, gap 10 -> 0.82   (scale = 2)
# A customer's very first transaction has no gap and gets `first_score`.
#
# Only earlier steps are used, never the current or future ones, and no fraud labels are used.
# Pass the full transaction history (all rows, e.g. the cleaned bs140513_032310.csv) as `history`
# when scoring a sampled split, otherwise gaps are measured over the sample's missing rows.

import numpy as np
import pandas as pd


def _clean(s):
    return s.astype(str).str.strip("'")


def gaps(df, history=None):
    """Days since the customer's previous transaction day (NaN for their first transaction)."""
    if history is None:
        history = df
    for name, frame in (("df", df), ("history", history)):
        missing = {"customer", "step"} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    h = pd.DataFrame({"customer": _clean(history["customer"]).to_numpy(),
                      "step": pd.to_numeric(history["step"]).to_numpy()})
    days = h.drop_duplicates().sort_values(["customer", "step"])
    days["gap"] = days["step"] - days.groupby("customer")["step"].shift(1)

    key = pd.DataFrame({"customer": _clean(df["customer"]).to_numpy(),
                        "step": pd.to_numeric(df["step"]).to_numpy()})
    merged = key.merge(days, on=["customer", "step"], how="left")

    # rows whose (customer, step) is not in history: use the latest earlier day in history
    todo = merged["gap"].isna() & ~key.set_index(["customer", "step"]).index.isin(
        days.set_index(["customer", "step"]).index)
    if todo.any():
        last = days.groupby("customer")["step"].apply(np.array).to_dict()
        for i in np.flatnonzero(todo.to_numpy()):
            arr = last.get(key.at[i, "customer"])
            if arr is not None:
                earlier = arr[arr < key.at[i, "step"]]
                if len(earlier):
                    merged.at[i, "gap"] = key.at[i, "step"] - earlier.max()

    return pd.Series(merged["gap"].to_numpy(), index=df.index, name="time_gap_days")


def score(df, history=None, scale=2.0, first_score=0.5):
    """Return a time-gap score between 0 and 1 for every row of df."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a finite positive number")
    if not 0 <= first_score <= 1:
        raise ValueError("first_score must be between 0 and 1")

    g = gaps(df, history)
    extra = (g - 1).clip(lower=0)
    s = extra / (extra + scale)
    return s.fillna(first_score).astype(float).rename("score_time_gap")
