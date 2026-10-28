# Merchant Novelty Detector - Pranav
# Scores how new a merchant is to the customer: 1.0 for a first purchase there,
# dropping as the customer's earlier purchases at that merchant add up.
#
#     score = scale / (prior_purchases + scale)      (scale=1: 0 -> 1.0, 1 -> 0.5, 2 -> 0.33)
#
# Only purchases in strictly earlier steps count, and fraud labels are never read,
# so passing the full dataset as history does not leak labels.

import numpy as np
import pandas as pd

REQUIRED = {"customer", "merchant", "step"}


def _check(df, label):
    missing = REQUIRED - set(df.columns)
    if missing:
        suffix = f" in {label}" if label else ""
        raise ValueError(f"Missing columns{suffix}: {sorted(missing)}")
    if df[list(REQUIRED)].isna().any().any():
        raise ValueError("customer, merchant and step cannot contain missing values")


def _keys(df):
    customers = df["customer"].astype(str).str.strip("'")
    merchants = df["merchant"].astype(str).str.strip("'")
    return (customers + "|" + merchants).to_numpy()


def _steps(df):
    steps = pd.to_numeric(df["step"], errors="raise").to_numpy()
    if (steps < 0).any() or (steps != np.floor(steps)).any():
        raise ValueError("step must contain nonnegative whole numbers")
    return steps.astype(np.int64)


def counts(df, history=None):
    """Number of purchases this customer made at this merchant in earlier steps."""
    _check(df, "")
    if history is None:
        history = df
    _check(history, "history")

    hist_keys, df_keys = _keys(history), _keys(df)
    hist_steps, df_steps = _steps(history), _steps(df)

    # Shared integer code for each customer|merchant pair across both frames.
    codes, _ = pd.factorize(np.concatenate([hist_keys, df_keys]))
    codes = codes.astype(np.int64)
    hist_codes, df_codes = codes[:len(hist_keys)], codes[len(hist_keys):]

    # Pack (pair, step) into one sortable number, then count earlier history rows per pair.
    m = int(max(hist_steps.max(initial=0), df_steps.max(initial=0))) + 1
    hist = np.sort(hist_codes * m + hist_steps)
    before_now = np.searchsorted(hist, df_codes * m + df_steps, side="left")
    pair_start = np.searchsorted(hist, df_codes * m, side="left")

    return pd.Series(before_now - pair_start, index=df.index, name="merchant_novelty_count")


def score(df, history=None, scale=1.0):
    """Return a merchant novelty score between 0 and 1 for every row of df."""
    if isinstance(scale, bool) or not np.isscalar(scale) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a finite positive number")

    prior = counts(df, history).astype(float)
    return (scale / (prior + scale)).rename("score_merchant_novelty")
