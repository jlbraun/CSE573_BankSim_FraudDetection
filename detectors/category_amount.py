# Category Amount Detector - Pranav
# Percentile of amount among normal train payments in the same category.

import numpy as np
import pandas as pd


def score(df, train=None):
    required = {"category", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if train is None:
        train = df
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"Missing columns in train: {sorted(missing)}")

    ref = train[train["fraud"] == 0] if "fraud" in train.columns else train
    if ref.empty:
        raise ValueError("train has no normal (fraud == 0) rows to compare against")

    ref_cats = ref["category"].astype(str).str.strip("'")
    ref_amounts = ref["amount"].astype(float)
    fallback = np.sort(ref_amounts.to_numpy())
    by_category = {c: np.sort(a.to_numpy()) for c, a in ref_amounts.groupby(ref_cats)}

    cats = df["category"].astype(str).str.strip("'").to_numpy()
    amounts = df["amount"].astype(float).to_numpy()
    result = np.zeros(len(df))

    for c in np.unique(cats):
        normal = by_category.get(c, fallback)
        rows = cats == c
        result[rows] = np.searchsorted(normal, amounts[rows], side="right") / len(normal)

    return pd.Series(result, index=df.index, name="score_category_amount")
