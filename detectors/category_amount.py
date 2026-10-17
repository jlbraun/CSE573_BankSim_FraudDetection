# Category Amount Detector
# Author: Pranav
#
# Fraudulent payments in BankSim tend to be much larger than normal payments of the same kind
# (a $500 grocery bill is strange, a $500 hotel bill is not). This detector asks:
# "what fraction of NORMAL payments in this category were smaller than or equal to this one?"
#
#     score = percentile of amount among non-fraud training payments in the same category
#
# Only non-fraud rows are used as the reference so fraud amounts don't make "normal" look bigger.
# Categories never seen in training are compared against all normal training payments.
# A score of 0.97 means "bigger than ~97% of normal payments in this category".

import numpy as np
import pandas as pd


def score(df, train=None):
    """
    Return a category amount score between 0 and 1 for every row of df.

    train: data to learn normal amounts from. ALWAYS pass the training split when scoring test data,
           e.g. category_amount.score(test, train). If omitted, df itself is used (only OK when df is
           the training data). If train has a "fraud" column, fraud rows are left out of the reference.
    """
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

    cats = df["category"].astype(str).str.strip("'")
    amounts = df["amount"].astype(float)
    result = pd.Series(0.0, index=df.index, name="score_category_amount")

    for c, a in amounts.groupby(cats):
        normal = by_category.get(c, fallback)
        result[a.index] = np.searchsorted(normal, a.to_numpy(), side="right") / len(normal)

    return result
