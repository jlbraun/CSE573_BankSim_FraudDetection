# Category Novelty Detector
# Author: Mansi Ganga Nayak
#
# Has this customer bought in this category before? A purchase in a category the customer
# rarely or never uses is suspicious, and more suspicious the longer their history is:
# a customer with 100 food purchases suddenly buying in a new category is flagged more than a
# brand-new customer making their first purchase.
#
#     prior_total = customer's transactions on earlier days
#     prior_cat   = customer's transactions in THIS category on earlier days
#     novelty     = 1 - prior_cat / prior_total            (1 = never bought this category)
#     confidence  = prior_total / (prior_total + k)        (how much history we have, k = 2)
#     score       = novelty * confidence
#
# In BankSim, purchases in the customer's usual category (>50% of history) are ~0.03% fraud,
# while purchases in a rare/new category are ~4-9% fraud.
#
# Only earlier steps are used and no fraud labels are needed. Pass the full transaction history
# (all rows, e.g. the cleaned bs140513_032310.csv) as `history` when scoring a sampled split.

import numpy as np
import pandas as pd


def _clean(s):
    return s.astype(str).str.strip("'")


def history_counts(df, history=None):
    """Return DataFrame with prior_total and prior_cat for every row of df."""
    if history is None:
        history = df
    for name, frame in (("df", df), ("history", history)):
        missing = {"customer", "category", "step"} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    h = pd.DataFrame({"customer": _clean(history["customer"]).to_numpy(),
                      "category": _clean(history["category"]).to_numpy(),
                      "step": pd.to_numeric(history["step"]).to_numpy()})

    # per customer: cumulative count of transactions strictly before each step
    tot = h.groupby(["customer", "step"]).size().rename("n").reset_index()
    tot = tot.sort_values(["customer", "step"])
    tot["prior_total"] = tot.groupby("customer")["n"].cumsum() - tot["n"]

    cat = h.groupby(["customer", "category", "step"]).size().rename("n").reset_index()
    cat = cat.sort_values(["customer", "category", "step"])
    cat["prior_cat"] = cat.groupby(["customer", "category"])["n"].cumsum() - cat["n"]

    key = pd.DataFrame({"customer": _clean(df["customer"]).to_numpy(),
                        "category": _clean(df["category"]).to_numpy(),
                        "step": pd.to_numeric(df["step"]).to_numpy()})
    out = key.merge(tot[["customer", "step", "prior_total"]], on=["customer", "step"], how="left")
    out = out.merge(cat[["customer", "category", "step", "prior_cat"]],
                    on=["customer", "category", "step"], how="left")

    # rows not present in history: count history rows on strictly earlier steps
    todo = out["prior_total"].isna() | out["prior_cat"].isna()
    for i in np.flatnonzero(todo.to_numpy()):
        c, k, st = key.at[i, "customer"], key.at[i, "category"], key.at[i, "step"]
        earlier = h[(h["customer"] == c) & (h["step"] < st)]
        out.at[i, "prior_total"] = len(earlier)
        out.at[i, "prior_cat"] = int((earlier["category"] == k).sum())

    out.index = df.index
    return out[["prior_total", "prior_cat"]].astype(float)


def score(df, history=None, k=2.0):
    """Return a category novelty score between 0 and 1 for every row of df."""
    if not np.isfinite(k) or k < 0:
        raise ValueError("k must be a finite number >= 0")

    c = history_counts(df, history)
    total = c["prior_total"]
    novelty = 1 - c["prior_cat"] / total.replace(0, np.nan)
    confidence = total / (total + k) if k > 0 else (total > 0).astype(float)
    s = (novelty * confidence).fillna(0.0)          # no history -> nothing to compare against
    return s.clip(0, 1).astype(float).rename("score_category_novelty")
