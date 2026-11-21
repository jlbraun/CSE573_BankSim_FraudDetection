"""Why merchant_risk and category_risk give (almost) the same output. (Mansi Ganga Nayak)

Usage: python analysis/merchant_vs_category.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.features import load
from detectors import category_risk, merchant_risk

train, test, full = load()

cats_per_merchant = full.groupby("merchant")["category"].nunique()
print(f"merchants: {len(cats_per_merchant)}, merchants selling in >1 category: {(cats_per_merchant > 1).sum()}")
print("=> every merchant belongs to exactly one category: merchant is a finer split of category\n")

print("Merchants per category and fraud rate of each merchant (full data):")
g = full.groupby(["category", "merchant"])["fraud"].agg(["mean", "size"]).reset_index()
for cat, grp in g.groupby("category"):
    rates = ", ".join(f"{r:.0%}" for r in grp.sort_values("mean")["mean"])
    print(f"  {cat:<22} {len(grp)} merchant(s): {rates}")

m = merchant_risk.score(test, train).to_numpy()
c = category_risk.score(test, train).to_numpy()
print(f"\nOn the 50/50 test split:")
print(f"  Pearson correlation of the two scores : {np.corrcoef(m, c)[0, 1]:.3f}")
print(f"  rows on the same side of the 0.05 cut : {((m >= .05) == (c >= .05)).mean():.1%}")
print(f"  mean |difference|                      : {np.abs(m - c).mean():.3f}")
print("\nConclusion: at the 0.05 cutoff both detectors flag exactly the same rows (zero-fraud")
print("categories -> below .05, everything else -> above), so their accuracy and confusion")
print("matrices are identical. They are not identical scores: merchant_risk also separates")
print("merchants inside a category (e.g. es_health 0% vs 47%), so it carries everything")
print("category_risk knows plus more.")
