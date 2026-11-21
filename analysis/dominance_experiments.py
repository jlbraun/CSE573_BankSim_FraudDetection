"""Why does merchant_risk get ~96% of the ensemble's weight? (Mansi Ganga Nayak)

Compares the current setup against (a) out-of-fold label encoding and (b) dropping
the redundant category_risk, on the 50/50 test split AND a realistic imbalanced
test set (full CSV minus training rows). Usage:
    python analysis/build_cache.py CACHE && python analysis/dominance_experiments.py CACHE
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.features import FEATURES

cache = sys.argv[1] if len(sys.argv) > 1 else "analysis_cache"
L = lambda n: pd.read_pickle(os.path.join(cache, n + ".pkl"))
tr_in, tr_oof, te, rt = L("train_insample"), L("train_oof"), L("test5050"), L("realistic")
GB = dict(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)


def metrics(y, p, thr):
    pred = p >= thr
    return dict(acc=(pred == y).mean(), prec=precision_score(y, pred, zero_division=0),
                rec=recall_score(y, pred), f1=f1_score(y, pred), pr_auc=average_precision_score(y, p),
                roc_auc=roc_auc_score(y, p))


configs = {
    "A current (in-sample, 8 det.)": (tr_in, FEATURES),
    "B out-of-fold, 8 det.":        (tr_oof, FEATURES),
    "C in-sample, no category_risk": (tr_in, [f for f in FEATURES if f != "category_risk"]),
    "D out-of-fold, no category_risk": (tr_oof, [f for f in FEATURES if f != "category_risk"]),
    "E out-of-fold, no merchant_risk": (tr_oof, [f for f in FEATURES if f != "merchant_risk"]),
}
rows, imps = [], {}
for name, (tr, feats) in configs.items():
    m = GradientBoostingClassifier(**GB).fit(tr[feats], tr["fraud"])
    for split, df in [("50/50 test", te), ("realistic", rt)]:
        p = m.predict_proba(df[feats])[:, 1]
        for thr in (0.05, 0.5):
            rows.append(dict(config=name, split=split, thr=thr, **metrics(df["fraud"].to_numpy(), p, thr)))
    pi = permutation_importance(m, te[feats], te["fraud"], scoring="average_precision",
                                n_repeats=10, random_state=0)
    imps[name] = pd.DataFrame({"gb_importance": m.feature_importances_,
                               "perm_importance_prauc": pi.importances_mean}, index=feats)

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
for name, df in imps.items():
    print(f"\n{name}\n{df.round(4).to_string()}")
res.to_csv(os.path.join(cache, "dominance_results.csv"), index=False)
