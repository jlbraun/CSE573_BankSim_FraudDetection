"""Hyperparameter tuning for detectors + ensemble.

Protocol (no test leakage):
  * train      = data/banksim_downsampled_train.csv (50/50)
  * realistic  = data/original_uneven_test.csv (~1.2% fraud), split 50/50 stratified
                 into VALIDATION (all tuning decisions) and TEST (reported once by main.py).
  * metric     = PR-AUC / F1, never accuracy.
Outputs:
  * data/tuned_thresholds.json  <- loaded by main.py (per-detector + ensemble)
  * analysis_cache/*.csv        <- search logs, final results
Usage: python analysis/tune.py [cache_dir]
"""
import itertools, json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.features import FEATURES, build_features, load   # [CHANGED] no realistic_test import

out = sys.argv[1] if len(sys.argv) > 1 else "analysis_cache"
os.makedirs(out, exist_ok=True)
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

tr, te, full = load()                             
rt = pd.read_csv("data/original_uneven_test.csv")
val, tst = train_test_split(rt, test_size=0.5, stratify=rt["fraud"], random_state=7)

log(f"train {len(tr)}  val {len(val)} ({val.fraud.mean():.3%})  "
    f"test {len(tst)} ({tst.fraud.mean():.3%})")

BASE_GB = dict(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)


def feats(params):
    Xtr = build_features(tr, tr, full, params, oof=True)
    Xva = build_features(val, tr, full, params)
    return Xtr, Xva


def gb_prauc(Xtr, Xva, gb=BASE_GB):
    m = GradientBoostingClassifier(**gb).fit(Xtr, tr["fraud"])
    return average_precision_score(val["fraud"], m.predict_proba(Xva)[:, 1])


# ---- 1) detector parameters
GRID = {
    "velocity_window": [1, 2, 3, 7],
    "cnov_k": [0.0, 2.0, 5.0, 10.0],
    "tgap_first": [0.0, 0.5, 1.0],
    "merchant_smoothing": [0.0, 5.0, 20.0, 50.0],
    "category_smoothing": [0.0, 5.0, 20.0, 50.0],
}
params = {}
Xtr, Xva = feats(params)
base = gb_prauc(Xtr, Xva)
log(f"baseline val PR-AUC {base:.4f}")
det_rows = []
for key, values in GRID.items():
    best_v, best_s = None, -1
    for v in values:
        s = gb_prauc(*feats({**params, key: v}))
        det_rows.append(dict(param=key, value=v, val_pr_auc=s))
        log(f"  {key}={v}: {s:.4f}")
        if s > best_s + 1e-4:
            best_v, best_s = v, s
    params[key] = best_v
log("chosen detector params:", params)
Xtr, Xva = feats(params)
Xte = build_features(tst, tr, full, params)

# ---- 2) Gradient Boosting grid
gb_grid = dict(n_estimators=[100, 300], max_depth=[2, 3, 4, 6], learning_rate=[0.03, 0.1],
               subsample=[0.8, 1.0], min_samples_leaf=[1, 50], max_features=[3])
gb_rows, best_gb, best_s = [], None, -1
for combo in itertools.product(*gb_grid.values()):
    gb = dict(zip(gb_grid, combo), random_state=42)
    s = gb_prauc(Xtr, Xva, gb)
    gb_rows.append({**gb, "val_pr_auc": s})
    if s > best_s:
        best_gb, best_s = gb, s
log("best GB:", best_gb, f"val PR-AUC {best_s:.4f}")

# ---- 3) ensemble threshold: maximise F1 on validation
model = GradientBoostingClassifier(**best_gb).fit(Xtr, tr["fraud"])
pv = model.predict_proba(Xva)[:, 1]
P, R, T = precision_recall_curve(val["fraud"], pv)
f1 = 2 * P * R / np.clip(P + R, 1e-12, None)
ens_thr = float(T[np.argmax(f1[:-1])])
log(f"best ensemble threshold {ens_thr:.4f}")


def report(name, m, X, y, thr):
    p = m.predict_proba(X)[:, 1]
    pred = p >= thr
    return dict(setup=name, threshold=round(thr, 4), pr_auc=average_precision_score(y, p),
                roc_auc=roc_auc_score(y, p), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), f1=f1_score(y, pred), alerts=int(pred.sum()),
                frauds=int(y.sum()))


# [CHANGED] baseline uses oof=True so only params differ from tuned
old = GradientBoostingClassifier(**BASE_GB).fit(build_features(tr, tr, full, oof=True), tr["fraud"])
Xte_old = build_features(tst, tr, full)
Xte50 = build_features(te, tr, full, params)
Xte50_old = build_features(te, tr, full)
final = pd.DataFrame([
    report("current main.py (thr .05)", old, Xte_old, tst["fraud"].to_numpy(), 0.05),
    report("current main.py (thr .50)", old, Xte_old, tst["fraud"].to_numpy(), 0.5),
    report("tuned (thr from val)", model, Xte, tst["fraud"].to_numpy(), ens_thr),
    report("current, 50/50 test (thr .05)", old, Xte50_old, te["fraud"].to_numpy(), 0.05),
    report("tuned, 50/50 test (thr .5)", model, Xte50, te["fraud"].to_numpy(), 0.5),
])

# ---- 4) per-detector thresholds — [CHANGED] F1-max on VALIDATION, not accuracy on train
Xval_in = build_features(val, tr, full, params)   # [CHANGED] validation features
det_thr = []
for f in FEATURES:
    s = Xval_in[f].to_numpy()
    y = val["fraud"].to_numpy()
    P, R, T = precision_recall_curve(y, s)
    f1s = 2 * P * R / np.clip(P + R, 1e-12, None)
    best_i = int(np.argmax(f1s[:-1]))
    c = float(T[best_i])
    # also record the precision@recall=0.9 threshold for a conservative operating point
    idx90 = np.where(R[:-1] >= 0.90)[0]
    thr90 = float(T[idx90[0]]) if len(idx90) else c
    det_thr.append(dict(detector=f, threshold_f1=round(c, 4), f1=float(f1s[best_i]),
                        precision=float(P[best_i]), recall=float(R[best_i]),
                        threshold_r90=round(thr90, 4),
                        standalone_val_pr_auc=average_precision_score(y, s)))

# [ADDED] write the file main.py loads
tuned_out = {d["detector"]: {"threshold": d["threshold_f1"], "f1": d["f1"],
                             "precision": d["precision"], "recall": d["recall"],
                             "pr_auc": d["standalone_val_pr_auc"]} for d in det_thr}
tuned_out["score"] = {"threshold": ens_thr, "f1": float(f1[best_i]) if False else float(
    f1_score(val["fraud"], pv >= ens_thr))}
json.dump(tuned_out, open("data/tuned_thresholds.json", "w"), indent=2)
log("wrote data/tuned_thresholds.json")

pd.set_option("display.width", 220)
print("\nDETECTOR PARAM SEARCH\n", pd.DataFrame(det_rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nTOP GB SETTINGS\n", pd.DataFrame(gb_rows).sort_values("val_pr_auc", ascending=False).head(8).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nFINAL (held-out realistic TEST unless noted)\n", final.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nPER-DETECTOR THRESHOLDS\n", pd.DataFrame(det_thr).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nFEATURE IMPORTANCE (tuned)\n", pd.Series(model.feature_importances_, FEATURES).round(4).to_string())

json.dump({"detector_params": params, "gb_params": best_gb, "threshold": ens_thr},
          open(os.path.join(out, "tuned_params.json"), "w"), indent=2)
pd.DataFrame(det_rows).to_csv(os.path.join(out, "detector_search.csv"), index=False)
pd.DataFrame(gb_rows).to_csv(os.path.join(out, "gb_search.csv"), index=False)
final.to_csv(os.path.join(out, "final_results.csv"), index=False)
pd.DataFrame(det_thr).to_csv(os.path.join(out, "detector_thresholds.csv"), index=False)
log("done")