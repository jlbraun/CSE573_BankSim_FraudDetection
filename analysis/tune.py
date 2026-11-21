"""Hyperparameter tuning for detectors + ensemble (Mansi Ganga Nayak).

Protocol (no test leakage):
  * train      = data/banksim_downsampled_train.csv (50/50), label detectors out-of-fold
  * realistic  = full CSV minus training rows (0.25% fraud), split 50/50 stratified into
                 VALIDATION (used for every tuning decision) and TEST (reported once).
  * metric     = PR-AUC on the fraud class (accuracy is meaningless at 0.25% fraud).
Steps: 1) detector parameters (one at a time), 2) GridSearch for Gradient Boosting,
3) decision threshold (max F1 on validation), 4) per-detector thresholds for main.py's
accuracy table. Usage: python analysis/tune.py [cache_dir]
"""
import itertools, json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.features import FEATURES, build_features, load, realistic_test

out = sys.argv[1] if len(sys.argv) > 1 else "analysis_cache"
os.makedirs(out, exist_ok=True)
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

tr, te, full = load()
rt = realistic_test(full, tr)
val, tst = train_test_split(rt, test_size=0.5, stratify=rt["fraud"], random_state=7)
log(f"train {len(tr)}  val {len(val)} ({val.fraud.mean():.3%})  test {len(tst)} ({tst.fraud.mean():.3%})")

BASE_GB = dict(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)


def feats(params):
    Xtr = build_features(tr, tr, full, params, oof=True)
    Xva = build_features(val, tr, full, params)
    return Xtr, Xva


def gb_prauc(Xtr, Xva, gb=BASE_GB):
    m = GradientBoostingClassifier(**gb).fit(Xtr, tr["fraud"])
    return average_precision_score(val["fraud"], m.predict_proba(Xva)[:, 1])


# ---- 1) detector parameters (only ones that change score ORDER matter to the trees)
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

# ---- 2) Gradient Boosting grid (validated on realistic VALIDATION set)
gb_grid = dict(n_estimators=[100, 300], max_depth=[2, 3, 4, 6], learning_rate=[0.03, 0.1],
               subsample=[0.8, 1.0], min_samples_leaf=[1, 50])
gb_rows, best_gb, best_s = [], None, -1
for combo in itertools.product(*gb_grid.values()):
    gb = dict(zip(gb_grid, combo), random_state=42)
    s = gb_prauc(Xtr, Xva, gb)
    gb_rows.append({**gb, "val_pr_auc": s})
    if s > best_s:
        best_gb, best_s = gb, s
log("best GB:", best_gb, f"val PR-AUC {best_s:.4f}")

# ---- 3) threshold: maximise F1 on validation
model = GradientBoostingClassifier(**best_gb).fit(Xtr, tr["fraud"])
pv = model.predict_proba(Xva)[:, 1]
P, R, T = precision_recall_curve(val["fraud"], pv)
f1 = 2 * P * R / np.clip(P + R, 1e-12, None)
thr = float(T[np.argmax(f1[:-1])])
log(f"best threshold {thr:.4f}")


def report(name, m, X, y, thr):
    p = m.predict_proba(X)[:, 1]
    pred = p >= thr
    return dict(setup=name, threshold=round(thr, 4), pr_auc=average_precision_score(y, p),
                roc_auc=roc_auc_score(y, p), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), f1=f1_score(y, pred), accuracy=(pred == y).mean(),
                alerts=int(pred.sum()), frauds=int(y.sum()))


old = GradientBoostingClassifier(**BASE_GB).fit(build_features(tr, tr, full), tr["fraud"])
Xte_old = build_features(tst, tr, full)
Xte50 = build_features(te, tr, full, params)
Xte50_old = build_features(te, tr, full)
final = pd.DataFrame([
    report("current main.py (thr .05)", old, Xte_old, tst["fraud"].to_numpy(), 0.05),
    report("current main.py (thr .50)", old, Xte_old, tst["fraud"].to_numpy(), 0.5),
    report("tuned (thr from val)", model, Xte, tst["fraud"].to_numpy(), thr),
    report("current, 50/50 test (thr .05)", old, Xte50_old, te["fraud"].to_numpy(), 0.05),
    report("tuned, 50/50 test (thr .5)", model, Xte50, te["fraud"].to_numpy(), 0.5),
])

# ---- 4) per-detector thresholds for main.py's accuracy table (chosen on the 50/50 TRAIN)
Xtr_in = build_features(tr, tr, full, params)
det_thr = []
for f in FEATURES:
    cands = np.unique(np.round(Xtr_in[f], 4))
    cands = cands if len(cands) < 400 else np.quantile(Xtr_in[f], np.linspace(0, 1, 401))
    accs = [((Xtr_in[f] >= c) == tr["fraud"]).mean() for c in cands]
    c = float(cands[int(np.argmax(accs))])
    det_thr.append(dict(detector=f, best_threshold=round(c, 4), train_acc=max(accs),
                        test_acc_at_best=((Xte50[f] >= c) == te["fraud"]).mean(),
                        test_acc_at_005=((Xte50[f] >= 0.05) == te["fraud"]).mean(),
                        standalone_val_pr_auc=average_precision_score(val["fraud"], Xva[f])))

pd.set_option("display.width", 220)
print("\nDETECTOR PARAM SEARCH\n", pd.DataFrame(det_rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nTOP GB SETTINGS\n", pd.DataFrame(gb_rows).sort_values("val_pr_auc", ascending=False).head(8).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nFINAL (held-out realistic TEST unless noted)\n", final.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nPER-DETECTOR THRESHOLDS\n", pd.DataFrame(det_thr).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nFEATURE IMPORTANCE (tuned)\n", pd.Series(model.feature_importances_, FEATURES).round(4).to_string())

json.dump({"detector_params": params, "gb_params": best_gb, "threshold": thr},
          open(os.path.join(out, "tuned_params.json"), "w"), indent=2)
pd.DataFrame(det_rows).to_csv(os.path.join(out, "detector_search.csv"), index=False)
pd.DataFrame(gb_rows).to_csv(os.path.join(out, "gb_search.csv"), index=False)
final.to_csv(os.path.join(out, "final_results.csv"), index=False)
pd.DataFrame(det_thr).to_csv(os.path.join(out, "detector_thresholds.csv"), index=False)
log("done")
