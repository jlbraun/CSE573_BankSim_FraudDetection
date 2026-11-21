"""Precision/recall trade-off of the tuned ensemble on the held-out realistic test set."""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.features import build_features, load, realistic_test

out = sys.argv[1] if len(sys.argv) > 1 else "analysis_cache"
cfg = json.load(open(os.path.join(out, "tuned_params.json")))
tr, te, full = load(); rt = realistic_test(full, tr)
val, tst = train_test_split(rt, test_size=0.5, stratify=rt["fraud"], random_state=7)
m = GradientBoostingClassifier(**cfg["gb_params"]).fit(
    build_features(tr, tr, full, cfg["detector_params"], oof=True), tr["fraud"])
pv = m.predict_proba(build_features(val, tr, full, cfg["detector_params"]))[:, 1]
pt = m.predict_proba(build_features(tst, tr, full, cfg["detector_params"]))[:, 1]
yv, yt = val["fraud"].to_numpy(), tst["fraud"].to_numpy()
P, R, T = precision_recall_curve(yv, pv)
rows = []
for target in [0.99, 0.95, 0.90, 0.80, None]:
    if target is None:
        f1 = 2*P*R/np.clip(P+R, 1e-12, None); i = int(np.argmax(f1[:-1])); label = "max F1"
    else:
        ok = np.where(R[:-1] >= target)[0]; i = int(ok[np.argmax(P[ok])]); label = f"recall >= {target:.0%}"
    thr = float(T[i]); pred = pt >= thr
    tp = int((pred & (yt == 1)).sum()); fp = int((pred & (yt == 0)).sum())
    rows.append(dict(rule=label, threshold=round(thr, 4), test_precision=tp/max(tp+fp, 1),
                     test_recall=tp/yt.sum(), caught=tp, false_alerts=fp, missed=int(yt.sum()-tp)))
df = pd.DataFrame(rows)
print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
df.to_csv(os.path.join(out, "operating_points.csv"), index=False)
