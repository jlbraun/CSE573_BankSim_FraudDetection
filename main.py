import pandas as pd
from detectors import amount_anomaly, merchant_risk, category_amount, category_risk, velocity
from detectors import merchant_novelty
from detectors import time_gap, category_novelty 
from sklearn.ensemble import GradientBoostingClassifier #this is for boosting the ensemble
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve  


import json, os
THRESHOLDS = {
    "amount_anomaly": 0.0346, "merchant_risk": 0.4167, "category_amount": 0.8667,
    "category_risk": 0.2273, "velocity": 0.05, "merchant_novelty": 0.1429,
    "time_gap": 0.3333, "category_novelty": 0.5, "score": 0.05,
}
if os.path.exists("data/tuned_thresholds.json"): 
    with open("data/tuned_thresholds.json") as f:
        _tuned = json.load(f)
    for k, v in _tuned.items():
        THRESHOLDS[k] = v["threshold"]
    print(f"[info] loaded tuned thresholds from data/tuned_thresholds.json")

#load the imported data files
train = pd.read_csv("data/banksim_downsampled_train.csv")
test = pd.read_csv("data/realistic_test_heldout.csv")


def report(name, scores, labels, thr):
    pred  = scores >= thr
    fraud = labels == 1
    tp = int(( pred &  fraud).sum()); fp = int(( pred & ~fraud).sum())
    fn = int((~pred &  fraud).sum()); tn = int((~pred & ~fraud).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    ap   = average_precision_score(labels, scores)
    auc  = roc_auc_score(labels, scores)
    print(f"{name:<20} PR-AUC={ap:>6.4f}  ROC-AUC={auc:>6.4f}  "
          f"P={prec:>6.3f}  R={rec:>6.3f}  F1={f1:>6.3f}  "
          f"TP={tp:>5}  FP={fp:>6}  FN={fn:>5}  thr={thr:.4f}")
    return ap

#run each detector (unchanged)
train["amount_anomaly"] = amount_anomaly.score(train)
test["amount_anomaly"]  = amount_anomaly.score(test)

train["merchant_risk"] = merchant_risk.score(train, train, smoothing=20.0)
test["merchant_risk"]  = merchant_risk.score(test, train, smoothing=20.0)

train["category_amount"] = category_amount.score(train, train)
test["category_amount"]  = category_amount.score(test, train)

train["category_risk"] = category_risk.score(train, train, smoothing=5.0)
test["category_risk"]  = category_risk.score(test, train, smoothing=5.0)

train["velocity"] = velocity.score(train)
test["velocity"]  = velocity.score(test)

history = pd.read_csv("data/bs140513_032310.csv")
train["merchant_novelty"] = merchant_novelty.score(train, history)
test["merchant_novelty"]  = merchant_novelty.score(test, history)

train["time_gap"] = time_gap.score(train, history)
test["time_gap"]  = time_gap.score(test, history)

train["category_novelty"] = category_novelty.score(train, history, k=5.0)
test["category_novelty"]  = category_novelty.score(test, history, k=5.0)

#GRADIENT BOOSTING ENSEMBLE (unchanged hyperparams; max_features=3 was already in your file)
model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=1.0,
    min_samples_leaf=1,
    random_state=42,
    max_features=3
)
FEATURES = ["amount_anomaly", "merchant_risk", "merchant_novelty", "velocity", "time_gap",
            "category_amount", "category_risk", "category_novelty"]

model.fit(train[FEATURES], train["fraud"])
train["score"] = model.predict_proba(train[FEATURES])[:, 1]
test["score"]  = model.predict_proba(test[FEATURES])[:, 1]

print("\nGradient Model Feature Importance: ")
for name, imp in zip(FEATURES, model.feature_importances_):
    print(f"{name:<20} {imp:.3f}")

bins = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]

COLS = [
    ("amount_anomaly",   "amount_anomaly"),
    ("merchant_risk",    "merchant_risk"),
    ("category_amount",  "category_amount"),
    ("category_risk",    "category_risk"),
    ("velocity",         "velocity"),
    ("merchant_novelty", "merchant_novelty"),
    ("time_gap",         "time_gap"),
    ("category_novelty", "category_novelty"),
    ("ensemble",         "score"),
]

for name, col in COLS:
    if name == "ensemble":
        continue
    df = test 
    bucket = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True, right=False)
    grouped = df.groupby(bucket, observed=False)["fraud"].agg(["mean", "count"])
    print(f"\n{name} — test")
    print(f"{'score range':<12} {'fraud %':>8} {'rows':>8}")
    for rng, row in grouped.iterrows():
        print(f"{rng:<12} {row['mean']*100:>7.1f}% {int(row['count']):>8}")


print(f"\nTEST (original_uneven_test, fraud rate={test['fraud'].mean():.4%}):")
print(f"{'name':<20} {'PR-AUC':>7} {'ROC-AUC':>8} {'P':>7} {'R':>7} {'F1':>7} {'TP':>6} {'FP':>7} {'FN':>5} {'thr':>7}")
for name, col in COLS:
    s, l = test[col], test["fraud"]
    t = THRESHOLDS.get(col, 0.05)
    report(name, s, l, t)

print("\nDiagnostic — best F1 on TEST (do NOT deploy these; tune on downsampled_test):")
print(f"{'name':<20} {'best_thr':>9} {'F1':>7} {'P':>7} {'R':>7}")
for name, col in COLS:
    s, l = test[col].values, test["fraud"].values
    prec, rec, thr = precision_recall_curve(l, s)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    i = int(f1[:-1].argmax())
    print(f"{name:<20} {thr[i]:>9.4f} {f1[i]:>7.3f} {prec[i]:>7.3f} {rec[i]:>7.3f}")