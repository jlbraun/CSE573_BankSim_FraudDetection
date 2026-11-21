import pandas as pd
#### HIRA ADDED VELOCITY####
from detectors import amount_anomaly, merchant_risk, category_amount, category_risk, velocity
from detectors import merchant_novelty
from detectors import time_gap, category_novelty   # Mansi Ganga Nayak
from sklearn.linear_model import LogisticRegression #this is for the ensemble model
from sklearn.ensemble import GradientBoostingClassifier #this is for boosting the ensemble
#from sklearn.model_selection import GridSearchCV

#load the imported data files, if you don't have them run dataCleaner.py, if it doens't work then read the README
train = pd.read_csv("data/banksim_downsampled_train.csv")
test = pd.read_csv("data/banksim_downsampled_test.csv")

#returns the percent of the time the predicted scores match the actual labels
#current score threshold is .05, this is arbitrary as it yielded a higher score when testing then the default of .5
#this is because of how my amount_anomaly.py created fraud scores and in the future I will probably make a more
#refined version for testing all detectors. Will probably automate it so we can find the ideal threshold
#per-detector decision thresholds, tuned on the 50/50 TRAIN split (analysis/tune.py, Mansi)
#a single .05 cutoff made strong detectors look weak (e.g. category_novelty 52% -> 89% at its own cutoff)
THRESHOLDS = {
    "amount_anomaly": 0.0346, "merchant_risk": 0.4167, "category_amount": 0.8667,
    "category_risk": 0.2273, "velocity": 0.05, "merchant_novelty": 0.1429,
    "time_gap": 0.3333, "category_novelty": 0.5, "score": 0.05,   #ensemble kept at .05 (0% missed fraud on the 50/50 split)
}
#NOTE: on realistic data (0.25% fraud) the ensemble needs a much higher cutoff, see analysis/RESULTS_tuning.md
#(0.05 -> 4.8% precision; 0.898 -> 30% precision at 89% recall; 0.966 -> 67% precision at 66% recall)

def acc(scores, labels, thr=0.05):
    return ((scores >= thr) == labels).mean()

def conf(scores, labels, thr=0.05):
    pred = scores >= thr
    fraud = labels == 1
    tp = ((pred) & (fraud)).mean()
    fp = ((pred) & (~fraud)).mean()
    tn = ((~pred) & (~fraud)).mean()
    fn = ((~pred) & (fraud)).mean()
    return tp, fp, tn, fn

#run each detector basically adding a new column at the end to record the "Score" or what number the detector gives each row
#This score will be used in acc function to compare against the actual labels
train["amount_anomaly"] = amount_anomaly.score(train)
test["amount_anomaly"]  = amount_anomaly.score(test)
#learned from train only
#smoothing tuned 20 -> 5 (analysis/tune.py)
train["merchant_risk"] = merchant_risk.score(train, train, smoothing=5.0)
test["merchant_risk"]  = merchant_risk.score(test, train, smoothing=5.0)

train["category_amount"] = category_amount.score(train, train)
test["category_amount"]  = category_amount.score(test, train)
#category_risk learns category fraud rates from the labeled train split, so test rows never see their own labels
train["category_risk"] = category_risk.score(train, train, smoothing=5.0)   #tuned 20 -> 5
test["category_risk"]  = category_risk.score(test, train, smoothing=5.0)

train["velocity"] = velocity.score(train)
test["velocity"]  = velocity.score(test)
#merchant_novelty counts earlier purchases from the full dataset (no labels used), since downsampling removes most history
history = pd.read_csv("data/bs140513_032310.csv")
train["merchant_novelty"] = merchant_novelty.score(train, history)
test["merchant_novelty"]  = merchant_novelty.score(test, history)
#time_gap and category_novelty only look at each customer's EARLIER transactions in the full history
train["time_gap"] = time_gap.score(train, history)
test["time_gap"]  = time_gap.score(test, history)

train["category_novelty"] = category_novelty.score(train, history, k=0.0)   #tuned k 2 -> 0
test["category_novelty"]  = category_novelty.score(test, history, k=0.0)


#START OF GRADIENT BOOSTING ENSEMBLE:
model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=6,          #tuned 4 -> 6 (analysis/tune.py, PR-AUC on realistic validation set)
    learning_rate=0.03,   #tuned 0.05 -> 0.03
    subsample=0.8,
    random_state=42,
    max_features=0.5
)
FEATURES = ["amount_anomaly", "merchant_risk", "merchant_novelty", "velocity", "time_gap", "category_amount", "category_risk", "category_novelty"]

model.fit(train[FEATURES], train["fraud"])
train["score"] = model.predict_proba(train[FEATURES])[:, 1]
test["score"]  = model.predict_proba(test[FEATURES])[:, 1]


# Show feature importance
print("\nGradient Model Feature Importance: ")
for name, imp in zip(FEATURES, model.feature_importances_):
    print(f"{name:<20} {imp:.3f}")
# END OF GRADIENT BOOSTING ENSEMBLE


#Calculate accuracy for each detector and the ensemble on both splits, more test data will be added here in the future,
#but Joseph will handle all that so focus on your individual detectors for now
amount_anomaly_train = acc(train["amount_anomaly"], train["fraud"])
amount_anomaly_test  = acc(test["amount_anomaly"],  test["fraud"])
merchant_risk_train = acc(train["merchant_risk"], train["fraud"])
merchant_risk_test  = acc(test["merchant_risk"],  test["fraud"])
category_amount_train = acc(train["category_amount"], train["fraud"])
category_amount_test  = acc(test["category_amount"],  test["fraud"])
category_risk_train = acc(train["category_risk"], train["fraud"])
category_risk_test  = acc(test["category_risk"],  test["fraud"])
velocity_train = acc(train["velocity"], train["fraud"])
velocity_test  = acc(test["velocity"],  test["fraud"])
merchant_novelty_train = acc(train["merchant_novelty"], train["fraud"])
merchant_novelty_test  = acc(test["merchant_novelty"],  test["fraud"])
time_gap_train = acc(train["time_gap"], train["fraud"])
time_gap_test  = acc(test["time_gap"],  test["fraud"])
category_novelty_train = acc(train["category_novelty"], train["fraud"]) 
category_novelty_test  = acc(test["category_novelty"],  test["fraud"])
ensemble_train = acc(train["score"], train["fraud"])
ensemble_test  = acc(test["score"],  test["fraud"])

# --- fraud rate per score bucket, for each detector ---
bins = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
#increased the bin size to include 1.0 for the fact that detectors can return a score of 1.0 and that is currently excluded from our logic
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

#bucket tables
for name, col in COLS:
    if name == "ensemble":
        continue
    for split, df in [("train", train), ("test", test)]:
        bucket = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True, right=False)
        grouped = df.groupby(bucket, observed=False)["fraud"].agg(["mean", "count"])
        print(f"\n{name} — {split}")
        print(f"{'score range':<12} {'fraud %':>8} {'rows':>8}")
        for rng, row in grouped.iterrows():
            print(f"{rng:<12} {row['mean']*100:>7.1f}% {int(row['count']):>8}")

#accuracy and confusion matrix
for split, df in [("train", train), ("test", test)]:
    print(f"\n {split.upper()}:")
    print(f"{'name':<20} {'acc':>7} {'TP%':>7} {'FP%':>7} {'TN%':>7} {'FN%':>7}")
    for name, col in COLS:
        s, l = df[col], df["fraud"]
        t = THRESHOLDS.get(col, 0.05)
        tp, fp, tn, fn = conf(s, l, t)
        print(f"{name:<20} {acc(s,l,t):>7.4f} {tp*100:>6.1f}% {fp*100:>6.1f}% {tn*100:>6.1f}% {fn*100:>6.1f}%")
