import pandas as pd
#### HIRA ADDED VELOCITY####
from detectors import amount_anomaly, merchant_risk, category_amount, category_risk, velocity
from detectors import merchant_novelty
from detectors import time_gap, category_novelty   # Mansi Ganga Nayak
#from detectors import SAMPLE_test

#load the imported data files, if you don't have them run dataCleaner.py, if it doens't work then read the README
train = pd.read_csv("data/banksim_downsampled_train.csv")
test = pd.read_csv("data/banksim_downsampled_test.csv")

#returns the percent of the time the predicted scores match the actual labels
#current score threshold is .05, this is arbitrary as it yielded a higher score when testing then the default of .5
#this is because of how my amount_anomaly.py created fraud scores and in the future I will probably make a more
#refined version for testing all detectors. Will probably automate it so we can find the ideal threshold
def acc(scores, labels):
    return ((scores >= 0.05) == labels).mean()

#run each detector basically adding a new column at the end to record the "Score" or what number the detector gives each row
#This score will be used in acc function to compare against the actual labels
train["amount_anomaly"] = amount_anomaly.score(train)
test["amount_anomaly"]  = amount_anomaly.score(test)
#learned from train only
train["merchant_risk"] = merchant_risk.score(train, train)
test["merchant_risk"]  = merchant_risk.score(test, train)

train["category_amount"] = category_amount.score(train, train)
test["category_amount"]  = category_amount.score(test, train)
#category_risk learns category fraud rates from the labeled train split, so test rows never see their own labels
train["category_risk"] = category_risk.score(train, train)
test["category_risk"]  = category_risk.score(test, train)

train["velocity"] = velocity.score(train)
test["velocity"]  = velocity.score(test)
#merchant_novelty counts earlier purchases from the full dataset (no labels used), since downsampling removes most history
history = pd.read_csv("data/bs140513_032310.csv")
train["merchant_novelty"] = merchant_novelty.score(train, history)
test["merchant_novelty"]  = merchant_novelty.score(test, history)
#time_gap and category_novelty only look at each customer's EARLIER transactions in the full history
train["time_gap"] = time_gap.score(train, history)
test["time_gap"]  = time_gap.score(test, history)

train["category_novelty"] = category_novelty.score(train, history)
test["category_novelty"]  = category_novelty.score(test, history)

#combine detecors into ensemble scores using weights
#right now only one detector exists, so weight = 1.0 and ensemble == amount_anomaly
#as other detectors are added, add weighted terms here like this:
#train["score"] = 0.6 * train["amount_anomaly"] + 0.3 * train["velocity"] + .1 * train["another_detector]"]

#NEW UPDATE I AM FULLY REDOING THE ENSEMBLE SO IGNORE THIS FOR NOW.
train["score"] = 1.0 * train["amount_anomaly"]
test["score"]  = 1.0 * test["amount_anomaly"]

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
ensemble_train = acc(train["score"], train["fraud"])
ensemble_test  = acc(test["score"],  test["fraud"])
#add time_gap and category_amount
time_gap_train = acc(train["time_gap"], train["fraud"])
time_gap_test  = acc(test["time_gap"],  test["fraud"])
category_novelty_train = acc(train["category_novelty"], train["fraud"]) 
category_novelty_test  = acc(test["category_novelty"],  test["fraud"])

# --- fraud rate per score bucket, for each detector ---
bins = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
#increased the bin size to include 1.0 for the fact that detectors can return a score of 1.0 and that is currently excluded from our logic
labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]

DETECTOR_COLUMNS = [
    ("amount_anomaly",  "amount_anomaly"),
    ("merchant_risk",   "merchant_risk"),
    ("category_amount", "category_amount"),
    ("category_risk",   "category_risk"),
    ("velocity",        "velocity"),
    ("merchant_novelty", "merchant_novelty"),
]

for name, col in DETECTOR_COLUMNS:
    for split, df in [("train", train), ("test", test)]:
        bucket = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True, right=False)        
        grouped = df.groupby(bucket, observed=False)["fraud"].agg(["mean", "count"])

        print(f"\n{name} — {split}")
        print(f"{'score range':<12} {'fraud %':>8} {'rows':>8}")
        for rng, row in grouped.iterrows():
            print(f"{rng:<12} {row['mean']*100:>7.1f}% {int(row['count']):>8}")


#output name and accuracy of each detector and the ensemble
print(f"{'name':<18} {'train':>7} {'test'}")
print(f"{'amount_anomaly':<20} {amount_anomaly_train:.4f}  {amount_anomaly_test:.4f}")
print(f"{'merchant_risk':<20} {merchant_risk_train:.4f}  {merchant_risk_test:.4f}")
print(f"{'category_amount':<20} {category_amount_train:.4f}  {category_amount_test:.4f}")
print(f"{'category_risk':<20} {category_risk_train:.4f}  {category_risk_test:.4f}")
print(f"{'velocity':<20} {velocity_train:.4f}  {velocity_test:.4f}")
print(f"{'merchant_novelty':<20} {merchant_novelty_train:.4f}  {merchant_novelty_test:.4f}")
#print(f"{'SAMPLE_detector':<20} {amount_anomaly_train:.4f}  {amount_anomaly_test:.4f}")
print(f"{'ensemble':<20} {ensemble_train:.4f}  {ensemble_test:.4f}")

# Mansi Ganga Nayak - time gap and category novelty
print(f"{'time_gap':<20} {acc(train['time_gap'], train['fraud']):.4f}  {acc(test['time_gap'], test['fraud']):.4f}")
print(f"{'category_novelty':<20} {acc(train['category_novelty'], train['fraud']):.4f}  {acc(test['category_novelty'], test['fraud']):.4f}")
