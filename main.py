import pandas as pd
from detectors import amount_anomaly
from detectors import category_risk   # Mansi Ganga Nayak
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
#category_risk learns category fraud rates from the labeled train split, so test rows never see their own labels
train["category_risk"] = category_risk.score(train, train)
test["category_risk"]  = category_risk.score(test, train)

#combine detecors into ensemble scores using weights
#right now only one detector exists, so weight = 1.0 and ensemble == amount_anomaly
#as other detectors are added, add weighted terms here like this:
#train["score"] = 0.6 * train["amount_anomaly"] + 0.3 * train["velocity"] + .1 * train["another_detector]"]
train["score"] = 1.0 * train["amount_anomaly"]
test["score"]  = 1.0 * test["amount_anomaly"]

#Calculate accuracy for each detector and the ensemble on both splits, more test data will be added here in the future,
#but Joseph will handle all that so focus on your individual detectors for now
amount_anomaly_train = acc(train["amount_anomaly"], train["fraud"])
amount_anomaly_test  = acc(test["amount_anomaly"],  test["fraud"])
category_risk_train = acc(train["category_risk"], train["fraud"])
category_risk_test  = acc(test["category_risk"],  test["fraud"])
ensemble_train = acc(train["score"], train["fraud"])
ensemble_test  = acc(test["score"],  test["fraud"])

#output name and accuracy of each detector and the ensemble
print(f"{'name':<18} {'train':>7} {'test'}")
print(f"{'amount_anomaly':<20} {amount_anomaly_train:.4f}  {amount_anomaly_test:.4f}")
print(f"{'category_risk':<20} {category_risk_train:.4f}  {category_risk_test:.4f}")
#print(f"{'SAMPLE_detector':<20} {amount_anomaly_train:.4f}  {amount_anomaly_test:.4f}")
print(f"{'ensemble':<20} {ensemble_train:.4f}  {ensemble_test:.4f}")
