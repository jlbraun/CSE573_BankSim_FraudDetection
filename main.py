import os
import glob
import numpy as np
import pandas as pd
import kagglehub
from dotenv import load_dotenv

from detectors import amount_anomaly
# from tests import velocity        # teammate
# from tests import category_risk   # teammate
# from tests import timing          # teammate

load_dotenv()

DATA = "data/bs140513_032310.csv"
os.makedirs("data", exist_ok=True)

# import the data from the Kaggle dataset
# if statement blocks the download if the file already exists
if not os.path.exists(DATA):
    path = kagglehub.dataset_download("ealaxi/banksim1")
    csv = glob.glob(f"{path}/**/*.csv", recursive=True)
    pd.read_csv(csv[0]).to_csv(DATA, index=False)

df = pd.read_csv(DATA)
for c in df.select_dtypes("object"):
    df[c] = df[c].str.strip("'")

# Run each imported test, which all return a value between 0 and 1
df["score_amount_anomaly"] = amount_anomaly.score(df)
# this is where you would add the other tests once they are built

#Combine all the scores into a single score, this may be changed later
df["score"] = df["score_amount_anomaly"]   # add weighted terms as tests arrive
df["confidence_%"] = (df["score"] * 100).round(1) # this line calculates the confidence percentage based on combined score
#rn it is only one test so it will look different later

#output to terminal for review and debugging, feel free to change this as you want
#This prints out the first 10 rows of the dataset with the listed columns
print(df[["customer", "amount", "category", "confidence_%"]].head(10))
#The following 2 lines prints a summary of the "score" column accross all rows
print("\nDistribution:")
print(df["score"].describe())
#prints out the 10 rows with the highest scores
print("\nTop 10 most suspicious:")
print(df.nlargest(10, "score")[["customer", "amount", "category", "confidence_%"]].to_string(index=False))
# will add more in depth testing later, someone will probably be assigned to making testing more automated so we can properly gauge our
# model, this is just for a more comprehensive evaluation.

# uncomment the line below to save output as csv file
# df.to_csv("scores.csv", index=False)