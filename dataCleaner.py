import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/bs140513_032310.csv")
for c in df.select_dtypes("object"):
    df[c] = df[c].str.strip("'")

#this selects all fraudulent transactions for use later
fraud = df[df["fraud"] == 1]
#this selects all normal transactions
normal = df[df["fraud"] == 0]

#these 2 lines create the downsampled and upsampled datasets, but don't store them to a file to save some space
down = pd.concat([fraud, normal.sample(n=len(fraud), random_state=1)]).sample(frac=1, random_state=1).reset_index(drop=True)
up = pd.concat([fraud.sample(n=len(normal), replace=True, random_state=1), normal]).sample(frac=1, random_state=1).reset_index(drop=True)

#uncomment these if you want the upsampled or downsampled without the splits
#down.to_csv("data/banksim_downsampled.csv", index=False)
#up.to_csv("data/banksim_upsampled.csv", index=False)

#this section creates the train and test splits for downsampled data
train, test = train_test_split(down, test_size=0.2, random_state=1, stratify=down["fraud"])
train.to_csv("data/banksim_downsampled_train.csv", index=False)
test.to_csv("data/banksim_downsampled_test.csv", index=False)

#this section creates the train and test splits for upsampled data
train, test = train_test_split(up, test_size=0.2, random_state=1, stratify=up["fraud"])
train.to_csv("data/banksim_upsampled_train.csv", index=False)
test.to_csv("data/banksim_upsampled_test.csv", index=False)