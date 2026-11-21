import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/bs140513_032310.csv")
for c in df.select_dtypes("object"):
    df[c] = df[c].str.strip("'")

original_uneven_test, remaining = train_test_split(
    df, test_size=0.20, random_state=1, stratify=df["fraud"]
)
original_uneven_test.to_csv("data/original_uneven_test.csv", index=False)

fraud  = remaining[remaining["fraud"] == 1]
normal = remaining[remaining["fraud"] == 0]

down = (
    pd.concat([fraud, normal.sample(n=len(fraud), random_state=1)])
    .sample(frac=1, random_state=1)
    .reset_index(drop=True)
)

up = (
    pd.concat([fraud.sample(n=len(normal), replace=True, random_state=1), normal])
    .sample(frac=1, random_state=1)
    .reset_index(drop=True)
)

down_train, down_test = train_test_split(
    down, test_size=0.2, random_state=1, stratify=down["fraud"]
)
down_train.to_csv("data/banksim_downsampled_train.csv", index=False)
down_test.to_csv("data/banksim_downsampled_test.csv", index=False)

up_train, up_test = train_test_split(
    up, test_size=0.2, random_state=1, stratify=up["fraud"]
)
up_train.to_csv("data/banksim_upsampled_train.csv", index=False)
up_test.to_csv("data/banksim_upsampled_test.csv", index=False)

key = ["step", "customer", "merchant", "amount"]
train_keys = set(map(tuple, down_train[key].values))
test_keys  = set(map(tuple, original_uneven_test[key].values))
assert not (train_keys & test_keys)