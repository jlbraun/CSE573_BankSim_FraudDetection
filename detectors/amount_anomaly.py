import numpy as np


def score(df):
    stats = df.groupby("customer")["amount"].agg(["mean", "std"])
    stats["std"] = stats["std"].fillna(0).clip(lower=1)
    z = (df["amount"] - df["customer"].map(stats["mean"])) / df["customer"].map(stats["std"])
    z = z.clip(lower=0)
    return 1 / (1 + np.exp(-(z - 5) / 1.5))
