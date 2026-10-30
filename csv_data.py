import pandas as pd

df = pd.read_csv("data/bs140513_032310.csv")
for c in df.select_dtypes("object"):
    df[c] = df[c].str.strip("'")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Rows            : {len(df):,}")
print(f"Columns         : {len(df.columns)}")
print(f"Fraud rows      : {int(df['fraud'].sum()):,}")
print(f"Non-fraud rows  : {int((df['fraud'] == 0).sum()):,}")
print(f"Fraud rate      : {df['fraud'].mean():.4%}")
print(f"Step range      : {df['step'].min()} to {df['step'].max()}")
print(f"Unique customers: {df['customer'].nunique():,}")
print(f"Unique merchants: {df['merchant'].nunique():,}")
print(f"Unique categories: {df['category'].nunique()}")
print(f"Unique ages     : {sorted(df['age'].unique())}")
print(f"Unique genders  : {sorted(df['gender'].unique())}")
print(f"Unique zipcodes : {df['zipcodeOri'].nunique()}")

print("\n" + "=" * 60)
print("FRAUD RATE BY CATEGORY")
print("=" * 60)
cat = df.groupby("category")["fraud"].agg(["mean", "count", "sum"])
cat.columns = ["fraud_rate", "total", "frauds"]
cat = cat.sort_values("fraud_rate", ascending=False)
cat["fraud_rate"] = (cat["fraud_rate"] * 100).round(2)
for name, row in cat.iterrows():
    print(f"{name:<25} {row['fraud_rate']:>6.2f}%  ({int(row['frauds']):>5} / {int(row['total']):>6})")

print("\n" + "=" * 60)
print("FRAUD RATE BY MERCHANT (top 15 by fraud count)")
print("=" * 60)
mer = df.groupby("merchant")["fraud"].agg(["mean", "count", "sum"])
mer.columns = ["fraud_rate", "total", "frauds"]
mer = mer.sort_values("frauds", ascending=False).head(15)
for name, row in mer.iterrows():
    print(f"{name:<15} {row['fraud_rate']*100:>6.2f}%  ({int(row['frauds']):>4} / {int(row['total']):>6})")

print("\n" + "=" * 60)
print("FRAUD RATE BY AGE")
print("=" * 60)
age = df.groupby("age")["fraud"].agg(["mean", "count"])
for name, row in age.iterrows():
    print(f"age {name:<3} {row['mean']*100:>6.2f}%  ({int(row['count']):>6} rows)")

print("\n" + "=" * 60)
print("FRAUD RATE BY GENDER")
print("=" * 60)
gen = df.groupby("gender")["fraud"].agg(["mean", "count"])
for name, row in gen.iterrows():
    print(f"{name:<3} {row['mean']*100:>6.2f}%  ({int(row['count']):>6} rows)")

print("\n" + "=" * 60)
print("AMOUNT DISTRIBUTION")
print("=" * 60)
print(df["amount"].describe().round(2))
print(f"\nZero-amount rows : {(df['amount'] == 0).sum():,}")
print(f"Amounts ending .00: {(df['amount'] % 1 == 0).sum():,}")

print("\n" + "=" * 60)
print("AMOUNT: FRAUD vs NORMAL")
print("=" * 60)
print(df.groupby("fraud")["amount"].describe().round(2))

print("\n" + "=" * 60)
print("TRANSACTIONS PER CUSTOMER")
print("=" * 60)
tpc = df.groupby("customer").size()
print(tpc.describe().round(2))
print(f"Customers with 1 txn : {(tpc == 1).sum()}")

print("\n" + "=" * 60)
print("TRANSACTIONS PER MERCHANT")
print("=" * 60)
tpm = df.groupby("merchant").size()
print(tpm.describe().round(2))

print("\n" + "=" * 60)
print("TRANSACTIONS PER STEP (first 10, last 10)")
print("=" * 60)
tps = df.groupby("step").size()
print("First 10 steps:")
print(tps.head(10).to_string())
print("\nLast 10 steps:")
print(tps.tail(10).to_string())

print("\n" + "=" * 60)
print("TOP 10 LARGEST TRANSACTIONS")
print("=" * 60)
top = df.nlargest(10, "amount")[["step", "customer", "merchant", "category", "amount", "fraud"]]
print(top.to_string(index=False))

print("\n" + "=" * 60)
print("ROWS PER COLUMN (missing values)")
print("=" * 60)
print(df.isna().sum().to_string())