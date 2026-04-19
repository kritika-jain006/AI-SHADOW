import pandas as pd

df = pd.read_csv("data/processed/final_time_series_dataset.csv")

print("Shape:", df.shape)

print("\nSample rows")
print(df[["District","Year","financial_metric_1"]].head(10))

print("\nMissing values")
print(df["financial_metric_1"].isna().sum())

print("\nUnique values")
print(df["financial_metric_1"].describe())