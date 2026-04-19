import pandas as pd

PROCESSED_PATH = "data/processed/final_time_series_dataset.csv"


def create_target():

    df = pd.read_csv(PROCESSED_PATH)

    df.sort_values(by=["District", "Year"], inplace=True)

    # Only original financial metrics
    financial_cols = [
        c for c in df.columns
        if "financial_metric" in c and "yoy" not in c and "rolling" not in c and "status" not in c
    ]

    # Convert to numeric
    for col in financial_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create financial score
    df["financial_score"] = df[financial_cols].sum(axis=1)

    # Future score (next year)
    df["future_score"] = df.groupby("District")["financial_score"].shift(-1)

    # Threshold
    threshold = df["financial_score"].median()

    # Target
    df["future_risk"] = (df["future_score"] < threshold).astype(int)

    # ❗ IMPORTANT: DO NOT DROP NaNs → keep 2024
    # df = df[df["future_score"].notna()]  ❌ REMOVE THIS

    print("Total rows:", len(df))
    print(df["future_risk"].value_counts())

    df.to_csv(PROCESSED_PATH, index=False)


if __name__ == "__main__":
    create_target()