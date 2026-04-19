import pandas as pd
import os

INTERIM_PATH = "data/master_dataset.csv"
PROCESSED_PATH = "data/processed/final_time_series_dataset.csv"

def create_time_series_features():

    df = pd.read_csv(INTERIM_PATH)

    df.sort_values(by=["District", "Year"], inplace=True)


    metric_cols = [
        col for col in df.columns
        if "metric" in col and "yoy" not in col and "rolling" not in col and "status" not in col
    ]

    print("Number of metric columns:", len(metric_cols))

    def interpret_metric(current, avg):
        if pd.isna(avg):
            return "No trend yet"
        elif current > avg * 1.1:
            return "High increase"
        elif current > avg:
            return "Slight increase"
        elif current < avg * 0.9:
            return "Decrease"
        else:
            return "Stable"

    for col in metric_cols:

        
        df[f"{col}_yoy_change"] = df.groupby("District")[col].diff()

        df[f"{col}_rolling_avg"] = (
            df.groupby("District")[col]
            .rolling(3)
            .mean()
            .reset_index(0, drop=True)
        )

        df[f"{col}_status"] = df.apply(
            lambda row: interpret_metric(
                row[col], row[f"{col}_rolling_avg"]
            ),
            axis=1
        )

    df = df.copy()

    
    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(PROCESSED_PATH, index=False)

    print("Time-series features created")
    print("Total columns now:", len(df.columns))
    print(df[metric_cols].head())


if __name__ == "__main__":
    create_time_series_features()