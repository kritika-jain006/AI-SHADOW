import pandas as pd
import pickle
import os

DATA_PATH = "data/processed/final_time_series_dataset.csv"
MODEL_PATH = "models/risk_model.pkl"
OUTPUT_PATH = "data/predictions/district_risk_predictions.csv"


def predict():

    df = pd.read_csv(DATA_PATH)

    # 🔹 Use latest year (should now be 2024)
    latest_year = df["Year"].max()
    df = df[df["Year"] == latest_year]

    print("Using data from year:", latest_year)

    model = pickle.load(open(MODEL_PATH, "rb"))

    drop_cols = [
        "District",
        "Year",
        "future_risk",
        "future_score",
        "financial_score"
    ]

    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include="number")

    df["predicted_risk"] = model.predict(X)
    df["risk_probability"] = model.predict_proba(X)[:, 1]

    result = df[["District", "Year", "predicted_risk", "risk_probability"]]

    os.makedirs("data/predictions", exist_ok=True)

    result.to_csv(OUTPUT_PATH, index=False)

    print("Predictions saved to:", OUTPUT_PATH)
    print(result.head())


if __name__ == "__main__":
    predict()