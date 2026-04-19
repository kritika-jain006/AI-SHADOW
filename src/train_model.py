import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

DATA_PATH = "data/processed/final_time_series_dataset.csv"
MODEL_PATH = "models/risk_model.pkl"


def train_model():

    df = pd.read_csv(DATA_PATH)

    # Sort by time
    df.sort_values(by=["Year"], inplace=True)

    # 🔹 Use only rows where future exists (exclude 2024 from training)
    df_train = df[df["future_score"].notna()].copy()

    # 🔹 Drop unwanted columns
    drop_cols = [
        "District",
        "Year",
        "future_risk",
        "future_score",
        "financial_score"
    ]

    X = df_train.drop(columns=drop_cols, errors="ignore")

    # 🔹 Keep only numeric columns (IMPORTANT FIX)
    X = X.select_dtypes(include="number")

    y = df_train["future_risk"]

    # 🔹 Time-based split
    years = sorted(df_train["Year"].unique())

    train_years = years[:-1]
    test_year = years[-1]

    train = df_train[df_train["Year"].isin(train_years)]
    test = df_train[df_train["Year"] == test_year]

    print("Train rows:", len(train))
    print("Test rows:", len(test))

    # 🔹 Align features properly
    X_train = train[X.columns]
    y_train = train["future_risk"]

    X_test = test[X.columns]
    y_test = test["future_risk"]

    # 🔹 Model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )

    # 🔹 Train
    model.fit(X_train, y_train)

    # 🔹 Predict
    preds = model.predict(X_test)

    print("\nModel Evaluation\n")
    print(classification_report(y_test, preds))

    # 🔹 Save model
    os.makedirs("models", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("\nModel saved to:", MODEL_PATH)


if __name__ == "__main__":
    train_model()