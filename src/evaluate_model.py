import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROCESSED_PATH = "data/processed/final_time_series_dataset.csv"
MODEL_PATH = "models/risk_model.pkl"


def evaluate():

    df = pd.read_csv(PROCESSED_PATH)

    target = "future_risk"

    drop_cols = [
        "district",
        "year",
        "future_risk",
        "future_utilization"
    ]

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target]

    X = X.select_dtypes(include="number")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = pickle.load(open(MODEL_PATH, "rb"))

    predictions = model.predict(X_test)

    print("\nClassification Report\n")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix\n")
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    evaluate()