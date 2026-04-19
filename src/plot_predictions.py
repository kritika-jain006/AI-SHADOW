import pandas as pd
import pickle
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/final_time_series_dataset.csv"
MODEL_PATH = "models/risk_model.pkl"

df = pd.read_csv(DATA_PATH)

# load model
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

y_actual = df["future_risk"]

y_pred = model.predict(X)

# plot
plt.figure(figsize=(10,5))

plt.plot(y_actual.values, label="Actual Risk", marker="o")
plt.plot(y_pred, label="Predicted Risk", marker="x")

plt.title("Actual vs Predicted Risk")
plt.xlabel("Samples")
plt.ylabel("Risk (0 or 1)")
plt.legend()

plt.show()