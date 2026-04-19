from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np

app = FastAPI()

# Load model and data
model = pickle.load(open("models/risk_model.pkl", "rb"))
data = pd.read_csv("data/processed/final_time_series_dataset.csv")


@app.get("/predict/{district}")
def predict(district: str):

    try:
        # Filter district data
        df = data[data["District"] == district]

        if df.empty:
            return {"error": "District not found"}

        # Get latest row
        latest_row = df.sort_values("Year").iloc[-1]

        current_year = int(latest_row["Year"])
        prediction_year = current_year + 1

        # Prepare features
        X = latest_row.drop([
            "District",
            "Year",
            "future_risk",
            "future_score",
            "financial_score"
        ], errors="ignore")

        X = pd.DataFrame([X])
        X = X.select_dtypes(include="number")

        # Fix data issues
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        X = X.astype(float)

        # ===== Prediction =====
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        # ===== DYNAMIC EXPLAINABILITY =====
        explanations = []

        feature_names = X.columns
        mean_values = data[feature_names].mean()
        feature_imp = model.feature_importances_

        scores = []

        for i, f in enumerate(feature_names):
            current_val = float(X[f].values[0])
            avg_val = float(mean_values[f])

            # HANDLE BAD DATA
            if pd.isna(current_val) or pd.isna(avg_val) or avg_val == 0:
                continue

            deviation = (current_val - avg_val) / avg_val
            score = abs(deviation) * feature_imp[i]

            scores.append((f, score, deviation, current_val, avg_val))

        # HANDLE EMPTY CASE (IMPORTANT FIX)
        if len(scores) == 0:
            explanations = [
                {
                    "feature": "No significant variation",
                    "impact": 0.0,
                    "current": 0.0,
                    "average": 0.0
                }
            ]
        else:
            top_features = sorted(scores, key=lambda x: x[1], reverse=True)[:3]

            for f, score, deviation, current_val, avg_val in top_features:
                explanations.append({
                    "feature": str(f),
                    "impact": float(deviation),
                    "current": current_val,
                    "average": avg_val
                })

        # ===== Status =====
        if pred == 1:
            status = "HIGH RISK"
            recommendation = "Monitor financial utilization and audit district activities"
        else:
            status = "NORMAL"
            recommendation = "No immediate action required"

        # ===== Response =====
        return {
            "district": district,
            "data_year_used": current_year,
            "predicted_risk_year": prediction_year,
            "risk_status": status,
            "risk_score": float(prob),
            "recommendation": recommendation,
            "explanations": explanations
        }

    except Exception as e:
        return {
            "error": str(e),
            "district": district
        }