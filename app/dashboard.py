import streamlit as st
import pandas as pd
import pickle
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "final_time_series_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "risk_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "..", "models", "feature_columns.pkl")

st.set_page_config(page_title="AI Governance Monitor", layout="wide")

st.title("AI Governance Monitoring System")

data = pd.read_csv(DATA_PATH)
districts = sorted(data["District"].unique())

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

feature_cols = joblib.load(FEATURE_PATH)

st.sidebar.header("Select District")
district = st.sidebar.selectbox("District", districts)

if st.sidebar.button("Predict Future Risk"):

    district_data = data[data["District"] == district]
    latest = district_data.sort_values("Year").iloc[-1:]

    X = latest.copy()
    X = X.drop(columns=["District", "Year"], errors="ignore")
    X = X.reindex(columns=feature_cols, fill_value=0)

    risk_score = model.predict_proba(X)[0][1]

    result = {
        "district": district,
        "predicted_risk_year": int(latest["Year"].values[0]) + 1,
        "data_year_used": int(latest["Year"].values[0]),
        "risk_score": float(risk_score),
        "explanations": []
    }

    for col in feature_cols:
        if col in data.columns:
            current = float(X[col].values[0])
            avg = float(data[col].median())

            if avg != 0:
                percent_diff = ((current - avg) / avg) * 100
            else:
                percent_diff = 0

            result["explanations"].append({
                "feature": col,
                "current": current,
                "average": avg,
                "percent_diff": percent_diff
            })

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)
    col1.metric("District", result["district"])
    col2.metric("Prediction Year", result["predicted_risk_year"])

    st.write("Data used:", result["data_year_used"])

    risk_percent = int(result["risk_score"] * 100)

    if result["risk_score"] < 0.3:
        label = "Low Risk"
        st.success(f"Status: {label}")
    elif result["risk_score"] < 0.7:
        label = "Moderate Risk"
        st.warning(f"Status: {label}")
    else:
        label = "High Risk"
        st.error(f"Status: {label}")

    st.write(f"Risk Probability: {risk_percent}%")

    st.subheader("Summary")

    if label == "Low Risk":
        st.write("The scheme is expected to perform normally with low risk.")
    elif label == "Moderate Risk":
        st.write("There are some potential risks that should be monitored.")
    else:
        st.write("The scheme is at high risk and needs urgent attention.")

    st.subheader("Recommended Action")

    if label == "Low Risk":
        st.write("Everything is stable. Continue regular monitoring.")
    elif label == "Moderate Risk":
        st.write("Some warning signs detected. Review the scheme performance.")
    else:
        st.write("High risk detected. Immediate intervention is required.")

    st.subheader("Key Observations")

    def clean_name(feature):
        name = feature.lower()
        if "rolling" in name:
            return "Financial Trend"
        if "metric" in name:
            return "Financial Indicator"
        return "Scheme Indicator"

    filtered = [
        exp for exp in result["explanations"]
        if "sno" not in exp["feature"].lower()
        and "id" not in exp["feature"].lower()
    ]

    filtered = sorted(filtered, key=lambda x: abs(x["percent_diff"]), reverse=True)[:3]

    for exp in filtered:
        feature = clean_name(exp["feature"])
        current = exp["current"]
        avg = exp["average"]
        percent_diff = exp["percent_diff"]

        if percent_diff > 10:
            status = "significantly higher than usual"
            box = st.warning
        elif percent_diff < -10:
            status = "significantly lower than usual"
            box = st.error
        else:
            status = "within normal range"
            box = st.success

        box(f"""
**{feature}**

Current value: ₹{current:,.0f}  
Typical value: ₹{avg:,.0f}  

Difference from normal: {percent_diff:.1f}%  
This is {status}.
""")