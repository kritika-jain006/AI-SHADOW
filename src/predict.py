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

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)
    col1.metric("District", district)
    col2.metric("Prediction Year", int(latest["Year"].values[0]) + 1)

    st.write("Data used:", int(latest["Year"].values[0]))

    risk_percent = int(risk_score * 100)

    if risk_score < 0.3:
        label = "Low Risk"
        st.success(f"Status: {label}")
    elif risk_score < 0.7:
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

    feature_map = {
        "metric_13": "Expenditure",
        "metric_5": "Fund Allocation",
        "metric_2": "Budget Usage",
        "metric_3": "Utilization Efficiency",
        "metric_4": "Total Spending",
    }

    explanations = []

    for col in feature_cols:
        if col not in data.columns:
            continue

        current = X[col].values[0]
        avg = data[col].mean()

        if pd.isna(current) or pd.isna(avg):
            continue

        impact = current - avg

        if current == 0:
            continue

        explanations.append({
            "feature": col,
            "impact": impact,
            "current": float(current),
            "average": float(avg)
        })

    explanations = sorted(explanations, key=lambda x: abs(x["impact"]), reverse=True)[:3]

    for exp in explanations:
        raw_feature = exp["feature"]
        feature = feature_map.get(raw_feature, raw_feature.replace("_", " ").title())

        impact = exp["impact"]
        current = exp["current"]
        avg = exp["average"]

        if impact > 0:
            status = "Slight increase"
            box = st.warning
        else:
            status = "Stable or decreasing"
            box = st.success

        box(f"""
{feature}
Current: ₹{current:.2f}
Average: ₹{avg:.2f}
Status: {status}
""")