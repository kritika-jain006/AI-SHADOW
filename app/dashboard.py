import streamlit as st
import pandas as pd
import requests

# Page config
st.set_page_config(page_title="AI Governance Monitor", layout="wide")

# Title
st.title("AI Governance Monitoring System")

# Load data
data = pd.read_csv("data/processed/final_time_series_dataset.csv")
districts = sorted(data["District"].unique())

# Sidebar
st.sidebar.header("Select District")
district = st.sidebar.selectbox("District", districts)

# Button action
if st.sidebar.button("Predict Future Risk"):

    # API call
    url = f"http://127.0.0.1:8000/predict/{district}"
    response = requests.get(url)
    result = response.json()

    # -------------------------------
    # Prediction Result
    # -------------------------------
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)
    col1.metric("District", result["district"])
    col2.metric("Prediction Year", result["predicted_risk_year"])

    # st.write("Data used:", result["data_year_used"])

    # -------------------------------
    # Risk Display (User Friendly)
    # -------------------------------
    risk_score = result["risk_score"]
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

    # -------------------------------
    # Summary
    # -------------------------------
    st.subheader("Summary")

    if label == "Low Risk":
        st.write("The scheme is expected to perform normally with low risk.")
    elif label == "Moderate Risk":
        st.write("There are some potential risks that should be monitored.")
    else:
        st.write("The scheme is at high risk and needs urgent attention.")

    # -------------------------------
    # Recommended Action
    # -------------------------------
    st.subheader("Recommended Action")

    if label == "Low Risk":
        st.write("Everything is stable. Continue regular monitoring.")
    elif label == "Moderate Risk":
        st.write("Some warning signs detected. Review the scheme performance.")
    else:
        st.write("High risk detected. Immediate intervention is required.")

    # -------------------------------
    # Key Observations (FIXED + CLEAN)
    # -------------------------------
    st.subheader("Key Observations")

    # Optional: Rename metrics to meaningful names
    feature_map = {
        "metric_13": "Expenditure",
        "metric_5": "Fund Allocation",
    }

    for exp in result["explanations"]:
        raw_feature = exp["feature"]
        feature = feature_map.get(raw_feature, raw_feature.replace("_", " ").title())

        impact = exp["impact"]
        current = exp["current"]
        avg = exp["average"]

        # Convert to simple human status
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