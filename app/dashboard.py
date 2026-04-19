import streamlit as st
import pandas as pd
import pickle
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "final_time_series_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "risk_model.pkl")

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="AI Governance Monitor", layout="wide")

st.title("AI Governance Monitoring System")

# -------------------------------
# Load data
# -------------------------------
data = pd.read_csv(DATA_PATH)
districts = sorted(data["District"].unique())

# -------------------------------
# Load model
# -------------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Sidebar
st.sidebar.header("Select District")
district = st.sidebar.selectbox("District", districts)

# -------------------------------
# Button action
# -------------------------------
if st.sidebar.button("Predict Future Risk"):

    # Get latest data for district
    district_data = data[data["District"] == district]
    latest = district_data.sort_values("Year").iloc[-1:]

    # Prepare input
    X = latest.copy()
    X = X.drop(columns=["District", "Year"], errors="ignore")
    X = X.select_dtypes(include=["number"])
    X = X.fillna(0)

    # -------------------------------
    # Prediction
    # -------------------------------
    try:
        risk_score = model.predict_proba(X)[0][1]
    except:
        st.error("Model feature mismatch. Please check training columns.")
        st.write("Columns passed:", list(X.columns))
        st.stop()

    risk_percent = int(risk_score * 100)

    # -------------------------------
    # Prediction Result
    # -------------------------------
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)
    col1.metric("District", district)
    col2.metric("Prediction Year", int(latest["Year"].values[0]) + 1)

    st.write("Data used:", int(latest["Year"].values[0]))

    # -------------------------------
    # Risk Display
    # -------------------------------
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
    # Key Observations (approx)
    # -------------------------------
    st.subheader("Key Observations")

    feature_map = {
        "metric_13": "Expenditure",
        "metric_5": "Fund Allocation",
    }

    for col in X.columns[:5]:
        name = feature_map.get(col, col.replace("_", " ").title())
        value = float(X[col].values[0])

        if value > X[col].mean():
            st.warning(f"{name}\nCurrent: ₹{value:.2f} (Above average)")
        else:
            st.success(f"{name}\nCurrent: ₹{value:.2f} (Normal)")