import streamlit as st
import pandas as pd
import joblib
import os


# --------------------------------------------------
# Path Handling (Production Safe)
# --------------------------------------------------

# Get current file directory (app/)
BASE_DIR = os.path.dirname(__file__)

# Move to project root
ROOT_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..")
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model_path = os.path.join(
    ROOT_DIR,
    "models",
    "balanced_random_forest.pkl"
)

model = joblib.load(model_path)


# --------------------------------------------------
# Load Threshold Config
# --------------------------------------------------

threshold_path = os.path.join(
    ROOT_DIR,
    "models",
    "threshold_config.pkl"
)

threshold_config = joblib.load(threshold_path)

THRESHOLD = threshold_config["threshold"]


# --------------------------------------------------
# Load Feature List
# --------------------------------------------------

features_path = os.path.join(
    ROOT_DIR,
    "models",
    "model_features.pkl"
)

model_features = joblib.load(features_path)


# Debug (optional — remove later)
print("Artifacts Loaded Successfully")


# Preprocessing Function
def preprocess_input(input_df):

    input_encoded = pd.get_dummies(
        input_df,
        drop_first=True
    )

    input_aligned = input_encoded.reindex(
        columns=model_features,
        fill_value=0
    )

    return input_aligned


# Prediction Function
def predict_fraud(input_df):

    processed_data = preprocess_input(input_df)

    fraud_prob = model.predict_proba(
        processed_data
    )[:, 1][0]

    prediction = int(fraud_prob >= THRESHOLD)

    return fraud_prob, prediction


# App Title
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Vehicle Insurance Fraud Detection")
st.markdown(
"""
Detect potentially fraudulent vehicle insurance claims using a machine learning model.

Fill in the claim details on the left and click **Predict Fraud Risk**.
"""
)


# User Inputs

st.sidebar.header("Enter Claim Details")

DriverRating = st.sidebar.slider(
    "Driver Rating", 1, 4, 2
)

Policyholder_At_Fault = st.sidebar.selectbox(
    "Policyholder At Fault?",
    [0, 1]
)

VehicleCategory = st.sidebar.selectbox(
    "Vehicle Category",
    ["Sedan", "Sport", "Utility"]
)

Deductible_Bin = st.sidebar.selectbox(
    "Deductible Bin",
    ["Low", "Medium", "High", "Very_High"]
)

Address_Change_Flag = st.sidebar.selectbox(
    "Recent Address Change?",
    [0, 1]
)

Repeat_Claimant = st.sidebar.selectbox(
    "Repeat Claimant?",
    [0, 1]
)

# Build Input DataFrame
input_data = pd.DataFrame({
    "DriverRating": [DriverRating],
    "Policyholder_At_Fault": [Policyholder_At_Fault],
    "VehicleCategory": [VehicleCategory],
    "Deductible_Bin": [Deductible_Bin],
    "Address_Change_Flag": [Address_Change_Flag],
    "Repeat_Claimant": [Repeat_Claimant]
})

# Predict Button
if st.button("Predict Fraud Risk"):

    fraud_prob, prediction = predict_fraud(
        input_data
    )

    st.subheader("Prediction Result")

    st.write(
        f"Fraud Probability: **{fraud_prob:.2f}**"
    )

    if prediction == 1:
        st.error("⚠️ Fraudulent Claim Detected")
    else:
        st.success("✅ Genuine Claim")



