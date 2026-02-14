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
    "Driver Rating",
    min_value=1,
    max_value=4,
    value=2,
    help="Higher rating indicates safer driving history."
)

fault_option = st.sidebar.selectbox(
    "Who is at Fault?",
    ["Policyholder", "Third Party"],
    help="Indicates who caused the accident."
)

Policyholder_At_Fault = 1 if fault_option == "Policyholder" else 0


VehicleCategory = st.sidebar.selectbox(
    "Vehicle Category",
    ["Sedan", "Sport", "Utility"],
    help="Type of vehicle involved in the claim."
)


Deductible_Bin = st.sidebar.selectbox(
    "Deductible Range",
    ["Low", "Medium", "High", "Very_High"],
    help="Higher deductibles may indicate risk‑sharing behavior."
)


address_option = st.sidebar.selectbox(
    "Recent Address Change?",
    ["No Change", "Address Changed"],
    help="Frequent address changes can be a fraud signal."
)

Address_Change_Flag = 1 if address_option == "Address Changed" else 0


repeat_option = st.sidebar.selectbox(
    "Repeat Claimant?",
    ["No", "Yes"],
    help="Indicates whether claimant has prior claims history."
)

Repeat_Claimant = 1 if repeat_option == "Yes" else 0


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
st.markdown("---")

if st.button("🔍 Predict Fraud Risk"):

    fraud_prob, prediction = predict_fraud(input_data)

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Fraud Probability",
            value=f"{fraud_prob:.2%}"
        )

    with col2:
        if prediction == 1:
            st.error("⚠️ High Fraud Risk Detected")
        else:
            st.success("✅ Claim Appears Genuine")


    st.progress(float(fraud_prob))


