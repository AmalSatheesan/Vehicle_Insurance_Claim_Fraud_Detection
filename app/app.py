import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load(
    "../models/balanced_random_forest.pkl"
)

# Load threshold
threshold_config = joblib.load(
    "../models/threshold_config.pkl"
)

THRESHOLD = threshold_config["threshold"]

# Load feature list
model_features = joblib.load(
    "../models/model_features.pkl"
)


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
st.set_page_config(page_title="Fraud Detection App")

st.title("🚗 Vehicle Insurance Fraud Detection")
st.write(
    "Predict whether a vehicle insurance claim is fraudulent."
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



