# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Streamlit UI
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📉 Customer Churn Prediction App")
st.write("Enter customer details below to predict the likelihood of churn.")

# Input fields
tenure_months = st.number_input("Tenure in Months", min_value=0, max_value=100, value=12)
monthly_usage_hours = st.number_input("Monthly Usage (hours)", min_value=0.0, max_value=100.0, value=28.5)
has_multiple_devices = st.selectbox("Has Multiple Devices?", ("Yes", "No"))
customer_support_calls = st.number_input("Number of Customer Support Calls", min_value=0, max_value=20, value=2)
payment_failures = st.number_input("Number of Payment Failures", min_value=0, max_value=10, value=0)
is_premium_plan = st.selectbox("Premium Plan?", ("Yes", "No"))

# Convert categorical input
has_multiple_devices_bin = 1 if has_multiple_devices == "Yes" else 0
is_premium_plan_bin = 1 if is_premium_plan == "Yes" else 0

# Predict button
if st.button("🔍 Predict Churn"):
    input_df = pd.DataFrame([{
        "tenure_months": tenure_months,
        "monthly_usage_hours": monthly_usage_hours,
        "has_multiple_devices": has_multiple_devices_bin,
        "customer_support_calls": customer_support_calls,
        "payment_failures": payment_failures,
        "is_premium_plan": is_premium_plan_bin
    }])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"🔴 This customer is likely to churn. (Confidence: {prediction_proba:.2%})")
    else:
        st.success(f"🟢 This customer is not likely to churn. (Confidence: {1 - prediction_proba:.2%})")

    # Display raw data
    with st.expander("See input details"):
        st.write(input_df)
