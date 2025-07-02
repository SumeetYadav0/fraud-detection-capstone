import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Title
st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("💳 Fraud Detection Prediction App")
st.markdown("Enter transaction details below and click **Predict** to know if it's potentially fraudulent.")
st.divider()

# Load model with error handling
try:
    model = joblib.load("fraud_detection_pipeline.pkl")
except FileNotFoundError:
    st.error("❌ Model file not found. Please ensure 'fraud_detection_pipeline.pkl' is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Input fields
amount = st.number_input("💰 Amount", min_value=0.0, value=1000.0)
oldbalanceorg = st.number_input("🏦 Old Balance (Sender)", min_value=0.0, value=10000.0)
newbalanceorg = st.number_input("🏦 New Balance (Sender)", min_value=0.0, value=10000.0)
oldbalancedest = st.number_input("🏦 Old Balance (Receiver)", min_value=0.0, value=10000.0)
newbalancedest = st.number_input("🏦 New Balance (Receiver)", min_value=0.0, value=10000.0)

# Predict button
if st.button("🔍 Predict"):
    # Construct input
    input_data = pd.DataFrame([{
        "amount": amount,
        "oldbalanceOrg": oldbalanceorg,
        "newbalanceOrig": newbalanceorg,
        "oldbalanceDest": oldbalancedest,
        "newbalanceDest": newbalancedest
    }])

    # Show input
    st.write("📋 Input Data Preview")
    st.dataframe(input_data)

    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    # Output result
    st.subheader("🔎 Prediction Result:")
    if prediction == 1:
        st.error("⚠️ This transaction **might be fraudulent**.")
    else:
        st.success("✅ This transaction looks **safe**.")


    