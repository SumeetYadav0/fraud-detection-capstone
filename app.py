import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("💳 Fraud Detection App")
st.markdown("Enter transaction details and predict if the transaction is potentially fraudulent.")
st.divider()

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load("fraud_detection_pipeline.pkl")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model = load_model()

# Load transformation function
try:
    from utils import transform_features
except ImportError:
    st.error("❌ Could not import `transform_features` from utils.py.")
    st.stop()

# Input fields
amount = st.number_input("💰 Amount", min_value=0.0, value=1000.0)
oldbalanceorg = st.number_input("🏦 Sender Old Balance", min_value=0.0, value=5000.0)
newbalanceorg = st.number_input("🏦 Sender New Balance", min_value=0.0, value=4000.0)
oldbalancedest = st.number_input("🏦 Receiver Old Balance", min_value=0.0, value=2000.0)
newbalancedest = st.number_input("🏦 Receiver New Balance", min_value=0.0, value=3000.0)

# Prediction
if st.button("🔍 Predict Fraud"):
    raw_input = pd.DataFrame([{
        "amount": amount,
        "oldbalanceOrg": oldbalanceorg,
        "newbalanceOrig": newbalanceorg,
        "oldbalanceDest": oldbalancedest,
        "newbalanceDest": newbalancedest
    }])

    st.markdown("### 📋 Input Summary")
    st.dataframe(raw_input)

    # Apply transformation
    try:
        transformed_input = transform_features(raw_input)
    except Exception as e:
        st.error(f"❌ Failed to apply feature engineering: {e}")
        st.stop()

    # Prediction
    try:
        prediction = model.predict(transformed_input)[0]
        probability = model.predict_proba(transformed_input)[0][1]
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.stop()

    # Result
    st.subheader("🔎 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Confidence: {probability:.2%})")
    else:
        st.success(f"✅ Safe Transaction. (Confidence: {1 - probability:.2%})")
