# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from tensorflow import keras

# ----------------- Load Models -----------------
rf = joblib.load("rf_fixed.pkl")                  # Random Forest
xgb = joblib.load("xgb.pkl")                # XGBoost (base model if used)
sarima_model = keras.models.load_model("sarima_model.keras", compile=False)  # Keras SARIMA
meta = joblib.load("stacked.pkl")           # Meta stacked model

# ----------------- Feature names -----------------
features = ['PM2.5 (µg/m³)', 'NO (µg/m³)', 'NO2 (µg/m³)']

# ----------------- Prediction Function -----------------
def stacked_predict(pm25, no, no2):
    # Prepare input DataFrame
    df = pd.DataFrame([{
        'PM2.5 (µg/m³)': pm25,
        'NO (µg/m³)': no,
        'NO2 (µg/m³)': no2
    }])
    df = df[features]

    # Base model predictions
    pred1 = rf.predict(df)[0]
    pred2 = xgb.predict(df)[0]  # Only if XGB was used as base model
    sarima_input = df.values.astype(np.float32)
    pred3 = sarima_model.predict(sarima_input)[0][0]

    # Combine predictions for meta-learner
    stacked_input = [[pred1, pred2, pred3]]
    final_pred = meta.predict(stacked_input)[0]
    return final_pred

# ----------------- Streamlit UI -----------------
st.title("🌍 Air Quality Index Forecasting (Stacked Model)")
st.markdown("Enter pollutant values to predict **AQI Index**:")

# User inputs
pm25 = st.number_input("PM2.5 (µg/m³)", value=50.0)
no = st.number_input("NO (µg/m³)", value=10.0)
no2 = st.number_input("NO2 (µg/m³)", value=20.0)

# Predict button
if st.button("Predict AQI"):
    prediction = stacked_predict(pm25, no, no2)
    st.success(f"Predicted AQI: {prediction:.2f}")

