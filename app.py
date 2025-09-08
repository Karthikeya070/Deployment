# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from tensorflow import keras

# ----------------- Load Models -----------------
rf = joblib.load("rf.pkl")                # Random Forest
xgb = joblib.load("xgb.pkl")              # XGBoost
sarima_model = keras.models.load_model("sarima_model.keras")  # Keras SARIMA
# Load models
rf = joblib.load("rf.pkl")
xgb = joblib.load("xgb.pkl")
sarima_model = keras.models.load_model("sarima_model.keras")  # Keras SARIMA

import pickle
with open("prophet_model.pkl", "rb") as f:
    prophet = pickle.load(f)

meta = joblib.load("stacked.pkl")         # Meta stacked model

# ----------------- Prediction Function -----------------
def stacked_predict(features):
    df = pd.DataFrame([features])
    
    # Base model predictions
    pred1 = rf.predict(df)[0]
    pred2 = xgb.predict(df)[0]
    
    # Keras SARIMA expects a 2D array
    sarima_input = df.values.astype(np.float32)
    pred3 = sarima_model.predict(sarima_input)[0][0]  # assuming single output
    
    # Prophet: forecast next step
    # Prophet requires a DataFrame with 'ds' column (dates)
    # We'll assume it's trained to use a 'y' column, so we just predict next step
    future = prophet.make_future_dataframe(periods=1)
    forecast = prophet.predict(future)
    pred4 = forecast['yhat'].iloc[-1]
    
    # Combine into meta model input
    stacked_input = [[pred1, pred2, pred3, pred4]]
    return meta.predict(stacked_input)[0]

# ----------------- Streamlit UI -----------------
st.title("🌍 Air Quality Index Forecasting (Stacked Model)")
st.markdown("Enter pollutant values to predict **AQI Index**:")

# Input features
pm25 = st.number_input("PM2.5 (µg/m³)", value=50.0)
no = st.number_input("NO (µg/m³)", value=10.0)
no2 = st.number_input("NO2 (µg/m³)", value=20.0)

if st.button("Predict AQI"):
    features = {
        "PM2.5 (µg/m³)": pm25,
        "NO (µg/m³)": no,
        "NO2 (µg/m³)": no2
    }
    prediction = stacked_predict(features)
    st.success(f"Predicted AQI: {prediction:.2f}")



