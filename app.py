# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from tensorflow import keras
import pickle

# ----------------- Load Models -----------------
rf = joblib.load("rf.pkl")                # Random Forest
xgb = joblib.load("xgb.pkl")              # XGBoost
sarima_model = keras.models.load_model("sarima_model.keras", compile=False)  # Keras SARIMA

with open("prophet_model.pkl", "rb") as f:
    prophet = pickle.load(f)

meta = joblib.load("stacked.pkl")         # Meta stacked model

# ----------------- Feature names -----------------
features = ['PM2.5 (µg/m³)', 'NO (µg/m³)', 'NO2 (µg/m³)']

# ----------------- Prediction Function -----------------
def stacked_predict(pm25, no, no2, pred_date):
    # Prepare input DataFrame with correct column names & order
    df = pd.DataFrame([{
        'PM2.5 (µg/m³)': pm25,
        'NO (µg/m³)': no,
        'NO2 (µg/m³)': no2
    }])
    df = df[features]

    # Base model predictions
    pred1 = rf.predict(df)[0]
    pred2 = xgb.predict(df)[0]

    # Keras SARIMA prediction
    sarima_input = df.values.astype(np.float32)
    pred3 = sarima_model.predict(sarima_input)[0][0]

    # Prophet prediction for the selected date
    future = prophet.make_future_dataframe(periods=(pd.Timestamp(pred_date) - prophet.history['ds'].max()).days)
    # If Prophet was trained with regressors, fill them
    for col, val in zip(features, [pm25, no, no2]):
        future.loc[future['ds'] == pd.Timestamp(pred_date), col] = val
    forecast = prophet.predict(future)
    pred4 = forecast.loc[forecast['ds'] == pd.Timestamp(pred_date), 'yhat'].values[0]

    # Combine predictions for meta-learner
    stacked_input = [[pred1, pred2, pred3, pred4]]
    return meta.predict(stacked_input)[0]

# ----------------- Streamlit UI -----------------
st.title("🌍 Air Quality Index Forecasting (Stacked Model)")
st.markdown("Enter pollutant values and select a future date to predict **AQI Index**:")

# User inputs
pm25 = st.number_input("PM2.5 (µg/m³)", value=50.0)
no = st.number_input("NO (µg/m³)", value=10.0)
no2 = st.number_input("NO2 (µg/m³)", value=20.0)
#pred_date = st.date_input("Select date for prediction")

# Predict button
if st.button("Predict AQI"):
    prediction = stacked_predict(pm25, no, no2)
    #st.success(f"Predicted AQI for {pred_date}: {prediction:.2f}")

