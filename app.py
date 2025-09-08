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

# ----------------- Prediction Function -----------------
def stacked_predict(pm25, no, no2, pred_date):
    # Create DataFrame for base models
    df = pd.DataFrame([{
        "PM2.5": pm25,
        "NO": no,
        "NO2": no2
    }])
    df = df[rf.feature_names_in_]  # Ensure correct column order

    # Base model predictions
    pred1 = rf.predict(df)[0]
    pred2 = xgb.predict(df)[0]

    # Keras SARIMA prediction
    sarima_input = df.values.astype(np.float32)
    pred3 = sarima_model.predict(sarima_input)[0][0]  # assuming single output

    # Prophet prediction for the selected date
    future = prophet.make_future_dataframe(periods=(pred_date - prophet.history['ds'].max()).days)
    # Merge regressors if Prophet was trained with them
    future.loc[future['ds'] == pred_date, 'PM2.5'] = pm25
    future.loc[future['ds'] == pred_date, 'NO'] = no
    future.loc[future['ds'] == pred_date, 'NO2'] = no2
    forecast = prophet.predict(future)
    pred4 = forecast.loc[forecast['ds'] == pred_date, 'yhat'].values[0]

    # Combine predictions for meta learner
    stacked_input = [[pred1, pred2, pred3, pred4]]
    return meta.predict(stacked_input)[0]

# ----------------- Streamlit UI -----------------
st.title("🌍 Air Quality Index Forecasting (Stacked Model)")
st.markdown("Enter pollutant values and select a future date to predict **AQI Index**:")

# Input features
pm25 = st.number_input("PM2.5 (µg/m³)", value=50.0)
no = st.number_input("NO (µg/m³)", value=10.0)
no2 = st.number_input("NO2 (µg/m³)", value=20.0)

# Calendar date picker
pred_date = st.date_input("Select date for prediction")

if st.button("Predict AQI"):
    prediction = stacked_predict(pm25, no, no2, pd.Timestamp(pred_date))
    st.success(f"Predicted AQI for {pred_date}: {prediction:.2f}")
