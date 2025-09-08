# app.py
import streamlit as st
import joblib
import pandas as pd

import statsmodels.api as sm   # important for SARIMA

sarima = joblib.load("sarima.pkl")

# Load models
rf = joblib.load("rf.pkl")
xgb = joblib.load("xgb.pkl")
sarima = joblib.load("sarima.pkl")
prophet = joblib.load("prophet.pkl")
meta = joblib.load("stacked.pkl")

def stacked_predict(features):
    df = pd.DataFrame([features])
    pred1 = rf.predict(df)[0]
    pred2 = xgb.predict(df)[0]
    pred3 = sarima.forecast(steps=1)[0]
    pred4 = prophet.predict(df)[0]
    stacked_input = [[pred1, pred2, pred3, pred4]]
    return meta.predict(stacked_input)[0]

# Streamlit UI
st.title("🌍 Air Quality Index Forecasting (Stacked Model)")

# Example inputs (adapt to your dataset features)
temp = st.number_input("Temperature", value=30.0)
humidity = st.number_input("Humidity", value=60.0)

if st.button("Predict AQI"):
    features = {"temp": temp, "humidity": humidity}
    prediction = stacked_predict(features)

    st.success(f"Predicted AQI: {prediction:.2f}")




