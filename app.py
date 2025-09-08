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
    import pandas as pd
    import numpy as np

    # Prepare input for SARIMAX
    sarima_input = pd.DataFrame(
        [[pm25, no, no2]],
        columns=['PM2.5 (µg/m³)', 'NO (µg/m³)', 'NO2 (µg/m³)']
    )

    # SARIMAX prediction
    pred_sarimax = sarimax_fit.predict(start=len(train), end=len(train), exog=sarima_input)[0]

    # Random Forest prediction
    pred_rf = rf.predict(sarima_input)[0]

    # Prophet prediction
    from prophet import Prophet
    future_date = pd.DataFrame({'ds': [pd.Timestamp.today()]})
    prophet_pred = prophet_model.predict(future_date)['yhat'].values[0]

    # Stack predictions
    stacked_input = pd.DataFrame({
        'sarimax': [pred_sarimax],
        'rf': [pred_rf],
        'prophet': [prophet_pred]
    })

    # Meta-learner prediction
    final_pred = xgb_meta.predict(stacked_input)[0]
    return final_pred


# Predict button
if st.button("Predict AQI"):
    prediction = stacked_predict(pm25, no, no2)
    st.success(f"Predicted AQI: {prediction:.2f}")


