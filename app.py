# app.py - AQI Forecasting with RF, XGB, tiny SARIMAX, Prophet
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import warnings
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AQI Forecasting", page_icon="🌍", layout="wide")

# ----------------- Load Models -----------------
@st.cache_resource
def load_models():
    models = {}

    # Random Forest
    try:
        models['rf'] = joblib.load("rf.pkl")
        st.sidebar.success("✅ RF loaded")
    except:
        models['rf'] = None
        st.sidebar.error("❌ RF not loaded")

    # XGBoost
    try:
        models['xgb'] = joblib.load("xgb.pkl")
        st.sidebar.success("✅ XGBoost loaded")
    except:
        models['xgb'] = None
        st.sidebar.error("❌ XGBoost not loaded")

    # Tiny SARIMAX
    try:
        with open("sarimax_tiny.pkl", "rb") as f:
            sarimax_small = pickle.load(f)
        models['sarima_small'] = sarimax_small
        st.sidebar.success("✅ SARIMA loaded")
    except Exception as e:
        models['sarima_small'] = None
        st.sidebar.warning(f"⚠️ SARIMA not loaded: {e}")

    # Prophet
    try:
        with open("prophet_model.pkl", "rb") as f:
            models['prophet'] = pickle.load(f)
        st.sidebar.success("✅ Prophet loaded")
    except:
        models['prophet'] = None
        st.sidebar.warning("⚠️ Prophet not loaded")

    return models

# ----------------- Create SARIMAXResults dynamically from tiny model -----------------
def create_sarimax_model(params, order, seasonal_order, trend, input_exog):
    # placeholder endog to match length of exog
    model = SARIMAX(endog=[0]*len(input_exog),
                    order=order,
                    seasonal_order=seasonal_order,
                    trend=trend,
                    exog=input_exog)
    res = model.filter(params)
    return res

# ----------------- AQI Category -----------------
def get_aqi_category(aqi):
    if aqi <= 50: return "Good", "#00e400"
    elif aqi <= 100: return "Moderate", "#ffff00"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif aqi <= 200: return "Unhealthy", "#ff0000"
    elif aqi <= 300: return "Very Unhealthy", "#8f3f97"
    else: return "Hazardous", "#7e0023"

# ----------------- Prediction -----------------
def predict_aqi(models, pm25, no, no2):
    input_df = pd.DataFrame([[pm25, no, no2]],
                            columns=['PM2.5 (µg/m³)', 'NO (µg/m³)', 'NO2 (µg/m³)'])
    preds = {}

    # Random Forest
    preds['RF'] = models['rf'].predict(input_df)[0] if models['rf'] else None

    # XGBoost
    preds['XGBoost'] = models['xgb'].predict(input_df)[0] if models['xgb'] else None

    # SARIMAX tiny
    if models.get('sarima_small'):
        try:
            sarima_model = create_sarimax_model(
                params=models['sarima_small']['params'],
                order=models['sarima_small']['order'],
                seasonal_order=models['sarima_small']['seasonal_order'],
                trend=models['sarima_small']['trend'],
                input_exog=input_df
            )
            preds['SARIMA'] = sarima_model.get_forecast(steps=1, exog=input_df).predicted_mean.iloc[0]
        except:
            preds['SARIMA'] = None
    else:
        preds['SARIMA'] = None

    # Prophet
    if models['prophet']:
        future = pd.DataFrame({'ds': [pd.Timestamp.today()]})
        try:
            preds['Prophet'] = models['prophet'].predict(future)['yhat'].values[0]
        except:
            preds['Prophet'] = None
    else:
        preds['Prophet'] = None

    # Final prediction: average of available models
    valid_preds = [v for v in preds.values() if v is not None]
    final_pred = np.mean(valid_preds) if valid_preds else 50.0
    return final_pred, preds

# ----------------- Main App -----------------
def main():
    st.title("🌍 AQI Forecasting")

    models = load_models()

    st.subheader("📊 Input Pollutants")
    col1, col2, col3 = st.columns(3)
    with col1: pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 500.0, 50.0)
    with col2: no = st.number_input("NO (µg/m³)", 0.0, 200.0, 10.0)
    with col3: no2 = st.number_input("NO2 (µg/m³)", 0.0, 200.0, 20.0)

    if st.button("🔮 Predict AQI"):
        prediction, base_preds = predict_aqi(models, pm25, no, no2)
        category, color = get_aqi_category(prediction)

        st.markdown(f"""
        <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="margin:0;">Predicted AQI: {prediction:.1f}</h2>
            <h3 style="margin:5px 0;">Category: {category}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("🔧 Base Model Predictions")
        for name, val in base_preds.items():
            st.metric(name, f"{val:.1f}" if val is not None else "N/A")


if __name__ == "__main__":
    main()
