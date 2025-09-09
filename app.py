# app.py - AQI Forecasting with RF, XGB, SARIMA (.pkl.gz), Prophet
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import warnings
import gzip, pickle
warnings.filterwarnings("ignore")

from prophet.serialize import model_from_json

st.set_page_config(page_title="AQI Forecasting", page_icon="🌍", layout="wide")

# ----------------- Load Models -----------------
@st.cache_resource
def load_models():
    models = {}
    try:
        models['rf'] = joblib.load("rf.pkl")
        st.sidebar.success("✅ RF loaded")
    except:
        models['rf'] = None
        st.sidebar.error("❌ RF not loaded")
    
    try:
        models['xgb'] = joblib.load("xgb.pkl")
        st.sidebar.success("✅ XGBoost loaded")
    except:
        models['xgb'] = None
        st.sidebar.error("❌ XGBoost not loaded")
    
    try:
        with gzip.open("sarimax.pkl.gz", "rb") as f:
            models['sarima'] = pickle.load(f)
        st.sidebar.success("✅ SARIMA loaded")
    except:
        models['sarima'] = None
        st.sidebar.warning("⚠️ SARIMA not loaded")
    
    try:
        with open("prophet_model.json", "r") as fin:
            models['prophet'] = model_from_json(fin.read())
        st.sidebar.success("✅ Prophet loaded")
    except:
        models['prophet'] = None
        st.sidebar.warning("⚠️ Prophet not loaded")
    
    return models

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
    
    # SARIMA (time-series forecast)
    if models['sarima']:
        preds['SARIMA'] = models['sarima'].forecast(steps=1)[0]
    else:
        preds['SARIMA'] = None
    
    # Prophet (forecast one step ahead)
    if models['prophet']:
        future = pd.DataFrame({'ds': [pd.Timestamp.today()]})
        preds['Prophet'] = models['prophet'].predict(future)['yhat'].values[0]
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
