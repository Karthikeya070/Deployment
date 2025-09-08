# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="AQI Forecasting",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .aqi-good { background-color: #00e400; }
    .aqi-moderate { background-color: #ffff00; }
    .aqi-unhealthy-sensitive { background-color: #ff7e00; }
    .aqi-unhealthy { background-color: #ff0000; }
    .aqi-very-unhealthy { background-color: #8f3f97; }
    .aqi-hazardous { background-color: #7e0023; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all models with error handling"""
    models = {}
    model_files = {
        'rf': 'rf.pkl',
        'xgb': 'xgb.pkl', 
        'meta': 'stacked.pkl'
    }
    
    # Load joblib models
    for name, file_path in model_files.items():
        try:
            models[name] = joblib.load(file_path)
            st.sidebar.success(f"✅ {name.upper()} model loaded")
        except FileNotFoundError:
            st.sidebar.error(f"❌ {file_path} not found")
            return None
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {name}: {str(e)}")
            return None
    
    # Load Keras model
    try:
        import tensorflow as tf
        models['sarima'] = tf.keras.models.load_model("sarima_model.keras", compile=False)
        st.sidebar.success("✅ SARIMA model loaded")
    except FileNotFoundError:
        st.sidebar.error("❌ sarima_model.keras not found")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading SARIMA: {str(e)}")
        return None
        
    return models

def get_aqi_category(aqi_value):
    """Return AQI category and color based on value"""
    if aqi_value <= 50:
        return "Good", "#00e400"
    elif aqi_value <= 100:
        return "Moderate", "#ffff00"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif aqi_value <= 200:
        return "Unhealthy", "#ff0000"
    elif aqi_value <= 300:
        return "Very Unhealthy", "#8f3f97"
    else:
        return "Hazardous", "#7e0023"

def stacked_predict(models, pm25, no, no2):
    """Enhanced prediction function with error handling"""
    try:
        # Prepare input DataFrame
        input_data = pd.DataFrame([{
            'PM2.5 (µg/m³)': pm25,
            'NO (µg/m³)': no,
            'NO2 (µg/m³)': no2
        }])
        
        features = ['PM2.5 (µg/m³)', 'NO (µg/m³)', 'NO2 (µg/m³)']
        input_data = input_data[features]
        
        # Base model predictions
        pred_rf = models['rf'].predict(input_data)[0]
        pred_xgb = models['xgb'].predict(input_data)[0]
        
        # SARIMA prediction
        sarima_input = input_data.values.astype(np.float32)
        pred_sarima = models['sarima'].predict(sarima_input, verbose=0)[0][0]
        
        # Combine predictions for meta-learner
        stacked_input = np.array([[pred_rf, pred_xgb, pred_sarima]])
        final_pred = models['meta'].predict(stacked_input)[0]
        
        return final_pred, {
            'Random Forest': pred_rf,
            'XGBoost': pred_xgb,
            'SARIMA': pred_sarima
        }
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def create_gauge_chart(aqi_value):
    """Create a gauge chart for AQI visualization"""
    category, color = get_aqi_category(aqi_value)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = aqi_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"AQI: {category}"},
        gauge = {
            'axis': {'range': [None, 500]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "#00e400"},
                {'range': [50, 100], 'color': "#ffff00"},
                {'range': [100, 150], 'color': "#ff7e00"},
                {'range': [150, 200], 'color': "#ff0000"},
                {'range': [200, 300], 'color': "#8f3f97"},
                {'range': [300, 500], 'color': "#7e0023"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': aqi_value
            }
        }
    ))
    
    fig.update_layout(height=400, font={'color': "darkblue", 'family': "Arial"})
    return fig

def create_base_models_comparison(base_predictions):
    """Create comparison chart of base model predictions"""
    fig = go.Figure()
    
    models = list(base_predictions.keys())
    values = list(base_predictions.values())
    
    fig.add_trace(go.Bar(
        x=models,
        y=values,
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
        text=[f'{val:.2f}' for val in values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Base Model Predictions Comparison",
        xaxis_title="Models",
        yaxis_title="Predicted AQI",
        showlegend=False,
        height=400
    )
    
    return fig

# ----------------- Main App -----------------
def main():
    st.markdown("<h1 class='main-header'>🌍 Air Quality Index Forecasting</h1>", unsafe_allow_html=True)
    st.markdown("### Ensemble Stacked Model for AQI Prediction")
    
    # Load models
    models = load_models()
    if models is None:
        st.error("❌ Failed to load models. Please check if all model files are present.")
        st.stop()
    
    # Sidebar for inputs
    st.sidebar.header("📊 Input Parameters")
    
    # Input validation ranges (based on typical pollutant ranges)
    pm25_range = (0.0, 500.0)
    no_range = (0.0, 200.0)
    no2_range = (0.0, 200.0)
    
    pm25 = st.sidebar.slider(
        "PM2.5 (µg/m³)", 
        min_value=pm25_range[0], 
        max_value=pm25_range[1], 
        value=50.0, 
        step=1.0,
        help="Fine particulate matter concentration"
    )
    
    no = st.sidebar.slider(
        "NO (µg/m³)", 
        min_value=no_range[0], 
        max_value=no_range[1], 
        value=10.0, 
        step=1.0,
        help="Nitrogen oxide concentration"
    )
    
    no2 = st.sidebar.slider(
        "NO2 (µg/m³)", 
        min_value=no2_range[0], 
        max_value=no2_range[1], 
        value=20.0, 
        step=1.0,
        help="Nitrogen dioxide concentration"
    )
    
    # Alternative input method
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Or enter values manually:**")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        manual_pm25 = st.number_input("PM2.5", value=pm25, min_value=0.0, max_value=500.0)
        manual_no = st.number_input("NO", value=no, min_value=0.0, max_value=200.0)
    with col2:
        manual_no2 = st.number_input("NO2", value=no2, min_value=0.0, max_value=200.0)
    
    # Use manual inputs if different from sliders
    if manual_pm25 != pm25:
        pm25 = manual_pm25
    if manual_no != no:
        no = manual_no
    if manual_no2 != no2:
        no2 = manual_no2
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Current Input Values")
        
        # Display current inputs
        input_df = pd.DataFrame({
            'Pollutant': ['PM2.5 (µg/m³)', 'NO (µg/m³)', 'NO2 (µg/m³)'],
            'Value': [pm25, no, no2]
        })
        st.dataframe(input_df, use_container_width=True)
        
        # Predict button
        if st.button("🔮 Predict AQI", type="primary", use_container_width=True):
            with st.spinner("Making prediction..."):
                prediction, base_predictions = stacked_predict(models, pm25, no, no2)
                
                if prediction is not None:
                    # Store prediction in session state for persistence
                    st.session_state['prediction'] = prediction
                    st.session_state['base_predictions'] = base_predictions
    
    with col2:
        st.subheader("ℹ️ AQI Scale")
        aqi_info = {
            "0-50": ("Good", "#00e400"),
            "51-100": ("Moderate", "#ffff00"), 
            "101-150": ("Unhealthy for Sensitive", "#ff7e00"),
            "151-200": ("Unhealthy", "#ff0000"),
            "201-300": ("Very Unhealthy", "#8f3f97"),
            "300+": ("Hazardous", "#7e0023")
        }
        
        for range_val, (category, color) in aqi_info.items():
            st.markdown(f"<div style='background-color: {color}; padding: 5px; margin: 2px; border-radius: 3px;
