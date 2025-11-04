import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd

# --- Model Definition ---
class WaterQualityMLP(nn.Module):
    def __init__(self, input_size=20):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.network(x)

# --- Load Model & Scaler ---
@st.cache_resource
def load_model():
    model = WaterQualityMLP()
    model.load_state_dict(torch.load("model/water_mlp.pth", map_location='cpu'))
    model.eval()
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# --- Premium Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
    }
    
    .block-container {
        max-width: 1400px;
        background: rgba(255, 255, 255, 0.98);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        animation: fadeInDown 0.8s ease;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 500;
        animation: fadeInUp 0.8s ease;
    }
    
    .input-section {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.8rem;
        border-radius: 16px;
        margin-bottom: 1.2rem;
        border: 2px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .input-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .input-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 24px -6px rgba(102, 126, 234, 0.15);
        border-color: #667eea;
    }
    
    .input-section h3 {
        margin: 0 0 1.2rem 0;
        color: #1e293b;
        font-size: 1.4rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .result-box {
        text-align: center;
        padding: 3rem 2rem;
        border-radius: 20px;
        font-size: 2rem;
        font-weight: 800;
        margin-top: 2rem;
        animation: scaleIn 0.5s ease;
        box-shadow: 0 20px 40px -12px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .result-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    .safe {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
    }
    
    .unsafe {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        border: none;
    }
    
    .confidence {
        font-size: 1.3rem;
        margin-top: 1rem;
        font-weight: 600;
        opacity: 0.95;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .footer {
        text-align: center;
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 2px solid #e5e7eb;
        color: #9ca3af;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    /* Streamlit Element Overrides */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.6rem;
        font-size: 0.95rem;
        transition: all 0.2s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 1rem 2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 10px 25px -5px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px -5px rgba(102, 126, 234, 0.5);
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes shimmer {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Loading Spinner Override */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

# --- App UI ---
st.markdown('<h1 class="main-header">💧 Water Safety Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced AI-powered water quality analysis • Enter parameters below to check drinkability</p>', unsafe_allow_html=True)

# Group inputs into logical sections
cols = [
    'aluminium', 'ammonia', 'arsenic', 'barium', 'cadmium',
    'chloramine', 'chromium', 'copper', 'flouride', 'bacteria',
    'viruses', 'lead', 'nitrates', 'nitrites', 'mercury',
    'perchlorate', 'radium', 'selenium', 'silver', 'uranium'
]

inputs = {}

# Split into 3 columns for better layout
col1, col2, col3 = st.columns(3, gap="large")

sections = [
    ("⚗️ Heavy Metals", ['aluminium', 'arsenic', 'barium', 'cadmium', 'chromium', 'copper', 'lead', 'mercury', 'selenium', 'silver', 'uranium']),
    ("🧬 Chemicals & Nutrients", ['ammonia', 'chloramine', 'flouride', 'nitrates', 'nitrites', 'perchlorate']),
    ("🦠 Biological & Radioactive", ['bacteria', 'viruses', 'radium'])
]

# Assign columns
cols_list = [col1, col2, col3]

for idx, (title, params) in enumerate(sections):
    with cols_list[idx]:
        with st.container():
            st.markdown(f"<div class='input-section'><h3>{title}</h3>", unsafe_allow_html=True)
            for col in params:
                default_val = 0.0
                step = 0.0001 if col in ['ammonia', 'flouride', 'nitrites'] else 0.01
                val = st.number_input(
                    col.replace('_', ' ').title(),
                    min_value=0.0,
                    value=float(default_val),
                    step=step,
                    format="%.6f" if step == 0.0001 else "%.4f",
                    key=col,
                    help=f"Enter {col} concentration (mg/L)"
                )
                inputs[col] = val
            st.markdown("</div>", unsafe_allow_html=True)

# Quick Preset Buttons
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### ⚡ Quick Presets")
preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

with preset_col1:
    if st.button("🟢 Safe Sample", use_container_width=True):
        st.rerun()

with preset_col2:
    if st.button("🟡 Borderline", use_container_width=True):
        for key in ['lead', 'mercury', 'arsenic']:
            if key in st.session_state:
                st.session_state[key] = 0.01

with preset_col3:
    if st.button("🔴 Contaminated", use_container_width=True):
        for key in ['bacteria', 'viruses', 'lead']:
            if key in st.session_state:
                st.session_state[key] = 0.5

with preset_col4:
    if st.button("🔄 Reset All", use_container_width=True):
        for key in cols:
            if key in st.session_state:
                st.session_state[key] = 0.0

# Predict Button
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍 Analyze Water Quality", use_container_width=True, type="primary")

if predict_btn:
    with st.spinner("🧪 Running advanced neural network analysis..."):
        # Prepare input
        input_df = pd.DataFrame([inputs])
        input_scaled = scaler.transform(input_df)
        input_tensor = torch.FloatTensor(input_scaled)
        
        # Predict
        with torch.no_grad():
            prob = model(input_tensor).item()
        
        label = "SAFE" if prob >= 0.5 else "UNSAFE"
        color_class = "safe" if label == "SAFE" else "unsafe"
        icon = "✓" if label == "SAFE" else "✗"
        
        # Display Result with enhanced styling
        st.markdown(f"""
        <div class="result-box {color_class}">
            <div style="font-size: 4rem; margin-bottom: 0.5rem;">{icon}</div>
            Water is <strong>{label}</strong> to drink
            <div class="confidence">Model Confidence: {prob:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional insights
        st.markdown("<br>", unsafe_allow_html=True)
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin:0; color: #667eea;">🎯 Prediction</h4>
                <p style="font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0 0 0; color: #1e293b;">{label}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with insight_col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin:0; color: #667eea;">📊 Confidence</h4>
                <p style="font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0 0 0; color: #1e293b;">{prob:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with insight_col3:
            risk_level = "Low" if prob > 0.7 or prob < 0.3 else "Medium" if prob > 0.6 or prob < 0.4 else "High"
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin:0; color: #667eea;">⚠️ Risk Level</h4>
                <p style="font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0 0 0; color: #1e293b;">{risk_level}</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p><strong>Advanced Neural Network Architecture</strong> • 20 Parameters • 3-Layer MLP with Dropout Regularization</p>
    <p style="margin-top: 0.5rem; font-size: 0.85rem;">Trained on comprehensive water quality datasets • For educational purposes only</p>
</div>
""", unsafe_allow_html=True)