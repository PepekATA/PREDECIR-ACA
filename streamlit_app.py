import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import os
import asyncio
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StreamlitApp")

# Configuración de página
st.set_page_config(
    page_title="🧠 PAPA-DINERO - AI Crypto Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Crear directorios si no existen
os.makedirs('modules', exist_ok=True)
os.makedirs('data', exist_ok=True)

# CSS mejorado para Render
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        font-family: 'Orbitron', monospace;
    }
    
    .ai-prediction {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    
    .never-sell-loss {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
        font-family: 'Orbitron', monospace;
        letter-spacing: 2px;
        text-transform: uppercase;
        animation: glow 3s ease-in-out infinite alternate;
    }
    
    .profit-card {
        background: linear-gradient(135deg, #00C851 0%, #00ff88 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem;
        text-align: center;
        font-family: 'Orbitron', monospace;
    }
    
    .loss-card {
        background: linear-gradient(135deg, #FF6B6B 0%, #FFE66D 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        margin: 0.5rem;
        text-align: center;
        font-family: 'Orbitron', monospace;
    }
    
    .ai-brain {
        background: #1a1a2e;
        color: #0ff1ce;
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #0ff1ce;
        margin: 1rem 0;
        font-family: 'Orbitron', monospace;
        animation: glow-ai 3s ease-in-out infinite alternate;
    }
    
    .credentials-setup {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        font-family: 'Orbitron', monospace;
    }
    
    @keyframes glow {
        from { box-shadow: 0 0 10px #f5576c, 0 0 20px #f5576c, 0 0 30px #f5576c; }
        to { box-shadow: 0 0 20px #f5576c, 0 0 30px #f5576c, 0 0 40px #f5576c; }
    }
    
    @keyframes glow-ai {
        from { box-shadow: 0 0 10px #0ff1ce, 0 0 20px #0ff1ce; }
        to { box-shadow: 0 0 20px #0ff1ce, 0 0 30px #0ff1ce; }
    }
    
    .metric-card {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.2);
        margin: 1rem 0;
    }
    
    .status-healthy { color: #00C851; }
    .status-warning { color: #ffbb33; }
    .status-error { color: #ff4444; }
</style>
""", unsafe_allow_html=True)

# Importar o crear gestor de credenciales
try:
    from modules.credentials_manager import CredentialsManager
    credentials_manager = CredentialsManager()
except ImportError:
    # Crear gestor básico si no existe el módulo
    class BasicCredentialsManager:
        def __init__(self):
            self.data_dir = Path('data')
            self.data_dir.mkdir(exist_ok=True)
            self.credentials_file = self.data_dir / 'credentials.json'
        
        def save_credentials(self, api_key, api_secret, paper_trading=True):
            try:
                credentials = {
                    'api_key': api_key,
                    'api_secret': api_secret,
                    'paper_trading': paper_trading,
                    'saved_at': str(datetime.now())
                }
                with open(self.credentials_file, 'w') as f:
                    json.dump(credentials, f, indent=2)
                return True
            except Exception as e:
                st.error(f"Error guardando credenciales: {e}")
                return False
        
        def load_credentials(self):
            try:
                if self.credentials_file.exists():
                    with open(self.credentials_file, 'r') as f:
                        return json.load(f)
     
