# streamlit_app_production.py — PAPA-DINERO AI Crypto Bot vFinal PRO
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time, json, os, logging
from datetime import datetime
from pathlib import Path

# ====================================================================
# CONFIGURACIÓN
# ====================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PAPA-DINERO-PRO")

st.set_page_config(
    page_title="🧠 PAPA-DINERO PRO - AI Crypto Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

os.makedirs('modules', exist_ok=True)
os.makedirs('data', exist_ok=True)

# ====================================================================
# CSS ESTÉTICO
# ====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
.main-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem; box-shadow: 0 10px 40px rgba(0,0,0,0.3); font-family: 'Orbitron', monospace; }
.ai-brain { background: #1a1a2e; color: #0ff1ce; padding: 1.5rem; border-radius: 15px; border: 2px solid #0ff1ce; margin: 1rem 0; font-family: 'Orbitron', monospace; animation: glow-ai 3s ease-in-out infinite alternate; }
.metric-card { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(255,255,255,0.2); margin: 1rem 0; }
@keyframes glow-ai { from { box-shadow: 0 0 10px #0ff1ce, 0 0 20px #0ff1ce; } to { box-shadow: 0 0 20px #0ff1ce, 0 0 30px #0ff1ce; } }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🧠 PAPA-DINERO PRO - AI Crypto Bot</h1>", unsafe_allow_html=True)

# ====================================================================
# IMPORTAR MÓDULOS INTERNOS
# ====================================================================
try:
    from modules.credentials_manager import CredentialsManager
    from modules.memory_system import MemorySystem
    from modules.data_manager import DataManager
    from modules.market_analyzer import MarketAnalyzer
    from modules.ai_predictor import AIPredictor
    from modules.trading_engine import TradingEngine
    from modules.portfolio_manager import PortfolioManager
    from modules.multi_symbol_trader import MultiSymbolTrader
except ImportError as e:
    st.error(f"No se pudieron cargar los módulos internos: {e}")
    st.stop()

# ====================================================================
# CREDENCIALES Y DETECCIÓN DE TIPO DE CUENTA
# ====================================================================
credentials_manager = CredentialsManager()
credentials = credentials_manager.load_credentials()

if not credentials:
    st.info("Configura tus credenciales Alpaca")
    api_key = st.text_input("API Key")
    api_secret = st.text_input("API Secret")
    paper_trading = st.checkbox("Paper Trading", value=True)
    if st.button("Guardar Credenciales"):
        credentials_manager.save_credentials(api_key, api_secret, paper_trading)
        st.experimental_rerun()
else:
    account_type = "Paper Trading" if credentials.get('paper_trading', True) else "Real Money"
    st.success(f"Cuenta detectada: {account_type}")

# ====================================================================
# INICIALIZAR SISTEMAS
# ====================================================================
memory = MemorySystem()
data_manager = DataManager()
market_analyzer = MarketAnalyzer()
ai_predictor = AIPredictor(memory)
portfolio_manager = PortfolioManager()
trading_engine = TradingEngine(credentials)
multi_trader = MultiSymbolTrader(trading_engine, market_analyzer, ai_predictor, portfolio_manager)

# ====================================================================
# SIDEBAR CONTROLS
# ====================================================================
st.sidebar.title("Controles Bot PRO")
auto_refresh = st.sidebar.checkbox("Auto-refresh", True)
refresh_interval = st.sidebar.slider("Intervalo actualización (segundos)", 5, 60, 15)
show_predictions = st.sidebar.checkbox("Mostrar predicciones AI", True)
show_portfolio = st.sidebar.checkbox("Mostrar portafolio", True)
show_metrics = st.sidebar.checkbox("Mostrar métricas del bot", True)
max_risk_percent = st.sidebar.slider("Máximo % riesgo por orden", 1, 10, 2)

# ====================================================================
# LOOP PRINCIPAL DE TRADING
# ====================================================================
placeholder = st.empty()
while True:
    # 1️⃣ Actualizar datos
    market_data = data_manager.fetch_market_data()
    
    # 2️⃣ Analizar mercado
    analysis = market_analyzer.analyze(market_data)
    
    # 3️⃣ Predicciones AI
    predictions = ai_predictor.predict(analysis)
    
    # 4️⃣ Ejecutar trading seguro
    trades = multi_trader.execute(predictions, max_risk_percent=max_risk_percent)
    
    # 5️⃣ Actualizar portafolio
    portfolio = portfolio_manager.update(trades)
    
    # 6️⃣ Mostrar resultados en Streamlit
    with placeholder.container():
        st.markdown("<div class='ai-brain'>🤖 Predicciones AI & Trading</div>", unsafe_allow_html=True)
        
        if show_predictions:
            st.dataframe(predictions)
        
        if show_portfolio:
            st.markdown("<div class='metric-card'>💰 Portafolio</div>", unsafe_allow_html=True)
            st.dataframe(portfolio)
        
        if show_metrics:
            st.markdown("<div class='metric-card'>📊 Métricas</div>", unsafe_allow_html=True)
            st.write({
                "Memoria bot (nº eventos)": memory.size(),
                "Última actualización": datetime.now()
            })
        
        # Gráfica de precio vs predicción
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=market_data['timestamp'], y=market_data['price'], name='Precio Real'))
        fig.add_trace(go.Scatter(x=predictions['timestamp'], y=predictions['predicted_price'], name='Predicción AI'))
        st.plotly_chart(fig, use_container_width=True)
    
    if not auto_refresh:
        break
    time.sleep(refresh_interval)
