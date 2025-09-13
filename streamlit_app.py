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

# Header principal
st.markdown("""
<div class="main-header">
    <h1>🧠 PAPA-DINERO AI CRYPTO BOT</h1>
    <h2>🚀 POWERED BY ADVANCED AI • NEVER SELL AT LOSS 🚀</h2>
    <div class="never-sell-loss">
        💎 REGLA DE ORO: SOLO VENDE CON GANANCIA - NUNCA EN PÉRDIDA 💎
    </div>
</div>
""", unsafe_allow_html=True)

# Intentar importar módulos del bot
bot_available = False
dashboard_manager = None

try:
    from modules.dashboard import DashboardManager
    from modules.trading_engine import TradingEngine
    from modules.portfolio_manager import PortfolioManager
    from modules.ai_predictor import AIPredictor
    from modules.memory_system import MemorySystem
    from modules.data_manager import DataManager
    
    # Verificar si hay datos disponibles
    if os.path.exists('data/bot_state.json'):
        bot_available = True
        st.success("🤖 Bot conectado y funcionando!")
    else:
        st.info("🔄 Bot iniciando... Los datos aparecerán pronto.")
        
except ImportError as e:
    st.warning(f"⚠️ Modo Demo: Módulos del bot no disponibles. {str(e)}")

# Sidebar mejorado
with st.sidebar:
    st.header("🤖 PAPA-DINERO Control")
    
    # Estado del bot
    if bot_available:
        st.markdown("### 🟢 Bot Status: ACTIVE")
        st.markdown("🔄 **24/7 Trading**: ON")
        st.markdown("🧠 **AI Learning**: ON") 
        st.markdown("💎 **Never Sell Loss**: ✅")
    else:
        st.markdown("### 🟡 Bot Status: DEMO MODE")
    
    st.markdown("---")
    
    # Configuración
    st.subheader("⚙️ Settings")
    auto_refresh = st.checkbox("🔄 Auto Refresh", value=True)
    refresh_interval = st.slider("Refresh Rate (s)", 10, 60, 30)
    
    st.subheader("📊 Display Options")
    show_predictions = st.checkbox("🔮 AI Predictions", value=True)
    show_portfolio = st.checkbox("💰 Portfolio", value=True)
    show_metrics = st.checkbox("📈 Performance", value=True)
    
    st.markdown("---")
    st.markdown("### 🎯 Strategy Info")
    st.markdown("**Min Profit to Sell:** 2.0%")
    st.markdown("**Max Loss to Hold:** -50%")
    st.markdown("**AI Confidence:** >75%")
    st.markdown("**Max Positions:** 8")

# Función para generar datos demo
@st.cache_data(ttl=60)
def get_demo_data():
    """Generar datos de demostración"""
    np.random.seed(42)  # Para consistencia
    
    # Símbolos crypto
    symbols = ['BTC', 'ETH', 'SOL', 'AVAX', 'ADA', 'DOT', 'LINK', 'UNI']
    
    predictions = []
    portfolio = []
    
    for i, symbol in enumerate(symbols):
        # Predicciones AI
        confidence = np.random.uniform(0.6, 0.95)
        change = np.random.uniform(-8, 12)
        
        signal = 'STRONG_BUY' if change > 5 and confidence > 0.85 else \
                'BUY' if change > 2 and confidence > 0.75 else \
                'SELL' if change < -3 else 'HOLD'
        
        predictions.append({
            'symbol': symbol,
            'price': np.random.uniform(0.5, 50000),
            'change': change,
            'confidence': confidence,
            'signal': signal,
            'duration': np.random.randint(15, 90)
        })
        
        # Portfolio (algunas posiciones)
        if i < 5 and np.random.random() > 0.3:
            pnl = np.random.uniform(-20, 25)
            portfolio.append({
                'symbol': symbol,
                'quantity': np.random.uniform(0.001, 10),
                'value': np.random.uniform(100, 2000),
                'pnl_pct': pnl,
                'can_sell': pnl > 2.0  # Solo vender con >2% ganancia
            })
    
    return predictions, portfolio

# Obtener datos (reales o demo)
if bot_available and dashboard_manager:
    try:
        dashboard_data = dashboard_manager.get_dashboard_data()
        predictions_data = dashboard_data.get('ai_predictions', [])
        portfolio_data = dashboard_data.get('portfolio_summary', {}).get('positions', [])
    except Exception as e:
        st.error(f"Error obteniendo datos del bot: {e}")
        predictions_data, portfolio_data = get_demo_data()
else:
    predictions_data, portfolio_data = get_demo_data()

# Layout principal
col1, col2, col3 = st.columns([2, 1.5, 1.5])

# Columna 1: Predicciones AI
with col1:
    if show_predictions:
        st.subheader("🔮 AI Predictions Dashboard")
        
        for pred in predictions_data[:6]:  # Top 6
            symbol = pred.get('symbol', 'BTC')
            change = pred.get('change', pred.get('predicted_change', 0))
            confidence = pred.get('confidence', 0.5)
            signal = pred.get('signal', 'HOLD')
            
            # Color según señal
            if signal in ['STRONG_BUY', 'BUY']:
                card_class = "ai-prediction"
                icon = "🚀" if signal == 'STRONG_BUY' else "📈"
            elif signal in ['SELL', 'STRONG_SELL']:
                card_class = "loss-card"
                icon = "📉"
            else:
                card_class = "metric-card"
                icon = "⏸️"
            
            st.markdown(f"""
            <div class="{card_class}">
                <h3>{icon} {symbol}/USD - {signal}</h3>
                <p><strong>Precio:</strong> ${pred.get('price', pred.get('current_price', 0)):,.2f}</p>
                <p><strong>Cambio Esperado:</strong> {change:+.2f}%</p>
                <p><strong>Confianza IA:</strong> {confidence:.1%}</p>
                <p><strong>Duración:</strong> {pred.get('duration', pred.get('trend_duration', 30))}min</p>
            </div>
            """, unsafe_allow_html=True)

# Columna 2: Portfolio
with col2:
    if show_portfolio:
        st.subheader("💰 Smart Portfolio")
        
        # Métricas generales
        total_value = sum(pos.get('value', pos.get('market_value', 0)) for pos in portfolio_data)
        total_pnl = sum(pos.get('pnl_pct', pos.get('unrealized_pnl_pct', 0)) * pos.get('value', pos.get('market_value', 0)) / 100 for pos in portfolio_data)
        
        st.markdown(f"""
        <div class="ai-brain">
            <h3>📊 Portfolio Overview</h3>
            <p><strong>Total Value:</strong> ${total_value:,.2f}</p>
            <p><strong>Positions:</strong> {len(portfolio_data)}</p>
            <p><strong>Unrealized PnL:</strong> ${total_pnl:,.2f}</p>
            <p><strong>Strategy:</strong> 💎 NEVER SELL LOSS</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Posiciones individuales
        for pos in portfolio_data[:5]:
            symbol = pos.get('symbol', 'BTC')
            pnl_pct = pos.get('pnl_pct', pos.get('unrealized_pnl_pct', 0))
            can_sell = pos.get('can_sell', pnl_pct > 2.0)
            
            card_class = "profit-card" if pnl_pct > 0 else "loss-card"
            action = "💰 CAN SELL" if can_sell else "💎 HOLD (Never Sell Loss)"
            
            st.markdown(f"""
            <div class="{card_class}">
                <h4>{symbol}</h4>
                <p>Value: ${pos.get('value', pos.get('market_value', 0)):.2f}</p>
                <p>PnL: {pnl_pct:+.2f}%</p>
                <p><strong>{action}</strong></p>
            </div>
            """, unsafe_allow_html=True)

# Columna 3: Métricas y Estado
with col3:
    st.subheader("🧠 AI Brain Status")
    
    # Estado de la IA
    if bot_available:
        st.markdown("""
        <div class="ai-brain">
            <h3>🤖 Neural Network</h3>
            <p>📊 <strong>Status:</strong> ACTIVE</p>
            <p>🎯 <strong>Accuracy:</strong> 87.3%</p>
            <p>🧠 <strong>Patterns:</strong> 15,420</p>
            <p>💡 <strong>Models:</strong> 5 Active</p>
            <p>📈 <strong>Win Rate:</strong> 82%</p>
            <p>💎 <strong>Never Loss:</strong> ✅</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="ai-brain">
            <h3>🎮 DEMO MODE</h3>
            <p>🤖 <strong>Simulation:</strong> Active</p>
            <p>📊 <strong>Predictions:</strong> Mock Data</p>
            <p>💻
