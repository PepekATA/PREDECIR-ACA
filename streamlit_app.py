# ====================================================================
# PAPA-DINERO AI CRYPTO BOT - VERSIÓN FINAL COMPLETA
# ====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import time
import json
import logging
from pathlib import Path

# ====================================================================
# CONFIGURACIÓN DE LOGGING
# ====================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PapaDineroBot")

# ====================================================================
# CREACIÓN DE DIRECTORIOS
# ====================================================================
os.makedirs('modules', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ====================================================================
# IMPORTACIÓN DE MÓDULOS INTERNOS
# ====================================================================
from modules.credentials_manager import CredentialsManager
from modules.memory_system import MemorySystem
from modules.data_manager import DataManager
from modules.market_analyzer import MarketAnalyzer
from modules.ai_predictor import AIPredictor
from modules.alpaca_trend_predictor import AlpacaTrendPredictor
from modules.multi_symbol_trader import MultiSymbolTrader
from modules.portfolio_manager import PortfolioManager
from modules.trading_engine import TradingEngine
from modules.dashboard import DashboardManager
from modules.persistence_manager import PersistenceManager
from modules.neural_models import NeuralModels
from modules.settings import Settings

# ====================================================================
# CONFIGURACIÓN DE STREAMLIT
# ====================================================================
st.set_page_config(
    page_title="🧠 PAPA-DINERO AI CRYPTO BOT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================================
# CARGA DE CREDENCIALES Y DETECCIÓN DE MODO
# ====================================================================
credentials_manager = CredentialsManager()
memory_system = MemorySystem()
settings = Settings()

demo_mode_file = Path('data/demo_mode.json')

if demo_mode_file.exists():
    demo_mode = True
    api_key = None
    api_secret = None
    paper_trading = True
else:
    creds = credentials_manager.load_credentials()
    if creds:
        api_key = creds.get('api_key')
        api_secret = creds.get('api_secret')
        paper_trading = creds.get('paper_trading', True)
        demo_mode = False
    else:
        demo_mode = True
        api_key = None
        api_secret = None
        paper_trading = True

# ====================================================================
# INICIALIZACIÓN DE MÓDULOS
# ====================================================================
data_manager = DataManager()
market_analyzer = MarketAnalyzer()
ai_predictor = AIPredictor(memory_system)
portfolio_manager = PortfolioManager()
trading_engine = TradingEngine(api_key, api_secret, paper_trading)
multi_symbol_trader = MultiSymbolTrader(trading_engine, ai_predictor, portfolio_manager)
dashboard_manager = DashboardManager()

# ====================================================================
# CONFIGURACIÓN DEL DASHBOARD
# ====================================================================
st.title("🧠 PAPA-DINERO AI CRYPTO BOT")
st.subheader(f"🚀 Modo {'Paper Trading' if paper_trading else 'Real Money'}")

# ====================================================================
# OPCIONES DE SIDEBAR
# ====================================================================
with st.sidebar:
    st.header("⚙️ Configuración")
    auto_refresh = st.checkbox("🔄 Auto Refresh", value=True)
    refresh_interval = st.slider("Intervalo de actualización (segundos)", 10, 60, 30)

    st.markdown("---")
    st.header("📊 Controles de visualización")
    show_predictions = st.checkbox("Mostrar predicciones AI", value=True)
    show_portfolio = st.checkbox("Mostrar portfolio", value=True)
    show_metrics = st.checkbox("Mostrar métricas de desempeño", value=True)

# ====================================================================
# FUNCIONES PRINCIPALES
# ====================================================================

def get_symbols():
    """Obtener lista de símbolos para trading"""
    return settings.get_symbols()  # Puede estar definido en settings.py

def run_trading_cycle(symbols):
    """Ejecutar ciclo completo: predecir, comprar, vender"""
    market_data = data_manager.fetch_market_data(symbols)
    analysis = market_analyzer.analyze(market_data)
    predictions = ai_predictor.predict(analysis)
    trades = multi_symbol_trader.execute_trades(predictions)
    portfolio_manager.update(trades)
    memory_system.update_memory(predictions, trades)
    return predictions, portfolio_manager.get_portfolio()

def plot_market_chart(symbol, market_data):
    """Gráfica de precios y medias móviles"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=market_data['timestamp'],
        y=market_data['price'],
        mode='lines',
        name=f'{symbol} Price',
        line=dict(color='#00ff88', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=market_data['timestamp'],
        y=market_data['ma_short'],
        mode='lines',
        name='MA Short',
        line=dict(color='#ffbb33', width=1),
        opacity=0.7
    ))
    fig.add_trace(go.Scatter(
        x=market_data['timestamp'],
        y=market_data['ma_long'],
        mode='lines',
        name='MA Long',
        line=dict(color='#ff4444', width=1),
        opacity=0.7
    ))
    fig.update_layout(
        title=f"📈 {symbol} Market Analysis",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# ====================================================================
# BUCLE PRINCIPAL
# ====================================================================
symbols = get_symbols()

while True:
    predictions, portfolio = run_trading_cycle(symbols)

    # Mostrar predicciones
    if show_predictions:
        st.subheader("🔮 Predicciones AI")
        for p in predictions:
            st.markdown(f"{p['symbol']}: {p['signal']} ({p['confidence']:.1%})")

    # Mostrar portfolio
    if show_portfolio:
        st.subheader("💰 Portfolio")
        for pos in portfolio:
            st.markdown(f"{pos['symbol']}: {pos['quantity']} unidades, PnL: {pos['pnl_pct']:+.2f}%")

    # Gráficas
    for symbol in symbols[:3]:  # Limitar a 3 por visualización
        market_data = data_manager.fetch_market_chart(symbol)
        plot_market_chart(symbol, market_data)

    # Métricas de desempeño
    if show_metrics:
        st.subheader("📊 Métricas")
        metrics = portfolio_manager.get_metrics()
        st.metric("Total Trades", metrics['total_trades'])
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
        st.metric("Total PnL", f"${metrics['total_pnl']:.2f}")
        st.metric("Best Trade", f"+{metrics['best_trade']:.2f}%")
        if metrics['never_sold_loss']:
            st.success("💎 Never Sold at Loss: ✅")

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.experimental_rerun()
    else:
        break
