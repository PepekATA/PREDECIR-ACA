import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StreamlitApp")

st.set_page_config(
    page_title="🧠 PAPA-DINERO - AI Crypto Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

os.makedirs('modules', exist_ok=True)
os.makedirs('data', exist_ok=True)

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    .main-header {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);padding: 2rem;border-radius: 15px;color: white;text-align: center;margin-bottom: 2rem;box-shadow: 0 10px 40px rgba(0,0,0,0.3);font-family: 'Orbitron', monospace;}
    .never-sell-loss {background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);padding: 1.5rem;border-radius: 15px;color: white;text-align: center;font-size: 1.2rem;font-weight: bold;margin: 1rem 0;font-family: 'Orbitron', monospace;letter-spacing: 2px;text-transform: uppercase;animation: glow 3s ease-in-out infinite alternate;}
    .ai-brain {background: #1a1a2e;color: #0ff1ce;padding: 1.5rem;border-radius: 15px;border: 2px solid #0ff1ce;margin: 1rem 0;font-family: 'Orbitron', monospace;animation: glow-ai 3s ease-in-out infinite alternate;}
    @keyframes glow {from { box-shadow: 0 0 10px #f5576c, 0 0 20px #f5576c, 0 0 30px #f5576c; }to { box-shadow: 0 0 20px #f5576c, 0 0 30px #f5576c, 0 0 40px #f5576c; }}
    @keyframes glow-ai {from { box-shadow: 0 0 10px #0ff1ce, 0 0 20px #0ff1ce; }to { box-shadow: 0 0 20px #0ff1ce, 0 0 30px #0ff1ce; }}
</style>
""", unsafe_allow_html=True)

# Importar o crear gestor de credenciales
try:
    from modules.credentials_manager import CredentialsManager
    credentials_manager = CredentialsManager()
except ImportError:
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
                else:
                    return None
            except Exception as e:
                st.error(f"Error cargando credenciales: {e}")
                return None
    credentials_manager = BasicCredentialsManager()

st.markdown("<div class='main-header'><h1>🤖 PAPA-DINERO - AI Crypto Trading Bot</h1></div>", unsafe_allow_html=True)
st.markdown("<div class='never-sell-loss'>💎 Estrategia: NUNCA VENDER EN PÉRDIDA - SIEMPRE GANANCIA</div>", unsafe_allow_html=True)

# Panel para credenciales
with st.expander("🔑 Configuración de credenciales Alpaca", expanded=True):
    cred = credentials_manager.load_credentials()
    if cred:
        st.success(f"Credenciales cargadas. Modo: {'Paper' if cred.get('paper_trading', True) else 'Live'}")
        st.write(cred)
    else:
        with st.form("credentials_form"):
            api_key = st.text_input("ALPACA_API_KEY")
            api_secret = st.text_input("ALPACA_SECRET_KEY", type="password")
            paper_trading = st.checkbox("Modo Paper Trading", value=True)
            submitted = st.form_submit_button("Guardar credenciales")
            if submitted:
                ok = credentials_manager.save_credentials(api_key, api_secret, paper_trading)
                if ok:
                    st.success("Credenciales guardadas correctamente")
                else:
                    st.error("No se pudieron guardar las credenciales.")

# Test de conexión Alpaca
st.markdown("## 🟢 Test de conexión Alpaca API")
try:
    import alpaca_trade_api as tradeapi
except ImportError:
    st.warning("El paquete alpaca-trade-api no está instalado. Añádelo a requirements.txt.")

creds = credentials_manager.load_credentials()
al_connected = False
if creds:
    api_key = creds['api_key']
    api_secret = creds['api_secret']
    base_url = "https://paper-api.alpaca.markets" if creds.get('paper_trading', True) else "https://api.alpaca.markets"
    try:
        api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        account = api.get_account()
        st.success(f"¡Conexión Alpaca exitosa! Estado de la cuenta: {account.status}")
        st.json({
            "account_id": account.id,
            "buying_power": account.buying_power,
            "equity": account.equity,
            "status": account.status
        })
        al_connected = True
    except Exception as e:
        st.error(f"Error al conectar con Alpaca API: {e}")
else:
    st.warning("No hay credenciales guardadas para conectar con Alpaca.")

# --- Panel Estado del bot y controles ---
bot_state_file = Path('data') / 'bot_state.json'

if bot_state_file.exists():
    with open(bot_state_file, 'r') as f:
        bot_state = json.load(f)
else:
    bot_state = {"running": False, "last_action": None, "alpaca_connected": al_connected}

st.markdown("<h2>🕹️ Control del Bot</h2>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Iniciar Bot"):
        bot_state["running"] = True
        bot_state["last_action"] = f"Iniciado el {datetime.now()}"
        bot_state["alpaca_connected"] = al_connected
        with open(bot_state_file, 'w') as f:
            json.dump(bot_state, f, indent=2)
        st.success("Bot iniciado")
with col2:
    if st.button("⏸️ Pausar Bot"):
        bot_state["running"] = False
        bot_state["last_action"] = f"Pausado el {datetime.now()}"
        bot_state["alpaca_connected"] = al_connected
        with open(bot_state_file, 'w') as f:
            json.dump(bot_state, f, indent=2)
        st.warning("Bot pausado")

st.markdown("<h2>📊 Estado actual del bot</h2>", unsafe_allow_html=True)
st.json(bot_state)

# --- Panel visual demo ---
st.markdown("<h2>📈 Panel de Trading y Predicciones</h2>", unsafe_allow_html=True)
st.info("Esta sección mostrará métricas, gráficos y predicciones de trading en tiempo real cuando el bot esté en ejecución.")

demo_data = pd.DataFrame({
    "Time": pd.date_range(datetime.now() - timedelta(hours=10), periods=10, freq="h"),
    "BTCUSD": np.random.normal(44000, 500, 10),
    "ETHUSD": np.random.normal(3300, 80, 10),
})
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=demo_data["Time"], y=demo_data["BTCUSD"], name="BTCUSD", line=dict(color="blue")), secondary_y=False)
fig.add_trace(go.Scatter(x=demo_data["Time"], y=demo_data["ETHUSD"], name="ETHUSD", line=dict(color="green")), secondary_y=True)
fig.update_layout(title_text="Demo precios BTC y ETH")
st.plotly_chart(fig, use_container_width=True)

st.markdown("<div class='ai-brain'>🧠 Módulo de IA: Aquí se mostrarán predicciones automáticas si el bot está conectado.</div>", unsafe_allow_html=True)
