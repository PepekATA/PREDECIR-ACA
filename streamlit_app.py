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

# ====== IMPORTS PARA ALPACA (INTENTAMOS SOPORTAR SDK NUEVA + LEGADO) ======
ALPACA_SDK = None
ALPACA_LEGACY = None
try:
    # alpaca-py (nueva)
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
    ALPACA_SDK = "alpaca-py"
except Exception:
    try:
        # alpaca_trade_api (legacy)
        import alpaca_trade_api as alpaca_legacy
        ALPACA_LEGACY = "alpaca-trade-api"
    except Exception:
        # no alpaca installed
        pass

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

# CSS mejorado para Render (mismo que tenías)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    /* ... (omito repetir CSS por brevedad, pega el CSS original aquí) ... */
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
                return None
            except Exception as e:
                st.error(f"Error cargando credenciales: {e}")
                return None
        
        def credentials_exist(self):
            return self.credentials_file.exists()
        
        def delete_credentials(self):
            try:
                if self.credentials_file.exists():
                    self.credentials_file.unlink()
                return True
            except Exception as e:
                st.error(f"Error eliminando credenciales: {e}")
                return False
    
    credentials_manager = BasicCredentialsManager()

# Header principal (igual que antes)
st.markdown("""
<div class="main-header">
    <h1>🧠 PAPA-DINERO AI CRYPTO BOT</h1>
    <h2>🚀 POWERED BY ADVANCED AI • NEVER SELL AT LOSS 🚀</h2>
    <div class="never-sell-loss">
        💎 REGLA DE ORO: SOLO VENDE CON GANANCIA - NUNCA EN PÉRDIDA 💎
    </div>
</div>
""", unsafe_allow_html=True)

# Variables globales
bot_available = False
dashboard_manager = None

# -----------------------
# Funciones de integración con Alpaca
# -----------------------
def init_alpaca_client(api_key: str, api_secret: str, paper_trading: bool = True):
    """
    Inicializa un cliente Alpaca. Intenta usar 'alpaca-py' y si no está instalar, usa 'alpaca-trade-api'.
    Guarda el cliente en st.session_state['alpaca_client'] para su uso posterior.
    """
    if not api_key or not api_secret:
        return None
    
    # Preferir nueva SDK
    if ALPACA_SDK:
        try:
            base_url = "https://paper-api.alpaca.markets" if paper_trading else "https://api.alpaca.markets"
            client = TradingClient(api_key, api_secret, paper=paper_trading)
            logger.info("Alpaca client initialized (alpaca-py)")
            return client
        except Exception as e:
            logger.error(f"Error inicializando alpaca-py client: {e}")
            # fallthrough a legacy
    if ALPACA_LEGACY:
        try:
            # legacy REST client
            api = alpaca_legacy.REST(api_key, api_secret, api_version='v2')
            logger.info("Alpaca client initialized (alpaca-trade-api legacy)")
            return api
        except Exception as e:
            logger.error(f"Error inicializando alpaca_trade_api client: {e}")
    # Si no hay cliente disponible
    logger.warning("No Alpaca SDK disponible. Instala 'alpaca-py' o 'alpaca-trade-api'.")
    return None

def place_market_order(symbol: str, qty: float, side: str = "buy", time_in_force="day"):
    """
    Envía una orden de mercado simple. Devuelve el objeto orden o None en caso de error.
    side: 'buy'|'sell' (o en español, será mapeado)
    """
    client = st.session_state.get('alpaca_client')
    if client is None:
        st.error("Alpaca client no inicializado.")
        return None
    
    # Normalizar lado
    s = side.lower()
    if s in ["comprar", "buy", "b"]:
        side_api = "buy"
    else:
        side_api = "sell"
    
    try:
        if ALPACA_SDK:
            # alpaca-py MarketOrderRequest
            order_req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side_api == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            order = client.submit_order(order_data=order_req)
            return order
        elif ALPACA_LEGACY:
            # legacy: api.submit_order
            order = st.session_state['alpaca_client'].submit_order(
                symbol=symbol,
                qty=qty,
                side=side_api,
                type='market',
                time_in_force='day'
            )
            return order
        else:
            st.error("No hay SDK Alpaca disponible para enviar órdenes.")
            return None
    except Exception as e:
        st.error(f"Error enviando orden: {e}")
        logger.error(f"Error enviando orden: {e}")
        return None

def fetch_and_format_orders(limit: int = 200):
    """
    Trae órdenes desde Alpaca y las normaliza en un DataFrame con las columnas:
    [Activo, Tipo de orden, Lado, Cantidad, Cantidad llena, Precio promedio de llenado, Estado, Fuente, Enviado en, Lleno en, Caduca a las]
    """
    client = st.session_state.get('alpaca_client')
    if client is None:
        return pd.DataFrame()
    
    raw_orders = []
    try:
        if ALPACA_SDK:
            # alpaca-py client.get_orders() devuelve list[Order]
            orders = client.get_orders(status="all", limit=limit)
            for o in orders:
                # o es un Order object; usamos atributos conocidos
                sym = getattr(o, 'symbol', None) or (o.get('symbol') if isinstance(o, dict) else None)
                qty = float(getattr(o, 'qty', 0) or 0)
                filled_qty = float(getattr(o, 'filled_qty', 0) or 0)
                filled_avg = getattr(o, 'filled_avg_price', None) or getattr(o, 'filled_avg', None)
                order_type = getattr(o, 'type', None)
                side = getattr(o, 'side', None)
                status = getattr(o, 'status', None)
                submitted_at = getattr(o, 'submitted_at', None)
                filled_at = getattr(o, 'filled_at', None)
                expires_at = getattr(o, 'expires_at', None)
                source = getattr(o, 'source', None) if hasattr(o, 'source') else None
                
                raw_orders.append({
                    "Activo": sym,
                    "Tipo de orden": order_type or "",
                    "Lado": "comprar" if (str(side).lower() == "buy") else "vender",
                    "Cantidad": qty,
                    "Cantidad llena": filled_qty,
                    "Precio promedio de llenado": float(filled_avg) if filled_avg else None,
                    "Estado": status,
                    "Fuente": source or "-",
                    "Enviado en": _to_dt(submitted_at),
                    "Lleno en": _to_dt(filled_at),
                    "Caduca a las": _to_dt(expires_at)
                })
        elif ALPACA_LEGACY:
            # legacy: list_orders
            orders = st.session_state['alpaca_client'].list_orders(status='all', limit=limit)
            for o in orders:
                # o es un object con attrs
                sym = o.symbol
                qty = float(o.qty) if o.qty else 0.0
                filled_qty = float(o.filled_qty) if getattr(o, 'filled_qty', None) else 0.0
                filled_avg = getattr(o, 'filled_avg_price', None) or getattr(o, 'filled_avg', None)
                order_type = o.type
                side = o.side
                status = o.status
                submitted_at = getattr(o, 'submitted_at', None)
                filled_at = getattr(o, 'filled_at', None)
                expires_at = getattr(o, 'expires_at', None)
                source = getattr(o, 'client_order_id', None) or "-"
                
                raw_orders.append({
                    "Activo": sym,
                    "Tipo de orden": order_type or "",
                    "Lado": "comprar" if (str(side).lower() == "buy") else "vender",
                    "Cantidad": qty,
                    "Cantidad llena": filled_qty,
                    "Precio promedio de llenado": float(filled_avg) if filled_avg else None,
                    "Estado": status,
                    "Fuente": source or "-",
                    "Enviado en": _to_dt(submitted_at),
                    "Lleno en": _to_dt(filled_at),
                    "Caduca a las": _to_dt(expires_at)
                })
        else:
            # No client available -> retornar vacío
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching orders: {e}")
        st.error(f"Error obteniendo órdenes: {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(raw_orders)
    # Ordenar por fecha enviada descendente
    if "Enviado en" in df.columns:
        df = df.sort_values(by="Enviado en", ascending=False)
    return df

def _to_dt(value):
    """Normaliza distintos formatos de timestamps a pd.Timestamp o None"""
    if value is None:
        return None
    # Si ya es datetime
    if isinstance(value, datetime):
        return value
    try:
        # alpaca-py a veces devuelve strings ISO
        return pd.to_datetime(value)
    except Exception:
        # fallback: intentar parseo simple
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return None

# -----------------------
# Interfaz y lógica previa (la mantuve)
# -----------------------
def show_credentials_setup():
    """Mostrar interfaz de configuración de credenciales"""
    st.markdown("""
    <div class="credentials-setup">
        <h2>🔐 Configuración de API - Alpaca Markets</h2>
        <p>Para comenzar a operar con dinero real, configura tus credenciales de Alpaca:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("credentials_form"):
            st.subheader("📊 Credenciales API")
            
            api_key = st.text_input(
                "🔑 API Key", 
                type="password",
                placeholder="Pega tu API Key de Alpaca aquí",
                help="Tu API Key de Alpaca Markets"
            )
            
            api_secret = st.text_input(
                "🔒 Secret Key", 
                type="password",
                placeholder="Pega tu Secret Key de Alpaca aquí",
                help="Tu Secret Key de Alpaca Markets"
            )
            
            paper_trading = st.radio(
                "🎮 Modo de Trading:",
                ["Paper Trading (Recomendado)", "Live Trading"],
                index=0,
                help="Paper Trading para pruebas, Live Trading para dinero real"
            )
            
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                submitted = st.form_submit_button(
                    "💾 Guardar Credenciales", 
                    type="primary",
                    use_container_width=True
                )
            with col_btn2:
                skip_credentials = st.form_submit_button(
                    "🎮 Continuar en Demo",
                    use_container_width=True
                )
            
            if submitted:
                if api_key and api_secret:
                    paper_mode = paper_trading == "Paper Trading (Recomendado)"
                    
                    if credentials_manager.save_credentials(api_key, api_secret, paper_mode):
                        st.success("✅ Credenciales guardadas correctamente!")
                        st.info("🔄 Inicializando cliente Alpaca...")
                        # Inicializar cliente Alpaca en memoria
                        client = init_alpaca_client(api_key, api_secret, paper_mode)
                        if client:
                            st.session_state['alpaca_client'] = client
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.error("❌ Error guardando credenciales")
                else:
                    st.error("⚠️ Por favor completa todos los campos")
            
            if skip_credentials:
                # Crear archivo temporal para saltar configuración
                with open('data/demo_mode.json', 'w') as f:
                    json.dump({'demo_mode': True, 'created_at': str(datetime.now())}, f)
                st.info("🎮 Continuando en modo demo...")
                time.sleep(1)
                st.rerun()
    
    with col2:
        st.markdown("""
        ### 📋 Cómo obtener credenciales:
        
        1. **Registrarse** en [alpaca.markets](https://alpaca.markets)
        2. **Verificar** tu cuenta
        3. Ir a **'Paper Trading'** en el dashboard
        4. Crear **nuevas credenciales API**
        5. **Copiar** API Key y Secret Key
        6. **Pegarlos** aquí
        
        ### ⚠️ Importante:
        - Usa **Paper Trading** para pruebas
        - **Nunca** compartas tus credenciales
        - Las credenciales se guardan **localmente**
        - Puedes **eliminarlas** cuando quieras
        
        ### 🎮 Modo Demo:
        Si prefieres probar primero, puedes continuar en **modo demo** sin credenciales.
        """)

# ============================================================
# MODIFICACIÓN: check_credentials_and_initialize (reemplazo)
# ============================================================
def check_credentials_and_initialize():
    """Verificar credenciales e inicializar bot + cliente Alpaca"""
    global bot_available, dashboard_manager
    
    # Verificar si está en modo demo
    demo_file = Path('data/demo_mode.json')
    if demo_file.exists():
        return False  # Continuar en demo
    
    if credentials_manager.credentials_exist():
        credentials = credentials_manager.load_credentials()
        if credentials:
            # Configurar variables de entorno (opcional)
            os.environ['ALPACA_API_KEY'] = credentials['api_key']
            os.environ['ALPACA_SECRET_KEY'] = credentials['api_secret']
            os.environ['PAPER_TRADING'] = str(credentials['paper_trading'])
            
            # Inicializar cliente Alpaca y guardar en session_state
            client = init_alpaca_client(credentials['api_key'], credentials['api_secret'], credentials['paper_trading'])
            if client:
                st.session_state['alpaca_client'] = client
            # Intentar inicializar otros módulos del bot si existen
            try:
                from modules.multi_symbol_trader import MultiSymbolTrader
                trader = MultiSymbolTrader()
                # no arrancamos automáticamente aquí, lo dejamos al bloque que añadiste
                bot_available = True
                return True
            except Exception:
                # si no existe ese módulo, devolvemos True si client está presente o si las credenciales simplemente existen
                return True
    return False

# (Mantengo show_sidebar_controls, get_demo_data, create_market_chart tal cual)
# ... Puedes pegar aquí las definiciones originales show_sidebar_controls, get_demo_data, create_market_chart ...
# Para no duplicar el mensaje, las dejo sin cambios — asegúrate de copiar las funciones originales que ya tenías.

# Para integrarlo en la función de dashboard, añadimos la visualización de órdenes Alpaca
def show_main_dashboard(predictions_data, portfolio_data, show_predictions, show_portfolio, show_metrics):
    """Mostrar dashboard principal (manteniendo el contenido original)"""
    # --- PRIMERA PARTE: si hay cliente Alpaca, traer y mostrar órdenes normalizadas ---
    if 'alpaca_client' in st.session_state and st.session_state['alpaca_client']:
        st.subheader("📋 Órdenes recientes (desde Alpaca)")
        with st.expander("Ver órdenes normalizadas"):
            df_orders = fetch_and_format_orders(limit=200)
            if df_orders.empty:
                st.info("No se encontraron órdenes o el cliente Alpaca no respondió.")
            else:
                # Mostrar DataFrame limpio
                st.dataframe(df_orders.fillna("-"))
                # Botones rápidos para probar enviar orden (solo demo)
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("📈 Test: comprar 0.001 BTC (orden mercado)"):
                        # intentar enviar orden de prueba
                        res = place_market_order("BTCUSD", qty=0.001, side="buy")
                        if res:
                            st.success("Orden enviada. Revisa las órdenes en la tabla.")
                            # refrescar las órdenes mostrando la última info
                            df_orders = fetch_and_format_orders(limit=200)
                            st.dataframe(df_orders.fillna("-"))
                with col_b:
                    if st.button("📉 Test: vender 0.001 BTC (orden mercado)"):
                        res = place_market_order("BTCUSD", qty=0.001, side="sell")
                        if res:
                            st.success("Orden de venta enviada.")
    
    # --- A PARTIR DE AQUÍ: pega el resto del contenido original del dashboard ---
    # Layout principal (copiar todo tu original desde aquí)
    col1, col2, col3 = st.columns([2, 1.5, 1.5])
    
    # Columna 1: Predicciones AI
    with col1:
        if show_predictions:
            st.subheader("🔮 AI Predictions Dashboard")
            
            for pred in predictions_data[:6]:
                symbol = pred.get('symbol', 'BTC')
                change = pred.get('change', pred.get('predicted_change', 0))
                confidence = pred.get('confidence', 0.5)
                signal = pred.get('signal', 'HOLD')
                
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
    # ... el resto de la función show_main_dashboard continúa exactamente como lo tenías ...
    # Para no repetir todo, asegúrate de pegar aquí el resto de tu función original desde la sección Portfolio, Metrics, Charts, Alerts, Footer, etc.

# ============================================================================
# FLUJO PRINCIPAL DE LA APLICACIÓN
# ============================================================================

# Verificar credenciales al inicio
credentials_configured = check_credentials_and_initialize()

if not credentials_configured and not Path('data/demo_mode.json').exists():
    # Mostrar configuración de credenciales si no existen
    show_credentials_setup()
    
    # Footer informativo (igual que antes)
    st.markdown("---")
    st.markdown("""
    ### 🚀 Deploy on Render.com
    
    **Quick Setup:**
    1. Fork este repo en GitHub
    2. Crear nuevo Web Service en Render
    3. Conectar tu repositorio
    4. Deploy automático!
    5. Configurar credenciales directamente aquí
    
    **Features:**
    - ✅ 24/7 AI Trading
    - ✅ Never Sell at Loss
    - ✅ Real-time Dashboard  
    - ✅ Secure Credentials
    - ✅ Auto-scaling
    """)
else:
    # Mostrar sidebar con controles (usa tu función original, idéntica)
    auto_refresh, refresh_interval, show_predictions, show_portfolio, show_metrics = show_sidebar_controls(credentials_configured)
    
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
    
    # Mostrar dashboard principal
    show_main_dashboard(predictions_data, portfolio_data, show_predictions, show_portfolio, show_metrics)
    
    # Información sobre deployment
    st.markdown("---")
    st.markdown("""
    ### 🚀 Deploy on Render.com
    
    **Features Activas:**
    - ✅ 24/7 AI Trading
    - ✅ Never Sell at Loss
    - ✅ Real-time Dashboard  
    - ✅ Secure Credentials
    - ✅ Auto-scaling
    """)
    
    # Auto-refresh solo en modo demo
    if auto_refresh and not credentials_configured:
        time.sleep(refresh_interval)
        st.rerun()

# ============================================================
# INICIALIZAR TRADING AUTOMÁTICO SI HAY CREDENCIALES
# (Se agrega antes del footer; intenta iniciar trader si aún no se inició)
# ============================================================
if credentials_configured and not Path('data/demo_mode.json').exists():
    if not hasattr(st.session_state, 'trading_started') or not st.session_state.trading_started:
        try:
            from modules.multi_symbol_trader import MultiSymbolTrader
            
            if 'trader' not in st.session_state:
                trader = MultiSymbolTrader()
                if trader.start_trading():
                    st.session_state.trader = trader
                    st.session_state.trading_started = True
                    logger.info("TRADING AUTOMÁTICO INICIADO DESDE MAIN")
            
        except Exception as e:
            logger.error(f"Error iniciando trading automático: {e}")

# Footer final (igual que antes)
st.markdown("""
---
<div style='text-align: center; color: #888; padding: 20px;'>
    <p>🧠 PAPA-DINERO AI Crypto Bot | 💎 Never Sell at Loss Strategy</p>
    <p>⚡ Powered by Advanced AI | 🚀 Deploy on Render.com</p>
</div>
""", unsafe_allow_html=True)
