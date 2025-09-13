import streamlit as st
import os

st.set_page_config(
    page_title="Comprobador de Variables de Entorno",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Comprobador de Variables de Entorno de Render.com")
st.markdown("Esta aplicación simple revisa si las variables de entorno de Alpaca están disponibles para el bot.")
st.markdown("---")

# Verificar las variables de entorno
alpaca_api_key = os.getenv("ALPACA_API_KEY")
alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
paper_trading_mode = os.getenv("PAPER_TRADING")

st.subheader("Estado de las Variables")

if alpaca_api_key:
    st.success(f"✅ `ALPACA_API_KEY` encontrada. (Valor: {alpaca_api_key[:4]}...{alpaca_api_key[-4:]})")
else:
    st.error("❌ `ALPACA_API_KEY` no encontrada.")
    st.warning("Asegúrate de haberla agregado en las variables de entorno de Render.com.")

if alpaca_secret_key:
    st.success(f"✅ `ALPACA_SECRET_KEY` encontrada. (Valor: {'*' * 10})")
else:
    st.error("❌ `ALPACA_SECRET_KEY` no encontrada.")
    st.warning("Asegúrate de haberla agregado en las variables de entorno de Render.com.")

if paper_trading_mode:
    st.info(f"ℹ️ `PAPER_TRADING` encontrada. (Valor: `{paper_trading_mode}`)")
else:
    st.info("ℹ️ `PAPER_TRADING` no encontrada. El bot usará el valor predeterminado.")

st.markdown("---")
st.markdown("Si todo se muestra en verde, significa que las variables de entorno están correctamente configuradas y el problema está en la lógica de conexión del bot. Si no, necesitarás agregarlas a la configuración de tu servicio en Render.com.")
