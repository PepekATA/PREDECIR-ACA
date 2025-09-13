# 🧠 PAPA-DINERO - AI Crypto Trading Bot

## 💎 NUNCA VENDER EN PÉRDIDA - SIEMPRE GANANCIA 💎

Bot de trading de criptomonedas con Inteligencia Artificial que implementa la estrategia "Never Sell at Loss" - nunca vende una posición en pérdida, solo en ganancia.

## 🚀 Características Principales

- **🤖 Inteligencia Artificial Avanzada**: Múltiples modelos de ML (LSTM, Random Forest, Gradient Boosting)
- **💎 Never Sell at Loss**: Regla fundamental - nunca vende en pérdida
- **📊 Trading 24/7**: Opera continuamente en el mercado de criptomonedas
- **🎯 Predicciones Precisas**: Alta confianza en predicciones antes de operar
- **📈 Dashboard en Tiempo Real**: Interfaz web con Streamlit
- **🔒 Seguridad**: Credenciales cifradas y almacenamiento seguro
- **🌐 Cloud Ready**: Optimizado para Render.com

## 🏗️ Arquitectura Modular

### Módulos Core
- **main.py**: Coordinador principal del bot
- **persistence_manager.py**: Gestión segura de credenciales
- **data_manager.py**: Recolección de datos de mercado
- **ai_predictor.py**: Motor de IA con múltiples modelos
- **trading_engine.py**: Ejecución de operaciones
- **portfolio_manager.py**: Gestión inteligente de cartera
- **memory_system.py**: Aprendizaje continuo
- **market_analyzer.py**: Análisis técnico avanzado
- **dashboard.py**: Backend del dashboard

## 🛠️ Instalación Local
```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/papa-dinero.git
cd papa-dinero

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales de Alpaca
