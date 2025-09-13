"""
Configuraciones globales del sistema PAPA-DINERO
"""

import os
from pathlib import Path

# Directorios del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Configuración de trading
TRADING_CONFIG = {
    # Regla principal: NUNCA VENDER EN PÉRDIDA
    'never_sell_at_loss': True,
    
    # Configuración de ganancias
    'min_profit_threshold': 2.0,  # Mínimo 2% para vender
    'take_profit_levels': [2.0, 5.0, 10.0, 20.0],  # Niveles de toma de ganancia
    
    # Gestión de riesgo
    'max_position_size': 15.0,     # Máximo 15% del capital por posición
    'max_positions': 8,            # Máximo 8 posiciones simultáneas
    'emergency_stop_loss': -50.0,  # Solo vender si pérdida > 50% y > 30 días
    'max_hold_days': 365,         # Máximo 1 año holding
    
    # Configuración de IA
    'ai_confidence_threshold': 0.75,  # Mínima confianza para trading
    'prediction_horizon_minutes': 30,  # Horizonte de predicción
    'learning_rate': 0.01,
    'model_retrain_frequency': 100,   # Cada 100 nuevos datos
    
    # Configuración de mercado
    'symbols': [
        'BTCUSD', 'ETHUSD', 'SOLUSD', 'AVAXUSD', 'ADAUSD',
        'DOTUSD', 'MATICUSD', 'LINKUSD', 'UNIUSD', 'AAVEUSD',
        'LTCUSD', 'BCHUSD', 'XTZUSD', 'ALGOUSD', 'ATOMUSD'
    ],
    
    # Timeframes para análisis
    'timeframes': {
        'primary': '1Min',
        'secondary': '5Min',
        'analysis': '1Hour',
        'longterm': '1Day'
    }
}

# Configuración de APIs
API_CONFIG = {
    'alpaca': {
        'paper_base_url': 'https://paper-api.alpaca.markets',
        'live_base_url': 'https://api.alpaca.markets',
        'api_version': 'v2',
        'timeout': 30
    }
}

# Configuración de logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_rotation': True,
    'max_file_size': '10MB',
    'backup_count': 5
}

# Configuración de dashboard
DASHBOARD_CONFIG = {
    'refresh_rate': 30,        # Segundos
    'cache_duration': 30,      # Segundos
    'max_chart_points': 1000,
    'default_timeframe': '1Hour'
}

# Configuración de memoria/persistencia
MEMORY_CONFIG = {
    'max_patterns': 10000,     # Máximo patrones en memoria
    'cleanup_frequency': 24,   # Horas
    'backup_frequency': 6,     # Horas
    'data_retention_days': 90
}

# Variables de entorno
def get_env_var(name: str, default: str = None) -> str:
    """Obtener variable de entorno con valor por defecto"""
    return os.getenv(name, default)

# Configuración específica para Render.com
RENDER_CONFIG = {
    'port': int(get_env_var('PORT', '8501')),
    'host': get_env_var('HOST', '0.0.0.0'),
    'debug': get_env_var('DEBUG', 'False').lower() == 'true',
    'redis_url': get_env_var('REDIS_URL'),  # Para cache distribuido si disponible
}

# Validar configuración crítica
def validate_config():
    """Validar configuración crítica"""
    errors = []
    
    # Verificar que las reglas de trading estén correctas
    if not TRADING_CONFIG['never_sell_at_loss']:
        errors.append("❌ CRÍTICO: never_sell_at_loss debe ser True")
    
    if TRADING_CONFIG['min_profit_threshold'] < 0.5:
        errors.append("❌ ADVERTENCIA: min_profit_threshold muy bajo")
    
    # Verificar directorios
    for directory in [DATA_DIR, LOGS_DIR, CONFIG_DIR]:
        directory.mkdir(exist_ok=True)
    
    if errors:
        for error in errors:
            print(error)
        return False
    
    return True

# Ejecutar validación al importar
if __name__ != '__main__':
    validate_config()
