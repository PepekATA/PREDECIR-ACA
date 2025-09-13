#!/usr/bin/env python3
"""
PAPA-DINERO - AI Crypto Trading Bot
Never Sell at Loss - Always Profit Strategy

Main entry point for the trading bot.
Handles initialization, coordination, and 24/7 operation.
"""

import os
import sys
import asyncio
import signal
import logging
import json
from datetime import datetime
import traceback
from pathlib import Path

# Configurar el path de módulos
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Configuración básica de logging
log_dir = current_dir / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'papa_dinero.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("PAPA-DINERO")

class PapaDineroBot:
    """Bot de trading principal que coordina todos los módulos"""
    
    def __init__(self):
        self.running = False
        self.cycle_count = 0
        
        # Estados de componentes
        self.components = {}
        self.api_connected = False
        self.credentials = None
        
        # Crear directorios necesarios
        self.create_directories()
        
        logger.info("🤖 PAPA-DINERO Bot inicializado")
        logger.info("💎 Estrategia: NUNCA VENDER EN PÉRDIDA - SIEMPRE GANANCIA")
    
    def create_directories(self):
        """Crear directorios necesarios"""
        directories = ['data', 'logs', 'config']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_credentials_from_env(self):
        """Obtener credenciales desde variables de entorno"""
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_SECRET_KEY')
        paper_trading = os.getenv('PAPER_TRADING', 'True').lower() == 'true'
        
        if not api_key or not api_secret:
            logger.error("❌ Variables ALPACA_API_KEY y ALPACA_SECRET_KEY no encontradas")
            return None
        
        self.credentials = {
            'api_key': api_key,
            'api_secret': api_secret,
            'paper_trading': paper_trading,
            'base_url': 'https://paper-api.alpaca.markets' if paper_trading else 'https://api.alpaca.markets'
        }
        
        logger.info(f"✅ Credenciales cargadas - Modo: {'Paper' if paper_trading else 'Live'}")
        return self.credentials
    
    async def initialize_components(self):
        """Inicializar componentes del bot de forma segura"""
        try:
            # Verificar credenciales
            if not self.get_credentials_from_env():
                raise Exception("Credenciales no disponibles")
            
            # Importar módulos dinámicamente para manejar errores
            try:
                from modules.memory_system import MemorySystem
                self.components['memory'] = MemorySystem()
                logger.info("✅ Sistema de memoria inicializado")
            except ImportError as e:
                logger.warning(f"⚠️ Memory system no disponible: {e}")
                self.components['memory'] = None
            
            try:
                from modules.trading_engine import TradingEngine
                if self.components['memory']:
                    self.components['trading'] = TradingEngine(self.components['memory'])
                    
                    # Conectar API
                    api_connected = self.components['trading'].initialize_api(
                        self.credentials['api_key'],
                        self.credentials['api_secret'],
                        self.credentials['paper_trading']
                    )
                    
                    if api_connected:
                        self.api_connected = True
                        logger.info("✅ Trading engine conectado a Alpaca")
                    else:
                        raise Exception("Error conectando a Alpaca API")
                else:
                    raise Exception("Memory system requerido para trading engine")
                    
            except Exception as e:
                logger.error(f"❌ Trading engine no disponible: {e}")
                self.components['trading'] = None
            
            # Otros componentes opcionales
            try:
                from modules.ai_predictor import AIPredictor
                if self.components['memory']:
                    self.components['ai'] = AIPredictor(self.components['memory'])
                    logger.info("✅ AI Predictor inicializado")
            except ImportError as e:
                logger.warning(f"⚠️ AI Predictor no disponible: {e}")
                self.components['ai'] = None
            
            try:
                from modules.portfolio_manager import PortfolioManager
                self.components['portfolio'] = PortfolioManager()
                logger.info("✅ Portfolio Manager inicializado")
            except ImportError as e:
                logger.warning(f"⚠️ Portfolio Manager no disponible: {e}")
                self.components['portfolio'] = None
            
            # Crear archivo de estado para indicar que el bot está activo
            await self.create_bot_state_file()
            
            return self.api_connected
            
        except Exception as e:
            logger.error(f"❌ Error inicializando componentes: {e}")
            return False
    
    async def create_bot_state_file(self):
        """Crear archivo de estado para el dashboard"""
        try:
            state = {
                'status': 'active' if self.api_connected else 'demo',
                'initialized_at': datetime.now().isoformat(),
                'components': {
                    'trading_engine': self.components.get('trading') is not None,
                    'ai_predictor': self.components.get('ai') is not None,
                    'memory_system': self.components.get('memory') is not None,
                    'portfolio_manager': self.components.get('portfolio') is not None,
                    'api_connected': self.api_connected
                },
                'credentials_found': self.credentials is not None,
                'paper_trading': self.credentials.get('paper_trading', True) if self.credentials else True
            }
            
            with open('data/bot_state.json', 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info("✅ Archivo de estado creado")
            
        except Exception as e:
            logger.error(f"❌ Error creando archivo de estado: {e}")
    
    def get_active_symbols(self):
        """Símbolos activos para trading"""
        return [
            'BTCUSD', 'ETHUSD', 'SOLUSD', 'AVAXUSD', 'ADAUSD',
            'DOTUSD', 'MATICUSD', 'LINKUSD', 'UNIUSD', 'AAVEUSD'
        ]
    
    async def run_trading_cycle(self):
        """Ejecutar un ciclo completo de trading"""
        if not self.api_connected or not self.components.get('trading'):
            logger.warning("⚠️ API no conectada, saltando ciclo de trading")
            return {'status': 'skipped', 'reason': 'api_not_connected'}
        
        try:
            self.cycle_count += 1
            start_time = datetime.now()
            
            logger.info(f"🔄 Ciclo #{self.cycle_count} - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Obtener símbolos activos
            symbols = self.get_active_symbols()
            
            # Ejecutar ciclo de trading automático
            trading_engine = self.components['trading']
            ai_predictor = self.components.get('ai')
            
            if ai_predictor:
                # Trading con IA
                results = trading_engine.auto_trading_cycle(symbols, ai_predictor, max_positions=5)
            else:
                # Trading básico sin IA
                logger.info("🤖 Trading sin IA - usando análisis técnico básico")
                results = await self.basic_trading_cycle(symbols)
            
            # Actualizar archivo de estado
            await self.update_bot_state(results)
            
            cycle_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"⏱️ Ciclo completado en {cycle_time:.2f}s - Acciones: {len(results.get('actions_taken', []))}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en ciclo de trading: {e}")
            traceback.print_exc()
            return {'status': 'error', 'error': str(e)}
    
    async def basic_trading_cycle(self, symbols):
        """Ciclo básico de trading sin IA"""
        actions = []
        
        try:
            trading_engine = self.components['trading']
            
            # Revisar posiciones existentes
            for symbol in list(trading_engine.positions.keys()):
                if trading_engine.positions[symbol]['status'] == 'active':
                    try:
                        # Obtener precio actual (simulado para demo)
                        current_price = 45000  # Precio demo
                        
                        analysis = trading_engine.analyze_position_profitability(symbol, current_price)
                        
                        if analysis['can_sell'] and analysis['recommended_action'] == 'sell_profit':
                            # Intentar venta (solo con ganancia)
                            logger.info(f"💰 Intentando venta con ganancia: {symbol}")
                            
                    except Exception as e:
                        logger.error(f"Error analizando {symbol}: {e}")
            
            return {
                'timestamp': datetime.now().isoformat(),
                'actions_taken': actions,
                'positions_analyzed': len(trading_engine.positions),
                'mode': 'basic_trading'
            }
            
        except Exception as e:
            logger.error(f"❌ Error en trading básico: {e}")
            return {'error': str(e)}
    
    async def update_bot_state(self, trading_results):
        """Actualizar estado del bot"""
        try:
            state = {
                'status': 'active' if self.api_connected else 'demo',
                'last_update': datetime.now().isoformat(),
                'total_cycles': self.cycle_count,
                'api_connected': self.api_connected,
                'last_trading_results': trading_results,
                'components': {
                    'trading_engine': self.components.get('trading') is not None,
                    'ai_predictor': self.components.get('ai') is not None,
                    'memory_system': self.components.get('memory') is not None,
                    'portfolio_manager': self.components.get('portfolio') is not None
                }
            }
            
            with open('data/bot_state.json', 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Error actualizando estado: {e}")
    
    async def run_main_loop(self):
        """Loop principal del bot"""
        logger.info("🚀 Iniciando loop principal...")
        self.running = True
        
        while self.running:
            try:
                # Ejecutar ciclo de trading
                await self.run_trading_cycle()
                
                # Esperar 60 segundos entre ciclos
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("🛑 Interrupción manual")
                break
            except Exception as e:
                logger.error(f"❌ Error en loop principal: {e}")
                await asyncio.sleep(120)  # Esperar más tiempo si hay error
    
    async def shutdown(self):
        """Cierre ordenado"""
        logger.info("🛑 Cerrando bot...")
        self.running = False
        
        try:
            # Guardar estado final
            if self.components.get('trading'):
                self.components['trading'].save_trading_state()
            
            # Actualizar estado a inactivo
            with open('data/bot_state.json', 'w') as f:
                json.dump({
                    'status': 'stopped',
                    'stopped_at': datetime.now().isoformat(),
                    'total_cycles': self.cycle_count
                }, f, indent=2)
            
            logger.info("✅ Bot cerrado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error en cierre: {e}")

async def main():
    """Función principal"""
    bot = PapaDineroBot()
    
    # Manejar señales de cierre
    def signal_handler(signum, frame):
        logger.info(f"🛑 Señal {signum} recibida")
        asyncio.create_task(bot.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Inicializar componentes
        if await bot.initialize_components():
            logger.info("✅ Bot inicializado, comenzando trading...")
            await bot.run_main_loop()
        else:
            logger.warning("⚠️ Bot iniciado en modo limitado (sin API)")
            # Crear archivo de estado para demo
            await bot.create_bot_state_file()
            # Mantener vivo para Streamlit
            while True:
                await asyncio.sleep(60)
                await bot.create_bot_state_file()  # Actualizar estado
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Salida manual")
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}")
        traceback.print_exc()
