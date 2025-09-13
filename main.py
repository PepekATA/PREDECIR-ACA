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
from datetime import datetime
import traceback
from pathlib import Path

# Agregar módulos al path
sys.path.append(str(Path(__file__).parent))

from modules.persistence_manager import PersistenceManager
from modules.data_manager import DataManager
from modules.ai_predictor import AIPredictor
from modules.trading_engine import TradingEngine
from modules.portfolio_manager import PortfolioManager
from modules.memory_system import MemorySystem
from modules.market_analyzer import MarketAnalyzer

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/papa_dinero.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("PAPA-DINERO")

class PapaDineroBot:
    """Bot de trading principal que coordina todos los módulos"""
    
    def __init__(self):
        self.running = False
        self.persistence_manager = None
        self.memory_system = None
        self.data_manager = None
        self.ai_predictor = None
        self.trading_engine = None
        self.portfolio_manager = None
        self.market_analyzer = None
        
        # Crear directorios necesarios
        self.create_directories()
        
        logger.info("🤖 PAPA-DINERO Bot inicializado")
        logger.info("💎 Estrategia: NUNCA VENDER EN PÉRDIDA - SIEMPRE GANANCIA")
    
    def create_directories(self):
        """Crear directorios necesarios"""
        directories = ['data', 'logs', 'config', 'modules']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    async def initialize(self):
        """Inicializar todos los módulos del bot"""
        try:
            logger.info("🔧 Inicializando módulos...")
            
            # 1. Sistema de persistencia (credenciales y estado)
            self.persistence_manager = PersistenceManager()
            await self.persistence_manager.initialize()
            
            # 2. Sistema de memoria y aprendizaje
            self.memory_system = MemorySystem()
            
            # 3. Gestor de datos de mercado
            self.data_manager = DataManager(
                self.persistence_manager.get_alpaca_credentials(),
                self.memory_system
            )
            
            # 4. Predictor de IA
            self.ai_predictor = AIPredictor(self.memory_system)
            
            # 5. Motor de trading
            self.trading_engine = TradingEngine(self.memory_system)
            
            # 6. Gestor de portafolio
            self.portfolio_manager = PortfolioManager()
            
            # 7. Analizador de mercado
            self.market_analyzer = MarketAnalyzer(
                self.data_manager, 
                self.ai_predictor
            )
            
            # Conectar API de trading
            if not await self.connect_trading_api():
                raise Exception("No se pudo conectar a la API de trading")
            
            logger.info("✅ Todos los módulos inicializados correctamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando bot: {e}")
            traceback.print_exc()
            return False
    
    async def connect_trading_api(self):
        """Conectar a la API de Alpaca"""
        try:
            # Obtener credenciales
            credentials = self.persistence_manager.get_alpaca_credentials()
            if not credentials:
                # Solicitar credenciales la primera vez
                credentials = await self.request_credentials()
                if not credentials:
                    return False
            
            # Inicializar APIs
            success_data = await self.data_manager.initialize_api(credentials)
            success_trading = self.trading_engine.initialize_api(
                credentials['api_key'],
                credentials['api_secret'],
                credentials.get('paper_trading', True)
            )
            
            if success_data and success_trading:
                logger.info("✅ APIs conectadas exitosamente")
                return True
            else:
                logger.error("❌ Error conectando APIs")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error conectando API: {e}")
            return False
    
    async def request_credentials(self):
        """Solicitar credenciales de API (primera vez)"""
        try:
            # En producción, usar variables de entorno
            api_key = os.getenv('ALPACA_API_KEY')
            api_secret = os.getenv('ALPACA_SECRET_KEY')
            paper_trading = os.getenv('PAPER_TRADING', 'True').lower() == 'true'
            
            if api_key and api_secret:
                credentials = {
                    'api_key': api_key,
                    'api_secret': api_secret,
                    'paper_trading': paper_trading,
                    'base_url': 'https://paper-api.alpaca.markets' if paper_trading else 'https://api.alpaca.markets'
                }
                
                # Guardar credenciales
                await self.persistence_manager.save_alpaca_credentials(credentials)
                logger.info(f"✅ Credenciales guardadas - Modo: {'Paper' if paper_trading else 'Live'}")
                return credentials
            else:
                logger.error("❌ Variables de entorno ALPACA_API_KEY y ALPACA_SECRET_KEY no encontradas")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error obteniendo credenciales: {e}")
            return None
    
    async def run_main_loop(self):
        """Loop principal de trading 24/7"""
        logger.info("🚀 Iniciando loop principal de trading...")
        self.running = True
        
        cycle_count = 0
        
        while self.running:
            try:
                cycle_count += 1
                start_time = datetime.now()
                
                logger.info(f"🔄 Ciclo #{cycle_count} - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 1. Actualizar datos de mercado
                await self.update_market_data()
                
                # 2. Generar predicciones de IA
                predictions = await self.generate_ai_predictions()
                
                # 3. Analizar portfolio actual
                portfolio_analysis = await self.analyze_portfolio()
                
                # 4. Ejecutar decisiones de trading
                trading_actions = await self.execute_trading_decisions(predictions, portfolio_analysis)
                
                # 5. Actualizar memoria y aprender
                await self.update_learning_system(trading_actions)
                
                # 6. Guardar estado
                await self.save_state()
                
                # Log del ciclo
                cycle_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"⏱️  Ciclo completado en {cycle_time:.2f}s")
                
                # Esperar antes del próximo ciclo (30 segundos)
                await asyncio.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("🛑 Interrupción manual detectada")
                break
            except Exception as e:
                logger.error(f"❌ Error en loop principal: {e}")
                traceback.print_exc()
                await asyncio.sleep(60)  # Esperar 1 minuto antes de reintentar
    
    async def update_market_data(self):
        """Actualizar datos de mercado para todos los símbolos"""
        try:
            symbols = self.get_active_symbols()
            await self.data_manager.update_all_symbols(symbols)
            
        except Exception as e:
            logger.error(f"❌ Error actualizando datos de mercado: {e}")
    
    async def generate_ai_predictions(self):
        """Generar predicciones de IA para todos los símbolos"""
        try:
            symbols = self.get_active_symbols()
            predictions = {}
            
            for symbol in symbols:
                # Obtener datos históricos
                price_data = self.data_manager.get_symbol_data(symbol)
                indicators = self.market_analyzer.calculate_indicators(price_data)
                
                # Generar predicción
                prediction = self.ai_predictor.predict_price_movement(
                    symbol, price_data, indicators
                )
                
                predictions[symbol] = prediction
                
                logger.debug(f"🔮 {symbol}: {prediction.get('signal', 'HOLD')} - "
                           f"Cambio: {prediction.get('predicted_change', 0):.2f}% - "
                           f"Confianza: {prediction.get('confidence', 0):.1%}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Error generando predicciones: {e}")
            return {}
    
    async def analyze_portfolio(self):
        """Analizar estado actual del portfolio"""
        try:
            # Obtener posiciones actuales
            current_positions = self.trading_engine.get_current_positions()
            
            # Analizar con el portfolio manager
            analysis = self.portfolio_manager.analyze_current_allocation(
                current_positions, 
                self.data_manager.get_current_prices()
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analizando portfolio: {e}")
            return {}
    
    async def execute_trading_decisions(self, predictions, portfolio_analysis):
        """Ejecutar decisiones de trading basadas en predicciones y análisis"""
        try:
            actions = []
            
            # Revisar posiciones existentes para posibles ventas (SOLO CON GANANCIA)
            for symbol, position in self.trading_engine.positions.items():
                if position['status'] == 'active':
                    current_price = self.data_manager.get_current_price(symbol)
                    analysis = self.trading_engine.analyze_position_profitability(symbol, current_price)
                    
                    if analysis['can_sell'] and analysis['recommended_action'] == 'sell_profit':
                        result = self.trading_engine.execute_sell_order(symbol)
                        if result['success']:
                            actions.append({
                                'action': 'sell',
                                'symbol': symbol,
                                'result': result,
                                'pnl_pct': analysis['unrealized_pnl_pct']
                            })
                            logger.info(f"💰 VENTA CON GANANCIA: {symbol} - PnL: {analysis['unrealized_pnl_pct']:.2f}%")
            
            # Buscar oportunidades de compra
            for symbol, prediction in predictions.items():
                # Solo comprar si no tenemos posición activa
                if symbol not in self.trading_engine.positions or \
                   self.trading_engine.positions[symbol]['status'] != 'active':
                    
                    if (prediction.get('signal') in ['strong_buy', 'buy'] and 
                        prediction.get('confidence', 0) > 0.75 and 
                        prediction.get('predicted_change', 0) > 2.0):
                        
                        # Calcular tamaño de posición
                        position_size = self.portfolio_manager.smart_position_sizing(
                            symbol, prediction, 
                            self.trading_engine.get_available_cash(),
                            self.trading_engine.get_current_positions()
                        )
                        
                        if position_size['recommended_size'] > 25:  # Mínimo $25
                            result = self.trading_engine.execute_buy_order(
                                symbol, 
                                position_size['recommended_size'], 
                                prediction
                            )
                            
                            if result['success']:
                                actions.append({
                                    'action': 'buy',
                                    'symbol': symbol,
                                    'amount': position_size['recommended_size'],
                                    'result': result
                                })
                                logger.info(f"🛒 COMPRA: {symbol} - ${position_size['recommended_size']:.2f}")
            
            return actions
            
        except Exception as e:
            logger.error(f"❌ Error ejecutando decisiones: {e}")
            return []
    
    async def update_learning_system(self, trading_actions):
        """Actualizar sistema de aprendizaje con resultados"""
        try:
            for action in trading_actions:
                if action['action'] == 'sell':
                    # Aprender del resultado de venta
                    outcome = 'profit' if action['pnl_pct'] > 0 else 'loss'
                    # El sistema ya aprende en trading_engine.execute_sell_order()
                    pass
            
            # Entrenar modelo si hay suficientes datos
            if len(self.memory_system.learned_signals) >= 50:
                await asyncio.to_thread(self.memory_system.train_model)
            
        except Exception as e:
            logger.error(f"❌ Error actualizando aprendizaje: {e}")
    
    async def save_state(self):
        """Guardar estado del bot"""
        try:
            await self.persistence_manager.save_bot_state({
                'last_update': datetime.now().isoformat(),
                'active_symbols': self.get_active_symbols(),
                'total_cycles': getattr(self, 'cycle_count', 0)
            })
            
            self.trading_engine.save_trading_state()
            self.portfolio_manager.save_portfolio_state()
            self.memory_system.save_memory()
            
        except Exception as e:
            logger.error(f"❌ Error guardando estado: {e}")
    
    def get_active_symbols(self):
        """Obtener símbolos activos para trading"""
        # Símbolos por defecto (se pueden configurar)
        return [
            'BTCUSD', 'ETHUSD', 'SOLUSD', 'AVAXUSD', 'ADAUSD',
            'DOTUSD', 'MATICUSD', 'LINKUSD', 'UNIUSD', 'AAVEUSD'
        ]
    
    async def shutdown(self):
        """Cierre ordenado del bot"""
        logger.info("🛑 Iniciando cierre ordenado...")
        self.running = False
        
        try:
            # Guardar estado final
            await self.save_state()
            
            # Cerrar conexiones
            if self.data_manager:
                await self.data_manager.close()
            
            logger.info("✅ Bot cerrado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error en cierre: {e}")

async def main():
    """Función principal"""
    bot = PapaDineroBot()
    
    # Manejar señales de cierre
    def signal_handler(signum, frame):
        logger.info(f"🛑 Señal {signum} recibida, cerrando bot...")
        asyncio.create_task(bot.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Inicializar y ejecutar bot
    if await bot.initialize():
        try:
            await bot.run_main_loop()
        finally:
            await bot.shutdown()
    else:
        logger.error("❌ No se pudo inicializar el bot")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🛑 Interrupción manual")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}")
        traceback.print_exc()
        sys.exit(1)
