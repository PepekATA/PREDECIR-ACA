"""
Gestor de Datos de Mercado
Obtiene y gestiona datos de precios, indicadores y market data
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("DataManager")

class DataManager:
    """Gestiona datos de mercado en tiempo real y históricos"""
    
    def __init__(self, credentials, memory_system):
        self.credentials = credentials
        self.memory_system = memory_system
        self.api = None
        self.market_data = {}
        self.last_update = {}
        
        # Cache de datos
        self.price_cache = {}
        self.indicators_cache = {}
        
    async def initialize_api(self, credentials):
        """Inicializar API de Alpaca"""
        try:
            if not credentials:
                return False
            
            self.api = tradeapi.REST(
                credentials['api_key'],
                credentials['api_secret'],
                credentials['base_url'],
                api_version='v2'
            )
            
            # Verificar conexión
            account = self.api.get_account()
            logger.info(f"✅ API Alpaca conectada - Portfolio: ${account.portfolio_value}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando API: {e}")
            return False
    
    async def update_all_symbols(self, symbols: List[str]):
        """Actualizar datos para todos los símbolos"""
        try:
            tasks = []
            for symbol in symbols:
                task = asyncio.create_task(self.update_symbol_data(symbol))
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.debug(f"📊 Datos actualizados para {len(symbols)} símbolos")
            
        except Exception as e:
            logger.error(f"❌ Error actualizando símbolos: {e}")
    
    async def update_symbol_data(self, symbol: str):
        """Actualizar datos de un símbolo específico"""
        try:
            # Obtener datos históricos
            end_time = datetime.now()
            start_time = end_time - timedelta(days=1)  # Último día
            
            # Usar asyncio.to_thread para operaciones síncronas de Alpaca
            bars = await asyncio.to_thread(
                self.api.get_crypto_bars,
                symbol,
                '1Min',
                start=start_time.isoformat(),
                end=end_time.isoformat()
            )
            
            if not bars.df.empty:
                df = bars.df.reset_index()
                
                # Procesar datos
                processed_data = self.process_market_data(df, symbol)
                
                # Actualizar cache
                self.market_data[symbol] = processed_data
                self.last_update[symbol] = datetime.now()
                
                # Guardar en memoria para AI
                await self.save_to_memory(symbol, processed_data)
                
                logger.debug(f"📈 {symbol}: {len(df)} barras actualizadas")
            
        except Exception as e:
            logger.error(f"❌ Error actualizando {symbol}: {e}")
    
    def process_market_data(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Procesar datos brutos de mercado"""
        try:
            # Calcular indicadores técnicos
            df = self.calculate_technical_indicators(df)
            
            # Preparar estructura de datos
            processed_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'data': df,
                'latest': {
                    'price': df['close'].iloc[-1],
                    'volume': df['volume'].iloc[-1],
                    'change_1h': self.calculate_change(df['close'], 60),
                    'change_4h': self.calculate_change(df['close'], 240),
                    'change_24h': self.calculate_change(df['close'], 1440),
                    'volatility': df['close'].rolling(20).std().iloc[-1]
                },
                'indicators': {
                    'rsi': df['rsi'].iloc[-1] if 'rsi' in df else 50,
                    'macd': df['macd'].iloc[-1] if 'macd' in df else 0,
                    'macd_signal': df['macd_signal'].iloc[-1] if 'macd_signal' in df else 0,
                    'bb_upper': df['bb_upper'].iloc[-1] if 'bb_upper' in df else 0,
                    'bb_lower': df['bb_lower'].iloc[-1] if 'bb_lower' in df else 0,
                    'bb_middle': df['bb_middle'].iloc[-1] if 'bb_middle' in df else 0,
                    'sma_20': df['sma_20'].iloc[-1] if 'sma_20' in df else 0,
                    'ema_12': df['ema_12'].iloc[-1] if 'ema_12' in df else 0,
                    'ema_26': df['ema_26'].iloc[-1] if 'ema_26' in df else 0
                }
            }
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Error procesando datos {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular indicadores técnicos"""
        try:
            df = df.copy()
            
            # RSI (Relative Strength Index)
            df['rsi'] = self.calculate_rsi(df['close'])
            
            # MACD
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['sma_20'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['sma_20'] - (df['bb_std'] * 2)
            df['bb_middle'] = df['sma_20']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Medias móviles adicionales
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Momentum
            df['momentum'] = df['close'] / df['close'].shift(10)
            df['rate_of_change'] = df['close'].pct_change(10) * 100
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculando indicadores: {e}")
            return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcular RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)
            
        except Exception:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def calculate_change(self, prices: pd.Series, periods: int) -> float:
        """Calcular cambio porcentual"""
        try:
            if len(prices) > periods:
                return ((prices.iloc[-1] - prices.iloc[-(periods+1)]) / prices.iloc[-(periods+1)]) * 100
            return 0.0
        except Exception:
            return 0.0
    
    async def save_to_memory(self, symbol: str, processed_data: Dict):
        """Guardar datos en el sistema de memoria para AI"""
        try:
            # Preparar datos para memoria
            memory_entry = {
                'symbol': symbol,
                'time': processed_data['timestamp'],
                'close': processed_data['latest']['price'],
                'volume': processed_data['latest']['volume'],
                'rsi': processed_data['indicators']['rsi'],
                'macd': processed_data['indicators']['macd'],
                'bollinger_pos': processed_data['indicators'].get('bb_position', 0.5),
                'volatility': processed_data['latest']['volatility'],
                'change_1h': processed_data['latest']['change_1h'],
                'change_4h': processed_data['latest']['change_4h'],
                'change_24h': processed_data['latest']['change_24h']
            }
            
            # Guardar en memoria del sistema
            if hasattr(self.memory_system, 'add_market_data'):
                await asyncio.to_thread(
                    self.memory_system.add_market_data, 
                    symbol, 
                    memory_entry
                )
            
        except Exception as e:
            logger.error(f"❌ Error guardando en memoria {symbol}: {e}")
    
    def get_symbol_data(self, symbol: str) -> Optional[Dict]:
        """Obtener datos de un símbolo"""
        return self.market_data.get(symbol)
    
    def get_current_price(self, symbol: str) -> float:
        """Obtener precio actual"""
        data = self.get_symbol_data(symbol)
        if data and 'latest' in data:
            return data['latest']['price']
        return 0.0
    
    def get_current_prices(self) -> Dict[str, float]:
        """Obtener precios actuales de todos los símbolos"""
        prices = {}
        for symbol, data in self.market_data.items():
            if data and 'latest' in data:
                prices[symbol] = data['latest']['price']
        return prices
    
    def get_market_summary(self) -> Dict:
        """Obtener resumen del mercado"""
        try:
            if not self.market_data:
                return {'error': 'No market data available'}
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'symbols_tracked': len(self.market_data),
                'market_overview': {},
                'top_gainers': [],
                'top_losers': [],
                'high_volume': []
            }
            
            # Analizar cada símbolo
            symbol_analysis = []
            for symbol, data in self.market_data.items():
                if data and 'latest' in data:
                    analysis = {
                        'symbol': symbol,
                        'price': data['latest']['price'],
                        'change_24h': data['latest']['change_24h'],
                        'volume': data['latest']['volume'],
                        'volatility': data['latest']['volatility'],
                        'rsi': data['indicators']['rsi']
                    }
                    symbol_analysis.append(analysis)
            
            # Ordenar por cambio 24h
            symbol_analysis.sort(key=lambda x: x['change_24h'], reverse=True)
            
            # Top gainers y losers
            summary['top_gainers'] = symbol_analysis[:3]
            summary['top_losers'] = symbol_analysis[-3:]
            
            # Alto volumen
            symbol_analysis.sort(key=lambda x: x['volume'], reverse=True)
            summary['high_volume'] = symbol_analysis[:3]
            
            # Overview general
            changes = [s['change_24h'] for s in symbol_analysis if s['change_24h'] != 0]
            if changes:
                summary['market_overview'] = {
                    'avg_change': np.mean(changes),
                    'bullish_count': len([c for c in changes if c > 0]),
                    'bearish_count': len([c for c in changes if c < 0]),
                    'market_sentiment': 'BULLISH' if np.mean(changes) > 1 else 'BEARISH' if np.mean(changes) < -1 else 'NEUTRAL'
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error generando resumen de mercado: {e}")
            return {'error': str(e)}
    
    async def get_historical_data(self, symbol: str, timeframe: str = '1Hour', limit: int = 100) -> Optional[pd.DataFrame]:
        """Obtener datos históricos"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=7)  # Última semana
            
            bars = await asyncio.to_thread(
                self.api.get_crypto_bars,
                symbol,
                timeframe,
                start=start_time.isoformat(),
                end=end_time.isoformat(),
                limit=limit
            )
            
            if not bars.df.empty:
                df = bars.df.reset_index()
                return self.calculate_technical_indicators(df)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos históricos {symbol}: {e}")
            return None
    
    def is_market_open(self) -> bool:
        """Verificar si el mercado crypto está abierto (siempre True para crypto)"""
        return True  # Crypto opera 24/7
    
    def get_data_freshness(self, symbol: str) -> int:
        """Obtener antiguedad de los datos en segundos"""
        if symbol in self.last_update:
            return int((datetime.now() - self.last_update[symbol]).total_seconds())
        return 999999  # Muy antiguo si no hay datos
    
    async def close(self):
        """Cerrar conexiones y limpiar recursos"""
        try:
            # Limpiar cache
            self.market_data.clear()
            self.price_cache.clear()
            self.indicators_cache.clear()
            
            logger.info("✅ DataManager cerrado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error cerrando DataManager: {e}")
