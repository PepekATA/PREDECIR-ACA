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
