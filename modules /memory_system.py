"""
Analizador de Mercado
Analiza patrones, tendencias y condiciones de mercado
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger("MarketAnalyzer")

class MarketAnalyzer:
    """Analiza condiciones y patrones de mercado"""
    
    def __init__(self, data_manager, ai_predictor):
        self.data_manager = data_manager
        self.ai_predictor = ai_predictor
        self.analysis_cache = {}
        
    def calculate_indicators(self, price_data: pd.DataFrame) -> Dict:
        """Calcular indicadores técnicos completos"""
        try:
            if price_data is None or price_data.empty:
                return self.get_default_indicators()
            
            # Obtener datos base
            if 'data' in price_data:
                df = price_data['data']
            else:
                df = price_data
            
            if df.empty or 'close' not in df.columns:
                return self.get_default_indicators()
            
            close_prices = df['close']
            
            indicators = {
                # Indicadores básicos
                'rsi': self.safe_get_last(df, 'rsi', 50),
                'macd': self.safe_get_last(df, 'macd', 0),
                'macd_signal': self.safe_get_last(df, 'macd_signal', 0),
                'bollinger_pos': self.safe_get_last(df, 'bb_position', 0.5),
                
                # Medias móviles
                'sma_10': self.safe_get_last(df, 'sma_10', close_prices.iloc[-1]),
                'sma_20': self.safe_get_last(df, 'sma_20', close_prices.iloc[-1]),
                'ema_12': self.safe_get_last(df, 'ema_12', close_prices.iloc[-1]),
                'ema_26': self.safe_get_last(df, 'ema_26', close_prices.iloc[-1]),
                
                # Volumen
                'volume': self.safe_get_last(df, 'volume', 1000),
                'volume_ratio': self.safe_get_last(df, 'volume_ratio', 1.0),
                
                # Cambios de precio
                'price_change_1h': self.calculate_price_change(close_prices, 60),
                'price_change_4h': self.calculate_price_change(close_prices, 240),
                'price_change_24h': self.calculate_price_change(close_prices, 1440),
                
                # Volatilidad y momentum
                'volatility': close_prices.rolling(20).std().iloc[-1] if len(close_prices) >= 20 else 0,
                'momentum': self.safe_get_last(df, 'momentum', 1.0),
                'rate_of_change': self.safe_get_last(df, 'rate_of_change', 0),
                
                # Análisis de tendencia
                'trend_strength': self.calculate_trend_strength(close_prices),
                'trend_direction': self.calculate_trend_direction(close_prices),
                'support_level': self.calculate_support_resistance(close_prices)[0],
                'resistance_level': self.calculate_support_resistance(close_prices)[1]
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Error calculando indicadores: {e}")
            return self.get_default_indicators()
    
    def safe_get_last(self, df: pd.DataFrame, column: str, default_value):
        """Obtener último valor de columna de forma segura"""
        try:
            if column in df.columns and not df[column].empty:
                last_val = df[column].iloc[-1]
                if pd.notna(last_val):
                    return float(last_val)
            return default_value
        except Exception:
            return default_value
    
    def get_default_indicators(self) -> Dict:
        """Indicadores por defecto cuando no hay datos"""
        return {
            'rsi': 50, 'macd': 0, 'macd_signal': 0, 'bollinger_pos': 0.5,
            'sma_10': 0, 'sma_20': 0, 'ema_12': 0, 'ema_26': 0,
            'volume': 1000, 'volume_ratio': 1.0,
            'price_change_1h': 0, 'price_change_4h': 0, 'price_change_24h': 0,
            'volatility': 0, 'momentum': 1.0, 'rate_of_change': 0,
            'trend_strength': 0, 'trend_direction': 'neutral',
            'support_level': 0, 'resistance_level': 0
        }
    
    def calculate_price_change(self, prices: pd.Series, periods: int) -> float:
        """Calcular cambio de precio en períodos específicos"""
        try:
            if len(prices) > periods:
                return ((prices.iloc[-1] - prices.iloc[-(periods+1)]) / prices.iloc[-(periods+1)]) * 100
            return 0.0
        except Exception:
            return 0.0
    
    def calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calcular fuerza de la tendencia (0-100)"""
        try:
            if len(prices) < 20:
                return 0
            
            # Usar ADX simplificado
            high = prices.rolling(window=1).max()  # Simplificado para close prices
            low = prices.rolling(window=1).min()
            
            # Direccional movement
            plus_dm = high.diff()
            minus_dm = -low.diff()
            
            plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
            minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
            
            # True Range simplificado
            tr = prices.diff().abs()
            
            # Smoothed
            plus_di = 100 * (plus_dm.rolling(14).sum() / tr.rolling(14).sum())
            minus_di = 100 * (minus_dm.rolling(14).sum() / tr.rolling(14).sum())
            
            # ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(14).mean()
            
            return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0
            
        except Exception:
            return 0
    
    def calculate_trend_direction(self, prices: pd.Series) -> str:
        """Determinar dirección de tendencia"""
        try:
            if len(prices) < 20:
                return 'neutral'
            
            # Comparar medias móviles
            short_ma = prices.rolling(10).mean().iloc[-1]
            long_ma = prices.rolling(20).mean().iloc[-1]
            current_price = prices.iloc[-1]
            
            if current_price > short_ma > long_ma:
                return 'bullish'
            elif current_price < short_ma < long_ma:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception:
            return 'neutral'
    
    def calculate_support_resistance(self, prices: pd.Series) -> Tuple[float, float]:
        """Calcular niveles de soporte y resistencia"""
        try:
            if len(prices) < 50:
                current = prices.iloc[-1] if len(prices) > 0 else 0
                return current * 0.95, current * 1.05  # 5% arriba y abajo
            
            # Usar máximos y mínimos locales
            recent_prices = prices.tail(50)
            
            # Soporte: mínimos recientes
            support = recent_prices.quantile(0.1)  # 10% percentil
            
            # Resistencia: máximos recientes  
            resistance = recent_prices.quantile(0.9)  # 90% percentil
            
            return float(support), float(resistance)
            
        except Exception:
            current = prices.iloc[-1] if len(prices) > 0 else 0
            return current * 0.95, current * 1.05
    
    def analyze_market_conditions(self, symbol: str) -> Dict:
        """Analizar condiciones generales del mercado"""
        try:
            # Obtener datos del símbolo
            market_data = self.data_manager.get_symbol_data(symbol)
            if not market_data:
                return {'error': 'No market data available'}
            
            indicators = self.calculate_indicators(market_data)
            
            # Análisis de condiciones
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'market_phase': self.determine_market_phase(indicators),
                'volatility_regime': self.determine_volatility_regime(indicators),
                'momentum_status': self.analyze_momentum(indicators),
                'volume_analysis': self.analyze_volume(indicators),
                'technical_score': self.calculate_technical_score(indicators),
                'risk_level': self.assess_risk_level(indicators),
                'trading_opportunities': self.identify_opportunities(indicators),
                'key_levels': {
                    'support': indicators['support_level'],
                    'resistance': indicators['resistance_level'],
                    'current_price': market_data['latest']['price']
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analizando condiciones {symbol}: {e}")
            return {'error': str(e)}
    
    def determine_market_phase(self, indicators: Dict) -> str:
        """Determinar fase del mercado"""
        try:
            rsi = indicators['rsi']
            trend_direction = indicators['trend_direction']
            volatility = indicators['volatility']
            
            if rsi > 70 and trend_direction == 'bullish':
                return 'overbought_trending'
            elif rsi < 30 and trend_direction == 'bearish':
                return 'oversold_trending'
            elif 45 <= rsi <= 55 and volatility < 2:
                return 'consolidation'
            elif trend_direction == 'bullish' and volatility > 3:
                return 'bull_market'
            elif trend_direction == 'bearish' and volatility > 3:
                return 'bear_market'
            else:
                return 'neutral'
                
        except Exception:
            return 'unknown'
    
    def determine_volatility_regime(self, indicators: Dict) -> str:
        """Determinar régimen de volatilidad"""
        try:
            volatility = indicators['volatility']
            
            if volatility > 5:
                return 'high'
            elif volatility > 2:
                return 'medium'
            else:
                return 'low'
                
        except Exception:
            return 'unknown'
    
    def analyze_momentum(self, indicators: Dict) -> Dict:
        """Analizar momentum del mercado"""
        try:
            momentum = indicators['momentum']
            roc = indicators['rate_of_change']
            macd = indicators['macd']
            macd_signal = indicators['macd_signal']
            
            analysis = {
                'strength': 'neutral',
                'direction': 'neutral',
                'divergence': False
            }
            
            # Fuerza del momentum
            if abs(roc) > 5:
                analysis['strength'] = 'strong'
            elif abs(roc) > 2:
                analysis['strength'] = 'moderate'
            else:
                analysis['strength'] = 'weak'
            
            # Dirección
            if momentum > 1.02 and roc > 1:
                analysis['direction'] = 'bullish'
            elif momentum < 0.98 and roc < -1:
                analysis['direction'] = 'bearish'
            
            # Divergencia MACD
            if macd > macd_signal:
                analysis['macd_signal'] = 'bullish'
            else:
                analysis['macd_signal'] = 'bearish'
            
            return analysis
            
        except Exception:
            return {'strength': 'unknown', 'direction': 'neutral', 'divergence': False}
    
    def analyze_volume(self, indicators: Dict) -> Dict:
        """Analizar patrones de volumen"""
        try:
            volume_ratio = indicators['volume_ratio']
            
            analysis = {
                'trend': 'normal',
                'strength': 'average',
                'confirmation': 'neutral'
            }
            
            if volume_ratio > 2:
                analysis['trend'] = 'increasing'
                analysis['strength'] = 'high'
                analysis['confirmation'] = 'strong'
            elif volume_ratio > 1.5:
                analysis['trend'] = 'increasing'
                analysis['strength'] = 'above_average'
                analysis['confirmation'] = 'moderate'
            elif volume_ratio < 0.5:
                analysis['trend'] = 'decreasing'
                analysis['strength'] = 'low'
                analysis['confirmation'] = 'weak'
            
            return analysis
            
        except Exception:
            return {'trend': 'unknown', 'strength': 'average', 'confirmation': 'neutral'}
    
    def calculate_technical_score(self, indicators: Dict) -> int:
        """Calcular score técnico (0-100)"""
        try:
            score = 50  # Base neutral
            
            # RSI
            rsi = indicators['rsi']
            if 40 <= rsi <= 60:
                score += 10  # Zona neutral es positiva
            elif 30 <= rsi <= 70:
                score += 5   # Zona aceptable
            else:
                score -= 10  # Zonas extremas
            
            # MACD
            if indicators['macd'] > indicators['macd_signal']:
                score += 10
            else:
                score -= 5
            
            # Bollinger Position
            bb_pos = indicators['bollinger_pos']
            if 0.2 <= bb_pos <= 0.8:
                score += 10  # Zona intermedia es buena
            else:
                score -= 5   # Zonas extremas de Bollinger
            
            # Tendencia
            if indicators['trend_direction'] == 'bullish':
                score += 15
            elif indicators['trend_direction'] == 'bearish':
                score -= 10
            
            # Volatilidad (baja volatilidad es mejor para nuestro estilo)
            if indicators['volatility'] < 2:
                score += 10
            elif indicators['volatility'] > 5:
                score -= 15
            
            return max(0, min(100, score))
            
        except Exception:
            return 50
    
    def assess_risk_level(self, indicators: Dict) -> str:
        """Evaluar nivel de riesgo"""
        try:
            risk_factors = 0
            
            # Volatilidad alta
            if indicators['volatility'] > 5:
                risk_factors += 2
            elif indicators['volatility'] > 3:
                risk_factors += 1
            
            # RSI extremo
            if indicators['rsi'] > 80 or indicators['rsi'] < 20:
                risk_factors += 2
            elif indicators['rsi'] > 70 or indicators['rsi'] < 30:
                risk_factors += 1
            
            # Bollinger extremo
            bb_pos = indicators['bollinger_pos']
            if bb_pos > 0.95 or bb_pos < 0.05:
                risk_factors += 2
            elif bb_pos > 0.9 or bb_pos < 0.1:
                risk_factors += 1
            
            # Cambios de precio grandes
            if abs(indicators['price_change_24h']) > 10:
                risk_factors += 2
            elif abs(indicators['price_change_24h']) > 5:
                risk_factors += 1
            
            if risk_factors >= 5:
                return 'very_high'
            elif risk_factors >= 3:
                return 'high'
            elif risk_factors >= 1:
                return 'medium'
            else:
                return 'low'
                
        except Exception:
            return 'unknown'
    
    def identify_opportunities(self, indicators: Dict) -> List[Dict]:
        """Identificar oportunidades de trading"""
        opportunities = []
        
        try:
            # Oportunidad de compra por RSI oversold
            if indicators['rsi'] < 35 and indicators['trend_direction'] != 'bearish':
                opportunities.append({
                    'type': 'buy',
                    'reason': 'RSI Oversold Recovery',
                    'confidence': 0.7,
                    'timeframe': 'short_term'
                })
            
            # Oportunidad por cruce MACD
            if indicators['macd'] > indicators['macd_signal'] and indicators['macd'] > 0:
                opportunities.append({
                    'type': 'buy',
                    'reason': 'MACD Bullish Crossover',
                    'confidence': 0.75,
                    'timeframe': 'medium_term'
                })
            
            # Oportunidad por breakout
            bb_pos = indicators['bollinger_pos']
            if bb_pos > 0.8 and indicators['volume_ratio'] > 1.5:
                opportunities.append({
                    'type': 'buy',
                    'reason': 'Bollinger Breakout with Volume',
                    'confidence': 0.8,
                    'timeframe': 'short_term'
                })
            
            # Oportunidad de venta (solo con ganancia)
            if indicators['rsi'] > 75 and indicators['trend_direction'] == 'bullish':
                opportunities.append({
                    'type': 'sell',
                    'reason': 'Take Profit - Overbought',
                    'confidence': 0.65,
                    'timeframe': 'short_term',
                    'note': 'Solo vender si hay ganancia'
                })
            
            return opportunities
            
        except Exception as e:
            logger.error(f"❌ Error identificando oportunidades: {e}")
            return []
    
    def get_market_sentiment(self) -> Dict:
        """Obtener sentimiento general del mercado"""
        try:
            # Obtener resumen de mercado
            market_summary = self.data_manager.get_market_summary()
            
            if 'error' in market_summary:
                return {'sentiment': 'unknown', 'confidence': 0}
            
            overview = market_summary.get('market_overview', {})
            sentiment = overview.get('market_sentiment', 'NEUTRAL')
            
            # Calcular confianza basada en datos disponibles
            symbols_count = market_summary.get('symbols_tracked', 0)
            confidence = min(1.0, symbols_count / 10)  # Máximo confianza con 10+ símbolos
            
            return {
                'sentiment': sentiment.lower(),
                'confidence': confidence,
                'bullish_count': overview.get('bullish_count', 0),
                'bearish_count': overview.get('bearish_count', 0),
                'avg_change': overview.get('avg_change', 0),
                'data_quality': 'good' if symbols_count >= 5 else 'limited'
            }
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo sentimiento: {e}")
            return {'sentiment': 'unknown', 'confidence': 0}
