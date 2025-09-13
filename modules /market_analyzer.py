# modules/market_analyzer.py — ANÁLISIS AVANZADO (completo)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class MarketAnalyzer:
    """Analizador avanzado de mercado con IA"""
    
    def __init__(self, memory_system=None):
        self.memory = memory_system
        self.indicators_cache = {}
        self.market_regime = "neutral"
        self.volatility_clusters = None
        self.volatility_scaler = None
        
    def analyze_market_structure(self, price_data):
        """Analizar estructura del mercado"""
        try:
            if len(price_data) < 50:
                return {'error': 'Insufficient data'}
            
            df = price_data.copy().reset_index(drop=True)
            if 'close' not in df:
                return {'error': 'price_data must contain a "close" column'}
            
            prices = df['close'].values
            volumes = df['volume'].values if 'volume' in df else np.ones(len(prices))
            
            # 1. Análisis de tendencia multi-timeframe
            trend_analysis = self.multi_timeframe_trend(prices)
            
            # 2. Análisis de volatilidad
            volatility_analysis = self.volatility_regime_detection(prices, df)
            
            # 3. Análisis de momentum
            momentum_analysis = self.momentum_analysis(df)
            
            # 4. Detección de patrones
            pattern_analysis = self.pattern_detection(df)
            
            # 5. Análisis de estructura de mercado
            market_structure = self.market_structure_analysis(df)
            
            overall = self.synthesize_signals(trend_analysis, momentum_analysis, volatility_analysis, market_structure)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'trend': trend_analysis,
                'volatility': volatility_analysis,
                'momentum': momentum_analysis,
                'patterns': pattern_analysis,
                'market_structure': market_structure,
                'overall_signal': overall
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def multi_timeframe_trend(self, prices):
        """Análisis de tendencia en múltiples marcos temporales"""
        try:
            trends = {}
            
            # Diferentes períodos de análisis
            periods = {'short': 10, 'medium': 20, 'long': 50}
            
            for period_name, period_length in periods.items():
                if len(prices) > period_length:
                    recent_prices = prices[-period_length:]
                    
                    # Regresión lineal simple para tendencia
                    x = np.arange(len(recent_prices))
                    coeffs = np.polyfit(x, recent_prices, 1)
                    slope = float(coeffs[0])
                    
                    # Determinar dirección y fuerza
                    price_change = float((recent_prices[-1] - recent_prices[0]) / recent_prices[0])
                    
                    if slope > 0 and price_change > 0.02:
                        direction = 'bullish'
                        strength = min(1.0, abs(price_change) * 10)
                    elif slope < 0 and price_change < -0.02:
                        direction = 'bearish' 
                        strength = min(1.0, abs(price_change) * 10)
                    else:
                        direction = 'neutral'
                        strength = 0.5
                    
                    trends[period_name] = {
                        'direction': direction,
                        'strength': float(np.round(strength, 3)),
                        'slope': float(np.round(slope, 8)),
                        'price_change': float(np.round(price_change, 6))
                    }
            
            # Consenso de tendencias
            bullish_count = sum(1 for t in trends.values() if t['direction'] == 'bullish')
            bearish_count = sum(1 for t in trends.values() if t['direction'] == 'bearish')
            
            if bullish_count > bearish_count:
                consensus = 'bullish'
                consensus_strength = bullish_count / len(trends)
            elif bearish_count > bullish_count:
                consensus = 'bearish'
                consensus_strength = bearish_count / len(trends)
            else:
                consensus = 'neutral'
                consensus_strength = 0.5
            
            return {'per_period': trends, 'consensus': consensus, 'consensus_strength': float(np.round(consensus_strength,3))}
        except Exception as e:
            return {'error': f"multi_timeframe_trend: {str(e)}"}
    
    def volatility_regime_detection(self, prices, df=None, window=20, n_clusters=3):
        """Detecta el régimen de volatilidad usando desviación estándar de retornos y clustering"""
        try:
            prices = np.asarray(prices)
            if len(prices) < window + 5:
                return {'error': 'Not enough data for volatility detection'}
            
            # calcular retornos log
            returns = np.diff(np.log(prices + 1e-12))
            rolling_vol = pd.Series(returns).rolling(window=window).std().dropna().values
            
            if len(rolling_vol) < n_clusters:
                # fallback: usar var simple
                current_vol = float(np.std(returns[-window:]))
                regime = 'unknown'
                return {'volatility': current_vol, 'regime': regime}
            
            # preparar features para clustering (usar últimos N valores)
            X = rolling_vol.reshape(-1,1)
            
            # ajustar scaler y cluster si no existen o si hay mucha diferencia en tamaño
            if (self.volatility_clusters is None) or (len(X) > getattr(self, "_last_vol_fit_size", 0) * 1.5):
                self.volatility_scaler = StandardScaler()
                Xs = self.volatility_scaler.fit_transform(X)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(Xs)
                self.volatility_clusters = kmeans
                self._last_vol_fit_size = len(X)
            
            Xs = self.volatility_scaler.transform(X)
            labels = self.volatility_clusters.predict(Xs)
            cluster_centers = self.volatility_clusters.cluster_centers_.flatten()
            
            # determinar etiqueta del cluster más alto (volatilidad alta = centro mayor)
            center_order = np.argsort(cluster_centers)  # ascending
            # map cluster index to regime label
            regime_map = {}
            regime_map[center_order[0]] = 'low'
            if len(center_order) == 3:
                regime_map[center_order[1]] = 'medium'
                regime_map[center_order[2]] = 'high'
            elif len(center_order) == 2:
                regime_map[center_order[1]] = 'high'
            
            current_label = labels[-1]
            current_regime = regime_map.get(current_label, 'unknown')
            current_vol = float(rolling_vol[-1])
            
            # guardar en memoria interna
            self.market_regime = current_regime
            
            return {
                'volatility': float(np.round(current_vol,8)),
                'regime': current_regime,
                'cluster_centers': [float(c) for c in np.sort(cluster_centers)],
                'label': int(current_label)
            }
        except Exception as e:
            return {'error': f"volatility_regime_detection: {str(e)}"}
    
    def momentum_analysis(self, df):
        """Calcula RSI, MACD, OBV y una puntuación simple de momentum"""
        try:
            data = df.copy()
            if 'close' not in data:
                return {'error': 'momentum_analysis requires close prices'}
            
            close = data['close']
            # RSI
            rsi = None
            try:
                rsi = ta.momentum.rsi(close, window=14)
                rsi_val = float(np.round(rsi.iloc[-1],3))
            except Exception:
                # fallback manual
                delta = close.diff()
                up = delta.clip(lower=0).rolling(14).mean()
                down = -delta.clip(upper=0).rolling(14).mean()
                rs = up / (down + 1e-12)
                rsi_val = float(np.round(100 - (100 / (1 + rs.iloc[-1])),3))
            
            # MACD
            try:
                macd = ta.trend.MACD(close)
                macd_val = float(np.round(macd.macd().iloc[-1],6))
                macd_signal = float(np.round(macd.macd_signal().iloc[-1],6))
            except Exception:
                macd_val = 0.0
                macd_signal = 0.0
            
            # OBV (requiere volume)
            try:
                vol = data['volume'] if 'volume' in data else pd.Series(np.ones(len(close)))
                obv = ta.volume.on_balance_volume(close, vol)
                obv_val = float(np.round(obv.iloc[-1],3))
            except Exception:
                obv_val = 0.0
            
            # Momentum score heurístico [-1..1]
            score = 0.0
            # RSI contribution
            if rsi_val > 70:
                score += 0.6
            elif rsi_val < 30:
                score -= 0.6
            else:
                score += (rsi_val - 50) / 50.0 * 0.2  # leve inclinación
            
            # MACD contribution
            if macd_val > macd_signal:
                score += 0.4
            else:
                score -= 0.4
            
            # normalizar score a [-1,1]
            score = max(-1.0, min(1.0, score))
            
            return {
                'rsi': rsi_val,
                'macd': macd_val,
                'macd_signal': macd_signal,
                'obv': obv_val,
                'momentum_score': float(np.round(score,3))
            }
        except Exception as e:
            return {'error': f"momentum_analysis: {str(e)}"}
    
    def pattern_detection(self, df):
        """
        Detección simple de patrones: doble techo/suelo (approx), máximos/mínimos locales y ruptura de soporte/resistencia.
        Nota: es una detección heurística y rápida.
        """
        try:
            data = df.copy()
            if 'close' not in data:
                return {'error': 'pattern_detection requires close prices'}
            close = data['close'].astype(float)
            n = len(close)
            patterns = []
            
            # detectar máximos y mínimos locales (ventana)
            window = 5
            peaks = []
            troughs = []
            for i in range(window, n-window):
                segment = close[i-window:i+window+1]
                if close[i] == segment.max():
                    peaks.append((i, float(close[i])))
                if close[i] == segment.min():
                    troughs.append((i, float(close[i])))
            
            # heurística doble techo (dos peaks cercanos con valle intermedio)
            if len(peaks) >= 2:
                # tomar últimos 3 peaks para evaluar
                for i in range(len(peaks)-1):
                    p1_idx, p1_val = peaks[i]
                    p2_idx, p2_val = peaks[i+1]
                    # condición: alturas similares y separación razonable
                    height_diff = abs(p1_val - p2_val) / max(p1_val, p2_val)
                    sep = p2_idx - p1_idx
                    if height_diff < 0.03 and 3 <= sep <= 50:
                        # comprobar valle entre ellos
                        valley = float(close[p1_idx+1:p2_idx].min()) if p2_idx - p1_idx > 1 else None
                        patterns.append({'type': 'double_top', 'peak1': p1_idx, 'peak2': p2_idx, 'valley': valley})
            
            # heurística doble suelo
            if len(troughs) >= 2:
                for i in range(len(troughs)-1):
                    t1_idx, t1_val = troughs[i]
                    t2_idx, t2_val = troughs[i+1]
                    height_diff = abs(t1_val - t2_val) / max(t1_val, t2_val)
                    sep = t2_idx - t1_idx
                    if height_diff < 0.03 and 3 <= sep <= 50:
                        peak_between = float(close[t1_idx+1:t2_idx].max()) if t2_idx - t1_idx > 1 else None
                        patterns.append({'type': 'double_bottom', 'trough1': t1_idx, 'trough2': t2_idx, 'peak_between': peak_between})
            
            # soporte/resistencia simple: niveles frecuentes (round)
            rounded = (close.round(2)).value_counts().nlargest(3)
            sr_levels = [float(x) for x in rounded.index.tolist()]
            patterns.append({'support_resistance_levels': sr_levels})
            
            return {'patterns': patterns, 'peaks_count': len(peaks), 'troughs_count': len(troughs)}
        except Exception as e:
            return {'error': f"pattern_detection: {str(e)}"}
    
    def market_structure_analysis(self, df):
        """
        Analiza estructura básica: HH/HL (uptrend), LH/LL (downtrend) o Rango.
        Usa últimos swings detectados.
        """
        try:
            if 'close' not in df:
                return {'error': 'market_structure_analysis requires close prices'}
            close = df['close'].astype(float).values
            n = len(close)
            if n < 20:
                return {'error': 'Not enough data for market structure analysis'}
            
            # calcular swings simples por diferencia de medias móviles
            short_ma = pd.Series(close).rolling(8).mean()
            long_ma = pd.Series(close).rolling(21).mean()
            ma_diff = (short_ma - long_ma).dropna()
            
            # contar estructura en últimas N observaciones
            last = ma_diff[-30:]
            up_count = int((last > 0).sum())
            down_count = int((last < 0).sum())
            
            if up_count > down_count * 1.2:
                structure = 'uptrend'
            elif down_count > up_count * 1.2:
                structure = 'downtrend'
            else:
                structure = 'range'
            
            # detect last swing highs/lows (simple)
            peaks_idx = (pd.Series(close).shift(1) < pd.Series(close)) & (pd.Series(close).shift(-1) < pd.Series(close))
            troughs_idx = (pd.Series(close).shift(1) > pd.Series(close)) & (pd.Series(close).shift(-1) > pd.Series(close))
            peaks = np.where(peaks_idx)[0].tolist()
            troughs = np.where(troughs_idx)[0].tolist()
            
            # último HH/HL/LH/LL heurístico
            last_three_extrema = []
            extrema = sorted([(i, 'peak') for i in peaks] + [(i, 'trough') for i in troughs], key=lambda x: x[0])
            for idx, kind in extrema[-6:]:
                last_three_extrema.append({'index': int(idx), 'type': kind, 'price': float(close[idx])})
            
            return {
                'structure': structure,
                'up_count': up_count,
                'down_count': down_count,
                'recent_extrema': last_three_extrema
            }
        except Exception as e:
            return {'error': f"market_structure_analysis: {str(e)}"}
    
    def synthesize_signals(self, trend_analysis, momentum_analysis, volatility_analysis, market_structure=None):
        """
        Combina las señales en una señal global: 'buy', 'sell' o 'neutral' con una confianza.
        Heurística que combina consenso de tendencia, momentum_score y régimen de volatilidad.
        """
        try:
            # manejo de errores o dicts con 'error'
            if isinstance(trend_analysis, dict) and 'error' in trend_analysis:
                return {'signal': 'neutral', 'confidence': 0.0}
            if isinstance(momentum_analysis, dict) and 'error' in momentum_analysis:
                return {'signal': 'neutral', 'confidence': 0.0}
            if isinstance(volatility_analysis, dict) and 'error' in volatility_analysis:
                vol_regime = 'unknown'
                vol_score = 0.5
            else:
                vol_regime = volatility_analysis.get('regime', 'unknown')
                # asignar penalización si la volatilidad es alta (se reduce confianza)
                vol_score = {'low': 1.0, 'medium': 0.9, 'high': 0.7}.get(vol_regime, 0.85)
            
            # tendencia
            consensus = trend_analysis.get('consensus', 'neutral')
            consensus_strength = float(trend_analysis.get('consensus_strength', 0.5))
            
            # momentum
            momentum_score = float(momentum_analysis.get('momentum_score', 0.0))
            
            # estructura
            struct = market_structure.get('structure') if market_structure and isinstance(market_structure, dict) else None
            
            # combinación heurística
            score = 0.0
            if consensus == 'bullish':
                score += 0.6 * consensus_strength
            elif consensus == 'bearish':
                score -= 0.6 * consensus_strength
            else:
                score += 0.0
            
            # añadir momentum
            score += 0.4 * momentum_score
            
            # ajustar por estructura
            if struct == 'uptrend':
                score += 0.1
            elif struct == 'downtrend':
                score -= 0.1
            
            # aplicar penalización/ajuste por volatilidad (reduce magnitude)
            score = score * vol_score
            
            # mapear a señal
            if score > 0.25:
                signal = 'buy'
            elif score < -0.25:
                signal = 'sell'
            else:
                signal = 'neutral'
            
            confidence = float(min(1.0, max(0.0, abs(score))))
            
            return {'signal': signal, 'confidence': float(np.round(confidence,3)), 'raw_score': float(np.round(score,4)), 'vol_regime': vol_regime}
        except Exception as e:
            return {'error': f"synthesize_signals: {str(e)}"}

# Fin del módulo
