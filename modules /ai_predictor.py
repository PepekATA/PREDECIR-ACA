import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class AIPredictor:
    """Sistema de predicción con múltiples modelos de IA"""
    
    def __init__(self, memory_system):
        self.memory = memory_system
        self.models = {}
        self.scalers = {}
        self.predictions_cache = {}
        self.model_weights = {
            'lstm': 0.4,
            'random_forest': 0.25,
            'gradient_boost': 0.20,
            'linear': 0.15
        }
        
        self.initialize_models()
    
    def initialize_models(self):
        """Inicializar todos los modelos de predicción"""
        # Random Forest para patrones complejos
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting para tendencias
        self.models['gradient_boost'] = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Linear Regression para tendencias lineales
        self.models['linear'] = LinearRegression()
        
        # Inicializar escaladores
        for model_name in self.models.keys():
            self.scalers[model_name] = MinMaxScaler(feature_range=(0, 1))
    
    def create_lstm_model(self, input_shape):
        """Crear modelo LSTM para series temporales"""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(50, return_sequences=False),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(25),
                tf.keras.layers.Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            return model
        except Exception as e:
            print(f"Error creating LSTM: {e}")
            return None
    
    def prepare_features(self, price_data, indicators):
        """Preparar características para predicción"""
        features = []
        
        if len(price_data) >= 20:
            # Características de precio
            price_array = np.array(price_data['close'])
            
            # Medias móviles
            ma5 = np.mean(price_array[-5:]) if len(price_array) >= 5 else price_array[-1]
            ma10 = np.mean(price_array[-10:]) if len(price_array) >= 10 else price_array[-1]
            ma20 = np.mean(price_array[-20:]) if len(price_array) >= 20 else price_array[-1]
            
            # Volatilidad
            volatility = np.std(price_array[-20:]) if len(price_array) >= 20 else 0
            
            # Momentum
            momentum = (price_array[-1] - price_array[-10]) / price_array[-10] if len(price_array) >= 10 else 0
            
            # Rate of Change
            roc = (price_array[-1] - price_array[-5]) / price_array[-5] if len(price_array) >= 5 else 0
            
            features = [
                price_array[-1],  # Precio actual
                ma5, ma10, ma20,  # Medias móviles
                volatility,       # Volatilidad
                momentum,         # Momentum
                roc,             # Rate of Change
                indicators.get('rsi', 50),
                indicators.get('macd', 0),
                indicators.get('bollinger_pos', 0.5),
                indicators.get('volume', 1000),
                indicators.get('price_change_1h', 0),
                indicators.get('price_change_4h', 0),
                indicators.get('price_change_24h', 0)
            ]
        else:
            # Características básicas si no hay suficientes datos
            features = [
                price_data['close'][-1] if len(price_data) > 0 else 0,
                indicators.get('rsi', 50),
                indicators.get('macd', 0),
                indicators.get('bollinger_pos', 0.5),
                indicators.get('volume', 1000),
                0, 0, 0, 0, 0, 0, 0  # Relleno con ceros
            ]
        
        return np.array(features).reshape(1, -1)
    
    def train_ensemble_models(self, historical_data):
        """Entrenar el conjunto de modelos"""
        if len(historical_data) < 50:
            return False
        
        try:
            # Preparar datos de entrenamiento
            X_train = []
            y_train = []
            
            for i in range(20, len(historical_data) - 5):  # Ventana de 20, predecir 5 períodos adelante
                features = self.extract_features_from_history(historical_data, i)
                target = historical_data.iloc[i + 5]['close']  # Precio 5 períodos adelante
                
                X_train.append(features)
                y_train.append(target)
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            if len(X_train) < 20:
                return False
            
            # Entrenar cada modelo
            for model_name, model in self.models.items():
                if model_name != 'lstm':  # LSTM se entrena por separado
                    try:
                        # Escalar datos
                        X_scaled = self.scalers[model_name].fit_transform(X_train)
                        
                        # Entrenar modelo
                        model.fit(X_scaled, y_train)
                        
                        print(f"✅ Modelo {model_name} entrenado exitosamente")
                    except Exception as e:
                        print(f"❌ Error entrenando {model_name}: {e}")
            
            return True
            
        except Exception as e:
            print(f"Error en entrenamiento: {e}")
            return False
    
    def extract_features_from_history(self, data, index):
        """Extraer características de datos históricos"""
        if index < 20:
            return [0] * 13  # 13 características por defecto
        
        price_slice = data.iloc[index-20:index]['close'].values
        
        # Calcular indicadores técnicos básicos
        rsi = self.calculate_rsi_simple(price_slice)
        ma5 = np.mean(price_slice[-5:])
        ma10 = np.mean(price_slice[-10:])
        ma20 = np.mean(price_slice[-20:])
        volatility = np.std(price_slice)
        momentum = (price_slice[-1] - price_slice[-10]) / price_slice[-10]
        
        return [
            price_slice[-1], ma5, ma10, ma20, volatility, momentum,
            rsi, 0, 0.5,  # RSI, MACD placeholder, Bollinger placeholder
            data.iloc[index]['volume'] if 'volume' in data.columns else 1000,
            0, 0, 0  # Price changes placeholder
        ]
    
    def calculate_rsi_simple(self, prices, period=14):
        """Calcular RSI simple"""
        if len(prices) < period + 1:
            return 50  # Valor neutro
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def predict_price_movement(self, symbol, price_data, indicators, horizon_minutes=30):
        """Predecir movimiento de precio usando ensemble de modelos"""
        try:
            features = self.prepare_features(price_data, indicators)
            
            predictions = {}
            ensemble_prediction = 0
            total_weight = 0
            
            # Obtener predicción de cada modelo
            for model_name, model in self.models.items():
                if model_name == 'lstm':
                    continue  # LSTM requiere procesamiento especial
                
                try:
                    # Escalar características
                    features_scaled = self.scalers[model_name].transform(features)
                    
                    # Hacer predicción
                    prediction = model.predict(features_scaled)[0]
                    predictions[model_name] = prediction
                    
                    # Agregar al ensemble
                    weight = self.model_weights[model_name]
                    ensemble_prediction += prediction * weight
                    total_weight += weight
                    
                except Exception as e:
                    print(f"Error en predicción {model_name}: {e}")
                    continue
            
            if total_weight > 0:
                ensemble_prediction /= total_weight
                current_price = price_data['close'][-1] if len(price_data) > 0 else 0
                
                if current_price > 0:
                    predicted_change = ((ensemble_prediction - current_price) / current_price) * 100
                    
                    # Calcular confianza basada en consistencia de modelos
                    if len(predictions) > 1:
                        pred_values = list(predictions.values())
                        consistency = 1 - (np.std(pred_values) / np.mean(pred_values)) if np.mean(pred_values) != 0 else 0
                        confidence = min(0.95, max(0.5, consistency))
                    else:
                        confidence = 0.6
                    
                    # Determinar señal de trading
                    if predicted_change > 2.0 and confidence > 0.7:
                        signal = 'strong_buy'
                    elif predicted_change > 0.5:
                        signal = 'buy'
                    elif predicted_change < -2.0 and confidence > 0.7:
                        signal = 'strong_sell'
                    elif predicted_change < -0.5:
                        signal = 'sell'
                    else:
                        signal = 'hold'
                    
                    result = {
                        'symbol': symbol,
                        'current_price': current_price,
                        'predicted_price': ensemble_prediction,
                        'predicted_change': predicted_change,
                        'signal': signal,
                        'confidence': confidence,
                        'horizon_minutes': horizon_minutes,
                        'timestamp': datetime.now().isoformat(),
                        'model_predictions': predictions,
                        'ensemble_used': True
                    }
                    
                    # Cachear predicción
                    cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                    self.predictions_cache[cache_key] = result
                    
                    return result
            
            # Predicción por defecto
            return {
                'symbol': symbol,
                'signal': 'hold',
                'confidence': 0.5,
                'predicted_change': 0,
                'reason': 'insufficient_model_data'
            }
            
        except Exception as e:
            print(f"Error en predicción: {e}")
            return {
                'symbol': symbol,
                'signal': 'hold',
                'confidence': 0.5,
                'error': str(e)
            }
    
    def get_market_sentiment(self, symbol, predictions_history):
        """Analizar sentimiento del mercado"""
        if not predictions_history:
            return {'sentiment': 'neutral', 'strength': 0.5}
        
        recent_predictions = predictions_history[-10:]  # Últimas 10 predicciones
        
        bullish_count = sum(1 for p in recent_predictions if p.get('predicted_change', 0) > 0)
        bearish_count = sum(1 for p in recent_predictions if p.get('predicted_change', 0) < 0)
        
        total_change = sum(p.get('predicted_change', 0) for p in recent_predictions)
        avg_confidence = np.mean([p.get('confidence', 0.5) for p in recent_predictions])
        
        if bullish_count > bearish_count * 1.5:
            sentiment = 'bullish'
            strength = min(0.95, (bullish_count / len(recent_predictions)) * avg_confidence)
        elif bearish_count > bullish_count * 1.5:
            sentiment = 'bearish'
            strength = min(0.95, (bearish_count / len(recent_predictions)) * avg_confidence)
        else:
            sentiment = 'neutral'
            strength = 0.5
        
        return {
            'sentiment': sentiment,
            'strength': strength,
            'avg_predicted_change': total_change / len(recent_predictions),
            'bullish_ratio': bullish_count / len(recent_predictions),
            'avg_confidence': avg_confidence
        }
    
    def predict_trend_duration(self, symbol, current_trend, price_data):
        """Predecir duración de tendencia actual"""
        try:
            if len(price_data) < 20:
                return {'duration_minutes': 30, 'confidence': 0.5}
            
            prices = np.array(price_data['close'])
            
            # Análisis de volatilidad
            volatility = np.std(prices[-20:])
            avg_price = np.mean(prices[-20:])
            volatility_ratio = volatility / avg_price if avg_price > 0 else 0
            
            # Análisis de momentum
            short_ma = np.mean(prices[-5:])
            long_ma = np.mean(prices[-20:])
            momentum_strength = abs(short_ma - long_ma) / long_ma if long_ma > 0 else 0
            
            # Estimación de duración basada en patrones históricos
            if momentum_strength > 0.05:  # Tendencia fuerte
                base_duration = 45
            elif momentum_strength > 0.02:  # Tendencia moderada
                base_duration = 30
            else:  # Tendencia débil
                base_duration = 15
            
            # Ajuste por volatilidad
            volatility_factor = 1 + (volatility_ratio * 10)
            estimated_duration = int(base_duration * volatility_factor)
            
            # Limitar duración
            estimated_duration = max(5, min(120, estimated_duration))
            
            # Confianza basada en datos disponibles
            confidence = min(0.9, len(price_data) / 100)
            
            return {
                'duration_minutes': estimated_duration,
                'confidence': confidence,
                'momentum_strength': momentum_strength,
                'volatility_factor': volatility_factor
            }
            
        except Exception as e:
            return {
                'duration_minutes': 30,
                'confidence': 0.5,
                'error': str(e)
            }
    
    def get_ai_insights(self, symbol, predictions_history):
        """Obtener insights avanzados de IA"""
        insights = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(predictions_history),
            'insights': []
        }
        
        if len(predictions_history) >= 5:
            recent = predictions_history[-5:]
            
            # Análisis de precisión
            if all('actual_outcome' in p for p in recent[-3:]):
                correct = sum(1 for p in recent[-3:] if p.get('prediction_correct', False))
                accuracy = correct / 3
                insights['insights'].append(f"Precisión reciente: {accuracy:.1%}")
            
            # Análisis de tendencias
            changes = [p.get('predicted_change', 0) for p in recent]
            if all(c > 0 for c in changes):
                insights['insights'].append("🔥 Tendencia alcista consistente detectada")
            elif all(c < 0 for c in changes):
                insights['insights'].append("📉 Tendencia bajista consistente detectada")
            
            # Análisis de confianza
            avg_confidence = np.mean([p.get('confidence', 0.5) for p in recent])
            if avg_confidence > 0.8:
                insights['insights'].append(f"🎯 Alta confianza del modelo: {avg_confidence:.1%}")
            
            # Análisis de volatilidad
            volatility = np.std(changes)
            if volatility > 5:
                insights['insights'].append("⚡ Alta volatilidad detectada - Oportunidades de trading")
        
        return insights
