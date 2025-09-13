import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class MemorySystem:
    """Sistema de memoria y aprendizaje del bot"""
    
    def __init__(self):
        self.memory_file = 'data/market_memory.json'
        self.model_file = 'data/ai_model.joblib'
        self.patterns = {}
        self.learned_signals = []
        self.success_rate = 0.85
        self.total_patterns = 0
        
        self.load_memory()
        self.initialize_ai_model()
    
    def load_memory(self):
        """Cargar memoria guardada"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.patterns = data.get('patterns', {})
                    self.learned_signals = data.get('learned_signals', [])
                    self.success_rate = data.get('success_rate', 0.85)
                    self.total_patterns = data.get('total_patterns', 0)
            except:
                self.initialize_empty_memory()
        else:
            self.initialize_empty_memory()
    
    def initialize_empty_memory(self):
        """Inicializar memoria vacía"""
        self.patterns = {}
        self.learned_signals = []
        self.success_rate = 0.85
        self.total_patterns = 0
        self.save_memory()
    
    def save_memory(self):
        """Guardar memoria"""
        os.makedirs('data', exist_ok=True)
        memory_data = {
            'patterns': self.patterns,
            'learned_signals': self.learned_signals[-1000:],  # Mantener últimas 1000
            'success_rate': self.success_rate,
            'total_patterns': self.total_patterns,
            'last_update': datetime.now().isoformat()
        }
        
        with open(self.memory_file, 'w') as f:
            json.dump(memory_data, f, indent=2)
    
    def learn_pattern(self, symbol, price_data, indicators, outcome):
        """Aprender de un patrón de trading"""
        pattern_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H')}"
        
        pattern = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'price_change': price_data['change'],
            'volume': price_data['volume'],
            'rsi': indicators['rsi'],
            'macd': indicators['macd'],
            'bollinger_position': indicators['bollinger_pos'],
            'outcome': outcome,  # 'profit', 'loss', 'hold'
            'success': outcome == 'profit'
        }
        
        # Guardar patrón
        if symbol not in self.patterns:
            self.patterns[symbol] = []
        
        self.patterns[symbol].append(pattern)
        self.learned_signals.append(pattern)
        self.total_patterns += 1
        
        # Actualizar tasa de éxito
        recent_signals = self.learned_signals[-100:]  # Últimas 100
        if recent_signals:
            success_count = sum(1 for s in recent_signals if s['success'])
            self.success_rate = success_count / len(recent_signals)
        
        self.save_memory()
        return pattern
    
    def get_prediction(self, symbol, current_indicators):
        """Obtener predicción basada en patrones aprendidos"""
        if symbol not in self.patterns or len(self.patterns[symbol]) < 10:
            return {'signal': 'hold', 'confidence': 0.5, 'reason': 'insufficient_data'}
        
        # Buscar patrones similares
        similar_patterns = self.find_similar_patterns(symbol, current_indicators)
        
        if not similar_patterns:
            return {'signal': 'hold', 'confidence': 0.5, 'reason': 'no_similar_patterns'}
        
        # Análisis de patrones similares
        successful_patterns = [p for p in similar_patterns if p['success']]
        success_ratio = len(successful_patterns) / len(similar_patterns)
        
        # Determinar señal
        if success_ratio > 0.75 and len(successful_patterns) >= 3:
            avg_outcome = np.mean([p['price_change'] for p in successful_patterns])
            
            if avg_outcome > 2.0:  # Ganancia esperada > 2%
                signal = 'buy'
                confidence = min(0.95, success_ratio + 0.1)
            elif avg_outcome < -2.0:  # Se espera caída
                signal = 'sell'
                confidence = min(0.90, success_ratio)
            else:
                signal = 'hold'
                confidence = 0.6
        else:
            signal = 'hold'
            confidence = 0.5
        
        return {
            'signal': signal,
            'confidence': confidence,
            'similar_patterns': len(similar_patterns),
            'success_rate': success_ratio,
            'expected_return': np.mean([p['price_change'] for p in similar_patterns])
        }
    
    def find_similar_patterns(self, symbol, indicators, threshold=0.15):
        """Encontrar patrones similares"""
        if symbol not in self.patterns:
            return []
        
        similar = []
        current_rsi = indicators['rsi']
        current_macd = indicators['macd']
        current_bb = indicators['bollinger_pos']
        
        for pattern in self.patterns[symbol][-200:]:  # Últimos 200 patrones
            rsi_diff = abs(pattern['rsi'] - current_rsi)
            macd_diff = abs(pattern['macd'] - current_macd)
            bb_diff = abs(pattern['bollinger_position'] - current_bb)
            
            # Calcular similitud
            similarity_score = (
                (1 - rsi_diff / 100) * 0.4 +
                (1 - min(macd_diff, 1)) * 0.3 +
                (1 - bb_diff) * 0.3
            )
            
            if similarity_score > threshold:
                pattern['similarity'] = similarity_score
                similar.append(pattern)
        
        return sorted(similar, key=lambda x: x['similarity'], reverse=True)[:20]
    
    def initialize_ai_model(self):
        """Inicializar modelo de ML"""
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
        # Cargar modelo si existe
        if os.path.exists(self.model_file):
            try:
                model_data = joblib.load(self.model_file)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.is_trained = True
            except:
                pass
    
    def train_model(self):
        """Entrenar modelo de ML con patrones aprendidos"""
        if len(self.learned_signals) < 50:
            return False
        
        # Preparar datos
        features = []
        targets = []
        
        for signal in self.learned_signals:
            if all(key in signal for key in ['rsi', 'macd', 'bollinger_position']):
                feature = [
                    signal['rsi'],
                    signal['macd'],
                    signal['bollinger_position'],
                    signal['volume'],
                    signal['price_change']
                ]
                
                target = 1 if signal['success'] else 0
                
                features.append(feature)
                targets.append(target)
        
        if len(features) < 20:
            return False
        
        # Entrenar modelo
        X = np.array(features)
        y = np.array(targets)
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Guardar modelo
        os.makedirs('data', exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, self.model_file)
        
        return True
    
    def predict_with_ml(self, indicators):
        """Predicción usando modelo ML"""
        if not self.is_trained:
            return None
        
        try:
            features = np.array([[
                indicators['rsi'],
                indicators['macd'],
                indicators['bollinger_pos'],
                indicators.get('volume', 1000),
                indicators.get('price_change', 0)
            ]])
            
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict_proba(features_scaled)[0]
            
            return {
                'success_probability': prediction[1],
                'confidence': max(prediction) - 0.5,
                'recommended': 'buy' if prediction[1] > 0.7 else 'hold'
            }
        except:
            return None
    
    def get_memory_stats(self):
        """Obtener estadísticas de memoria"""
        return {
            'total_patterns': self.total_patterns,
            'success_rate': self.success_rate,
            'symbols_learned': len(self.patterns),
            'recent_signals': len(self.learned_signals),
            'model_trained': self.is_trained,
            'memory_size_mb': self.get_memory_size()
        }
    
    def get_memory_size(self):
        """Calcular tamaño de memoria"""
        try:
            if os.path.exists(self.memory_file):
                return os.path.getsize(self.memory_file) / 1024 / 1024  # MB
        except:
            pass
        return 0.0
