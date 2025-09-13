"""
Dashboard Backend
Proporciona datos para el dashboard web de Streamlit
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("Dashboard")

class DashboardManager:
    """Gestor del dashboard que provee datos para la interfaz web"""
    
    def __init__(self, trading_engine, portfolio_manager, ai_predictor, memory_system, data_manager):
        self.trading_engine = trading_engine
        self.portfolio_manager = portfolio_manager
        self.ai_predictor = ai_predictor
        self.memory_system = memory_system
        self.data_manager = data_manager
        
        # Cache para dashboard
        self.dashboard_cache = {}
        self.last_update = None
    
    def get_dashboard_data(self, force_refresh: bool = False) -> Dict:
        """Obtener todos los datos para el dashboard"""
        try:
            # Cache de 30 segundos para evitar sobrecarga
            if (not force_refresh and 
                self.last_update and 
                (datetime.now() - self.last_update).seconds < 30):
                return self.dashboard_cache
            
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'bot_status': self.get_bot_status(),
                'portfolio_summary': self.get_portfolio_summary(),
                'ai_predictions': self.get_ai_predictions_summary(),
                'market_overview': self.get_market_overview(),
                'recent_trades': self.get_recent_trades(),
                'performance_metrics': self.get_performance_metrics(),
                'alerts': self.get_alerts(),
                'system_health': self.get_system_health()
            }
            
            # Actualizar cache
            self.dashboard_cache = dashboard_data
            self.last_update = datetime.now()
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos dashboard: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def get_bot_status(self) -> Dict:
        """Estado general del bot"""
        try:
            # Obtener estado de trading
            portfolio = self.trading_engine.get_portfolio_summary()
            
            active_positions = 0
            total_value = 0
            total_pnl = 0
            
            if 'positions' in portfolio:
                active_positions = len([p for p in portfolio['positions'] if p.get('quantity', 0) != 0])
                total_value = portfolio.get('account_value', 0)
                total_pnl = portfolio.get('total_unrealized_pnl', 0)
            
            # Estado de memoria AI
            memory_stats = self.memory_system.get_memory_stats()
            
            status = {
                'is_active': True,  # Asumimos que está activo si estamos obteniendo datos
                'uptime_hours': self.calculate_uptime(),
                'trading_mode': 'paper' if portfolio.get('paper_mode', True) else 'live',
                'active_positions': active_positions,
                'portfolio_value': total_value,
                'unrealized_pnl': total_pnl,
                'pnl_percentage': (total_pnl / (total_value - total_pnl)) * 100 if total_value > total_pnl else 0,
                'ai_learning_active': memory_stats.get('model_trained', False),
                'total_patterns_learned': memory_stats.get('total_patterns', 0),
                'success_rate': memory_stats.get('success_rate', 0),
                'never_sold_at_loss': True,  # Nuestra regla principal
                'last_action_time': self.get_last_action_time()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo estado del bot: {e}")
            return {'is_active': False, 'error': str(e)}
    
    def get_portfolio_summary(self) -> Dict:
        """Resumen del portfolio"""
        try:
            portfolio = self.trading_engine.get_portfolio_summary()
            
            if 'error' in portfolio:
                return portfolio
            
            # Procesar posiciones
            positions_data = []
            for position in portfolio.get('positions', []):
                position_data = {
                    'symbol': position['symbol'],
                    'quantity': position['quantity'],
                    'current_price': position['current_price'],
                    'market_value': position['market_value'],
                    'unrealized_pnl': position['unrealized_pnl'],
                    'unrealized_pnl_pct': position['unrealized_pnl_pct'],
                    'can_sell': position['unrealized_pnl_pct'] > 2.0,  # Solo vender con >2% ganancia
                    'recommendation': position['recommendation'],
                    'days_held': self.calculate_days_held(position['symbol']),
                    'status_color': 'green' if position['unrealized_pnl'] > 0 else 'red',
                    'action_icon': '💰' if position['unrealized_pnl'] > 0 else '💎'  # Diamond hands para pérdidas
                }
                positions_data.append(position_data)
            
            summary = {
                'total_value': portfolio.get('account_value', 0),
                'cash_available': portfolio.get('cash', 0),
                'buying_power': portfolio.get('buying_power', 0),
                'total_positions': len(positions_data),
                'positions': positions_data,
                'diversification_score': self.calculate_diversification_score(positions_data),
                'risk_level': self.assess_portfolio_risk(positions_data),
                'performance': {
                    'total_pnl': portfolio.get('total_unrealized_pnl', 0),
                    'daily_change': self.calculate_daily_change(),
                    'weekly_change': self.calculate_weekly_change(),
                    'best_performer': self.get_best_performer(positions_data),
                    'worst_performer': self.get_worst_performer(positions_data)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo resumen portfolio: {e}")
            return {'error': str(e)}
    
    def get_ai_predictions_summary(self) -> List[Dict]:
        """Resumen de predicciones de IA"""
        try:
            predictions = []
            
            # Obtener símbolos activos
            symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'AVAXUSD', 'ADAUSD', 'DOTUSD']
            
            for symbol in symbols:
                try:
                    # Obtener datos de mercado
                    market_data = self.data_manager.get_symbol_data(symbol)
                    if not market_data:
                        continue
                    
                    # Calcular indicadores
                    from modules.market_analyzer import MarketAnalyzer
                    analyzer = MarketAnalyzer(self.data_manager, self.ai_predictor)
                    indicators = analyzer.calculate_indicators(market_data)
                    
                    # Generar predicción
                    prediction = self.ai_predictor.predict_price_movement(
                        symbol, market_data, indicators
                    )
                    
                    # Procesar predicción para dashboard
                    pred_data = {
                        'symbol': symbol.replace('USD', ''),
                        'current_price': market_data['latest']['price'],
                        'predicted_change': prediction.get('predicted_change', 0),
                        'confidence': prediction.get('confidence', 0.5),
                        'signal': prediction.get('signal', 'hold').upper(),
                        'trend_duration': prediction.get('duration_minutes', 30),
                        'signal_strength': self.classify_signal_strength(prediction),
                        'signal_color': self.get_signal_color(prediction.get('signal', 'hold')),
                        'signal_icon': self.get_signal_icon(prediction.get('signal', 'hold')),
                        'ai_confidence_level': self.classify_confidence(prediction.get('confidence', 0.5)),
                        'recommendation': self.generate_recommendation(prediction),
                        'risk_reward': self.calculate_risk_reward(prediction)
                    }
                    
                    predictions.append(pred_data)
                    
                except Exception as e:
                    logger.error(f"❌ Error procesando predicción {symbol}: {e}")
                    continue
            
            # Ordenar por confianza y cambio esperado
            predictions.sort(key=lambda x: (x['confidence'] * abs(x['predicted_change'])), reverse=True)
            
            return predictions[:10]  # Top 10
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo predicciones: {e}")
            return []
    
    def get_market_overview(self) -> Dict:
        """Vista general del mercado"""
        try:
            market_summary = self.data_manager.get_market_summary()
            
            if 'error' in market_summary:
                return market_summary
            
            overview = market_summary.get('market_overview', {})
            
            market_data = {
                'timestamp': market_summary['timestamp'],
                'symbols_tracked': market_summary['symbols_tracked'],
                'market_sentiment': overview.get('market_sentiment', 'NEUTRAL'),
                'sentiment_score': self.calculate_sentiment_score(overview),
                'avg_change_24h': overview.get('avg_change', 0),
                'bullish_assets': overview.get('bullish_count', 0),
                'bearish_assets': overview.get('bearish_count', 0),
                'top_gainers': market_summary.get('top_gainers', [])[:3],
                'top_losers': market_summary.get('top_losers', [])[:3],
                'high_volume': market_summary.get('high_volume', [])[:3],
                'market_phase': self.determine_market_phase(overview),
                'volatility_index': self.calculate_volatility_index(),
                'fear_greed_index': self.calculate_fear_greed_index(overview)
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo overview de mercado: {e}")
            return {'error': str(e)}
    
    def get_recent_trades(self) -> List[Dict]:
        """Trades recientes"""
        try:
            trading_stats = self.trading_engine.get_trading_statistics()
            
            if 'no_trades' in trading_stats or 'completed_trades' not in trading_stats:
                return []
            
            # Obtener trades del log
            recent_trades = []
            for trade in self.trading_engine.trading_log[-10:]:  # Últimos 10
                trade_data = {
                    'timestamp': trade['timestamp'],
                    'action': trade['action'].upper(),
                    'symbol': trade['symbol'],
                    'amount': trade.get('amount', 0),
                    'pnl': trade.get('pnl', 0),
                    'pnl_pct': trade.get('pnl_pct', 0),
                    'reason': trade.get('reason', 'N/A'),
                    'confidence': trade.get('confidence', 0),
                    'action_color': 'green' if trade['action'] == 'buy' else 'blue' if trade.get('pnl', 0) > 0 else 'orange',
                    'action_icon': '🛒' if trade['action'] == 'buy' else '💰' if trade.get('pnl', 0) > 0 else '⏰',
                    'formatted_time': self.format_trade_time(trade['timestamp'])
                }
                recent_trades.append(trade_data)
            
            return list(reversed(recent_trades))  # Más recientes primero
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo trades recientes: {e}")
            return []
    
    def get_performance_metrics(self) -> Dict:
        """Métricas de rendimiento"""
        try:
            trading_stats = self.trading_engine.get_trading_statistics()
            memory_stats = self.memory_system.get_memory_stats()
            
            if 'no_trades' in trading_stats:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_trade': 0,
                    'best_trade': 0,
                    'ai_accuracy': memory_stats.get('success_rate', 0),
                    'never_sold_at_loss': True,
                    'streak': 0
                }
            
            metrics = {
                'total_trades': trading_stats.get('total_trades', 0),
                'win_rate': trading_stats.get('win_rate', 0) * 100,
                'total_pnl': trading_stats.get('total_pnl', 0),
                'avg_profit': trading_stats.get('avg_profit', 0),
                'avg_loss': trading_stats.get('avg_loss', 0),
                'best_trade': trading_stats.get('largest_win', 0),
                'worst_trade': trading_stats.get('largest_loss', 0),
                'profit_factor': self.calculate_profit_factor(trading_stats),
                'sharpe_ratio': self.calculate_sharpe_ratio(),
                'max_drawdown': self.calculate_max_drawdown(),
                'ai_accuracy': memory_stats.get('success_rate', 0) * 100,
                'patterns_learned': memory_stats.get('total_patterns', 0),
                'never_sold_at_loss': True,
                'current_streak': self.calculate_current_streak(),
                'performance_grade': self.calculate_performance_grade(trading_stats)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculando métricas: {e}")
            return {'error': str(e)}
    
    def get_alerts(self) -> List[Dict]:
        """Alertas del sistema"""
        alerts = []
        
        try:
            # Alertas de portfolio
            portfolio = self.trading_engine.get_portfolio_summary()
            if 'positions' in portfolio:
                for position in portfolio['positions']:
                    # Alerta de posición con gran ganancia
                    if position['unrealized_pnl_pct'] > 20:
                        alerts.append({
                            'type': 'profit_opportunity',
                            'level': 'success',
                            'message': f"🎯 {position['symbol']} tiene +{position['unrealized_pnl_pct']:.1f}% ganancia - Considerar venta",
                            'timestamp': datetime.now().isoformat(),
                            'action': 'consider_sell'
                        })
                    
                    # Alerta de posición con pérdida (pero NUNCA vender)
                    elif position['unrealized_pnl_pct'] < -10:
                        alerts.append({
                            'type': 'loss_position',
                            'level': 'warning',
                            'message': f"💎 {position['symbol']} en {position['unrealized_pnl_pct']:.1f}% - MANTENER (No vender en pérdida)",
                            'timestamp': datetime.now().isoformat(),
                            'action': 'diamond_hands'
                        })
            
            # Alertas de AI y predicciones
            predictions = self.get_ai_predictions_summary()
            for pred in predictions[:3]:  # Top 3
                if pred['confidence'] > 0.85 and abs(pred['predicted_change']) > 5:
                    alerts.append({
                        'type': 'ai_opportunity',
                        'level': 'info',
                        'message': f"🤖 IA detecta oportunidad en {pred['symbol']}: {pred['predicted_change']:+.1f}% con {pred['confidence']:.0%} confianza",
                        'timestamp': datetime.now().isoformat(),
                        'action': 'review_prediction'
                    })
            
            # Alertas de sistema
            memory_stats = self.memory_system.get_memory_stats()
            if memory_stats.get('success_rate', 0) > 0.9:
                alerts.append({
                    'type': 'ai_performance',
                    'level': 'success',
                    'message': f"🧠 IA funcionando excelente: {memory_stats['success_rate']:.0%} precisión",
                    'timestamp': datetime.now().isoformat(),
                    'action': 'none'
                })
            
            return sorted(alerts, key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            logger.error(f"❌ Error generando alertas: {e}")
            return []
    
    def get_system_health(self) -> Dict:
        """Estado de salud del sistema"""
        try:
            health = {
                'overall_status': 'healthy',
                'components': {
                    'trading_engine': self.check_trading_engine_health(),
                    'ai_predictor': self.check_ai_health(),
                    'data_manager': self.check_data_manager_health(),
                    'memory_system': self.check_memory_health()
                },
                'uptime_hours': self.calculate_uptime(),
                'last_error': self.get_last_error(),
                'memory_usage_mb': self.get_memory_usage(),
                'data_freshness': self.check_data_freshness(),
                'api_connection': self.check_api_connection()
            }
            
            # Determinar estado general
            component_statuses = list(health['components'].values())
            if all(status == 'healthy' for status in component_statuses):
                health['overall_status'] = 'healthy'
            elif any(status == 'error' for status in component_statuses):
                health['overall_status'] = 'error'
            else:
                health['overall_status'] = 'warning'
            
            return health
            
        except Exception as e:
            logger.error(f"❌ Error verificando salud del sistema: {e}")
            return {'overall_status': 'error', 'error': str(e)}
    
    # Métodos auxiliares
    def calculate_uptime(self) -> float:
        """Calcular tiempo de funcionamiento en horas"""
        # Placeholder - implementar con timestamp de inicio real
        return 24.5
    
    def get_last_action_time(self) -> str:
        """Obtener tiempo de última acción"""
        try:
            if self.trading_engine.trading_log:
                return self.trading_engine.trading_log[-1]['timestamp']
            return datetime.now().isoformat()
        except Exception:
            return datetime.now().isoformat()
    
    def calculate_days_held(self, symbol: str) -> int:
        """Calcular días que se ha mantenido una posición"""
        try:
            if symbol in self.trading_engine.positions:
                entry_date = datetime.fromisoformat(
                    self.trading_engine.positions[symbol]['entry_date']
                )
                return (datetime.now() - entry_date).days
            return 0
        except Exception:
            return 0
    
    def classify_signal_strength(self, prediction: Dict) -> str:
        """Clasificar fuerza de señal"""
        confidence = prediction.get('confidence', 0)
        change = abs(prediction.get('predicted_change', 0))
        
        if confidence > 0.85 and change > 5:
            return 'very_strong'
        elif confidence > 0.75 and change > 3:
            return 'strong'
        elif confidence > 0.65 and change > 1:
            return 'moderate'
        else:
            return 'weak'
    
    def get_signal_color(self, signal: str) -> str:
        """Obtener color para señal"""
        colors = {
            'strong_buy': '#00ff88',
            'buy': '#00C851',
            'hold': '#ffbb33',
            'sell': '#ff4444',
            'strong_sell': '#cc0000'
        }
        return colors.get(signal.lower(), '#888888')
    
    def get_signal_icon(self, signal: str) -> str:
        """Obtener icono para señal"""
        icons = {
            'strong_buy': '🚀',
            'buy': '📈',
            'hold': '⏸️',
            'sell': '📉',
            'strong_sell': '⬇️'
        }
        return icons.get(signal.lower(), '➖')
    
    def classify_confidence(self, confidence: float) -> str:
        """Clasificar nivel de confianza"""
        if confidence > 0.9:
            return 'very_high'
        elif confidence > 0.8:
            return 'high'
        elif confidence > 0.7:
            return 'medium'
        else:
            return 'low'
    
    def generate_recommendation(self, prediction: Dict) -> str:
        """Generar recomendación basada en predicción"""
        signal = prediction.get('signal', 'hold').lower()
        confidence = prediction.get('confidence', 0)
        change = prediction.get('predicted_change', 0)
        
        if signal in ['strong_buy', 'buy'] and confidence > 0.75 and change > 2:
            return f"Comprar - Ganancia esperada: {change:.1f}%"
        elif signal in ['sell', 'strong_sell']:
            return "Solo vender si hay ganancia"
        else:
            return "Mantener y observar"
    
    def calculate_risk_reward(self, prediction: Dict) -> Dict:
        """Calcular riesgo/beneficio"""
        change = prediction.get('predicted_change', 0)
        confidence = prediction.get('confidence', 0.5)
        
        # Estimaciones simples
        potential_reward = abs(change) * confidence
        potential_risk = abs(change) * (1 - confidence) * 0.5  # Riesgo estimado menor
        
        ratio = potential_reward / potential_risk if potential_risk > 0 else float('inf')
        
        return {
            'reward': potential_reward,
            'risk': potential_risk,
            'ratio': ratio,
            'assessment': 'good' if ratio > 2 else 'fair' if ratio > 1 else 'poor'
        }
    
    def check_trading_engine_health(self) -> str:
        """Verificar salud del motor de trading"""
        try:
            if self.trading_engine.api:
                return 'healthy'
            return 'warning'
        except Exception:
            return 'error'
    
    def check_ai_health(self) -> str:
        """Verificar salud de la IA"""
        try:
            memory_stats = self.memory_system.get_memory_stats()
            if memory_stats.get('model_trained', False):
                return 'healthy'
            return 'warning'
        except Exception:
            return 'error'
    
    def check_data_manager_health(self) -> str:
        """Verificar salud del gestor de datos"""
        try:
            if self.data_manager.api and len(self.data_manager.market_data) > 0:
                return 'healthy'
            return 'warning'
        except Exception:
            return 'error'
    
    def check_memory_health(self) -> str:
        """Verificar salud del sistema de memoria"""
        try:
            if len(self.memory_system.learned_signals) > 10:
                return 'healthy'
            return 'warning'
        except Exception:
            return 'error'
    
    def format_trade_time(self, timestamp: str) -> str:
        """Formatear tiempo de trade"""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%H:%M:%S')
        except Exception:
            return timestamp[:8] if len(timestamp) > 8 else timestamp
