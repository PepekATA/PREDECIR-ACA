import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class PortfolioManager:
    """Gestor inteligente de cartera con diversificación automática"""
    
    def __init__(self):
        self.portfolio_state = {}
        self.allocation_rules = {
            'max_single_asset': 25,    # Máximo 25% en un activo
            'min_diversification': 5,   # Mínimo 5 activos diferentes
            'rebalance_threshold': 10,  # Rebalancear si desviación > 10%
            'cash_reserve': 10,         # Mantener 10% en cash
            'profit_reinvestment': 80   # Reinvertir 80% de ganancias
        }
        
        self.load_portfolio_state()
    
    def load_portfolio_state(self):
        """Cargar estado de la cartera"""
        state_file = 'data/portfolio_state.json'
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    self.portfolio_state = json.load(f)
            except:
                self.initialize_empty_portfolio()
        else:
            self.initialize_empty_portfolio()
    
    def save_portfolio_state(self):
        """Guardar estado de la cartera"""
        os.makedirs('data', exist_ok=True)
        with open('data/portfolio_state.json', 'w') as f:
            json.dump(self.portfolio_state, f, indent=2)
    
    def initialize_empty_portfolio(self):
        """Inicializar cartera vacía"""
        self.portfolio_state = {
            'total_value': 0,
            'cash_balance': 0,
            'positions': {},
            'target_allocation': {},
            'last_rebalance': datetime.now().isoformat(),
            'performance_history': []
        }
    
    def calculate_optimal_allocation(self, symbols, market_predictions, total_capital):
        """Calcular asignación óptima basada en predicciones de IA"""
        try:
            # Filtrar símbolos con predicciones positivas y alta confianza
            viable_assets = []
            
            for symbol in symbols:
                prediction = market_predictions.get(symbol, {})
                predicted_change = prediction.get('predicted_change', 0)
                confidence = prediction.get('confidence', 0)
                
                if predicted_change > 1.0 and confidence > 0.7:
                    score = predicted_change * confidence
                    viable_assets.append({
                        'symbol': symbol,
                        'predicted_return': predicted_change,
                        'confidence': confidence,
                        'score': score
                    })
            
            if not viable_assets:
                return {'error': 'No viable assets found'}
            
            # Ordenar por score
            viable_assets.sort(key=lambda x: x['score'], reverse=True)
            
            # Tomar los mejores activos (máximo según reglas de diversificación)
            max_assets = max(self.allocation_rules['min_diversification'], 
                           min(len(viable_assets), 10))
            selected_assets = viable_assets[:max_assets]
            
            # Calcular pesos basados en scores
            total_score = sum(asset['score'] for asset in selected_assets)
            
            allocation = {}
            cash_reserve_pct = self.allocation_rules['cash_reserve']
            investable_capital = total_capital * (1 - cash_reserve_pct / 100)
            
            for asset in selected_assets:
                # Peso base según score
                base_weight = asset['score'] / total_score
                
                # Aplicar límite máximo por activo
                max_weight = self.allocation_rules['max_single_asset'] / 100
                actual_weight = min(base_weight, max_weight)
                
                allocation[asset['symbol']] = {
                    'target_weight': actual_weight,
                    'target_amount': investable_capital * actual_weight,
                    'predicted_return': asset['predicted_return'],
                    'confidence': asset['confidence'],
                    'score': asset['score']
                }
            
            # Normalizar pesos si es necesario
            total_weight = sum(alloc['target_weight'] for alloc in allocation.values())
            if total_weight > 0:
                for symbol in allocation:
                    allocation[symbol]['target_weight'] /= total_weight
                    allocation[symbol]['target_amount'] = investable_capital * allocation[symbol]['target_weight']
            
            return {
                'allocation': allocation,
                'cash_reserve': total_capital * (cash_reserve_pct / 100),
                'total_investable': investable_capital,
                'selected_assets': len(selected_assets),
                'diversification_score': len(selected_assets) / max_assets
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_current_allocation(self, current_positions, market_prices):
        """Analizar asignación actual vs objetivo"""
        if not current_positions:
            return {'status': 'empty_portfolio'}
        
        try:
            total_value = 0
            position_values = {}
            
            # Calcular valores actuales
            for symbol, position in current_positions.items():
                if symbol in market_prices:
                    current_price = market_prices[symbol]
                    position_value = position['quantity'] * current_price
                    position_values[symbol] = position_value
                    total_value += position_value
            
            if total_value == 0:
                return {'status': 'no_value'}
            
            # Calcular pesos actuales
            current_weights = {}
            for symbol, value in position_values.items():
                current_weights[symbol] = value / total_value
            
            # Comparar con objetivos
            target_allocation = self.portfolio_state.get('target_allocation', {})
            deviations = {}
            
            for symbol in set(list(current_weights.keys()) + list(target_allocation.keys())):
                current_weight = current_weights.get(symbol, 0)
                target_weight = target_allocation.get(symbol, {}).get('target_weight', 0)
                deviation = abs(current_weight - target_weight)
                deviations[symbol] = {
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'deviation': deviation,
                    'needs_rebalance': deviation > (self.allocation_rules['rebalance_threshold'] / 100)
                }
            
            # Determinar si necesita rebalance
            max_deviation = max(dev['deviation'] for dev in deviations.values()) if deviations else 0
            needs_rebalance = max_deviation > (self.allocation_rules['rebalance_threshold'] / 100)
            
            return {
                'total_value': total_value,
                'current_weights': current_weights,
                'deviations': deviations,
                'max_deviation': max_deviation,
                'needs_rebalance': needs_rebalance,
                'rebalance_threshold': self.allocation_rules['rebalance_threshold']
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def generate_rebalance_orders(self, current_analysis, target_allocation, available_cash):
        """Generar órdenes de rebalance"""
        if not current_analysis.get('needs_rebalance'):
            return {'rebalance_needed': False}
        
        try:
            orders = []
            total_value = current_analysis['total_value'] + available_cash
            
            for symbol, deviation in current_analysis['deviations'].items():
                if not deviation['needs_rebalance']:
                    continue
                
                current_weight = deviation['current_weight']
                target_weight = deviation['target_weight']
                
                target_value = total_value * target_weight
                current_value = total_value * current_weight
                value_diff = target_value - current_value
                
                if abs(value_diff) > 50:  # Solo si diferencia > $50
                    if value_diff > 0:
                        # Necesitamos comprar más
                        orders.append({
                            'symbol': symbol,
                            'action': 'buy',
                            'amount': value_diff,
                            'reason': f'Aumentar peso de {current_weight:.1%} a {target_weight:.1%}'
                        })
                    else:
                        # Necesitamos vender (solo si hay ganancia)
                        orders.append({
                            'symbol': symbol,
                            'action': 'potential_sell',
                            'amount': abs(value_diff),
                            'reason': f'Reducir peso de {current_weight:.1%} a {target_weight:.1%}',
                            'note': 'Solo ejecutar si hay ganancia'
                        })
            
            return {
                'rebalance_needed': True,
                'orders': orders,
                'total_adjustments': len(orders)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_portfolio_performance(self, historical_positions):
        """Calcular rendimiento del portfolio"""
        if not historical_positions:
            return {'no_history': True}
        
        try:
            performance_data = []
            
            for position in historical_positions:
                if position.get('status') == 'sold' and 'final_pnl_pct' in position:
                    entry_date = datetime.fromisoformat(position['entry_date'])
                    exit_date = datetime.fromisoformat(position['exit_date'])
                    days_held = (exit_date - entry_date).days
                    
                    performance_data.append({
                        'symbol': position['symbol'],
                        'return_pct': position['final_pnl_pct'],
                        'days_held': days_held,
                        'annualized_return': (position['final_pnl_pct'] / days_held) * 365 if days_held > 0 else 0
                    })
            
            if not performance_data:
                return {'no_completed_trades': True}
            
            returns = [p['return_pct'] for p in performance_data]
            
            performance = {
                'total_trades': len(performance_data),
                'avg_return': np.mean(returns),
                'total_return': sum(returns),
                'win_rate': len([r for r in returns if r > 0]) / len(returns),
                'best_trade': max(returns),
                'worst_trade': min(returns),
                'volatility': np.std(returns),
                'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
                'avg_holding_period': np.mean([p['days_held'] for p in performance_data])
            }
            
            return performance
            
        except Exception as e:
            return {'error': str(e)}
    
    def smart_position_sizing(self, symbol, prediction, available_capital, current_positions):
        """Calcular tamaño de posición inteligente"""
        try:
            # Factores para el sizing
            confidence = prediction.get('confidence', 0.5)
            predicted_return = prediction.get('predicted_return', 0)
            
            # Base size según confianza
            base_size_pct = min(15, confidence * 20)  # Máximo 15%
            
            # Ajuste por retorno esperado
            return_multiplier = min(1.5, (predicted_return / 5) + 0.5)
            adjusted_size_pct = base_size_pct * return_multiplier
            
            # Límites de diversificación
            max_position_pct = self.allocation_rules['max_single_asset']
            current_position_value = 0
            
            if symbol in current_positions:
                # Ya tenemos posición - calcular valor actual
                current_position_value = current_positions[symbol].get('market_value', 0)
            
            total_portfolio_value = available_capital + current_position_value
            max_position_value = total_portfolio_value * (max_position_pct / 100)
            proposed_size = available_capital * (adjusted_size_pct / 100)
            
            # No exceder límites
            final_size = min(proposed_size, max_position_value - current_position_value)
            final_size = max(0, final_size)  # No negativo
            
            # Mínimo viable
            if final_size < 25:  # Mínimo $25
                final_size = 0
            
            return {
                'recommended_size': final_size,
                'size_percentage': (final_size / available_capital) * 100 if available_capital > 0 else 0,
                'confidence_factor': confidence,
                'return_factor': return_multiplier,
                'diversification_limit': max_position_value,
                'rationale': f"Size based on {confidence:.0%} confidence and {predicted_return:.1f}% expected return"
            }
            
        except Exception as e:
            return {'recommended_size': 0, 'error': str(e)}
    
    def get_portfolio_health_score(self, current_state, performance_data):
        """Calcular score de salud del portfolio"""
        try:
            health_score = 0
            factors = {}
            
            # Factor 1: Diversificación (0-25 puntos)
            num_positions = len(current_state.get('positions', {}))
            if num_positions >= self.allocation_rules['min_diversification']:
                diversification_score = min(25, num_positions * 5)
            else:
                diversification_score = num_positions * 3
            
            factors['diversification'] = diversification_score
            health_score += diversification_score
            
            # Factor 2: Performance (0-25 puntos)
            if performance_data and not performance_data.get('no_completed_trades'):
                win_rate = performance_data.get('win_rate', 0)
                avg_return = performance_data.get('avg_return', 0)
                
                performance_score = (win_rate * 15) + min(10, avg_return / 2)
            else:
                performance_score = 15  # Score neutro sin historial
            
            factors['performance'] = performance_score
            health_score += performance_score
            
            # Factor 3: Risk Management - NUNCA VENDER EN PÉRDIDA (0-25 puntos)
            # Este bot siempre tiene 25 puntos por su política
            risk_score = 25
            factors['risk_management'] = risk_score
            health_score += risk_score
            
            # Factor 4: Balance de Cash (0-25 puntos)
            cash_ratio = current_state.get('cash_balance', 0) / max(current_state.get('total_value', 1), 1)
            target_cash = self.allocation_rules['cash_reserve'] / 100
            
            if 0.05 <= cash_ratio <= 0.15:  # Entre 5% y 15% es óptimo
                cash_score = 25
            elif cash_ratio < 0.05:
                cash_score = cash_ratio * 500  # Penalizar poco cash
            else:
                cash_score = max(0, 25 - ((cash_ratio - 0.15) * 100))  # Penalizar mucho cash
            
            factors['cash_balance'] = cash_score
            health_score += cash_score
            
            # Normalizar a 100
            health_score = min(100, health_score)
            
            # Determinar nivel de salud
            if health_score >= 80:
                health_level = "EXCELENTE"
            elif health_score >= 60:
                health_level = "BUENA"
            elif health_score >= 40:
                health_level = "REGULAR"
            else:
                health_level = "NECESITA MEJORAS"
            
            return {
                'health_score': health_score,
                'health_level': health_level,
                'factors': factors,
                'recommendations': self.generate_health_recommendations(factors)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def generate_health_recommendations(self, factors):
        """Generar recomendaciones de mejora"""
        recommendations = []
        
        if factors.get('diversification', 0) < 20:
            recommendations.append("🌐 Aumentar diversificación - agregar más activos")
        
        if factors.get('performance', 0) < 15:
            recommendations.append("📈 Mejorar selección de activos - usar predicciones con mayor confianza")
        
        if factors.get('cash_balance', 0) < 15:
            recommendations.append("💰 Optimizar balance de efectivo - mantener reserva adecuada")
        
        if not recommendations:
            recommendations.append("✅ Portfolio en excelente estado - continuar con estrategia actual")
        
        return recommendations
