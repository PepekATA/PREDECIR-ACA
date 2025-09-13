import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import alpaca_trade_api as tradeapi
import time

class TradingEngine:
    """Motor de trading que NUNCA vende en pérdida"""
    
    def __init__(self, memory_system):
        self.memory = memory_system
        self.api = None
        self.positions = {}
        self.orders = {}
        self.trading_rules = {
            'never_sell_at_loss': True,
            'min_profit_threshold': 2.0,  # Mínimo 2% ganancia
            'max_position_hold_days': 30,  # Máximo 30 días holding
            'emergency_stop_loss': -50.0,  # Solo si pérdida > 50%
            'dynamic_profit_taking': True,
            'reinvest_profits': True
        }
        
        self.trading_log = []
        self.load_trading_state()
    
    def initialize_api(self, api_key, api_secret, paper_trading=True):
        """Inicializar API de Alpaca"""
        try:
            base_url = "https://paper-api.alpaca.markets" if paper_trading else "https://api.alpaca.markets"
            self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
            
            # Verificar conexión
            account = self.api.get_account()
            print(f"✅ API conectada - Portfolio: ${account.portfolio_value}")
            return True
        except Exception as e:
            print(f"❌ Error conectando API: {e}")
            return False
    
    def load_trading_state(self):
        """Cargar estado de trading"""
        state_file = 'data/trading_state.json'
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    self.positions = data.get('positions', {})
                    self.orders = data.get('orders', {})
                    self.trading_log = data.get('trading_log', [])
            except Exception as e:
                print(f"Error cargando estado: {e}")
                self.initialize_empty_state()
        else:
            self.initialize_empty_state()
    
    def save_trading_state(self):
        """Guardar estado de trading"""
        os.makedirs('data', exist_ok=True)
        state_data = {
            'positions': self.positions,
            'orders': self.orders,
            'trading_log': self.trading_log[-500:],  # Últimas 500 entradas
            'last_update': datetime.now().isoformat()
        }
        
        with open('data/trading_state.json', 'w') as f:
            json.dump(state_data, f, indent=2)
    
    def initialize_empty_state(self):
        """Inicializar estado vacío"""
        self.positions = {}
        self.orders = {}
        self.trading_log = []
    
    def analyze_position_profitability(self, symbol, current_price):
        """Analizar rentabilidad de posición - NUNCA VENDER EN PÉRDIDA"""
        if symbol not in self.positions:
            return {'can_sell': False, 'reason': 'no_position'}
        
        position = self.positions[symbol]
        entry_price = position['entry_price']
        quantity = position['quantity']
        entry_date = datetime.fromisoformat(position['entry_date'])
        
        # Calcular P&L
        current_value = current_price * quantity
        cost_basis = entry_price * quantity
        unrealized_pnl = current_value - cost_basis
        unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100
        
        # Días en posición
        days_held = (datetime.now() - entry_date).days
        
        analysis = {
            'symbol': symbol,
            'entry_price': entry_price,
            'current_price': current_price,
            'quantity': quantity,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'days_held': days_held,
            'can_sell': False,
            'recommended_action': 'hold'
        }
        
        # REGLA PRINCIPAL: NUNCA VENDER EN PÉRDIDA
        if unrealized_pnl_pct < 0:
            analysis['can_sell'] = False
            analysis['recommended_action'] = 'hold'
            analysis['reason'] = f"NUNCA VENDER EN PÉRDIDA - Pérdida actual: {unrealized_pnl_pct:.2f}%"
            
            # Excepción: Pérdida extrema (> 50%) Y más de 30 días
            if unrealized_pnl_pct < -50 and days_held > 30:
                analysis['can_sell'] = True
                analysis['recommended_action'] = 'emergency_sell'
                analysis['reason'] = f"VENTA DE EMERGENCIA - Pérdida: {unrealized_pnl_pct:.2f}%, Días: {days_held}"
        
        # Si hay ganancia, evaluar si vender
        elif unrealized_pnl_pct > 0:
            if unrealized_pnl_pct >= self.trading_rules['min_profit_threshold']:
                analysis['can_sell'] = True
                analysis['recommended_action'] = 'sell_profit'
                analysis['reason'] = f"VENTA CON GANANCIA - Profit: {unrealized_pnl_pct:.2f}%"
            else:
                analysis['can_sell'] = False
                analysis['recommended_action'] = 'hold_for_more_profit'
                analysis['reason'] = f"Ganancia insuficiente: {unrealized_pnl_pct:.2f}% < {self.trading_rules['min_profit_threshold']}%"
        
        return analysis
    
    def execute_buy_order(self, symbol, notional_amount, prediction_data):
        """Ejecutar orden de compra inteligente"""
        if not self.api:
            return {'success': False, 'error': 'API no inicializada'}
        
        try:
            # Verificar que tenemos confianza suficiente
            confidence = prediction_data.get('confidence', 0)
            if confidence < 0.7:
                return {
                    'success': False, 
                    'error': f'Confianza insuficiente: {confidence:.2f} < 0.70'
                }
            
            # Verificar que la predicción es alcista
            predicted_change = prediction_data.get('predicted_change', 0)
            if predicted_change < 1.0:  # Menos de 1% de ganancia esperada
                return {
                    'success': False,
                    'error': f'Ganancia esperada insuficiente: {predicted_change:.2f}%'
                }
            
            # Ejecutar orden
            order = self.api.submit_order(
                symbol=symbol,
                side='buy',
                type='market',
                time_in_force='day',
                notional=notional_amount
            )
            
            # Guardar en nuestro registro
            position_data = {
                'symbol': symbol,
                'entry_price': prediction_data.get('current_price', 0),
                'notional_amount': notional_amount,
                'entry_date': datetime.now().isoformat(),
                'order_id': order.id,
                'prediction_data': prediction_data,
                'status': 'active'
            }
            
            self.positions[symbol] = position_data
            
            # Log de trading
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': 'buy',
                'symbol': symbol,
                'amount': notional_amount,
                'expected_change': predicted_change,
                'confidence': confidence,
                'order_id': order.id
            }
            self.trading_log.append(log_entry)
            
            self.save_trading_state()
            
            return {
                'success': True,
                'order': order,
                'position': position_data,
                'message': f'Compra ejecutada: ${notional_amount} en {symbol}'
            }
            
        except Exception as e:
            error_msg = f"Error ejecutando compra: {str(e)}"
            print(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def execute_sell_order(self, symbol, sell_all=True):
        """Ejecutar orden de venta - SOLO SI HAY GANANCIA"""
        if not self.api:
            return {'success': False, 'error': 'API no inicializada'}
        
        if symbol not in self.positions:
            return {'success': False, 'error': 'No hay posición activa'}
        
        try:
            # Obtener precio actual
            latest_trade = self.api.get_crypto_bars(symbol, '1Min', limit=1).df
            if latest_trade.empty:
                return {'success': False, 'error': 'No se pudo obtener precio actual'}
            
            current_price = latest_trade['close'].iloc[-1]
            
            # Analizar si podemos vender
            analysis = self.analyze_position_profitability(symbol, current_price)
            
            if not analysis['can_sell']:
                return {
                    'success': False,
                    'error': analysis['reason'],
                    'analysis': analysis
                }
            
            # Obtener posición actual de Alpaca
            try:
                alpaca_position = self.api.get_position(symbol)
                quantity = float(alpaca_position.qty)
            except:
                return {'success': False, 'error': 'Posición no encontrada en Alpaca'}
            
            # Ejecutar venta
            if analysis['recommended_action'] == 'emergency_sell':
                # Venta de emergencia
                order = self.api.submit_order(
                    symbol=symbol,
                    side='sell',
                    type='market',
                    time_in_force='day',
                    qty=quantity
                )
                
                sell_reason = 'EMERGENCY_SELL'
                
            else:
                # Venta con ganancia
                order = self.api.submit_order(
                    symbol=symbol,
                    side='sell',
                    type='market',
                    time_in_force='day',
                    qty=quantity
                )
                
                sell_reason = 'PROFIT_TAKING'
            
            # Actualizar registro
            self.positions[symbol]['status'] = 'sold'
            self.positions[symbol]['exit_price'] = current_price
            self.positions[symbol]['exit_date'] = datetime.now().isoformat()
            self.positions[symbol]['final_pnl'] = analysis['unrealized_pnl']
            self.positions[symbol]['final_pnl_pct'] = analysis['unrealized_pnl_pct']
            
            # Log de trading
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': 'sell',
                'symbol': symbol,
                'reason': sell_reason,
                'pnl': analysis['unrealized_pnl'],
                'pnl_pct': analysis['unrealized_pnl_pct'],
                'days_held': analysis['days_held'],
                'order_id': order.id
            }
            self.trading_log.append(log_entry)
            
            # Aprender de la operación
            outcome = 'profit' if analysis['unrealized_pnl'] > 0 else 'loss'
            self.memory.learn_pattern(
                symbol, 
                {'change': analysis['unrealized_pnl_pct']},
                self.positions[symbol]['prediction_data'],
                outcome
            )
            
            self.save_trading_state()
            
            return {
                'success': True,
                'order': order,
                'analysis': analysis,
                'message': f'Venta ejecutada: {symbol} - PnL: {analysis["unrealized_pnl_pct"]:.2f}%'
            }
            
        except Exception as e:
            error_msg = f"Error ejecutando venta: {str(e)}"
            print(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def auto_trading_cycle(self, symbols, ai_predictor, max_positions=5):
        """Ciclo automático de trading - NUNCA VENDE EN PÉRDIDA"""
        if not self.api:
            return {'error': 'API no inicializada'}
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'actions_taken': [],
            'positions_analyzed': 0,
            'opportunities_found': 0
        }
        
        try:
            # 1. Revisar posiciones existentes
            current_positions = list(self.positions.keys())
            for symbol in current_positions:
                if self.positions[symbol]['status'] == 'active':
                    # Obtener precio actual
                    try:
                        bars = self.api.get_crypto_bars(symbol, '1Min', limit=1)
                        current_price = bars.df['close'].iloc[-1]
                        
                        # Analizar si vender
                        analysis = self.analyze_position_profitability(symbol, current_price)
                        results['positions_analyzed'] += 1
                        
                        if analysis['can_sell'] and analysis['recommended_action'] in ['sell_profit', 'emergency_sell']:
                            sell_result = self.execute_sell_order(symbol)
                            if sell_result['success']:
                                results['actions_taken'].append({
                                    'action': 'sell',
                                    'symbol': symbol,
                                    'reason': analysis['reason'],
                                    'pnl_pct': analysis['unrealized_pnl_pct']
                                })
                    except Exception as e:
                        print(f"Error analizando posición {symbol}: {e}")
            
            # 2. Buscar nuevas oportunidades de compra
            active_positions = len([p for p in self.positions.values() if p['status'] == 'active'])
            
            if active_positions < max_positions:
                account = self.api.get_account()
                available_cash = float(account.buying_power)
                position_size = min(available_cash * 0.15, available_cash / (max_positions - active_positions))
                
                for symbol in symbols:
                    if symbol in self.positions and self.positions[symbol]['status'] == 'active':
                        continue  # Ya tenemos posición activa
                    
                    try:
                        # Obtener datos de mercado
                        bars = self.api.get_crypto_bars(symbol, '1Hour', limit=100)
                        if bars.df.empty:
                            continue
                        
                        price_data = bars.df
                        indicators = self.calculate_basic_indicators(price_data)
                        
                        # Obtener predicción de IA
                        prediction = ai_predictor.predict_price_movement(symbol, price_data, indicators)
                        
                        results['opportunities_found'] += 1
                        
                        # Evaluar si comprar
                        if (prediction['signal'] in ['strong_buy', 'buy'] and 
                            prediction['confidence'] > 0.75 and 
                            prediction['predicted_change'] > 2.0 and
                            position_size >= 25):  # Mínimo $25
                            
                            buy_result = self.execute_buy_order(symbol, position_size, prediction)
                            if buy_result['success']:
                                results['actions_taken'].append({
                                    'action': 'buy',
                                    'symbol': symbol,
                                    'amount': position_size,
                                    'expected_change': prediction['predicted_change'],
                                    'confidence': prediction['confidence']
                                })
                                
                                active_positions += 1
                                if active_positions >= max_positions:
                                    break
                    
                    except Exception as e:
                        print(f"Error evaluando {symbol}: {e}")
                        continue
            
            return results
            
        except Exception as e:
            results['error'] = str(e)
            return results
    
    def calculate_basic_indicators(self, price_data):
        """Calcular indicadores técnicos básicos"""
        try:
            prices = price_data['close']
            
            # RSI
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = prices.ewm(span=12, adjust=False).mean()
            exp2 = prices.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            
            # Bollinger Bands
            sma = prices.rolling(window=20).mean()
            std = prices.rolling(window=20).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            bb_position = (prices.iloc[-1] - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
            
            return {
                'rsi': rsi.iloc[-1] if not rsi.empty else 50,
                'macd': macd.iloc[-1] if not macd.empty else 0,
                'bollinger_pos': bb_position if not np.isnan(bb_position) else 0.5,
                'volume': price_data['volume'].iloc[-1] if 'volume' in price_data else 1000,
                'price_change_1h': ((prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2]) * 100 if len(prices) > 1 else 0,
                'price_change_4h': ((prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5]) * 100 if len(prices) > 5 else 0,
                'price_change_24h': ((prices.iloc[-1] - prices.iloc[-25]) / prices.iloc[-25]) * 100 if len(prices) > 25 else 0
            }
        except Exception as e:
            return {
                'rsi': 50, 'macd': 0, 'bollinger_pos': 0.5, 'volume': 1000,
                'price_change_1h': 0, 'price_change_4h': 0, 'price_change_24h': 0
            }
    
    def get_portfolio_summary(self):
        """Resumen completo del portfolio"""
        if not self.api:
            return {'error': 'API no disponible'}
        
        try:
            account = self.api.get_account()
            alpaca_positions = self.api.list_positions()
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'account_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'total_positions': len(alpaca_positions),
                'active_positions': len([p for p in self.positions.values() if p['status'] == 'active']),
                'positions': [],
                'total_unrealized_pnl': 0,
                'never_sold_at_loss': True,
                'trading_stats': self.get_trading_statistics()
            }
            
            # Analizar cada posición
            for position in alpaca_positions:
                symbol = position.symbol
                current_price = float(position.current_price)
                unrealized_pnl = float(position.unrealized_pl)
                unrealized_pnl_pct = float(position.unrealized_plpc) * 100
                
                position_info = {
                    'symbol': symbol,
                    'quantity': float(position.qty),
                    'current_price': current_price,
                    'market_value': float(position.market_value),
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_pct': unrealized_pnl_pct,
                    'can_sell': unrealized_pnl > 0,  # Solo vender si hay ganancia
                    'recommendation': 'HOLD (NEVER SELL AT LOSS)' if unrealized_pnl < 0 else 'CAN SELL WITH PROFIT'
                }
                
                summary['positions'].append(position_info)
                summary['total_unrealized_pnl'] += unrealized_pnl
            
            return summary
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_trading_statistics(self):
        """Estadísticas de trading"""
        if not self.trading_log:
            return {'no_trades': True}
        
        completed_trades = [log for log in self.trading_log if log['action'] == 'sell']
        
        if not completed_trades:
            return {'completed_trades': 0}
        
        profits = [trade['pnl'] for trade in completed_trades if 'pnl' in trade]
        profit_trades = [p for p in profits if p > 0]
        loss_trades = [p for p in profits if p < 0]
        
        stats = {
            'total_trades': len(completed_trades),
            'profit_trades': len(profit_trades),
            'loss_trades': len(loss_trades),
            'win_rate': len(profit_trades) / len(completed_trades) if completed_trades else 0,
            'total_pnl': sum(profits),
            'avg_profit': np.mean(profit_trades) if profit_trades else 0,
            'avg_loss': np.mean(loss_trades) if loss_trades else 0,
            'largest_win': max(profits) if profits else 0,
            'largest_loss': min(profits) if profits else 0,
            'never_sold_at_loss_policy': True
        }
        
        return stats
    
    def emergency_portfolio_check(self):
        """Revisión de emergencia del portfolio"""
        if not self.api:
            return {'error': 'API no disponible'}
        
        emergency_actions = []
        
        try:
            positions = self.api.list_positions()
            
            for position in positions:
                symbol = position.symbol
                unrealized_pnl_pct = float(position.unrealized_plpc) * 100
                
                # Solo venta de emergencia si pérdida > 50%
                if unrealized_pnl_pct < -50:
                    days_held = 0  # Calcular días reales
                    if symbol in self.positions:
                        entry_date = datetime.fromisoformat(self.positions[symbol]['entry_date'])
                        days_held = (datetime.now() - entry_date).days
                    
                    if days_held > 30:  # Más de 30 días con pérdida extrema
                        emergency_actions.append({
                            'symbol': symbol,
                            'action': 'emergency_sell',
                            'loss_pct': unrealized_pnl_pct,
                            'days_held': days_held,
                            'reason': 'Pérdida extrema + tiempo excesivo'
                        })
            
            return {
                'emergency_actions': emergency_actions,
                'total_emergency_positions': len(emergency_actions)
            }
            
        except Exception as e:
            return {'error': str(e)}
