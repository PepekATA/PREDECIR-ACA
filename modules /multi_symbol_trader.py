# modules/multi_symbol_trader.py
import threading
import time
import logging
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Any, Optional, List
import os

import alpaca_trade_api as tradeapi
import pandas as pd

logger = logging.getLogger("multi_symbol_trader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)


class SymbolWorker:
    def __init__(self, symbol: str, alpaca_api: tradeapi.REST, memory_system, account_alloc_pct: float, config: Dict[str, Any], dry_run: bool = True):
        self.symbol = symbol
        self.api = alpaca_api
        self.memory = memory_system
        self.alloc_pct = float(account_alloc_pct)
        self.config = config
        self.dry_run = bool(dry_run)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.position = None

    def start(self):
        if self._thread and self._thread.is_alive():
            logger.info(f"{self.symbol}: worker ya corriendo")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name=f"worker-{self.symbol}", daemon=True)
        self._thread.start()
        logger.info(f"{self.symbol}: worker iniciado")

    def stop(self):
        logger.info(f"{self.symbol}: deteniendo worker...")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(f"{self.symbol}: worker detenido")

    def _get_memory_signals(self) -> List[Dict[str, Any]]:
        try:
            if hasattr(self.memory, "get_signals"):
                return self.memory.get_signals(self.symbol) or []
            elif hasattr(self.memory, "get"):
                return self.memory.get(self.symbol) or []
            else:
                logger.debug(f"{self.symbol}: memory_system no posee métodos esperados")
                return []
        except Exception as e:
            logger.exception(f"{self.symbol}: error leyendo memoria: {e}")
            return []

    def _refresh_position_cache(self):
        try:
            acct_positions = self.api.list_positions()
            pos = next((p for p in acct_positions if p.symbol == self.symbol), None)
            if pos:
                self.position = {
                    "qty": float(pos.qty),
                    "avg_entry": float(pos.avg_entry_price) if pos.avg_entry_price else 0.0,
                    "market_value": float(pos.market_value),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "side": "long" if float(pos.qty) > 0 else "flat",
                }
            else:
                self.position = None
        except Exception as e:
            logger.exception(f"{self.symbol}: error refrescando posición: {e}")
            self.position = None

    def _get_account(self):
        return self.api.get_account()

    def _calc_order_size_usd(self, cash_available_usd: float):
        desired = cash_available_usd * self.alloc_pct
        min_usd = float(self.config.get("min_order_usd", 10.0))
        invest_usd = max(min_usd, desired)
        return invest_usd

    def _place_market_buy(self, usd_amount: float):
        try:
            bar = self.api.get_latest_trade(self.symbol)
            last_price = float(bar.price)
            qty = float(Decimal(str(usd_amount / last_price)).quantize(Decimal('0.0001'), rounding=ROUND_DOWN))
            if qty < self.config.get("min_shares", 0.0001):
                logger.info(f"{self.symbol}: cantidad {qty} menor al mínimo; no ordén")
                return None
            if self.dry_run:
                logger.info(f"{self.symbol} [DRY_RUN] Comprar {qty} shares ~ ${usd_amount:.2f} @ {last_price}")
                return {"id": "dry_run", "qty": qty, "filled_avg_price": last_price}
            else:
                order = self.api.submit_order(symbol=self.symbol, qty=qty, side="buy", type="market", time_in_force="gtc")
                logger.info(f"{self.symbol}: orden BUY enviada id={order.id} qty={qty}")
                return order
        except Exception as e:
            logger.exception(f"{self.symbol}: error placing buy: {e}")
            return None

    def _place_market_sell_all(self):
        try:
            if not self.position:
                logger.info(f"{self.symbol}: no hay posición para vender")
                return None
            qty = self.position["qty"]
            if self.dry_run:
                logger.info(f"{self.symbol} [DRY_RUN] VENDER {qty} shares")
                return {"id": "dry_run_sell", "qty": qty}
            else:
                order = self.api.submit_order(symbol=self.symbol, qty=qty, side="sell", type="market", time_in_force="gtc")
                logger.info(f"{self.symbol}: orden SELL enviada id={order.id} qty={qty}")
                return order
        except Exception as e:
            logger.exception(f"{self.symbol}: error placing sell: {e}")
            return None

    def _compute_unrealized_profit_pct(self):
        if not self.position:
            return 0.0
        try:
            avg = float(self.position["avg_entry"])
            bar = self.api.get_latest_trade(self.symbol)
            last_price = float(bar.price)
            pct = (last_price - avg) / avg
            return pct
        except Exception as e:
            logger.exception(f"{self.symbol}: error calculando profit pct: {e}")
            return 0.0

    def _should_buy(self, latest_pred: Dict[str, Any], recent_changes: Dict[str, float]) -> bool:
        if self.position:
            return False
        if not latest_pred:
            return False
        direction = latest_pred.get("direction") or latest_pred.get("prediction") or latest_pred.get("trend")
        if direction is None:
            return False
        direction = str(direction).lower()
        if direction not in ["up", "bullish", "buy"]:
            return False
        min_dur = float(self.config.get("min_expected_duration_min", 2))
        expected = float(latest_pred.get("expected_duration_min", 0))
        if expected < min_dur:
            logger.debug(f"{self.symbol}: predicción duration {expected} < min {min_dur}")
            return False
        avg_change = sum(recent_changes.values()) / max(1, len(recent_changes))
        min_avg_change = float(self.config.get("min_avg_change_pct", 0.0005))
        if avg_change < min_avg_change:
            logger.debug(f"{self.symbol}: avg_change {avg_change:.6f} < threshold {min_avg_change}")
            return False
        return True

    def _should_sell(self, latest_pred: Dict[str, Any]) -> bool:
        if not self.position:
            return False
        if latest_pred:
            direction = str(latest_pred.get("direction") or latest_pred.get("prediction") or "").lower()
            if direction in ["down", "sell", "bearish"]:
                logger.info(f"{self.symbol}: predicción indica bajada -> vender")
                return True
        profit_pct = self._compute_unrealized_profit_pct()
        take_profit = float(self.config.get("take_profit_pct", 0.004))
        stop_loss = float(self.config.get("stop_loss_pct", -0.01))
        commission_pct = float(self.config.get("commission_pct", 0.0005))
        adjusted_take_profit = take_profit + commission_pct * 2
        if profit_pct >= adjusted_take_profit:
            logger.info(f"{self.symbol}: profit {profit_pct:.4f} >= adjusted_take_profit {adjusted_take_profit:.4f} -> vender")
            return True
        if profit_pct <= stop_loss:
            logger.info(f"{self.symbol}: loss {profit_pct:.4f} <= stop_loss {stop_loss:.4f} -> vender (cut loss)")
            return True
        return False

    def _get_cash_available(self):
        try:
            acct = self._get_account()
            cash = float(acct.cash) if hasattr(acct, "cash") else float(acct.cash)
            return max(0.0, cash)
        except Exception as e:
            logger.exception(f"{self.symbol}: error obteniendo cash account: {e}")
            return 0.0

    def _run_loop(self):
        poll = float(self.config.get("polling_seconds", 5.0))
        logger.info(f"{self.symbol}: loop iniciado polling={poll}s dry_run={self.dry_run}")
        while not self._stop_event.is_set():
            try:
                self._refresh_position_cache()
                signals = self._get_memory_signals()
                latest_pred = signals[-1] if signals else None
                recent_changes = {}
                if signals and len(signals) >= 2:
                    closes = [s.get("close") for s in signals if s.get("close") is not None]
                    windows = {"1m": 1, "5m": 5, "10m": 10, "20m": 20, "30m": 30}
                    for k, w in windows.items():
                        if len(closes) > w:
                            recent_changes[k] = (closes[-1] - closes[-(w+1)]) / closes[-(w+1)]
                if self.position:
                    if self._should_sell(latest_pred):
                        self._place_market_sell_all()
                        time.sleep(0.5)
                        self._refresh_position_cache()
                else:
                    if self._should_buy(latest_pred, recent_changes):
                        cash = self._get_cash_available()
                        invest_usd = self._calc_order_size_usd(cash)
                        if invest_usd >= float(self.config.get("min_order_usd", 10.0)):
                            self._place_market_buy(invest_usd)
                            time.sleep(0.8)
                            self._refresh_position_cache()
                time.sleep(poll)
            except Exception as e:
                logger.exception(f"{self.symbol}: error en loop principal: {e}")
                time.sleep(poll)
        logger.info(f"{self.symbol}: loop terminado")


class MultiSymbolManager:
    def __init__(self, alpaca_api: tradeapi.REST, memory_system, config: Optional[Dict[str, Any]] = None, dry_run: bool = True):
        self.api = alpaca_api
        self.memory = memory_system
        self.config = config or {}
        self.dry_run = dry_run
        self.workers: Dict[str, SymbolWorker] = {}
        self.lock = threading.Lock()

    def add_symbol(self, symbol: str, alloc_pct: float):
        with self.lock:
            if symbol in self.workers:
                logger.info(f"{symbol}: ya existe worker")
                return
            worker = SymbolWorker(symbol, self.api, self.memory, alloc_pct, self.config, dry_run=self.dry_run)
            self.workers[symbol] = worker
            worker.start()

    def remove_symbol(self, symbol: str):
        with self.lock:
            w = self.workers.pop(symbol, None)
            if w:
                w.stop()
                logger.info(f"{symbol}: worker removido")

    def stop_all(self):
        with self.lock:
            for s, w in list(self.workers.items()):
                try:
                    w.stop()
                except Exception:
                    logger.exception(f"{s}: fallo al detener worker")
            self.workers.clear()

    def list_active(self) -> List[str]:
        return list(self.workers.keys())

    def rebalance_allocations(self, allocations: Dict[str, float]):
        with self.lock:
            for s, pct in allocations.items():
                if s in self.workers:
                    self.workers[s].alloc_pct = pct
                    logger.info(f"{s}: alloc_pct actualizado a {pct}")

    def get_status(self) -> Dict[str, Any]:
        status = {}
        for s, w in self.workers.items():
            status[s] = {
                "alloc_pct": w.alloc_pct,
                "position": w.position,
                "dry_run": w.dry_run
            }
        return status


class MultiSymbolTrader:
    def __init__(self):
        self.logger = logging.getLogger("MultiSymbolTrader")
        self.api = None
        self.manager = None
        self.memory_system = None
        self.is_running = False
        self._stop_event = threading.Event()
        self.trading_thread = None
        
        self.config = {
            "polling_seconds": 15,
            "min_order_usd": 50.0,
            "take_profit_pct": 0.02,
            "stop_loss_pct": -0.05,
            "commission_pct": 0.001,
            "min_expected_duration_min": 3,
            "min_avg_change_pct": 0.001,
            "max_positions": 8,
            "paper": True
        }
        
        self.symbol_allocations = {
            "BTCUSD": 0.25,
            "ETHUSD": 0.25,
            "SOLUSD": 0.15,
            "AVAXUSD": 0.15,
            "DOTUSD": 0.20
        }

    def initialize(self):
        try:
            api_key = os.environ.get('ALPACA_API_KEY')
            api_secret = os.environ.get('ALPACA_SECRET_KEY')
            paper_trading = os.environ.get('PAPER_TRADING', 'True').lower() == 'true'
            
            if not api_key or not api_secret:
                raise Exception("Credenciales no encontradas")
            
            base_url = "https://paper-api.alpaca.markets" if paper_trading else "https://api.alpaca.markets"
            self.api = tradeapi.REST(api_key, api_secret, base_url=base_url, api_version='v2')
            
            account = self.api.get_account()
            self.logger.info(f"Conectado a Alpaca. Paper: {paper_trading} | Balance: ${account.cash}")
            
            try:
                from modules.memory_system import MemorySystem
                self.memory_system = MemorySystem()
                self.logger.info("Memory System inicializado")
            except ImportError:
                self.memory_system = self._create_mock_memory()
                self.logger.info("Usando Mock Memory System")
            
            self.manager = MultiSymbolManager(
                alpaca_api=self.api,
                memory_system=self.memory_system,
                config=self.config,
                dry_run=False
            )
            
            self.config['paper'] = paper_trading
            return True
            
        except Exception as e:
            self.logger.error(f"Error inicializando: {e}")
            return False

    def _create_mock_memory(self):
        class MockMemorySystem:
            def get_signals(self, symbol):
                import random
                signals = []
                base_prices = {
                    "BTCUSD": 45000, "ETHUSD": 2500, "SOLUSD": 140, 
                    "AVAXUSD": 25, "DOTUSD": 4.5
                }
                base = base_prices.get(symbol, 100)
                for i in range(50):
                    signals.append({
                        'time': datetime.now() - timedelta(minutes=i*5),
                        'close': base + random.uniform(-base*0.1, base*0.1),
                        'direction': random.choice(['up', 'up', 'down', 'up']),
                        'expected_duration_min': random.randint(5, 45),
                        'confidence': random.uniform(0.75, 0.95)
                    })
                return signals
        return MockMemorySystem()

    def start_trading(self):
        if self.is_running:
            self.logger.info("Trading ya está corriendo")
            return True
            
        if not self.initialize():
            return False
        
        self.is_running = True
        self._stop_event.clear()
        
        for symbol, allocation in self.symbol_allocations.items():
            self.logger.info(f"Agregando {symbol} con asignación {allocation*100:.1f}%")
            self.manager.add_symbol(symbol, allocation)
        
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()
        
        self.logger.info("TRADING AUTOMÁTICO INICIADO - Bot corriendo 24/7!")
        return True

    def _trading_loop(self):
        while not self._stop_event.is_set() and self.is_running:
            try:
                status = self.manager.get_status()
                active_positions = sum(1 for s in status.values() if s.get('position'))
                cash = float(self.api.get_account().cash)
                
                self.logger.info(f"Estado: {len(status)} símbolos, {active_positions} posiciones, ${cash:.2f} cash")
                
                if active_positions < 2:
                    self._adjust_strategy_aggressive()
                
                if self._stop_event.wait(180):
                    break
                    
            except Exception as e:
                self.logger.error(f"Error en loop principal: {e}")
                time.sleep(60)

    def _adjust_strategy_aggressive(self):
        adjusted_config = self.config.copy()
        adjusted_config['take_profit_pct'] = 0.015
        adjusted_config['min_avg_change_pct'] = 0.0008
        
        for worker in self.manager.workers.values():
            worker.config.update(adjusted_config)
        
        self.logger.info("Estrategia ajustada: modo más agresivo activado")

    def stop_trading(self):
        self.logger.info("Deteniendo trading automático...")
        self.is_running = False
        self._stop_event.set()
        
        if self.manager:
            self.manager.stop_all()
        
        self.logger.info("Trading automático detenido")

    def get_status(self):
        if not self.manager:
            return {"status": "not_initialized", "positions": 0}
        
        status = self.manager.get_status()
        return {
            "status": "running" if self.is_running else "stopped",
            "symbols_monitored": len(status),
            "positions": sum(1 for s in status.values() if s.get('position')),
            "paper_trading": self.config.get('paper', True),
            "details": status
        }

    def start_24_7_trading(self):
        return self.start_trading()

    def run_continuous_trading(self):
        return self.start_trading()
