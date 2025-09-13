# modules/multi_symbol_trader.py
import threading
import time
import logging
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Any, Optional, List

import alpaca_trade_api as tradeapi
import pandas as pd

# Ajusta logger
logger = logging.getLogger("multi_symbol_trader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)


class SymbolWorker:
    """
    Worker por símbolo: vigila señales, decide comprar/vender, y ejecuta órdenes.
    Corre en su propio thread y consulta memory_system y alpaca_api.
    """

    def __init__(
        self,
        symbol: str,
        alpaca_api: tradeapi.REST,
        memory_system,
        account_alloc_pct: float,
        config: Dict[str, Any],
        dry_run: bool = True,
    ):
        """
        symbol: ticker (ej: 'AAPL')
        alpaca_api: instancia tradeapi.REST (ya autenticada)
        memory_system: objeto que almacena señales/historial (debe exponer get(symbol) o get_signals(symbol))
        account_alloc_pct: fracción del capital total a asignar a este símbolo (0..1)
        config: diccionario con parámetros (commission_pct, stop_loss_pct, take_profit_pct, min_order_usd, polling_seconds, paper)
        dry_run: si True no envía órdenes reales, solo registra lo que haría.
        """
        self.symbol = symbol
        self.api = alpaca_api
        self.memory = memory_system
        self.alloc_pct = float(account_alloc_pct)
        self.config = config
        self.dry_run = bool(dry_run)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.position = None  # cache local de posición para este símbolo (dict) o None

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
        """
        Lee la memoria para este símbolo. Soporta memory_system.get_signals(symbol) o memory_system.get(symbol).
        Debe devolver lista de dicts con al menos: time, close, prediction/direction, expected_duration_min.
        """
        try:
            if hasattr(self.memory, "get_signals"):
                return self.memory.get_signals(self.symbol) or []
            elif hasattr(self.memory, "get"):
                # puede devolver lista histórica
                return self.memory.get(self.symbol) or []
            else:
                logger.debug(f"{self.symbol}: memory_system no posee métodos esperados")
                return []
        except Exception as e:
            logger.exception(f"{self.symbol}: error leyendo memoria: {e}")
            return []

    def _refresh_position_cache(self):
        """Actualiza cache local de posición desde Alpaca (si existe)"""
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
            # no invalidar cache por completo
            self.position = None

    def _get_account(self):
        return self.api.get_account()

    def _calc_order_size_usd(self, cash_available_usd: float):
        """
        Decide cuanto USD invertir en este símbolo (según alloc_pct y min_order_usd).
        """
        desired = cash_available_usd * self.alloc_pct
        min_usd = float(self.config.get("min_order_usd", 10.0))
        invest_usd = max(min_usd, desired)
        return invest_usd

    def _place_market_buy(self, usd_amount: float):
        """
        Coloca market buy por USD usando Alpaca. Convierte a cantidad de acciones
        """
        try:
            # obtener precio actual (último trade)
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
        """Vende toda la posición en mercado"""
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
        """Calcula ganancia porcentual no realizada considerando avg_entry y market value"""
        if not self.position:
            return 0.0
        try:
            avg = float(self.position["avg_entry"])
            # pedir precio actual
            bar = self.api.get_latest_trade(self.symbol)
            last_price = float(bar.price)
            pct = (last_price - avg) / avg
            return pct
        except Exception as e:
            logger.exception(f"{self.symbol}: error calculando profit pct: {e}")
            return 0.0

    def _should_buy(self, latest_pred: Dict[str, Any], recent_changes: Dict[str, float]) -> bool:
        """
        Heurística para decidir compra:
         - No tener posición.
         - Predicción de 'up' (o 'bullish') y expected_duration >= min_duration_min.
         - Cambio promedio positivo y suficiente.
        """
        if self.position:
            return False

        if not latest_pred:
            return False

        direction = latest_pred.get("direction") or latest_pred.get("prediction") or latest_pred.get("trend")
        if direction is None:
            return False

        # normalizar
        direction = str(direction).lower()
        if direction not in ["up", "bullish", "buy"]:
            return False

        # duración mínima requerida
        min_dur = float(self.config.get("min_expected_duration_min", 2))
        expected = float(latest_pred.get("expected_duration_min", 0))
        if expected < min_dur:
            logger.debug(f"{self.symbol}: predicción duration {expected} < min {min_dur}")
            return False

        # comprobar momentum básico (recent_changes es dict de windows->pct)
        avg_change = sum(recent_changes.values()) / max(1, len(recent_changes))
        min_avg_change = float(self.config.get("min_avg_change_pct", 0.0005))  # 0.05%
        if avg_change < min_avg_change:
            logger.debug(f"{self.symbol}: avg_change {avg_change:.6f} < threshold {min_avg_change}")
            return False

        return True

    def _should_sell(self, latest_pred: Dict[str, Any]) -> bool:
        """
        Heurística para vender:
         - Si predicción actual sugiere 'down' o 'sell'
         - Si ganancia realizada supera take_profit ajustada por comisión
         - Si pérdida supera stop_loss
        """
        if not self.position:
            return False

        # predicción contraria
        if latest_pred:
            direction = str(latest_pred.get("direction") or latest_pred.get("prediction") or "").lower()
            if direction in ["down", "sell", "bearish"]:
                logger.info(f"{self.symbol}: predicción indica bajada -> vender")
                return True

        # verificar profit/loss %
        profit_pct = self._compute_unrealized_profit_pct()
        take_profit = float(self.config.get("take_profit_pct", 0.004))  # ej 0.4%
        stop_loss = float(self.config.get("stop_loss_pct", -0.01))      # ej -1%
        commission_pct = float(self.config.get("commission_pct", 0.0005))  # ej 0.05%

        # ajustar ganancia target por comisión (necesitamos más que comisión para ser rentable)
        adjusted_take_profit = take_profit + commission_pct * 2  # ida y vuelta
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
            cash = float(acct.cash) if hasattr(acct, "cash") else float(acct.cash)  # alpaca object
            return max(0.0, cash)
        except Exception as e:
            logger.exception(f"{self.symbol}: error obteniendo cash account: {e}")
            # fallback conservador
            return 0.0

    def _run_loop(self):
        poll = float(self.config.get("polling_seconds", 5.0))
        logger.info(f"{self.symbol}: loop iniciado polling={poll}s dry_run={self.dry_run}")

        while not self._stop_event.is_set():
            try:
                # actualizar posición cache
                self._refresh_position_cache()

                # leer señales desde memoria
                signals = self._get_memory_signals()
                latest_pred = signals[-1] if signals else None

                # construir recent_changes: cambios porcentuales en distintas ventanas (si datos históricos presentes)
                recent_changes = {}
                if signals and len(signals) >= 2:
                    closes = [s.get("close") for s in signals if s.get("close") is not None]
                    # windows mapping: use last 1,5,10... if exists
                    windows = {"1m": 1, "5m": 5, "10m": 10, "20m": 20, "30m": 30}
                    for k, w in windows.items():
                        if len(closes) > w:
                            recent_changes[k] = (closes[-1] - closes[-(w+1)]) / closes[-(w+1)]
                # decisión: vender primero si ya hay posición y should_sell true
                if self.position:
                    if self._should_sell(latest_pred):
                        self._place_market_sell_all()
                        # tras vender refrescar cache
                        time.sleep(0.5)
                        self._refresh_position_cache()
                        # continue loop
                else:
                    # no hay posición -> evaluar compra
                    if self._should_buy(latest_pred, recent_changes):
                        # calcular orden según capital
                        cash = self._get_cash_available()
                        invest_usd = self._calc_order_size_usd(cash)
                        if invest_usd >= float(self.config.get("min_order_usd", 10.0)):
                            self._place_market_buy(invest_usd)
                            # esperar un poco para que la orden rellene y actualizar cache
                            time.sleep(0.8)
                            self._refresh_position_cache()

                # espera
                time.sleep(poll)
            except Exception as e:
                logger.exception(f"{self.symbol}: error en loop principal: {e}")
                time.sleep(poll)

        logger.info(f"{self.symbol}: loop terminado")


class MultiSymbolManager:
    """
    Manager general que crea workers por símbolo y orquesta la asignación de capital.
    """

    def __init__(self, alpaca_api: tradeapi.REST, memory_system, config: Optional[Dict[str, Any]] = None, dry_run: bool = True):
        self.api = alpaca_api
        self.memory = memory_system
        self.config = config or {}
        self.dry_run = dry_run
        self.workers: Dict[str, SymbolWorker] = {}
        self.lock = threading.Lock()

    def add_symbol(self, symbol: str, alloc_pct: float):
        """
        Agrega y arranca un worker para un símbolo.
        alloc_pct: fracción del capital a asignar (0..1). La suma idealmente <= 1.0
        """
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
        """
        Cambia porcentajes asignados a cada worker. allocations: {symbol: alloc_pct}
        """
        with self.lock:
            for s, pct in allocations.items():
                if s in self.workers:
                    self.workers[s].alloc_pct = pct
                    logger.info(f"{s}: alloc_pct actualizado a {pct}")

    def get_status(self) -> Dict[str, Any]:
        """Estado simple de manager y workers"""
        status = {}
        for s, w in self.workers.items():
            status[s] = {
                "alloc_pct": w.alloc_pct,
                "position": w.position,
                "dry_run": w.dry_run
            }
        return status
