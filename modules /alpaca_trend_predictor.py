# modules/alpaca_trend_predictor.py
import os
import json
import base64
import pickle
from cryptography.fernet import Fernet
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi

class AlpacaConnector:
    """Manejador de credenciales y conexión con Alpaca"""
    def __init__(self, cred_file="alpaca_credentials.enc", key_file="alpaca_key.key"):
        self.cred_file = cred_file
        self.key_file = key_file
        self.api = None

    def _generate_key(self):
        key = Fernet.generate_key()
        with open(self.key_file, "wb") as f:
            f.write(key)
        return key

    def _load_key(self):
        if not os.path.exists(self.key_file):
            return self._generate_key()
        with open(self.key_file, "rb") as f:
            return f.read()

    def save_credentials(self, api_key, api_secret, base_url="https://paper-api.alpaca.markets"):
        key = self._load_key()
        f = Fernet(key)
        creds = {"api_key": api_key, "api_secret": api_secret, "base_url": base_url}
        token = f.encrypt(json.dumps(creds).encode())
        with open(self.cred_file, "wb") as f_out:
            f_out.write(token)

    def load_credentials(self):
        if not os.path.exists(self.cred_file):
            return None
        key = self._load_key()
        f = Fernet(key)
        with open(self.cred_file, "rb") as f_in:
            token = f_in.read()
        creds = json.loads(f.decrypt(token).decode())
        return creds

    def connect(self):
        creds = self.load_credentials()
        if not creds:
            return None
        self.api = tradeapi.REST(
            creds["api_key"], creds["api_secret"], creds["base_url"]
        )
        return self.api


class TrendPredictor:
    """Predicción simple de tendencias usando memoria de señales"""

    def __init__(self, memory_system):
        self.memory = memory_system

    def predict_trend(self, symbol, timeframe="1Min"):
        """
        Analiza memoria de señales para predecir tendencia futura
        Basado en ventanas de 1, 5, 10, 20, 30 min
        """
        try:
            data = self.memory.get(symbol, [])
            if len(data) < 30:
                return {"error": "Datos insuficientes en memoria"}

            df = pd.DataFrame(data)
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time")

            close = df["close"].values

            # Cambios porcentuales pasados
            changes = {
                "1m": (close[-1] - close[-2]) / close[-2],
                "5m": (close[-1] - close[-5]) / close[-5] if len(close) >= 5 else 0,
                "10m": (close[-1] - close[-10]) / close[-10] if len(close) >= 10 else 0,
                "20m": (close[-1] - close[-20]) / close[-20] if len(close) >= 20 else 0,
                "30m": (close[-1] - close[-30]) / close[-30] if len(close) >= 30 else 0,
            }

            avg_change = np.mean(list(changes.values()))

            if avg_change > 0.001:  # 0.1%
                direction = "up"
                color = "green"
            elif avg_change < -0.001:
                direction = "down"
                color = "red"
            else:
                direction = "neutral"
                color = "gray"

            # Duración estimada heurística
            if direction == "up":
                duration = int(abs(avg_change) * 1000)  # en minutos
            elif direction == "down":
                duration = int(abs(avg_change) * 800)
            else:
                duration = 1

            return {
                "symbol": symbol,
                "direction": direction,
                "color": color,
                "expected_duration_min": max(1, min(duration, 60)),
                "timestamp": datetime.now().isoformat(),
                "changes": changes,
            }

        except Exception as e:
            return {"error": str(e)}
