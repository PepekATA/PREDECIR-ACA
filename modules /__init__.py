# Import the actual classes from their respective files
from .ai_predictor import AIPredictor
from .administrador_de_datos import AdministradorDeDatos
from .panel import Panel

# Expose them under the names that main.py is expecting
ForexPredictor = AIPredictor
DataStorage = AdministradorDeDatos
Dashboard = Panel
