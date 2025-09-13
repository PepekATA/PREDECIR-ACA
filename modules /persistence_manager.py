"""
Gestor de Persistencia y Credenciales
Maneja el almacenamiento seguro de credenciales y estado del bot
"""

import os
import json
import aiofiles
import asyncio
from datetime import datetime
from pathlib import Path
from cryptography.fernet import Fernet
import logging

logger = logging.getLogger("PersistenceManager")

class PersistenceManager:
    """Maneja persistencia de credenciales, estado y configuración"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.config_dir = Path("config")
        self.credentials_file = self.data_dir / "credentials.enc"
        self.key_file = self.data_dir / "master.key"
        self.state_file = self.data_dir / "bot_state.json"
        
        self.cipher = None
        self.bot_state = {}
        
    async def initialize(self):
        """Inicializar sistema de persistencia"""
        try:
            # Crear directorios
            self.data_dir.mkdir(exist_ok=True)
            self.config_dir.mkdir(exist_ok=True)
            
            # Inicializar cifrado
            await self.initialize_encryption()
            
            # Cargar estado del bot
            await self.load_bot_state()
            
            logger.info("✅ Sistema de persistencia inicializado")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando persistencia: {e}")
            return False
    
    async def initialize_encryption(self):
        """Inicializar sistema de cifrado"""
        if self.key_file.exists():
            # Cargar clave existente
            async with aiofiles.open(self.key_file, 'rb') as f:
                key = await f.read()
        else:
            # Generar nueva clave
            key = Fernet.generate_key()
            async with aiofiles.open(self.key_file, 'wb') as f:
                await f.write(key)
            logger.info("🔐 Nueva clave de cifrado generada")
        
        self.cipher = Fernet(key)
    
    async def save_alpaca_credentials(self, credentials):
        """Guardar credenciales de Alpaca cifradas"""
        try:
            # Cifrar credenciales
            credentials_json = json.dumps(credentials)
            encrypted_data = self.cipher.encrypt(credentials_json.encode())
            
            # Guardar archivo cifrado
            async with aiofiles.open(self.credentials_file, 'wb') as f:
                await f.write(encrypted_data)
            
            logger.info("🔐 Credenciales guardadas y cifradas")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error guardando credenciales: {e}")
            return False
    
    def get_alpaca_credentials(self):
        """Obtener credenciales descifradas"""
        try:
            if not self.credentials_file.exists():
                return None
            
            # Leer y descifrar
            with open(self.credentials_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher.decrypt(encrypted_data)
            credentials = json.loads(decrypted_data.decode())
            
            return credentials
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo credenciales: {e}")
            return None
    
    async def save_bot_state(self, state_update):
        """Guardar estado del bot"""
        try:
            self.bot_state.update(state_update)
            self.bot_state['last_save'] = datetime.now().isoformat()
            
            async with aiofiles.open(self.state_file, 'w') as f:
                await f.write(json.dumps(self.bot_state, indent=2))
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error guardando estado: {e}")
            return False
    
    async def load_bot_state(self):
        """Cargar estado del bot"""
        try:
            if self.state_file.exists():
                async with aiofiles.open(self.state_file, 'r') as f:
                    content = await f.read()
                    self.bot_state = json.loads(content)
            else:
                self.bot_state = {
                    'created': datetime.now().isoformat(),
                    'version': '1.0.0'
                }
            
            return self.bot_state
            
        except Exception as e:
            logger.error(f"❌ Error cargando estado: {e}")
            return {}
    
    def get_bot_state(self):
        """Obtener estado actual del bot"""
        return self.bot_state.copy()
    
    async def cleanup_old_data(self, days_to_keep=30):
        """Limpiar datos antiguos"""
        try:
            cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
            
            # Limpiar logs antiguos
            logs_dir = Path("logs")
            if logs_dir.exists():
                for log_file in logs_dir.glob("*.log"):
                    if log_file.stat().st_mtime < cutoff_date:
                        log_file.unlink()
                        logger.info(f"🗑️  Log eliminado: {log_file.name}")
            
            logger.info(f"🧹 Limpieza completada - datos > {days_to_keep} días eliminados")
            
        except Exception as e:
            logger.error(f"❌ Error en limpieza: {e}")
    
    async def backup_data(self):
        """Crear respaldo de datos importantes"""
        try:
            backup_dir = self.data_dir / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"backup_{timestamp}.json"
            
            backup_data = {
                'timestamp': datetime.now().isoformat(),
                'bot_state': self.bot_state,
                'credentials_exist': self.credentials_file.exists()
            }
            
            async with aiofiles.open(backup_file, 'w') as f:
                await f.write(json.dumps(backup_data, indent=2))
            
            logger.info(f"💾 Respaldo creado: {backup_file.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creando respaldo: {e}")
            return False
