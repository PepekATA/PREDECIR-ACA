import streamlit as st
import json
import os
from pathlib import Path
from cryptography.fernet import Fernet

class CredentialsManager:
    """Gestor seguro de credenciales para Streamlit"""
    
    def __init__(self):
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        self.credentials_file = self.data_dir / 'credentials.enc'
        self.key_file = self.data_dir / '.key'
        
    def generate_key(self):
        """Generar clave de cifrado"""
        if not self.key_file.exists():
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            # Ocultar archivo en sistemas Unix
            if os.name != 'nt':
                os.chmod(self.key_file, 0o600)
        
        with open(self.key_file, 'rb') as f:
            return f.read()
    
    def save_credentials(self, api_key, api_secret, paper_trading=True):
        """Guardar credenciales cifradas"""
        try:
            key = self.generate_key()
            fernet = Fernet(key)
            
            credentials = {
                'api_key': api_key,
                'api_secret': api_secret,
                'paper_trading': paper_trading,
                'saved_at': str(datetime.now())
            }
            
            encrypted_data = fernet.encrypt(json.dumps(credentials).encode())
            
            with open(self.credentials_file, 'wb') as f:
                f.write(encrypted_data)
            
            return True
            
        except Exception as e:
            st.error(f"Error guardando credenciales: {e}")
            return False
    
    def load_credentials(self):
        """Cargar credenciales cifradas"""
        try:
            if not self.credentials_file.exists():
                return None
            
            key = self.generate_key()
            fernet = Fernet(key)
            
            with open(self.credentials_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = fernet.decrypt(encrypted_data)
            credentials = json.loads(decrypted_data.decode())
            
            return credentials
            
        except Exception as e:
            st.error(f"Error cargando credenciales: {e}")
            return None
    
    def credentials_exist(self):
        """Verificar si existen credenciales guardadas"""
        return self.credentials_file.exists()
    
    def delete_credentials(self):
        """Eliminar credenciales guardadas"""
        try:
            if self.credentials_file.exists():
                self.credentials_file.unlink()
            if self.key_file.exists():
                self.key_file.unlink()
            return True
        except Exception as e:
            st.error(f"Error eliminando credenciales: {e}")
            return False
