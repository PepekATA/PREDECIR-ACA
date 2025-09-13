# credentials_manager.py
import json
import os
from pathlib import Path
from datetime import datetime
import streamlit as st

class CredentialsManager:
    def __init__(self):
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        self.credentials_file = self.data_dir / 'credentials.json'
    
    def save_credentials(self, api_key, api_secret, paper_trading=True):
        try:
            credentials = {
                'api_key': api_key,
                'api_secret': api_secret,
                'paper_trading': paper_trading,
                'saved_at': str(datetime.now())
            }
            with open(self.credentials_file, 'w') as f:
                json.dump(credentials, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Error guardando credenciales: {e}")
            return False
    
    def load_credentials(self):
        try:
            if self.credentials_file.exists():
                with open(self.credentials_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            st.error(f"Error cargando credenciales: {e}")
            return None
    
    def credentials_exist(self):
        return self.credentials_file.exists()
    
    def delete_credentials(self):
        try:
            if self.credentials_file.exists():
                self.credentials_file.unlink()
            return True
        except Exception as e:
            st.error(f"Error eliminando credenciales: {e}")
            return False
