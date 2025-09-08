from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
from typing import Union

class PHIEncryption:
    """HIPAA-compliant encryption for Protected Health Information"""
    
    def __init__(self, password: str = None):
        if password is None:
            password = os.getenv("SECRET_KEY", "default-key")
        
        password_bytes = password.encode()
        salt = b'healthcare_rag_salt_2024'
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        self.cipher = Fernet(key)
    
    def encrypt_phi(self, data: Union[str, dict]) -> str:
        """Encrypt PHI data"""
        if isinstance(data, dict):
            data = str(data)
        
        encrypted_data = self.cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_phi(self, encrypted_data: str) -> str:
        """Decrypt PHI data"""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            raise ValueError(f"Decryption failed: {str(e)}")
    
    def anonymize_patient_id(self, patient_id: str) -> str:
        """Create anonymized patient identifier"""
        hash_digest = hashes.Hash(hashes.SHA256())
        hash_digest.update(patient_id.encode())
        return base64.urlsafe_b64encode(hash_digest.finalize())[:16].decode()

# Global encryption instance
phi_encryption = PHIEncryption()
