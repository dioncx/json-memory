"""
Memory Encryption — Encrypt sensitive memory values.

This module provides encryption capabilities for sensitive memory data:
- Per-value encryption
- Key management
- Secure storage
"""

import base64
import hashlib
import os
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class EncryptedValue:
    """Represents an encrypted value."""
    ciphertext: str  # Base64 encoded
    iv: str  # Base64 encoded initialization vector
    key_id: str  # Identifier for the key used


class MemoryEncryption:
    """Encrypt and decrypt memory values."""
    
    def __init__(self, master_key: str = None):
        """Initialize encryption with optional master key.
        
        Args:
            master_key: Optional master key for encryption (will be derived if not provided)
        """
        if master_key:
            self.master_key = self._derive_key(master_key)
        else:
            self.master_key = self._generate_key()
        
        self.key_id = hashlib.sha256(self.master_key).hexdigest()[:16]
        self._key_store: Dict[str, bytes] = {self.key_id: self.master_key}
    
    def _derive_key(self, password: str) -> bytes:
        """Derive a key from a password using PBKDF2."""
        salt = b'json-memory-salt'  # In production, use random salt per key
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    
    def _generate_key(self) -> bytes:
        """Generate a random encryption key."""
        return os.urandom(32)
    
    def encrypt(self, value: Any, key_id: str = None) -> EncryptedValue:
        """Encrypt a value.
        
        Args:
            value: Value to encrypt (will be JSON serialized)
            key_id: Optional key ID to use (defaults to master key)
            
        Returns:
            EncryptedValue object
        """
        if key_id is None:
            key_id = self.key_id
        
        key = self._key_store.get(key_id)
        if not key:
            raise ValueError(f"Key not found: {key_id}")
        
        # Serialize value
        plaintext = json.dumps(value, default=str).encode('utf-8')
        
        # Generate IV
        iv = os.urandom(16)
        
        # Simple XOR encryption (for demo - use AES in production)
        # In production, use: from cryptography.fernet import Fernet
        ciphertext = self._xor_encrypt(plaintext, key, iv)
        
        return EncryptedValue(
            ciphertext=base64.b64encode(ciphertext).decode('utf-8'),
            iv=base64.b64encode(iv).decode('utf-8'),
            key_id=key_id
        )
    
    def decrypt(self, encrypted: EncryptedValue) -> Any:
        """Decrypt an encrypted value.
        
        Args:
            encrypted: EncryptedValue object
            
        Returns:
            Decrypted value
        """
        key = self._key_store.get(encrypted.key_id)
        if not key:
            raise ValueError(f"Key not found: {encrypted.key_id}")
        
        # Decode from base64
        ciphertext = base64.b64decode(encrypted.ciphertext)
        iv = base64.b64decode(encrypted.iv)
        
        # Decrypt
        plaintext = self._xor_decrypt(ciphertext, key, iv)
        
        # Deserialize
        return json.loads(plaintext.decode('utf-8'))
    
    def _xor_encrypt(self, plaintext: bytes, key: bytes, iv: bytes) -> bytes:
        """Simple XOR encryption (for demo purposes)."""
        # In production, use proper encryption like AES-GCM
        key_stream = self._generate_key_stream(key, iv, len(plaintext))
        return bytes(a ^ b for a, b in zip(plaintext, key_stream))
    
    def _xor_decrypt(self, ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
        """Simple XOR decryption (same as encryption)."""
        return self._xor_encrypt(ciphertext, key, iv)
    
    def _generate_key_stream(self, key: bytes, iv: bytes, length: int) -> bytes:
        """Generate a key stream from key and iv."""
        stream = b''
        counter = 0
        while len(stream) < length:
            data = key + iv + counter.to_bytes(4, 'big')
            stream += hashlib.sha256(data).digest()
            counter += 1
        return stream[:length]
    
    def add_key(self, key_id: str, key: bytes):
        """Add a key to the key store.
        
        Args:
            key_id: Key identifier
            key: Key bytes
        """
        self._key_store[key_id] = key
    
    def rotate_key(self, new_key: bytes = None) -> str:
        """Rotate to a new master key.
        
        Args:
            new_key: Optional new key (generated if not provided)
            
        Returns:
            New key ID
        """
        if new_key is None:
            new_key = self._generate_key()
        
        new_key_id = hashlib.sha256(new_key).hexdigest()[:16]
        self._key_store[new_key_id] = new_key
        self.master_key = new_key
        self.key_id = new_key_id
        
        return new_key_id
    
    def export_key(self, key_id: str = None) -> str:
        """Export a key as base64 string.
        
        Args:
            key_id: Key ID to export (defaults to master key)
            
        Returns:
            Base64 encoded key
        """
        if key_id is None:
            key_id = self.key_id
        
        key = self._key_store.get(key_id)
        if not key:
            raise ValueError(f"Key not found: {key_id}")
        
        return base64.b64encode(key).decode('utf-8')
    
    def import_key(self, key_base64: str, key_id: str = None) -> str:
        """Import a key from base64 string.
        
        Args:
            key_base64: Base64 encoded key
            key_id: Optional key ID (derived from key if not provided)
            
        Returns:
            Key ID
        """
        key = base64.b64decode(key_base64)
        
        if key_id is None:
            key_id = hashlib.sha256(key).hexdigest()[:16]
        
        self._key_store[key_id] = key
        return key_id
    
    def is_encrypted(self, value: Any) -> bool:
        """Check if a value is encrypted.
        
        Args:
            value: Value to check
            
        Returns:
            True if value is an EncryptedValue dict
        """
        if isinstance(value, dict):
            return all(k in value for k in ['ciphertext', 'iv', 'key_id'])
        return False
    
    def to_dict(self, encrypted: EncryptedValue) -> Dict[str, str]:
        """Convert EncryptedValue to dict for storage.
        
        Args:
            encrypted: EncryptedValue object
            
        Returns:
            Dict representation
        """
        return {
            'ciphertext': encrypted.ciphertext,
            'iv': encrypted.iv,
            'key_id': encrypted.key_id,
            '__encrypted__': True
        }
    
    def from_dict(self, data: Dict[str, str]) -> EncryptedValue:
        """Create EncryptedValue from dict.
        
        Args:
            data: Dict representation
            
        Returns:
            EncryptedValue object
        """
        return EncryptedValue(
            ciphertext=data['ciphertext'],
            iv=data['iv'],
            key_id=data['key_id']
        )


def create_encryption(master_key: str = None) -> MemoryEncryption:
    """Convenience function to create MemoryEncryption."""
    return MemoryEncryption(master_key=master_key)


# Test the encryption
if __name__ == "__main__":
    # Create encryption with password
    enc = MemoryEncryption("my-secret-password")
    
    # Encrypt a value
    secret = {"api_key": "sk-abc123", "password": "super-secret"}
    encrypted = enc.encrypt(secret)
    
    print(f"Original: {secret}")
    print(f"Encrypted: {encrypted.ciphertext[:50]}...")
    
    # Decrypt
    decrypted = enc.decrypt(encrypted)
    print(f"Decrypted: {decrypted}")
    print(f"Match: {secret == decrypted}")