import pytest
import json
import base64
from json_memory.encryption import MemoryEncryption, EncryptedValue, create_encryption

def test_encrypt_basic_string():
    enc = MemoryEncryption("test-password")
    value = "super secret data"

    encrypted = enc.encrypt(value)

    assert isinstance(encrypted, EncryptedValue)
    assert encrypted.key_id == enc.key_id
    assert encrypted.ciphertext
    assert encrypted.iv

def test_encrypt_complex_data():
    enc = MemoryEncryption("test-password")
    value = {
        "user_id": 123,
        "api_keys": ["sk-1", "sk-2"],
        "is_active": True
    }

    encrypted = enc.encrypt(value)

    assert isinstance(encrypted, EncryptedValue)
    assert encrypted.ciphertext

def test_encrypt_custom_key():
    enc = MemoryEncryption()
    # Add a custom key
    custom_key = b'0' * 32
    enc.add_key("custom-key-1", custom_key)

    value = "test value"
    encrypted = enc.encrypt(value, key_id="custom-key-1")

    assert isinstance(encrypted, EncryptedValue)
    assert encrypted.key_id == "custom-key-1"

def test_encrypt_unknown_key():
    enc = MemoryEncryption()

    with pytest.raises(ValueError, match="Key not found: unknown-key"):
        enc.encrypt("test value", key_id="unknown-key")

def test_encrypt_decrypt_integration():
    enc = MemoryEncryption("integration-password")

    values_to_test = [
        "simple string",
        12345,
        True,
        False,
        None,
        ["list", "of", "items"],
        {"nested": {"dict": True}}
    ]

    for value in values_to_test:
        encrypted = enc.encrypt(value)
        decrypted = enc.decrypt(encrypted)
        assert decrypted == value
