import pytest
from json_memory.encryption import MemoryEncryption, EncryptedValue, create_encryption

def test_encrypt_decrypt_roundtrip():
    """Test that a value can be encrypted and then decrypted back to its original form."""
    enc = create_encryption("test-master-key")

    payload = {
        "secret_token": "sk_test_12345",
        "user_id": 42,
        "is_admin": True,
        "tags": ["secure", "auth"],
    }

    # Encrypt
    encrypted = enc.encrypt(payload)
    assert isinstance(encrypted, EncryptedValue)
    assert encrypted.iv == ""
    assert encrypted.key_id == enc.key_id
    assert encrypted.ciphertext != ""

    # Verify the ciphertext is base64 Fernet token starting with 'gAAAAA'
    assert encrypted.ciphertext.startswith("gAAAAA")

    # Decrypt
    decrypted = enc.decrypt(encrypted)

    assert decrypted == payload
    assert decrypted["secret_token"] == "sk_test_12345"
    assert decrypted["user_id"] == 42


def test_multiple_keys():
    """Test that rotating keys correctly allows decryption with old and new keys."""
    enc = create_encryption("initial-key")

    payload1 = {"data": "first payload"}
    enc_value1 = enc.encrypt(payload1)

    # Rotate key
    old_key_id = enc.key_id
    new_key_id = enc.rotate_key()

    assert new_key_id != old_key_id

    payload2 = {"data": "second payload"}
    enc_value2 = enc.encrypt(payload2)

    # Verify enc_value2 uses new key
    assert enc_value2.key_id == new_key_id

    # Decrypt both
    dec1 = enc.decrypt(enc_value1)
    dec2 = enc.decrypt(enc_value2)

    assert dec1 == payload1
    assert dec2 == payload2


def test_invalid_key():
    """Test that decryption fails with missing or invalid keys."""
    enc = create_encryption("valid-key")

    payload = {"data": "secret"}
    encrypted = enc.encrypt(payload)

    # Create an invalid EncryptedValue (wrong key_id)
    invalid_encrypted = EncryptedValue(
        ciphertext=encrypted.ciphertext, iv=encrypted.iv, key_id="nonexistent_key_id"
    )

    with pytest.raises(ValueError, match="Key not found: nonexistent_key_id"):
        enc.decrypt(invalid_encrypted)

def test_encrypt_basic_string():
    enc = MemoryEncryption("test-password")
    value = "super secret data"

    encrypted = enc.encrypt(value)

    assert isinstance(encrypted, EncryptedValue)
    assert encrypted.key_id == enc.key_id
    assert encrypted.ciphertext
    assert encrypted.iv == ""

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
    assert encrypted.iv == ""

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

def test_export_default_key():
    enc = MemoryEncryption("test-password")
    exported = enc.export_key()

    # Verify we can import it back and it matches the original master key
    assert exported is not None
    assert len(exported) > 0
    import base64
    assert base64.b64decode(exported) == enc.master_key

def test_export_specific_key():
    enc = MemoryEncryption("test-password")

    # Add a custom key
    custom_key = b'1' * 32
    enc.add_key("custom-key-2", custom_key)

    exported = enc.export_key("custom-key-2")

    import base64
    assert base64.b64decode(exported) == custom_key

def test_export_invalid_key():
    enc = MemoryEncryption("test-password")

    with pytest.raises(ValueError, match="Key not found: unknown-key"):
        enc.export_key("unknown-key")
