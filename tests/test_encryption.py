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
