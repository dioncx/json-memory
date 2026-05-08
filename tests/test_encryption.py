"""Tests for MemoryEncryption."""

import pytest
import base64
import json
from json_memory.encryption import MemoryEncryption, EncryptedValue, create_encryption


class TestMemoryEncryption:
    def test_encryption_roundtrip(self):
        """Verify that encrypting and then decrypting returns the original value."""
        enc = MemoryEncryption("test-password")

        test_cases = [
            "simple string",
            {"key": "value", "nested": [1, 2, 3]},
            [1, 2, 3, "mixed"],
            12345,
            True,
            None
        ]

        for original in test_cases:
            encrypted = enc.encrypt(original)
            assert isinstance(encrypted, EncryptedValue)

            decrypted = enc.decrypt(encrypted)
            assert decrypted == original

    def test_decrypt_with_missing_key(self):
        """Verify that ValueError is raised when trying to decrypt with a missing key_id."""
        enc = MemoryEncryption("test-password")
        encrypted = enc.encrypt("secret")

        # Create a new encryption instance with a different key store
        enc2 = MemoryEncryption("different-password")

        with pytest.raises(ValueError, match="Key not found"):
            enc2.decrypt(encrypted)

    def test_decrypt_with_wrong_key(self):
        """Verify that decrypting with the wrong key fails or returns garbage."""
        enc1 = MemoryEncryption("password-1")
        enc2 = MemoryEncryption("password-2")

        original = {"secret": "data"}
        encrypted = enc1.encrypt(original)

        # Manually swap the key_id so enc2 tries to decrypt it with its own key
        encrypted.key_id = enc2.key_id

        # XOR with wrong key will yield garbage, likely failing JSON decoding
        with pytest.raises((json.JSONDecodeError, UnicodeDecodeError)):
            enc2.decrypt(encrypted)

    def test_to_from_dict_roundtrip(self):
        """Verify that to_dict and from_dict preserve EncryptedValue data."""
        enc = MemoryEncryption("test-password")
        original = "some data"
        encrypted = enc.encrypt(original)

        as_dict = enc.to_dict(encrypted)
        assert as_dict["__encrypted__"] is True
        assert "ciphertext" in as_dict
        assert "iv" in as_dict
        assert "key_id" in as_dict

        recovered = enc.from_dict(as_dict)
        assert recovered == encrypted

        assert enc.decrypt(recovered) == original

    def test_key_rotation_decryption(self):
        """Verify that values encrypted with old keys can still be decrypted after rotation."""
        enc = MemoryEncryption("test-password")
        old_key_id = enc.key_id

        original = "persistent secret"
        encrypted = enc.encrypt(original)

        # Rotate key
        new_key_id = enc.rotate_key()
        assert new_key_id != old_key_id
        assert enc.key_id == new_key_id

        # Should still be able to decrypt with the old key_id present in key_store
        assert enc.decrypt(encrypted) == original

        # New encryption uses new key
        new_encrypted = enc.encrypt("new secret")
        assert new_encrypted.key_id == new_key_id

    def test_is_encrypted(self):
        """Verify the is_encrypted helper method."""
        enc = MemoryEncryption("test-password")
        encrypted = enc.encrypt("data")
        encrypted_dict = enc.to_dict(encrypted)

        assert enc.is_encrypted(encrypted_dict) is True
        assert enc.is_encrypted({"not": "encrypted"}) is False
        assert enc.is_encrypted("not a dict") is False

    def test_create_encryption_convenience(self):
        """Verify the convenience function."""
        enc = create_encryption("password")
        assert isinstance(enc, MemoryEncryption)

        # Test basic functionality
        data = "test"
        assert enc.decrypt(enc.encrypt(data)) == data

    def test_import_export_key(self):
        """Verify key export and import."""
        enc1 = MemoryEncryption("password")
        key_id = enc1.key_id
        key_b64 = enc1.export_key(key_id)

        enc2 = MemoryEncryption("other")
        new_key_id = enc2.import_key(key_b64)

        assert new_key_id == key_id

        # Test cross-instance decryption
        encrypted = enc1.encrypt("cross-instance")
        assert enc2.decrypt(encrypted) == "cross-instance"
