from json_memory.encryption import create_encryption
enc = create_encryption("test")
e = enc.encrypt("val")
print(f"IV: |{e.iv}|")
