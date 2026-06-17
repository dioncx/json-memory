from json_memory.smart import SmartMemory


def test_auto_protect_user_paths():
    mem = SmartMemory()
    mem.remember("user.name", "Alice")
    assert mem._meta["user.name"].protected is True

    mem.remember("app.setting", "dark_mode")
    assert mem._meta["app.setting"].protected is False
