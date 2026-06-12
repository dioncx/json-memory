import os
import json
import sqlite3
import pytest
from unittest.mock import patch

from json_memory.adapters import FileAdapter, SQLiteAdapter

def test_file_adapter(tmp_path):
    # Initialize with a path
    file_path = tmp_path / "test_memory.json"
    adapter = FileAdapter(str(file_path))

    # Verify load returns None initially
    assert adapter.load() is None

    # Call save and verify
    data1 = {'key': 'value'}
    adapter.save(data1)
    assert adapter.load() == data1

    # Overwrite and verify
    data2 = {'new_key': 123}
    adapter.save(data2)
    assert adapter.load() == data2


def test_sqlite_adapter(tmp_path):
    # Initialize with a path
    db_path = tmp_path / "test_memory.db"
    adapter = SQLiteAdapter(str(db_path))

    # Verify load returns None initially
    assert adapter.load() is None

    # Call save and verify
    data1 = {'test': 123}
    adapter.save(data1)
    assert adapter.load() == data1

    # Overwrite and verify
    data2 = {'another': 'data'}
    adapter.save(data2)
    assert adapter.load() == data2

    # Test missing file
    os.remove(str(db_path))
    assert adapter.load() is None

    # Recreate to test error
    adapter = SQLiteAdapter(str(db_path))
    adapter.save(data1)

    # Test loading when SQLite throws an error
    with patch('sqlite3.connect', side_effect=sqlite3.Error("Mock error")):
        assert adapter.load() is None
