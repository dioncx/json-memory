import pytest
import time
from json_memory.versioning import MemoryVersioning, MemoryDiff

@pytest.fixture
def versioning():
    return MemoryVersioning()

def test_diff_basic(versioning):
    t0 = time.time()

    # Add new value
    versioning.record_change("user.name", None, "Alice")
    t1 = time.time()

    # Modify value and add another
    versioning.record_change("user.name", "Alice", "Bob")
    versioning.record_change("user.age", None, 30)
    t2 = time.time()

    # Delete a value
    versioning.record_change("user.age", 30, None)
    t3 = time.time()

    # Diff between t0 and t1 (Added "Alice")
    diff1 = versioning.diff(t0, t1)
    assert diff1.added == {"user.name": "Alice"}
    assert diff1.modified == {}
    assert diff1.deleted == {}

    # Diff between t1 and t2 (Modified "Alice" -> "Bob", Added "user.age")
    diff2 = versioning.diff(t1, t2)
    assert diff2.added == {"user.age": 30}
    assert diff2.modified == {"user.name": ("Alice", "Bob")}
    assert diff2.deleted == {}

    # Diff between t2 and t3 (Deleted "user.age")
    diff3 = versioning.diff(t2, t3)
    assert diff3.added == {}
    assert diff3.modified == {}
    assert diff3.deleted == {"user.age": 30}

    # Diff across everything t0 and t3
    diff4 = versioning.diff(t0, t3)
    assert diff4.added == {"user.name": "Bob"}
    assert diff4.modified == {}
    assert diff4.deleted == {}

def test_diff_empty(versioning):
    versioning.record_change("user.name", None, "Alice")
    t1 = time.time()

    # Same timestamps should yield empty diff
    diff = versioning.diff(t1, t1)
    assert diff.added == {}
    assert diff.modified == {}
    assert diff.deleted == {}

def test_diff_no_changes(versioning):
    t0 = time.time()
    versioning.record_change("user.name", None, "Alice")
    t1 = time.time()

    # No changes between t1 and t2
    time.sleep(0.01)
    t2 = time.time()

    diff = versioning.diff(t1, t2)
    assert diff.added == {}
    assert diff.modified == {}
    assert diff.deleted == {}
