import pytest
import time
from unittest.mock import patch
from json_memory.versioning import MemoryVersioning

@pytest.fixture
def versioning():
    return MemoryVersioning(max_history=100)

@pytest.fixture
def populated_versioning(versioning):
    # Predictable timestamps starting from 1000.0
    timestamps = [1000.0, 1010.0, 1020.0, 1030.0, 1040.0]

    with patch('time.time') as mock_time:
        mock_time.side_effect = timestamps

        # v0: path1
        versioning.record_change("path1", None, "value1", "set")
        # v1: path2
        versioning.record_change("path2", None, "value2", "set")
        # v2: path1
        versioning.record_change("path1", "value1", "value1_updated", "update")
        # v3: path3
        versioning.record_change("path3", None, "value3", "set")
        # v4: path1 deleted
        versioning.record_change("path1", "value1_updated", None, "delete")

    return versioning

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

def test_get_history_all(populated_versioning):
    """Test getting all history with no path filter."""
    history = populated_versioning.get_history()

    assert len(history) == 5
    # Should be sorted newest first
    assert history[0].version_id == "v4"
    assert history[0].path == "path1"
    assert history[0].operation == "delete"

    assert history[-1].version_id == "v0"
    assert history[-1].path == "path1"
    assert history[-1].operation == "set"

def test_get_history_by_path(populated_versioning):
    """Test getting history filtered by a specific path."""
    history = populated_versioning.get_history(path="path1")

    assert len(history) == 3
    # Check that it's only path1, sorted newest first
    assert history[0].version_id == "v4"
    assert history[1].version_id == "v2"
    assert history[2].version_id == "v0"

    history_path2 = populated_versioning.get_history(path="path2")
    assert len(history_path2) == 1
    assert history_path2[0].version_id == "v1"

def test_get_history_with_limit(populated_versioning):
    """Test getting history with a limit."""
    history = populated_versioning.get_history(limit=2)

    assert len(history) == 2
    # Should be the two newest entries (v4, v3)
    assert history[0].version_id == "v4"
    assert history[1].version_id == "v3"

    history_path1_limited = populated_versioning.get_history(path="path1", limit=1)
    assert len(history_path1_limited) == 1
    assert history_path1_limited[0].version_id == "v4"

def test_get_history_time_range(populated_versioning):
    """Test getting history filtered by start_time and end_time."""
    # get versions between 1010.0 (inclusive) and 1030.0 (inclusive)
    # v1 (1010.0), v2 (1020.0), v3 (1030.0)
    history = populated_versioning.get_history(start_time=1010.0, end_time=1030.0)

    assert len(history) == 3
    # Sorted newest first: v3, v2, v1
    assert history[0].version_id == "v3"
    assert history[1].version_id == "v2"
    assert history[2].version_id == "v1"

    # Test only start_time
    history_start = populated_versioning.get_history(start_time=1020.0)
    assert len(history_start) == 3 # v2 (1020), v3 (1030), v4 (1040)
    assert [v.version_id for v in history_start] == ["v4", "v3", "v2"]

    # Test only end_time
    history_end = populated_versioning.get_history(end_time=1010.0)
    assert len(history_end) == 2 # v0 (1000), v1 (1010)
    assert [v.version_id for v in history_end] == ["v1", "v0"]

def test_get_history_time_range_with_path(populated_versioning):
    """Test combining path filter and time range."""
    # path1 versions are at 1000, 1020, 1040
    history = populated_versioning.get_history(path="path1", start_time=1010.0, end_time=1030.0)

    assert len(history) == 1
    assert history[0].version_id == "v2"

def test_get_history_empty(populated_versioning):
    """Test get_history when returning empty results."""
    # Non-existent path
    history = populated_versioning.get_history(path="nonexistent_path")
    assert len(history) == 0

    # Time range with no events (e.g. before the first event)
    history = populated_versioning.get_history(end_time=900.0)
    assert len(history) == 0

    # Start time after the last event
    history = populated_versioning.get_history(start_time=2000.0)
    assert len(history) == 0

    # Empty versioning instance
    empty_versioning = MemoryVersioning()
    assert len(empty_versioning.get_history()) == 0
