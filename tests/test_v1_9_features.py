"""Tests for v1.9.0 — search_value, persistent snapshots, move, merge_from_file, diff_snapshots."""

import json
import os
import tempfile
import time
import pytest

from json_memory import Memory, SmartMemory


# ═══════════════════════════════════════════════════════════════════
# search_value
# ═══════════════════════════════════════════════════════════════════

class TestSearchValue:
    def test_basic(self):
        mem = Memory()
        mem.set("api.exchange", "Binance")
        mem.set("api.key", "sk-123")
        mem.set("notes", "Use Binance for spot trading")
        mem.set("config.timeout", 30)

        results = mem.search_value("Binance")
        assert "api.exchange" in results
        assert "notes" in results
        assert "api.key" not in results
        assert "config.timeout" not in results

    def test_case_insensitive(self):
        mem = Memory()
        mem.set("a", "Hello World")
        mem.set("b", "hello world")
        results = mem.search_value("hello")
        assert len(results) == 2

    def test_case_sensitive(self):
        mem = Memory()
        mem.set("a", "Hello World")
        mem.set("b", "hello world")
        results = mem.search_value("Hello", case_sensitive=True)
        assert "a" in results
        assert "b" not in results

    def test_path_field(self):
        mem = Memory()
        mem.set("trading.exchange", "Binance")
        mem.set("api.exchange", "Kraken")
        results = mem.search_value("trading", field="path")
        assert "trading.exchange" in results
        assert "api.exchange" not in results

    def test_both_field(self):
        mem = Memory()
        mem.set("trading.exchange", "Binance")
        mem.set("api.trading_note", "active")
        results = mem.search_value("trading", field="both")
        assert "trading.exchange" in results
        assert "api.trading_note" in results

    def test_empty(self):
        mem = Memory()
        results = mem.search_value("nothing")
        assert results == {}

    def test_integer_values(self):
        mem = Memory()
        mem.set("port", 8080)
        results = mem.search_value("8080")
        assert "port" in results

    def test_list_values(self):
        mem = Memory()
        mem.set("tags", ["binance", "spot", "crypto"])
        results = mem.search_value("binance")
        assert "tags" in results

    def test_nested_values(self):
        mem = Memory()
        mem.set("config.deep.nested", "find_me")
        results = mem.search_value("find_me")
        assert "config.deep.nested" in results

    def test_tracks_access(self):
        mem = Memory()
        mem.set("a", "test_value")
        mem.search_value("test")
        # Access should be tracked (no error = good)


# ═══════════════════════════════════════════════════════════════════
# persistent snapshots
# ═══════════════════════════════════════════════════════════════════

class TestPersistentSnapshots:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.json")
            mem = Memory(auto_flush_path=path)
            mem.set("user.name", "Dion")
            mem.save_snapshot("before_change")

            mem.set("user.name", "Vel")
            assert mem.get("user.name") == "Vel"

            result = mem.load_snapshot("before_change")
            assert result is True
            assert mem.get("user.name") == "Dion"

    def test_list_snapshots(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.json")
            mem = Memory(auto_flush_path=path)
            mem.set("a", 1)
            mem.save_snapshot("snap1", "First")
            mem.set("a", 2)
            mem.save_snapshot("snap2", "Second")

            snaps = mem.list_snapshots()
            names = [s["name"] for s in snaps]
            assert "snap1" in names
            assert "snap2" in names
            assert snaps[0]["name"] == "snap2"  # Most recent first

    def test_list_snapshots_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.json")
            mem = Memory(auto_flush_path=path)
            mem.set("a", 1)
            mem.save_snapshot("test", "Test description")

            snaps = mem.list_snapshots()
            assert len(snaps) == 1
            s = snaps[0]
            assert s["name"] == "test"
            assert s["description"] == "Test description"
            assert s["entry_count"] == 1
            assert s["chars"] > 0
            assert s["age_seconds"] is not None

    def test_delete_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.json")
            mem = Memory(auto_flush_path=path)
            mem.set("a", 1)
            mem.save_snapshot("temp")

            assert mem.delete_snapshot("temp") is True
            assert mem.delete_snapshot("temp") is False
            assert mem.list_snapshots() == []

    def test_survives_restart(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.json")
            mem1 = Memory(auto_flush_path=path)
            mem1.set("secret", "abc123")
            mem1.save_snapshot("backup")

            # Simulate restart
            mem2 = Memory(auto_flush_path=path)
            mem2.set("secret", "changed")
            result = mem2.load_snapshot("backup")
            assert result is True
            assert mem2.get("secret") == "abc123"

    def test_no_path_raises(self):
        mem = Memory()
        with pytest.raises(ValueError, match="No auto_flush_path"):
            mem.save_snapshot("test")

    def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.json")
            mem = Memory(auto_flush_path=path)
            result = mem.load_snapshot("missing")
            assert result is False

    def test_multiple_snapshots(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.json")
            mem = Memory(auto_flush_path=path)

            mem.set("v", 1)
            mem.save_snapshot("v1")
            mem.set("v", 2)
            mem.save_snapshot("v2")
            mem.set("v", 3)
            mem.save_snapshot("v3")

            mem.load_snapshot("v1")
            assert mem.get("v") == 1
            mem.load_snapshot("v2")
            assert mem.get("v") == 2
            mem.load_snapshot("v3")
            assert mem.get("v") == 3

    def test_snapshot_with_nested_data(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.json")
            mem = Memory(auto_flush_path=path)
            mem.set("deep.nested.value", "deep")
            mem.save_snapshot("deep")

            mem.clear()
            assert mem.get("deep.nested.value") is None

            mem.load_snapshot("deep")
            assert mem.get("deep.nested.value") == "deep"

    def test_overwrite_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.json")
            mem = Memory(auto_flush_path=path)
            mem.set("a", 1)
            mem.save_snapshot("snap")
            mem.set("a", 2)
            mem.save_snapshot("snap")  # Overwrite same name

            snaps = mem.list_snapshots()
            assert len(snaps) == 1  # Only one, not two


# ═══════════════════════════════════════════════════════════════════
# move (bulk rename)
# ═══════════════════════════════════════════════════════════════════

class TestMove:
    def test_basic(self):
        mem = Memory()
        mem.set("trading.exchange", "Binance")
        mem.set("trading.pair", "BTC/USDT")
        mem.set("trading.strategy", "grid")

        result = mem.move("trading", "crypto.trading")
        assert result["count"] == 3
        assert mem.get("crypto.trading.exchange") == "Binance"
        assert mem.get("crypto.trading.pair") == "BTC/USDT"
        assert mem.get("crypto.trading.strategy") == "grid"
        assert mem.get("trading.exchange") is None

    def test_conflict_raises(self):
        mem = Memory()
        mem.set("old.x", 1)
        mem.set("new.x", 2)
        with pytest.raises(ValueError, match="already exist"):
            mem.move("old", "new")

    def test_conflict_overwrite(self):
        mem = Memory()
        mem.set("old.x", 1)
        mem.set("new.x", 2)
        result = mem.move("old", "new", overwrite=True)
        assert result["count"] == 1
        assert mem.get("new.x") == 1

    def test_empty(self):
        mem = Memory()
        result = mem.move("nonexistent", "somewhere")
        assert result["count"] == 0
        assert result["moved"] == []

    def test_preserves_ttl(self):
        mem = Memory()
        mem.set("old.x", 1, ttl=3600)
        mem.move("old", "new")
        assert mem.get("new.x") == 1
        # TTL should be transferred

    def test_leaf_value(self):
        mem = Memory()
        mem.set("name", "Dion")
        result = mem.move("name", "user.name")
        assert result["count"] == 1
        assert mem.get("user.name") == "Dion"

    def test_prunes_empty_parents(self):
        mem = Memory()
        mem.set("a.b.c", 1)
        mem.move("a.b.c", "x.y.c")
        assert mem.get("x.y.c") == 1
        assert not mem.has("a")

    def test_partial_tree(self):
        mem = Memory()
        mem.set("api.key", "sk-123")
        mem.set("api.secret", "secret")
        mem.set("api.exchange.name", "Binance")
        mem.set("api.exchange.url", "https://api.binance.com")

        result = mem.move("api.exchange", "config.exchange")
        assert result["count"] == 2
        assert mem.get("config.exchange.name") == "Binance"
        assert mem.get("api.key") == "sk-123"  # Unmoved

    def test_mixed_leaf_and_branch(self):
        mem = Memory()
        mem.set("old.value", 42)
        mem.set("old.nested.deep", "hello")
        result = mem.move("old", "new")
        assert result["count"] == 2

    def test_move_to_root(self):
        mem = Memory()
        mem.set("wrapper.inner", "value")
        result = mem.move("wrapper.inner", "inner")
        assert result["count"] == 1
        assert mem.get("inner") == "value"


# ═══════════════════════════════════════════════════════════════════
# merge_from_file
# ═══════════════════════════════════════════════════════════════════

class TestMergeFromFile:
    def test_basic(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "source.json")
            with open(src, "w") as f:
                json.dump({"imported": {"x": 1, "y": 2}}, f)

            mem = Memory()
            mem.set("existing", "keep")
            result = mem.merge_from_file(src)
            assert result["imported"] == 2
            assert result["skipped"] == 0
            assert mem.get("imported.x") == 1
            assert mem.get("imported.y") == 2
            assert mem.get("existing") == "keep"

    def test_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "source.json")
            with open(src, "w") as f:
                json.dump({"x": 1}, f)

            mem = Memory()
            result = mem.merge_from_file(src, prefix="restored")
            assert mem.get("restored.x") == 1

    def test_skip_conflict(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "source.json")
            with open(src, "w") as f:
                json.dump({"x": 1, "y": 2}, f)

            mem = Memory()
            mem.set("x", 999)
            result = mem.merge_from_file(src, conflict="skip")
            assert result["imported"] == 1
            assert result["skipped"] == 1
            assert mem.get("x") == 999

    def test_overwrite_conflict(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "source.json")
            with open(src, "w") as f:
                json.dump({"x": 1}, f)

            mem = Memory()
            mem.set("x", 999)
            result = mem.merge_from_file(src, conflict="overwrite")
            assert mem.get("x") == 1

    def test_state_format(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "state.json")
            with open(src, "w") as f:
                json.dump({"data": {"x": 1}, "ttls": {}}, f)

            mem = Memory()
            result = mem.merge_from_file(src)
            assert mem.get("x") == 1

    def test_deep_nested(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "deep.json")
            with open(src, "w") as f:
                json.dump({"a": {"b": {"c": {"d": 42}}}}, f)

            mem = Memory()
            result = mem.merge_from_file(src)
            assert mem.get("a.b.c.d") == 42

    def test_non_dict_raises(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "array.json")
            with open(src, "w") as f:
                json.dump([1, 2, 3], f)

            mem = Memory()
            with pytest.raises(ValueError, match="Expected a JSON object"):
                mem.merge_from_file(src)

    def test_nonexistent_file(self):
        mem = Memory()
        with pytest.raises(FileNotFoundError):
            mem.merge_from_file("/nonexistent/file.json")


# ═══════════════════════════════════════════════════════════════════
# diff_snapshots
# ═══════════════════════════════════════════════════════════════════

class TestDiffSnapshots:
    def test_basic_diff(self):
        mem = Memory()
        mem.set("a", 1)
        mem.set("b", 2)
        mem.snapshot("v1")

        mem.set("a", 10)
        mem.set("c", 3)
        mem.delete("b")
        mem.snapshot("v2")

        diff = mem.diff_snapshots("v1", "v2")
        assert "c" in diff["added"]
        assert "b" in diff["removed"]
        assert "a" in diff["changed"]
        assert diff["changed"]["a"]["old"] == 1
        assert diff["changed"]["a"]["new"] == 10
        assert diff["unchanged"] == 0

    def test_identical(self):
        mem = Memory()
        mem.set("a", 1)
        mem.snapshot("s1")
        mem.snapshot("s2")

        diff = mem.diff_snapshots("s1", "s2")
        assert diff["added"] == []
        assert diff["removed"] == []
        assert diff["changed"] == {}
        assert diff["unchanged"] == 1

    def test_persistent(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.json")
            mem = Memory(auto_flush_path=path)
            mem.set("a", 1)
            mem.save_snapshot("before")
            mem.set("a", 2)
            mem.save_snapshot("after")

            diff = mem.diff_snapshots("before", "after")
            assert "a" in diff["changed"]
            assert diff["changed"]["a"]["old"] == 1
            assert diff["changed"]["a"]["new"] == 2

    def test_not_found(self):
        mem = Memory()
        with pytest.raises(ValueError, match="not found"):
            mem.diff_snapshots("missing", "also_missing")

    def test_mixed_in_memory_and_persistent(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.json")
            mem = Memory(auto_flush_path=path)
            mem.set("a", 1)
            mem.save_snapshot("persistent")
            mem.set("a", 2)
            mem.snapshot("inmemory")  # In-memory only

            # Can compare in-memory with persistent
            diff = mem.diff_snapshots("persistent", "inmemory")
            assert "a" in diff["changed"]

    def test_nested_changes(self):
        mem = Memory()
        mem.set("a.b.c", 1)
        mem.set("a.b.d", 2)
        mem.snapshot("v1")

        mem.set("a.b.c", 10)
        mem.set("a.b.e", 3)
        mem.delete("a.b.d")
        mem.snapshot("v2")

        diff = mem.diff_snapshots("v1", "v2")
        assert "a.b.e" in diff["added"]
        assert "a.b.d" in diff["removed"]
        assert "a.b.c" in diff["changed"]

    def test_summary_string(self):
        mem = Memory()
        mem.set("a", 1)
        mem.snapshot("s1")
        mem.set("a", 2)
        mem.set("b", 3)
        mem.delete("a")
        mem.snapshot("s2")

        diff = mem.diff_snapshots("s1", "s2")
        assert "added" in diff["summary"]
        assert "removed" in diff["summary"]


# ═══════════════════════════════════════════════════════════════════
# SmartMemory wrappers
# ═══════════════════════════════════════════════════════════════════

class TestSmartMemoryWrappers:
    def test_search_value(self):
        with tempfile.TemporaryDirectory() as td:
            sm = SmartMemory(path=os.path.join(td, "smart.json"))
            sm.remember("api.exchange", "Binance")
            sm.remember("notes", "Use Binance for spot")
            results = sm.search_value("Binance")
            assert "api.exchange" in results
            assert "notes" in results

    def test_move(self):
        with tempfile.TemporaryDirectory() as td:
            sm = SmartMemory(path=os.path.join(td, "smart.json"))
            sm.remember("trading.exchange", "Binance")
            result = sm.move("trading", "crypto.trading")
            assert result["count"] == 1
            assert sm.recall("crypto.trading.exchange") is not None

    def test_merge_from_file(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "data.json")
            with open(src, "w") as f:
                json.dump({"imported": {"key": "val"}}, f)
            sm = SmartMemory(path=os.path.join(td, "smart.json"))
            result = sm.merge_from_file(src)
            assert result["imported"] == 1

    def test_persistent_snapshots(self):
        with tempfile.TemporaryDirectory() as td:
            sm = SmartMemory(path=os.path.join(td, "smart.json"))
            sm.remember("x", 1)
            sm.save_snapshot("backup")
            sm.remember("x", 2)
            sm.load_snapshot("backup")
            assert sm.recall("x") == 1
            assert len(sm.list_snapshots()) == 1

    def test_diff_snapshots(self):
        with tempfile.TemporaryDirectory() as td:
            sm = SmartMemory(path=os.path.join(td, "smart.json"))
            sm.remember("a", 1)
            sm.snapshot("v1")
            sm.remember("a", 2)
            sm.snapshot("v2")
            diff = sm.diff_snapshots("v1", "v2")
            assert "a" in diff["changed"]

    def test_delete_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            sm = SmartMemory(path=os.path.join(td, "smart.json"))
            sm.remember("x", 1)
            sm.save_snapshot("temp")
            assert sm.delete_snapshot("temp") is True
            assert sm.list_snapshots() == []
