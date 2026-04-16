"""Tests for json-memory."""

import json
import pytest
from json_memory import Memory, Synapse, Schema, WeightGate, compress, decompress, savings_report


# ── Memory Tests ──────────────────────────────────────────────────

class TestMemory:
    def test_create_empty(self):
        mem = Memory(max_chars=500)
        assert mem.export() == "{}"
        assert mem.stats()["leaf_count"] == 0

    def test_set_and_get(self):
        mem = Memory()
        mem.set("u.name", "Alice")
        mem.set("u.tz", "UTC")
        assert mem.get("u.name") == "Alice"
        assert mem.get("u.tz") == "UTC"

    def test_nested_set(self):
        mem = Memory()
        mem.set("bot.binance.rst", "kill && nohup ./bot > log")
        mem.set("bot.binance.wl", "BNB,KITE")
        assert mem.get("bot.binance.rst") == "kill && nohup ./bot > log"
        assert mem.get("bot.binance.wl") == "BNB,KITE"

    def test_deep_nesting(self):
        mem = Memory()
        mem.set("a.b.c.d.e", "deep")
        assert mem.get("a.b.c.d.e") == "deep"

    def test_get_default(self):
        mem = Memory()
        assert mem.get("missing.path", "default") == "default"
        assert mem.get("missing.path") is None

    def test_delete(self):
        mem = Memory()
        mem.set("x.y", "val")
        assert mem.has("x.y")
        mem.delete("x.y")
        assert not mem.has("x.y")

    def test_has(self):
        mem = Memory()
        mem.set("x", 1)
        assert mem.has("x")
        assert not mem.has("y")

    def test_keys(self):
        mem = Memory()
        mem.set("a.x", 1)
        mem.set("a.y", 2)
        mem.set("b.z", 3)
        assert sorted(mem.keys("a")) == ["x", "y"]

    def test_paths(self):
        mem = Memory()
        mem.set("a.b", 1)
        mem.set("a.c", 2)
        mem.set("d", 3)
        assert sorted(mem.paths()) == ["a.b", "a.c", "d"]

    def test_export_minified(self):
        mem = Memory()
        mem.set("u.name", "Alice")
        exported = mem.export()
        assert " " not in exported  # minified
        assert json.loads(exported) == {"u": {"name": "Alice"}}

    def test_from_json(self):
        mem = Memory.from_json('{"u":{"name":"Alice"}}')
        assert mem.get("u.name") == "Alice"

    def test_merge(self):
        mem = Memory()
        data = {"u": {"name": "Alice", "tz": "UTC"}, "bot": {"bal": "$12K"}}
        count = mem.merge(data)
        assert count == 3
        assert mem.get("u.name") == "Alice"
        assert mem.get("bot.bal") == "$12K"

    def test_merge_with_prefix(self):
        mem = Memory()
        mem.merge({"name": "Alice"}, prefix="u")
        assert mem.get("u.name") == "Alice"

    def test_max_chars_overflow(self):
        mem = Memory(max_chars=20)
        with pytest.raises(ValueError, match="Memory overflow"):
            mem.set("key", "a" * 50)

    def test_stats(self):
        mem = Memory(max_chars=1000)
        mem.set("x", 1)
        stats = mem.stats()
        assert stats["leaf_count"] == 1
        assert stats["chars_used"] > 0
        assert stats["chars_max"] == 1000

    def test_to_dict(self):
        mem = Memory()
        mem.set("x.y", 1)
        d = mem.to_dict()
        assert d == {"x": {"y": 1}}

    def test_getitem_setitem(self):
        mem = Memory()
        mem["u.name"] = "Alice"
        assert mem["u.name"] == "Alice"

    def test_contains(self):
        mem = Memory()
        mem.set("x", 1)
        assert "x" in mem
        assert "y" not in mem

    def test_len(self):
        mem = Memory()
        mem.set("x", 1)
        assert len(mem) == len(mem.export())

    def test_repr(self):
        mem = Memory()
        mem.set("x", 1)
        r = repr(mem)
        assert "Memory" in r
        assert "leafs=1" in r


# ── Synapse Tests ─────────────────────────────────────────────────

class TestSynapse:
    def test_create(self):
        s = Synapse()
        assert repr(s) == "Synapse(concepts=0, links=0)"

    def test_link(self):
        s = Synapse()
        s.link("trading", ["binance", "strategy"])
        assert "binance" in s.activate("trading")
        assert "strategy" in s.activate("trading")

    def test_bidirectional(self):
        s = Synapse()
        s.link("trading", ["binance"], bidirectional=True)
        assert "trading" in s.activate("binance")

    def test_unidirectional(self):
        s = Synapse()
        s.link("trading", ["binance"], bidirectional=False)
        assert "binance" in s.activate("trading")
        assert "trading" not in s.activate("binance")

    def test_depth_traversal(self):
        s = Synapse()
        s.link("trading", ["binance"])
        s.link("binance", ["api", "demo"])
        result = s.activate("trading", depth=1)
        assert "binance" in result
        assert "api" not in result

        result = s.activate("trading", depth=2)
        assert "api" in result
        assert "demo" in result

    def test_connections(self):
        s = Synapse()
        s.link("hub", ["a", "b", "c", "d", "e"])
        info = s.connections("hub")
        assert info["count"] == 5
        assert info["is_hub"] is True

    def test_find_path(self):
        s = Synapse()
        s.link("a", ["b"])
        s.link("b", ["c"])
        path = s.find_path("a", "c")
        assert path == ["a", "b", "c"]

    def test_find_path_no_connection(self):
        s = Synapse()
        s.link("a", ["b"])
        s.link("c", ["d"])
        assert s.find_path("a", "d") is None

    def test_hubs(self):
        s = Synapse()
        s.link("hub1", ["a", "b", "c", "d"])
        s.link("hub2", ["x", "y", "z"])
        s.link("small", ["a"])
        hubs = s.hubs(min_connections=3)
        assert hubs[0][0] == "hub1"
        assert hubs[0][1] == 4  # 4 children

    def test_export_import(self):
        s = Synapse()
        s.link("a", ["b", "c"])
        exported = s.export()
        s2 = Synapse.from_json(exported)
        assert s2.activate("a") == s.activate("a")

    def test_no_cycles(self):
        s = Synapse()
        s.link("a", ["b"])
        s.link("b", ["c"])
        s.link("c", ["a"])  # cycle
        result = s.activate("a", depth=10)
        assert len(result) == len(set(result))  # no duplicates


class TestSynapseWeights:
    def test_weighted_link(self):
        s = Synapse()
        s.link("coffee", ["cappuccino", "americano", "espresso"],
               weights={"cappuccino": 0.9, "americano": 0.3, "espresso": 0.6})
        assert s.get_weight("coffee", "cappuccino") == 0.9
        assert s.get_weight("coffee", "americano") == 0.3
        assert s.get_weight("coffee", "espresso") == 0.6

    def test_default_weight(self):
        s = Synapse()
        s.link("coffee", ["latte"])  # no weight specified
        assert s.get_weight("coffee", "latte") == 0.5  # default

    def test_activate_sorted_by_weight(self):
        s = Synapse()
        s.link("coffee", ["cappuccino", "americano", "espresso"],
               weights={"cappuccino": 0.9, "americano": 0.3, "espresso": 0.6})
        result = s.activate("coffee", depth=1)
        assert result[0] == "cappuccino"  # highest weight first
        assert result[1] == "espresso"
        assert result[2] == "americano"

    def test_unweighted_activate(self):
        s = Synapse()
        s.link("coffee", ["cappuccino", "americano"],
               weights={"cappuccino": 0.9, "americano": 0.3})
        result = s.activate("coffee", depth=1, weighted=False)
        # Original insertion order
        assert result == ["cappuccino", "americano"]

    def test_strengthen(self):
        s = Synapse()
        s.link("coffee", ["cappuccino"], weights={"cappuccino": 0.5})
        new_w = s.strengthen("coffee", "cappuccino", boost=0.2)
        assert new_w == 0.7
        assert s.get_weight("coffee", "cappuccino") == 0.7

    def test_strengthen_cap(self):
        s = Synapse()
        s.link("coffee", ["cappuccino"], weights={"cappuccino": 0.9})
        new_w = s.strengthen("coffee", "cappuccino", boost=0.5)
        assert new_w == 1.0  # capped at 1.0

    def test_weaken(self):
        s = Synapse()
        s.link("coffee", ["americano"], weights={"americano": 0.5})
        new_w = s.weaken("coffee", "americano", decay=0.2)
        assert new_w == 0.3
        assert s.get_weight("coffee", "americano") == 0.3

    def test_weaken_floor(self):
        s = Synapse()
        s.link("coffee", ["americano"], weights={"americano": 0.1})
        new_w = s.weaken("coffee", "americano", decay=0.5)
        assert new_w == 0.0  # floored at 0.0

    def test_frequency_tracking(self):
        s = Synapse()
        s.link("coffee", ["cappuccino", "americano"])
        s.activate("coffee", depth=1)  # first activation
        s.activate("coffee", depth=1)  # second
        s.activate("coffee", depth=1)  # third
        assert s.get_frequency("coffee", "cappuccino") == 3
        assert s.get_frequency("coffee", "americano") == 3

    def test_top_associations(self):
        s = Synapse()
        s.link("coffee", ["cappuccino", "americano", "espresso", "latte"],
               weights={"cappuccino": 0.9, "americano": 0.3, "espresso": 0.6, "latte": 0.7})
        top = s.top_associations("coffee", limit=2)
        assert top[0] == ("cappuccino", 0.9)
        assert top[1] == ("latte", 0.7)

    def test_learning_simulation(self):
        """Simulate learning: user always picks cappuccino, rarely picks americano."""
        s = Synapse()
        s.link("coffee", ["cappuccino", "americano"],
               weights={"cappuccino": 0.5, "americano": 0.5})

        # User picks cappuccino 10 times → strengthens
        for _ in range(10):
            s.strengthen("coffee", "cappuccino", boost=0.05)

        # User never picks americano → weakens
        for _ in range(10):
            s.weaken("coffee", "americano", decay=0.03)

        # Now cappuccino should be dominant
        assert s.get_weight("coffee", "cappuccino") > s.get_weight("coffee", "americano")
        top = s.top_associations("coffee", limit=1)
        assert top[0][0] == "cappuccino"

    def test_export_import_preserves_weights(self):
        s = Synapse()
        s.link("coffee", ["cappuccino"], weights={"cappuccino": 0.8})
        s.strengthen("coffee", "cappuccino", boost=0.1)

        exported = s.export()
        s2 = Synapse.from_json(exported)
        assert s2.get_weight("coffee", "cappuccino") == s.get_weight("coffee", "cappuccino")

    def test_personalization_scenario(self):
        """Two people with different coffee preferences."""
        # Person A: loves cappuccino
        person_a = Synapse()
        person_a.link("coffee", ["cappuccino", "americano", "espresso"],
                      weights={"cappuccino": 0.95, "americano": 0.2, "espresso": 0.5})

        # Person B: loves americano
        person_b = Synapse()
        person_b.link("coffee", ["cappuccino", "americano", "espresso"],
                      weights={"cappuccino": 0.2, "americano": 0.9, "espresso": 0.4})

        # Same concept, different recall order
        a_first = person_a.activate("coffee", depth=1)[0]
        b_first = person_b.activate("coffee", depth=1)[0]

        assert a_first == "cappuccino"
        assert b_first == "americano"


# ── Schema Tests ──────────────────────────────────────────────────

class TestSchema:
    def test_validate_simple(self):
        schema = Schema({"u": {"n": "str", "tz": "str"}})
        assert schema.validate({"u": {"n": "Alice", "tz": "UTC"}})
        assert not schema.validate({"u": {"n": 123}})

    def test_defaults(self):
        schema = Schema({"u": {"n": "str", "tz": "str"}})
        d = schema.defaults()
        assert d == {"u": {"n": None, "tz": None}}

    def test_diff(self):
        schema = Schema({"u": {"n": "str", "tz": "str", "email": "str"}})
        diff = schema.diff({"u": {"n": "Alice", "extra": 1}})
        assert "u.tz" in diff["missing"]
        assert "u.email" in diff["missing"]
        assert "u.extra" in diff["extra"]

    def test_strict(self):
        schema = Schema({"u": {"n": "str"}})
        assert schema.validate({"u": {"n": "Alice"}}, strict=True)
        assert not schema.validate({"u": {"n": "Alice", "extra": 1}}, strict=True)

    def test_export(self):
        schema = Schema({"x": "str"})
        assert schema.export() == '{"x":"str"}'


# ── Compress Tests ────────────────────────────────────────────────

class TestCompress:
    def test_compress_dict(self):
        data = {"user": {"name": "Alice"}, "server": {"restart": "cmd"}}
        compressed = compress(data)
        assert "u" in compressed
        assert "n" in compressed["u"]
        assert "srv" in compressed
        assert "rst" in compressed["srv"]

    def test_decompress_roundtrip(self):
        original = {"user": {"name": "Alice"}, "watchlist": ["BTC"]}
        compressed = compress(original)
        decompressed = decompress(compressed)
        assert decompressed == original

    def test_savings_report(self):
        report = savings_report("hello world", "hw")
        assert report["original_chars"] == 11
        assert report["compressed_chars"] == 2
        assert report["chars_saved"] == 9
        assert report["savings_pct"] > 80

    def test_prose_to_json(self):
        prose = "name: Alice\ntz: UTC\nplatform: Telegram"
        result = json.loads(json.dumps(
            __import__("json_memory.compress", fromlist=["prose_to_json"]).prose_to_json(prose)
        ))
        assert result["name"] == "Alice"
        assert result["tz"] == "UTC"


# ── WeightGate Tests ─────────────────────────────────────────────

class TestWeightGate:
    def _make_gate(self, tmp_path, enabled=True):
        path = str(tmp_path / "test_synapse.json")
        gate = WeightGate(path, enabled=enabled)
        gate.add_concept("coffee", {"cappuccino": 0.9, "americano": 0.3, "espresso": 0.6})
        gate.add_concept("debug", {"check_logs": 0.9, "ask_user": 0.2})
        return gate

    def test_disabled_by_default(self):
        from json_memory import WeightGate
        gate = WeightGate("/tmp/test_gate.json")
        assert gate.enabled is False

    def test_enable_disable(self, tmp_path):
        gate = self._make_gate(tmp_path, enabled=False)
        assert gate.enabled is False
        gate.enable()
        assert gate.enabled is True
        gate.disable()
        assert gate.enabled is False

    def test_toggle(self, tmp_path):
        gate = self._make_gate(tmp_path, enabled=False)
        assert gate.toggle() is True
        assert gate.toggle() is False

    def test_disabled_is_noop(self, tmp_path):
        gate = self._make_gate(tmp_path, enabled=False)
        result = gate.process_input("I love cappuccino")
        assert result == {}
        result = gate.process_output("making cappuccino")
        assert result == {}

    def test_process_input_strengthen(self, tmp_path):
        gate = self._make_gate(tmp_path, enabled=True)
        gate.process_input("I love cappuccino coffee")
        weights = gate.get_weights("coffee")
        assert weights["cappuccino"] > 0.9  # strengthened

    def test_process_input_decay(self, tmp_path):
        gate = self._make_gate(tmp_path, enabled=True)
        gate.process_input("I love coffee")  # concept mentioned, but not americano
        weights = gate.get_weights("coffee")
        assert weights["americano"] < 0.3  # decayed (not mentioned)

    def test_process_output_strengthen(self, tmp_path):
        gate = self._make_gate(tmp_path, enabled=True)
        gate.process_output("Let me check logs for errors")
        weights = gate.get_weights("debug")
        assert weights["check_logs"] > 0.9  # mentioned in output

    def test_process_conversation(self, tmp_path):
        gate = self._make_gate(tmp_path, enabled=True)
        result = gate.process_conversation(
            "check the logs",
            "Looking at logs now, found the error"
        )
        assert "input" in result
        assert "output" in result
        assert result["total_interactions"] == 1

    def test_disabled_process_conversation(self, tmp_path):
        gate = self._make_gate(tmp_path, enabled=False)
        result = gate.process_conversation("test", "response")
        assert result == {}

    def test_context_manager(self, tmp_path):
        path = str(tmp_path / "ctx_synapse.json")
        with WeightGate(path) as gate:
            gate.add_concept("x", {"a": 0.5})
            assert gate.enabled is True
        assert gate.enabled is False

    def test_add_remove_concept(self, tmp_path):
        gate = self._make_gate(tmp_path)
        gate.add_concept("new", {"a": 0.5, "b": 0.3})
        assert "new" in gate.synapse["concepts"]
        assert gate.remove_concept("new") is True
        assert "new" not in gate.synapse["concepts"]

    def test_top_associations(self, tmp_path):
        gate = self._make_gate(tmp_path)
        top = gate.top_associations("coffee", limit=2)
        assert top[0][0] == "cappuccino"
        assert top[1][0] == "espresso"

    def test_manual_strengthen_weaken(self, tmp_path):
        gate = self._make_gate(tmp_path)
        new = gate.strengthen("coffee", "americano", boost=0.2)
        assert abs(new - 0.5) < 0.001  # 0.3 + 0.2
        new = gate.weaken("coffee", "cappuccino", decay=0.3)
        assert abs(new - 0.6) < 0.001  # 0.9 - 0.3

    def test_export_compact(self, tmp_path):
        gate = self._make_gate(tmp_path)
        compact = gate.export_compact()
        data = json.loads(compact)
        assert "coffee" in data
        assert "cappuccino" in data["coffee"]

    def test_learning_simulation(self, tmp_path):
        gate = self._make_gate(tmp_path, enabled=True)
        # User always asks about cappuccino
        for _ in range(50):
            gate.process_input("make me a cappuccino coffee")
        top = gate.top_associations("coffee", limit=1)
        assert top[0][0] == "cappuccino"
        assert top[0][1] > 0.95

    def test_repr(self, tmp_path):
        gate = self._make_gate(tmp_path, enabled=True)
        r = repr(gate)
        assert "[ON]" in r
        gate.disable()
        r = repr(gate)
        assert "[OFF]" in r
