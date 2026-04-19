"""Tests for SmartMemory — weighted retrieval, auto-extraction, tiered memory."""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from json_memory.smart import (
    SmartMemory,
    TieredMemory,
    _normalize_tokens,
    _recency_score,
    _frequency_score,
    _keyword_relevance,
    PathMeta,
)
from json_memory.semantic import SemanticIndex, HAS_SEMANTIC


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def mem(tmp_dir):
    path = os.path.join(tmp_dir, "test_smart.json")
    return SmartMemory(path, max_chars=3000)


# ── Token Normalization ───────────────────────────────────────────────

class TestNormalizeTokens:
    def test_basic(self):
        tokens = _normalize_tokens("Hello World Test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_short_words_excluded(self):
        tokens = _normalize_tokens("I am the bot")
        assert "bot" in tokens
        assert "the" in tokens
        assert "am" not in tokens  # too short
        assert "i" not in tokens  # too short

    def test_synonym_expansion(self):
        tokens = _normalize_tokens("who am I")
        assert "name" in tokens  # synonym of 'who'
        assert "user" in tokens  # synonym of 'me'

    def test_trading_synonyms(self):
        tokens = _normalize_tokens("What trades the bot")
        assert "exchange" in tokens or "bot" in tokens


# ── Scoring Functions ──────────────────────────────────────────────────

class TestScoring:
    def test_recency_decay(self):
        now = 1000.0
        recent = _recency_score(999.0, now, half_life=100)
        old = _recency_score(500.0, now, half_life=100)
        assert recent > old

    def test_recency_zero(self):
        assert _recency_score(0, 1000) == 0.1

    def test_frequency_log_scale(self):
        low = _frequency_score(1, 100)
        high = _frequency_score(100, 100)
        assert high > low
        assert high == 1.0

    def test_keyword_overlap(self):
        tokens_a = {"user", "name", "alice"}
        tokens_b = {"user", "timezone", "gmt"}
        score = _keyword_relevance(tokens_a, tokens_b)
        assert 0 < score < 1  # partial overlap

    def test_keyword_no_overlap(self):
        tokens_a = {"user", "name"}
        tokens_b = {"server", "ip"}
        assert _keyword_relevance(tokens_a, tokens_b) == 0.0

    def test_keyword_perfect_match(self):
        tokens = {"user", "name"}
        assert _keyword_relevance(tokens, tokens) == 1.0


# ── Basic Operations ──────────────────────────────────────────────────

class TestSmartMemory:
    def test_remember_and_recall(self, mem):
        mem.remember("user.name", "Alice")
        assert mem.recall("user.name") == "Alice"

    def test_forget(self, mem):
        mem.remember("user.name", "Alice")
        mem.forget("user.name")
        assert mem.recall("user.name") is None

    def test_search(self, mem):
        mem.remember("user.name", "Alice")
        mem.remember("user.tz", "UTC")
        results = mem.search("user.*")
        assert "user.name" in results
        assert "user.tz" in results

    def test_context_export(self, mem):
        mem.remember("user.name", "Alice")
        ctx = mem.context()
        data = json.loads(ctx)
        assert data["user"]["name"] == "Alice"

    def test_snapshot_rollback(self, mem):
        mem.remember("task", "original")
        mem.snapshot("before")
        mem.remember("task", "changed")
        mem.rollback("before")
        assert mem.recall("task") == "original"


# ── Smart Retrieval ───────────────────────────────────────────────────

class TestSmartRetrieval:
    def test_relevant_returns_only_matches(self, mem):
        mem.remember("user.timezone", "GMT+7")
        mem.remember("server.ip", "10.0.0.1")
        mem.remember("bot.symbol", "BNBUSDT")

        relevant = mem.recall_relevant("What's my timezone?")
        assert "user.timezone" in relevant
        assert len(relevant) <= 3  # should be tight

    def test_bot_query_returns_bot_fields(self, mem):
        mem.remember("bot.symbol", "BNBUSDT")
        mem.remember("bot.strategy", "RSI")
        mem.remember("user.name", "Alice")

        relevant = mem.recall_relevant("What's the bot trading?")
        assert "bot.symbol" in relevant
        assert "bot.strategy" in relevant

    def test_no_query_returns_top_scored(self, mem):
        mem.remember("a", "1")
        mem.remember("b", "2")
        mem.remember("c", "3")
        relevant = mem.recall_relevant()
        assert len(relevant) > 0

    def test_prompt_context_format(self, mem):
        mem.remember("user.name", "Alice")
        ctx = mem.prompt_context("Who am I?")
        assert "## Memory" in ctx
        assert "user.name" in ctx or "Alice" in ctx

    def test_prompt_context_empty_no_data(self, mem):
        ctx = mem.prompt_context("random unrelated query about quantum physics")
        assert ctx == "" or "## Memory" in ctx

    def test_score_explanation(self, mem):
        mem.remember("user.name", "Alice")
        explanation = mem.explain_score("user.name", "What's my name?")
        assert "final_score" in explanation
        assert "recency" in explanation
        assert "frequency" in explanation
        assert "keyword_relevance" in explanation


# ── Auto-Extraction ───────────────────────────────────────────────────

class TestAutoExtraction:
    def test_name_extraction(self, mem):
        extracted = mem.process_conversation("My name is Bob")
        paths = [e["path"] for e in extracted]
        assert "user.name" in paths
        values = [e["value"] for e in extracted]
        assert "Bob" in values

    def test_timezone_extraction(self, mem):
        extracted = mem.process_conversation("My timezone is UTC+9")
        paths = [e["path"] for e in extracted]
        assert "user.timezone" in paths

    def test_remember_extraction(self, mem):
        extracted = mem.process_conversation("Remember that the password is secret123")
        assert len(extracted) > 0
        assert any("secret123" in e["value"] for e in extracted)

    def test_no_extraction_on_empty(self, mem):
        extracted = mem.process_conversation("Hello, how are you?")
        assert len(extracted) == 0

    def test_confidence_threshold(self, mem):
        # Set high threshold
        mem.extract_confidence = 0.9
        extracted = mem.process_conversation("I prefer Python")
        # Preferences have 0.6 confidence, should be filtered
        assert len(extracted) == 0 or all(e["confidence"] >= 0.9 for e in extracted)


# ── Associative Memory ────────────────────────────────────────────────

class TestAssociativeMemory:
    def test_link_and_associate(self, mem):
        mem.link("debugging", ["logs", "reproduce"])
        result = mem.associate("debugging")
        assert "logs" in result
        assert "reproduce" in result

    def test_depth_traversal(self, mem):
        mem.link("a", ["b"])
        mem.link("b", ["c"])
        result = mem.associate("a", depth=2)
        assert "b" in result
        assert "c" in result


# ── Tiered Memory ─────────────────────────────────────────────────────

class TestTieredMemory:
    def test_hot_warm_cold(self, tmp_dir):
        path = os.path.join(tmp_dir, "tiered.json")
        tiered = TieredMemory(path, max_hot_chars=1000, max_warm_chars=2000)
        tiered.set("hot_key", "hot_value", tier="hot")
        tiered.set("warm_key", "warm_value", tier="warm")
        tiered.set("cold_key", "cold_value", tier="cold")

        assert tiered.get("hot_key") == "hot_value"
        assert tiered.get("warm_key") == "warm_value"
        assert tiered.get("cold_key") == "cold_value"

    def test_promote(self, tmp_dir):
        path = os.path.join(tmp_dir, "tiered.json")
        tiered = TieredMemory(path)
        tiered.set("key", "value", tier="cold")
        assert tiered.get_tier("key") == "cold"
        tiered.promote("key")
        assert tiered.get_tier("key") == "hot"

    def test_demote(self, tmp_dir):
        path = os.path.join(tmp_dir, "tiered.json")
        tiered = TieredMemory(path)
        tiered.set("key", "value", tier="hot")
        tiered.demote("key", target="cold")
        assert tiered.get_tier("key") == "cold"


# ── Persistence ───────────────────────────────────────────────────────

class TestPersistence:
    def test_remember_persists(self, tmp_dir):
        path = os.path.join(tmp_dir, "persist.json")
        mem1 = SmartMemory(path, max_chars=3000)
        mem1.remember("user.name", "Alice")

        # New instance loads from disk
        mem2 = SmartMemory(path, max_chars=3000)
        assert mem2.recall("user.name") == "Alice"

    def test_meta_persists(self, tmp_dir):
        path = os.path.join(tmp_dir, "persist.json")
        mem1 = SmartMemory(path, max_chars=3000)
        mem1.remember("user.name", "Alice")
        mem1.recall("user.name")  # bump access count

        mem2 = SmartMemory(path, max_chars=3000)
        assert "user.name" in mem2._meta
        assert mem2._meta["user.name"].access_count >= 1


# ── Semantic Layer (optional) ─────────────────────────────────────────

class TestSemanticIndex:
    @pytest.mark.skipif(not HAS_SEMANTIC, reason="sentence-transformers not installed")
    def test_add_and_search(self):
        index = SemanticIndex()
        index.add("user.timezone", "GMT+7 based in Jakarta")
        index.add("bot.restart", "kill and restart the bot process")

        results = index.search("What time zone am I in?")
        assert len(results) > 0
        assert results[0][0] == "user.timezone"

    def test_graceful_without_deps(self):
        """SemanticIndex works without dependencies (returns empty)."""
        # Force unavailable
        index = SemanticIndex.__new__(SemanticIndex)
        index.available = False
        index._index = None
        index._paths = []
        index._path_to_idx = {}
        index._model = None
        index._dim = 384
        index._model_name = "all-MiniLM-L6-v2"
        import threading
        index._lock = threading.RLock()

        assert index.search("anything") == []
        assert index.stats()["available"] is False


# ── Edge Cases ─────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_memory_relevant(self, mem):
        relevant = mem.recall_relevant("anything")
        assert relevant == {}

    def test_unicode_values(self, mem):
        mem.remember("user.location", "東京, 日本")
        assert mem.recall("user.location") == "東京, 日本"

    def test_nested_values(self, mem):
        mem.remember("config.db", {"host": "localhost", "port": 5432})
        val = mem.recall("config.db")
        assert val["host"] == "localhost"

    def test_list_values(self, mem):
        mem.remember("skills", ["python", "go", "rust"])
        assert len(mem.recall("skills")) == 3

    def test_overwrite(self, mem):
        mem.remember("user.name", "Alice")
        mem.remember("user.name", "Bob")
        assert mem.recall("user.name") == "Bob"

    def test_stats(self, mem):
        mem.remember("a", "1")
        mem.remember("b", "2")
        stats = mem.stats()
        assert stats["entries"] == 2
        assert stats["paths"] == 2
        assert "top_scored" in stats
