"""Tests for AdvancedSearch in json_memory/search.py."""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from json_memory.smart import SmartMemory
from json_memory.search import AdvancedSearch, create_search, SearchResult

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d

@pytest.fixture
def mem(tmp_dir):
    mem_file = os.path.join(tmp_dir, "test_search.json")
    memory = SmartMemory(mem_file, max_chars=5000)

    # Store some facts
    memory.remember("user.name", "Alice")
    memory.remember("user.email", "alice@example.com")
    memory.remember("project.name", "Trading Bot")
    memory.remember("project.description", "A bot for trading cryptocurrencies")
    memory.remember("project.tags", ["crypto", "trading", "bot"])
    memory.remember("settings.theme", "dark")
    memory.remember("settings.notifications", True)

    return memory

@pytest.fixture
def search(mem):
    return AdvancedSearch(mem)

def test_create_search(mem):
    s = create_search(mem)
    assert isinstance(s, AdvancedSearch)
    assert s.memory is mem

def test_regex_search(search):
    # Search by value
    results = search.regex_search(r"alice.*com")
    assert len(results) == 1
    assert results[0].path == "user.email"
    assert results[0].value == "alice@example.com"
    assert results[0].match_type == "regex"

    # Search by path
    results = search.regex_search(r"^project\.", field="path")
    assert len(results) >= 2
    paths = {r.path for r in results}
    assert paths == {"project.name", "project.description", "project.tags"}

    # Search with case sensitive
    results_insensitive = search.regex_search(r"ALICE")
    assert len(results_insensitive) > 0

    results_sensitive = search.regex_search(r"ALICE", case_sensitive=True)
    assert len(results_sensitive) == 0

    # Invalid regex
    with pytest.raises(ValueError):
        search.regex_search(r"[invalid")
    # assert len(results) == 0

def test_fuzzy_search(search):
    # Typo in query
    results = search.fuzzy_search("tradng", threshold=0.4)
    assert len(results) > 0
    paths = {r.path for r in results}
    assert "project.name" in paths or "project.description" in paths or "project.tags" in paths

    # Fuzzy match on path
    results = search.fuzzy_search("projct", field="path", threshold=0.5)
    assert len(results) >= 2
    assert all(r.path.startswith("project.") for r in results)

def test_full_text_search(search):
    # Multiple words
    results = search.full_text_search("bot trading")
    assert len(results) > 0
    paths = {r.path for r in results}
    assert "project.description" in paths
    assert "project.name" in paths

    # Empty query
    assert search.full_text_search("") == []
    assert search.full_text_search("   ") == []

    # Case sensitive
    results = search.full_text_search("Bot Trading", case_sensitive=True)
    # assert len(results) == 0

    results = search.full_text_search("Trading Bot", case_sensitive=True)
    assert len(results) > 0
    assert results[0].path == "project.name"

def test_semantic_search_with_mock(search):
    mock_index = MagicMock()
    mock_index.search.return_value = [
        {"path": "project.description", "score": 0.95},
        {"path": "user.name", "score": 0.45}
    ]

    with patch("json_memory.semantic.get_semantic_index", return_value=mock_index, create=True):
        results = search.semantic_search("crypto bot", top_k=5)

        assert len(results) == 2
        assert results[0].path == "project.description"
        assert results[0].score == 0.95
        assert results[0].match_type == "semantic"
        assert results[1].path == "user.name"
        assert results[1].score == 0.45

def test_semantic_search_unavailable(search):
    with patch("json_memory.semantic.get_semantic_index", return_value=None, create=True):
        results = search.semantic_search("crypto bot")
        assert results == []

    # Mocking ImportError
    with patch("builtins.__import__") as mock_import:
        def import_side_effect(name, *args, **kwargs):
            if name == "json_memory.search":
                # We need to let it import search itself
                pass
            if 'semantic' in name:
                raise ImportError("No module named 'json_memory.semantic'")
            import importlib
            return importlib.__import__(name, *args, **kwargs)

        # We can just mock the import directly inside semantic_search
        # by patching the internal logic, or simpler: patching sys.modules

    with patch.dict("sys.modules", {"json_memory.semantic": None}):
        try:
            results = search.semantic_search("crypto bot")
            assert results == []
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")
            pass # Module mocking might behave differently, so fallback is fine

def test_search_unified(search):
    # Auto: Regex
    with patch.object(search, "regex_search") as mock_regex:
        search.search("/^project/")
        mock_regex.assert_called_once_with("^project")

    # Auto: Full text (multiple words)
    with patch.object(search, "full_text_search") as mock_full_text:
        search.search("crypto trading bot")
        mock_full_text.assert_called_once_with("crypto trading bot")

    # Auto: Fuzzy (single word)
    with patch.object(search, "fuzzy_search") as mock_fuzzy:
        search.search("tradng")
        mock_fuzzy.assert_called_once_with("tradng")

    # Explicit search types
    with patch.object(search, "regex_search") as mock_regex:
        search.search("pattern", search_type="regex")
        mock_regex.assert_called_once_with("pattern")

    with patch.object(search, "semantic_search") as mock_semantic:
        search.search("crypto", search_type="semantic")
        mock_semantic.assert_called_once_with("crypto")

    # Invalid search type
    with pytest.raises(ValueError, match="Unknown search type: invalid"):
        search.search("query", search_type="invalid")

def test_suggest(search):
    suggestions = search.suggest("proje")
    assert len(suggestions) == 3
    assert all(s.startswith("project.") for s in suggestions)

    # Case insensitive
    suggestions = search.suggest("PROJE")
    assert len(suggestions) == 3

    # Limit
    suggestions = search.suggest("proje", limit=2)
    assert len(suggestions) == 2

    # No matches
    assert search.suggest("nonexistent") == []
