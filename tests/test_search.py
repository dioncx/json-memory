import pytest
from json_memory.smart import SmartMemory
from json_memory.search import AdvancedSearch

@pytest.fixture
def mem(tmp_path):
    path = tmp_path / "test_search.json"
    mem = SmartMemory(str(path))
    mem.remember("user.name", "Alice")
    mem.remember("user.email", "alice@example.com")
    mem.remember("project.name", "Trading Bot")
    mem.remember("project.description", "A bot for trading cryptocurrencies")
    return mem

@pytest.fixture
def search(mem):
    return AdvancedSearch(mem)

def test_regex_search_value(search):
    """Test regex search on values."""
    results = search.regex_search("alice", field="value")
    assert len(results) == 2
    paths = {r.path for r in results}
    assert "user.name" in paths
    assert "user.email" in paths

    # Check score and highlights
    result = next(r for r in results if r.path == "user.name")
    assert result.score == 0.5
    assert result.match_type == "regex"
    assert len(result.highlights) == 1
    assert "value:" in result.highlights[0]

def test_regex_search_path(search):
    """Test regex search on paths."""
    results = search.regex_search(r"^user\.", field="path")
    assert len(results) == 2
    paths = {r.path for r in results}
    assert "user.name" in paths
    assert "user.email" in paths

def test_regex_search_both(search):
    """Test regex search on both path and value."""
    # Add an entry that matches both in path and value
    search.memory.remember("bot.trading.status", "Trading is active")

    results = search.regex_search("trading", field="both")
    assert len(results) >= 1

    # The new entry should score 1.0 since it matches both
    result = next(r for r in results if r.path == "bot.trading.status")
    assert result.score == 1.0
    assert len(result.highlights) == 2
    assert any("path:" in h for h in result.highlights)
    assert any("value:" in h for h in result.highlights)

def test_regex_search_case_sensitive(search):
    """Test case-sensitive regex search."""
    results = search.regex_search("Alice", case_sensitive=True, field="value")
    assert len(results) == 1
    assert results[0].path == "user.name"

    results_lower = search.regex_search("alice", case_sensitive=True, field="value")
    assert len(results_lower) == 1
    assert results_lower[0].path == "user.email"

def test_regex_search_invalid_regex(search):
    """Test handling of invalid regex patterns."""
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        search.regex_search("[invalid")

def test_regex_search_no_results(search):
    """Test when no results match the regex."""
    results = search.regex_search("nonexistent_pattern")
    assert len(results) == 0
