import pytest
from json_memory.smart import SmartMemory
from json_memory.search import AdvancedSearch, SearchResult

class TestAdvancedSearch:
    @pytest.fixture
    def smart_memory(self, tmp_path):
        mem = SmartMemory(str(tmp_path / "test_search.json"), max_chars=5000)
        return mem

    @pytest.fixture
    def advanced_search(self, smart_memory):
        return AdvancedSearch(smart_memory)

    def test_full_text_search(self, smart_memory, advanced_search):
        smart_memory.remember("user.name", "Alice")
        smart_memory.remember("user.email", "alice@example.com")
        smart_memory.remember("project.name", "Trading Bot")
        smart_memory.remember("project.description", "A bot for trading cryptocurrencies")

        results = advanced_search.full_text_search("bot trading")
        assert len(results) > 0

        # Sort results to have a predictable order
        paths = [r.path for r in results]

        # Verify correct matching
        assert "project.description" in paths
        assert "project.name" in paths

    def test_full_text_search_empty_query(self, advanced_search):
        results = advanced_search.full_text_search("")
        assert results == []

        results = advanced_search.full_text_search("   ")
        assert results == []

    def test_full_text_search_case_sensitivity(self, smart_memory, advanced_search):
        smart_memory.remember("a", "Hello World")
        smart_memory.remember("b", "hello world")

        # Case insensitive (default)
        results = advanced_search.full_text_search("Hello")
        assert len(results) == 2

        # Case sensitive
        results = advanced_search.full_text_search("Hello", case_sensitive=True)
        assert len(results) == 1
        assert results[0].path == "a"

    def test_full_text_search_score_calculation(self, smart_memory, advanced_search):
        # A match on both path and value should have higher score than only one
        smart_memory.remember("apple.fruit", "A delicious red apple")
        smart_memory.remember("banana.fruit", "A long yellow banana")

        results = advanced_search.full_text_search("apple")
        assert len(results) == 1
        assert results[0].path == "apple.fruit"
        # Two words searched, "apple" matched in both path and value
        # Score calculation is matches / (len(words) * 2) => 2 / (1 * 2) = 1.0
        assert results[0].score == 1.0

    def test_full_text_search_none_value(self, smart_memory, advanced_search):
        # Force a None value in the underlying memory
        smart_memory.mem.set("test.none", None)
        smart_memory.remember("test.valid", "valid text")

        results = advanced_search.full_text_search("test")
        paths = [r.path for r in results]
        assert "test.valid" in paths
        assert "test.none" not in paths

    def test_full_text_search_non_string_value(self, smart_memory, advanced_search):
        # Test searching against non-string values
        smart_memory.remember("test.int", 12345)
        smart_memory.remember("test.bool", True)

        results = advanced_search.full_text_search("12345")
        assert len(results) == 1
        assert results[0].path == "test.int"

        results = advanced_search.full_text_search("True")
        assert len(results) == 1
        assert results[0].path == "test.bool"
