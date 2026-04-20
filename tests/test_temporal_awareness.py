"""Tests for temporal awareness in SmartMemory."""
import json
import os
import tempfile
import time

import pytest

from json_memory.smart import SmartMemory, _detect_temporal_intent, _temporal_score, PathMeta


class TestTemporalIntentDetection:
    """Test temporal intent detection from queries."""
    
    def test_detect_recent(self):
        """Should detect 'recent' intent."""
        result = _detect_temporal_intent("What did I learn recently?")
        assert result['intent'] == 'recent'
        assert result['range_seconds'] == 604800  # 7 days
    
    def test_detect_past_with_range(self):
        """Should detect 'past' intent with explicit range."""
        result = _detect_temporal_intent("Show me facts from last 7 days")
        assert result['intent'] == 'past'
        assert result['range_seconds'] == 604800  # 7 days
    
    def test_detect_past_week(self):
        """Should detect 'past week' intent."""
        result = _detect_temporal_intent("What happened in the last week?")
        assert result['intent'] == 'past'
        assert result['range_seconds'] == 604800  # 7 days
    
    def test_detect_old(self):
        """Should detect 'old' intent."""
        result = _detect_temporal_intent("What old facts do I have?")
        assert result['intent'] == 'old'
        assert result['range_seconds'] == 7776000  # 90 days
    
    def test_detect_future(self):
        """Should detect 'future' intent."""
        result = _detect_temporal_intent("What's upcoming?")
        assert result['intent'] == 'future'
        assert result['range_seconds'] == 2592000  # 30 days
    
    def test_detect_no_temporal(self):
        """Should return None for non-temporal queries."""
        result = _detect_temporal_intent("What's my name?")
        assert result['intent'] is None
        assert result['range_seconds'] is None
    
    def test_detect_empty_query(self):
        """Should handle empty query."""
        result = _detect_temporal_intent("")
        assert result['intent'] is None
        assert result['range_seconds'] is None
    
    def test_detect_none_query(self):
        """Should handle None query."""
        result = _detect_temporal_intent(None)
        assert result['intent'] is None
        assert result['range_seconds'] is None


class TestTemporalScoring:
    """Test temporal scoring function."""
    
    @pytest.fixture
    def meta_recent(self):
        """Create metadata for a recently created fact."""
        meta = PathMeta()
        meta.created_at = time.time() - 3600  # 1 hour ago
        meta.last_accessed = time.time() - 1800  # 30 minutes ago
        return meta
    
    @pytest.fixture
    def meta_old(self):
        """Create metadata for an old fact."""
        meta = PathMeta()
        meta.created_at = time.time() - 7776000  # 90 days ago
        meta.last_accessed = time.time() - 7776000  # 90 days ago
        return meta
    
    def test_temporal_score_recent_intent(self, meta_recent, meta_old):
        """Recent intent should favor recent facts."""
        now = time.time()
        temporal_intent = {'intent': 'recent', 'range_seconds': 604800}  # 7 days
        
        score_recent = _temporal_score(meta_recent, temporal_intent, now)
        score_old = _temporal_score(meta_old, temporal_intent, now)
        
        assert score_recent > score_old
        assert score_recent == 1.0  # Within 7 days
        assert score_old == 0.0  # Older than 7 days
    
    def test_temporal_score_past_intent(self, meta_recent, meta_old):
        """Past intent should favor facts within range."""
        now = time.time()
        temporal_intent = {'intent': 'past', 'range_seconds': 2592000}  # 30 days
        
        score_recent = _temporal_score(meta_recent, temporal_intent, now)
        score_old = _temporal_score(meta_old, temporal_intent, now)
        
        assert score_recent == 1.0  # Within 30 days
        assert score_old == 0.0  # Older than 30 days
    
    def test_temporal_score_old_intent(self, meta_recent, meta_old):
        """Old intent should favor old facts."""
        now = time.time()
        temporal_intent = {'intent': 'old', 'range_seconds': 7776000}  # 90 days
        
        score_recent = _temporal_score(meta_recent, temporal_intent, now)
        score_old = _temporal_score(meta_old, temporal_intent, now)
        
        assert score_old > score_recent
        assert score_old == 1.0  # Older than 90 days
        assert score_recent < 1.0  # Recent
    
    def test_temporal_score_no_intent(self, meta_recent):
        """No temporal intent should return neutral score."""
        now = time.time()
        temporal_intent = {'intent': None, 'range_seconds': None}
        
        score = _temporal_score(meta_recent, temporal_intent, now)
        
        assert score == 0.5  # Neutral
    
    def test_temporal_score_future_intent(self, meta_recent):
        """Future intent should return neutral score."""
        now = time.time()
        temporal_intent = {'intent': 'future', 'range_seconds': 2592000}
        
        score = _temporal_score(meta_recent, temporal_intent, now)
        
        assert score == 0.5  # Neutral (can't predict future)


class TestSmartMemoryTemporal:
    """Test SmartMemory with temporal awareness."""
    
    @pytest.fixture
    def mem(self):
        """Create SmartMemory with temporal awareness."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            temp_path = f.name
        
        try:
            memory = SmartMemory(path=temp_path, max_chars=5000)
            yield memory
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_recent_query_finds_recent_facts(self, mem):
        """Query with 'recent' should find recently created facts."""
        # Add old fact
        mem.remember("test.old", "old fact")
        mem._meta["test.old"].created_at = time.time() - 7776000  # 90 days ago
        
        # Add recent fact
        mem.remember("test.recent", "recent fact")
        
        # Query for recent
        results = mem.recall_relevant("What did I learn recently?", max_results=2)
        
        assert "test.recent" in results
        assert "test.old" not in results
    
    def test_past_query_finds_facts_in_range(self, mem):
        """Query with 'last week' should find facts from last week."""
        # Add old fact
        mem.remember("test.old", "old fact")
        mem._meta["test.old"].created_at = time.time() - 7776000  # 90 days ago
        
        # Add fact from last week
        mem.remember("test.last_week", "last week fact")
        mem._meta["test.last_week"].created_at = time.time() - 432000  # 5 days ago
        
        # Query for last week
        results = mem.recall_relevant("Show me facts from last week", max_results=2)
        
        assert "test.last_week" in results
        assert "test.old" not in results
    
    def test_old_query_finds_old_facts(self, mem):
        """Query with 'old' should find old facts."""
        # Add recent fact
        mem.remember("test.recent", "recent fact")
        
        # Add old fact
        mem.remember("test.old", "old fact")
        mem._meta["test.old"].created_at = time.time() - 7776000  # 90 days ago
        
        # Query for old
        results = mem.recall_relevant("What old facts do I have?", max_results=2)
        
        assert "test.old" in results
        assert "test.recent" not in results
    
    def test_temporal_scoring_boosts_relevant_facts(self, mem):
        """Temporal scoring should boost facts matching temporal intent."""
        # Add facts with different ages
        mem.remember("test.recent", "recent fact about trading")
        mem.remember("test.old", "old fact about trading")
        mem._meta["test.old"].created_at = time.time() - 7776000  # 90 days ago
        
        # Query with temporal intent
        results = mem.recall_relevant("What did I learn recently about trading?", max_results=2)
        
        # Recent fact should be first
        assert list(results.keys())[0] == "test.recent"
    
    def test_no_temporal_query_uses_normal_scoring(self, mem):
        """Query without temporal intent should use normal scoring."""
        # Add facts with common keywords
        mem.remember("test.recent", "trading strategy recent")
        mem.remember("test.old", "trading strategy old")
        mem._meta["test.old"].created_at = time.time() - 7776000  # 90 days ago
        
        # Query without temporal intent but with matching keywords
        results = mem.recall_relevant("What trading strategies do I have?", max_results=2)
        
        # Should return both facts (normal scoring)
        assert len(results) == 2
    
    def test_temporal_with_keyword_matching(self, mem):
        """Temporal + keyword matching should work together."""
        # Add facts with different ages
        mem.remember("test.recent_trading", "recent trading strategy")
        mem.remember("test.old_trading", "old trading strategy")
        mem._meta["test.old_trading"].created_at = time.time() - 7776000  # 90 days ago
        mem.remember("test.recent_cooking", "recent cooking recipe")
        
        # Query with both temporal and keyword intent
        results = mem.recall_relevant("What did I learn recently about trading?", max_results=2)
        
        # Should find recent trading (should be first due to temporal boost)
        assert "test.recent_trading" in results
        # Old trading might still be in results if it has strong keyword match
        # but recent trading should be ranked higher
        if "test.old_trading" in results:
            # If both are present, recent should be first
            result_keys = list(results.keys())
            recent_index = result_keys.index("test.recent_trading")
            old_index = result_keys.index("test.old_trading")
            assert recent_index < old_index
        # Cooking should not be in results (no trading keyword)
        assert "test.recent_cooking" not in results
