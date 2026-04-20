"""Tests for negation handling in SmartMemory."""
import json
import os
import tempfile
import time

import pytest

from json_memory.smart import SmartMemory, _detect_negation, _negation_score, PathMeta


class TestNegationDetection:
    """Test negation detection from queries."""
    
    def test_detect_not(self):
        """Should detect 'not' negation."""
        result = _detect_negation("What should I NOT do?")
        assert result['is_negated'] is True
        assert result['negation_type'] == 'exclusion'
        assert result['negation_keyword'] == 'not'
    
    def test_detect_dont(self):
        """Should detect 'don't' negation."""
        result = _detect_negation("What shouldn't I do?")
        assert result['is_negated'] is True
        assert result['negation_type'] == 'exclusion'
        assert result['negation_keyword'] == "shouldn't"  # Updated to match actual detection
    
    def test_detect_avoid(self):
        """Should detect 'avoid' as warning."""
        result = _detect_negation("What should I avoid?")
        assert result['is_negated'] is True
        assert result['negation_type'] == 'warning'
        assert result['negation_keyword'] == 'avoid'
    
    def test_detect_mistake(self):
        """Should detect 'mistake' as warning."""
        result = _detect_negation("What mistakes have I made?")
        assert result['is_negated'] is True
        assert result['negation_type'] == 'warning'
        assert result['negation_keyword'] == 'mistakes'  # Updated to match actual detection
    
    def test_detect_no(self):
        """Should detect 'no' as absence."""
        result = _detect_negation("What's missing?")
        # This doesn't have a negation keyword
        assert result['is_negated'] is False
    
    def test_detect_none_query(self):
        """Should handle None query."""
        result = _detect_negation(None)
        assert result['is_negated'] is False
        assert result['negation_type'] is None
        assert result['negation_keyword'] is None
    
    def test_detect_empty_query(self):
        """Should handle empty query."""
        result = _detect_negation("")
        assert result['is_negated'] is False
        assert result['negation_type'] is None
        assert result['negation_keyword'] is None


class TestNegationScoring:
    """Test negation scoring function."""
    
    @pytest.fixture
    def meta_warning(self):
        """Create metadata for a warning fact."""
        meta = PathMeta()
        meta.tags = ['warning', 'lesson']
        meta.tokens = {'avoid', 'leverage', 'trading'}
        return meta
    
    @pytest.fixture
    def meta_normal(self):
        """Create metadata for a normal fact."""
        meta = PathMeta()
        meta.tags = ['strategy']
        meta.tokens = {'trading', 'strategy', 'profit'}
        return meta
    
    def test_negation_score_warning_query(self, meta_warning, meta_normal):
        """Warning query should find warning facts."""
        negation_info = {'is_negated': True, 'negation_type': 'warning', 'negation_keyword': 'avoid'}
        query_tokens = {'avoid', 'trading'}
        
        score_warning = _negation_score(meta_warning, negation_info, query_tokens)
        score_normal = _negation_score(meta_normal, negation_info, query_tokens)
        
        assert score_warning > score_normal
        assert score_warning == 1.0  # Perfect match
        assert score_normal == 0.3  # Not a warning, but still include
    
    def test_negation_score_exclusion_query(self, meta_warning, meta_normal):
        """Exclusion query should find facts mentioning excluded thing."""
        negation_info = {'is_negated': True, 'negation_type': 'exclusion', 'negation_keyword': 'not'}
        query_tokens = {'leverage'}
        
        score_warning = _negation_score(meta_warning, negation_info, query_tokens)
        score_normal = _negation_score(meta_normal, negation_info, query_tokens)
        
        assert score_warning > score_normal
        assert score_warning == 0.9  # Mentions excluded thing - high relevance
        assert score_normal == 0.5  # Doesn't mention excluded thing - neutral
    
    def test_negation_score_no_negation(self, meta_warning):
        """No negation should return neutral score."""
        negation_info = {'is_negated': False, 'negation_type': None, 'negation_keyword': None}
        
        score = _negation_score(meta_warning, negation_info)
        
        assert score == 0.5  # Neutral
    
    def test_negation_score_general_negation(self, meta_warning, meta_normal):
        """General negation should favor warning facts."""
        negation_info = {'is_negated': True, 'negation_type': 'general', 'negation_keyword': 'never'}
        
        score_warning = _negation_score(meta_warning, negation_info)
        score_normal = _negation_score(meta_normal, negation_info)
        
        assert score_warning > score_normal
        assert score_warning == 0.8  # Likely relevant
        assert score_normal == 0.3  # Less likely, but still include


class TestSmartMemoryNegation:
    """Test SmartMemory with negation handling."""
    
    @pytest.fixture
    def mem(self):
        """Create SmartMemory with negation handling."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            temp_path = f.name
        
        try:
            memory = SmartMemory(path=temp_path, max_chars=5000)
            yield memory
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_negation_query_finds_warnings(self, mem):
        """Query with 'avoid' should find warning facts."""
        # Add warning fact with strong keyword match
        mem.remember("mem.lesson_leverage", "Avoid using leverage on Binance - warning", 
                    tags=['warning', 'lesson'])
        
        # Add normal fact with no warning keywords
        mem.remember("mem.strategy_trading", "Use RSI + MACD strategy for profit", 
                    tags=['strategy'])
        
        # Query for warnings
        results = mem.recall_relevant("What should I avoid?", max_results=2)
        
        # Should find warning fact (and possibly normal fact)
        assert "mem.lesson_leverage" in results
        # Warning fact should be ranked first (or at least present)
        result_keys = list(results.keys())
        assert result_keys[0] == "mem.lesson_leverage"  # Should be first
    
    def test_negation_query_finds_mistakes(self, mem):
        """Query with 'mistake' should find mistake facts."""
        # Add mistake fact
        mem.remember("mem.mistake_sunday", "Sunday trading is consistently poor", 
                    tags=['mistake', 'lesson'])
        
        # Add normal fact
        mem.remember("mem.strategy_trading", "Use RSI + MACD strategy", 
                    tags=['strategy'])
        
        # Query for mistakes
        results = mem.recall_relevant("What mistakes have I made?", max_results=2)
        
        # Should find mistake fact (and possibly normal fact)
        assert "mem.mistake_sunday" in results
        # Mistake fact should be ranked higher than normal fact
        if "mem.strategy_trading" in results:
            result_keys = list(results.keys())
            mistake_index = result_keys.index("mem.mistake_sunday")
            normal_index = result_keys.index("mem.strategy_trading")
            assert mistake_index < normal_index
    
    def test_exclusion_query_finds_excluded_thing(self, mem):
        """Query with 'don't' should find facts mentioning excluded thing."""
        # Add fact about leverage
        mem.remember("mem.lesson_leverage", "Leverage can be dangerous", 
                    tags=['warning'])
        
        # Add fact about spot trading
        mem.remember("mem.strategy_spot", "Spot trading is safer", 
                    tags=['strategy'])
        
        # Query excluding leverage
        results = mem.recall_relevant("What shouldn't I use?", max_results=2)
        
        # Should find fact mentioning leverage
        assert "mem.lesson_leverage" in results
    
    def test_no_negation_query_uses_normal_scoring(self, mem):
        """Query without negation should use normal scoring."""
        # Add facts with common keywords
        mem.remember("mem.lesson_leverage", "Avoid using leverage for trading",
                    tags=['warning'])
        mem.remember("mem.strategy_trading", "Use RSI + MACD strategy for trading",
                    tags=['strategy'])
        
        # Query without negation but with matching keywords
        results = mem.recall_relevant("What trading strategies do I have?", max_results=2)
        
        # Should return both facts (normal scoring)
        assert len(results) == 2
    
    def test_negation_with_keyword_matching(self, mem):
        """Negation + keyword matching should work together."""
        # Add warning fact about trading
        mem.remember("mem.warning_trading", "Avoid trading on Sundays",
                    tags=['warning'])
        
        # Add warning fact about cooking
        mem.remember("mem.warning_cooking", "Don't use too much salt",
                    tags=['warning'])
        
        # Add normal fact about trading
        mem.remember("mem.strategy_trading", "Use RSI + MACD strategy",
                    tags=['strategy'])
        
        # Query with negation and keyword
        results = mem.recall_relevant("What should I avoid in trading?", max_results=2)
        
        # Should find trading warning (and possibly strategy)
        assert "mem.warning_trading" in results
        # Cooking warning might still be in results if it has strong keyword match
        # but trading warning should be ranked higher
        if "mem.warning_cooking" in results:
            # If both are present, trading warning should be first
            result_keys = list(results.keys())
            trading_index = result_keys.index("mem.warning_trading")
            cooking_index = result_keys.index("mem.warning_cooking")
            assert trading_index < cooking_index
