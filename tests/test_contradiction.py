import pytest
from json_memory.contradiction import ContradictionDetector, detect_contradictions

class TestContradictionDetector:
    @pytest.fixture
    def detector(self):
        return ContradictionDetector()

    def test_direct_contradiction(self, detector):
        existing = {"user.status": "online"}
        # Same path, different value should be a contradiction
        results = detector.detect("user.status", "offline", existing, allow_same_path=True)
        assert len(results) > 0
        assert results[0].contradiction_type == 'semantic'
        assert "online" in str(results[0].existing_value)
        assert "offline" in str(results[0].new_value)

    def test_direct_contradiction_generic(self, detector):
        existing = {"user.name": "Alice"}
        # Same path, different value, not semantic
        results = detector.detect("user.name", "Bob", existing, allow_same_path=True)
        assert len(results) > 0
        assert results[0].contradiction_type == 'direct'

    def test_semantic_contradiction(self, detector):
        existing = {"bot.running": "yes"}
        results = detector.detect("bot.running", "no", existing, allow_same_path=True)
        assert len(results) > 0
        assert results[0].contradiction_type == 'semantic'

    def test_negation_contradiction(self, detector):
        existing = {"user.status": "active"}
        results = detector.detect("user.status", "not active", existing, allow_same_path=True)
        assert len(results) > 0
        assert "negation" in results[0].explanation.lower()

    def test_temporal_contradiction(self, detector):
        existing = {"meeting.time": "after lunch"}
        results = detector.detect("meeting.time", "before lunch", existing, allow_same_path=True)
        assert len(results) > 0
        assert results[0].contradiction_type == 'temporal'

    def test_no_contradiction(self, detector):
        existing = {"user.name": "Alice"}
        results = detector.detect("user.age", 30, existing)
        assert len(results) == 0

    def test_different_subject_no_contradiction(self, detector):
        existing = {"user1.status": "active"}
        results = detector.detect("user2.status", "inactive", existing)
        assert len(results) == 0

    def test_direct_pattern_contradiction(self, detector):
        existing = {"fact1": "Alice is happy"}
        results = detector.detect("fact2", "Alice is sad", existing)
        assert len(results) > 0
        assert results[0].contradiction_type == 'direct'
        assert "Direct pattern" in results[0].explanation

    def test_temporal_pattern_contradiction(self, detector):
        existing = {"event1": "lunch after meeting"}
        results = detector.detect("event2", "lunch before meeting", existing)
        assert len(results) > 0
        assert results[0].contradiction_type == 'temporal'

    def test_non_string_values(self, detector):
        existing = {"user.age": 25}
        results = detector.detect("user.age", 26, existing, allow_same_path=True)
        assert len(results) > 0
        assert results[0].contradiction_type == 'direct'

    def test_no_meaningful_difference(self, detector):
        existing = {"user.name": "Alice"}
        # Case and whitespace should not be a contradiction
        results = detector.detect("user.name", " alice ", existing, allow_same_path=True)
        assert len(results) == 0

        # Small typo (1 char) should not be a contradiction
        results = detector.detect("user.name", "Alic", existing, allow_same_path=True)
        assert len(results) == 0
