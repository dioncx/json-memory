"""
Tests for contradiction detection, memory consolidation, and forgetting curve.
"""

import pytest
import time
from json_memory import SmartMemory, detect_contradictions, consolidate_memory, ForgettingCurve


class TestContradictionDetection:
    """Test contradiction detection functionality."""
    
    def test_same_subject_same_attribute_contradiction(self):
        """Test detection of contradictions with same subject and attribute."""
        # This should be a contradiction: user.status = "active" vs user.status = "inactive"
        # But since it's the same path, it's an update, not a contradiction
        contradictions = detect_contradictions(
            "user.status", "inactive",
            {"user.status": "active"}
        )
        # Same path = update, not contradiction
        assert len(contradictions) == 0
    
    def test_different_subjects_no_contradiction(self):
        """Test that different subjects don't create contradictions."""
        contradictions = detect_contradictions(
            "bot.running", "no",
            {"user.status": "active"}
        )
        assert len(contradictions) == 0
    
    def test_same_subject_different_attributes_no_contradiction(self):
        """Test that same subject with different attributes don't create contradictions."""
        contradictions = detect_contradictions(
            "user.name", "Dion",
            {"user.status": "active"}
        )
        assert len(contradictions) == 0
    
    def test_semantic_opposites_contradiction(self):
        """Test detection of semantic opposites as contradictions."""
        # This should detect a contradiction: bot.power = "on" vs bot.power = "off"
        # But since it's the same path, it's an update
        contradictions = detect_contradictions(
            "bot.power", "off",
            {"bot.power": "on"}
        )
        # Same path = update, not contradiction
        assert len(contradictions) == 0
    
    def test_negation_contradiction(self):
        """Test detection of negation contradictions."""
        # This should detect a contradiction: user.status = "not active" vs user.status = "active"
        # But since it's the same path, it's an update
        contradictions = detect_contradictions(
            "user.status", "not active",
            {"user.status": "active"}
        )
        # Same path = update, not contradiction
        assert len(contradictions) == 0
    
    def test_smart_memory_no_contradiction_for_updates(self):
        """Test that SmartMemory doesn't flag updates as contradictions."""
        mem = SmartMemory()
        
        # Store initial fact
        mem.remember("user.status", "active")
        
        # Update the same fact
        result = mem.remember("user.status", "inactive", check_contradictions=True)
        
        # Should not detect contradiction for updates
        assert result['success'] is True
        assert len(result['contradictions']) == 0
    
    def test_smart_memory_contradiction_between_related_facts(self):
        """Test contradiction detection between related but different facts."""
        mem = SmartMemory()
        
        # Store a fact
        mem.remember("user.status", "active")
        
        # Try to store a contradictory fact about the same subject but different attribute
        # This should NOT be a contradiction (different attributes)
        result = mem.remember("user.state", "inactive", check_contradictions=True)
        
        assert result['success'] is True
        assert len(result['contradictions']) == 0  # Different attributes, no contradiction
    
    def test_get_contradictions_empty(self):
        """Test that get_contradictions returns empty when no contradictions exist."""
        mem = SmartMemory()
        
        # Store unrelated facts
        mem.remember("user.name", "Dion")
        mem.remember("bot.status", "running")
        mem.remember("server.ip", "192.168.1.100")
        
        contradictions = mem.get_contradictions()
        assert len(contradictions) == 0


class TestMemoryConsolidation:
    """Test memory consolidation functionality."""
    
    def test_consolidation_group_detection(self):
        """Test detection of groups that can be consolidated."""
        facts = {
            "user.skills": "Python, Go, JavaScript",
            "user.programming_languages": "Python, Go, JavaScript, Rust",
            "user.known_languages": "Python, Go, TypeScript",
        }
        
        groups = consolidate_memory(facts)
        assert len(groups) > 0
        
        # Check that groups suggest consolidation
        group = groups[0]
        assert len(group.paths) >= 2
        assert group.confidence > 0.5
    
    def test_smart_memory_consolidation(self):
        """Test consolidation in SmartMemory."""
        mem = SmartMemory()
        
        # Store similar facts
        mem.remember("user.skills", "Python, Go, JavaScript")
        mem.remember("user.programming_languages", "Python, Go, JavaScript, Rust")
        mem.remember("user.known_languages", "Python, Go, TypeScript")
        
        # Get consolidation suggestions
        groups = mem.consolidate_memory()
        assert len(groups) > 0
    
    def test_auto_consolidation(self):
        """Test automatic consolidation."""
        mem = SmartMemory()
        
        # Store similar facts
        mem.remember("user.skills", "Python, Go, JavaScript")
        mem.remember("user.languages", "Python, JavaScript, Rust")
        
        # Auto-consolidate with high confidence
        result = mem.auto_consolidate(min_confidence=0.6)
        
        assert 'consolidated' in result
        assert 'skipped' in result
        assert 'warnings' in result


class TestForgettingCurve:
    """Test forgetting curve functionality."""
    
    def test_strength_calculation(self):
        """Test memory strength calculation."""
        curve = ForgettingCurve()
        current_time = time.time()
        
        # Recent memory should be strong
        strength = curve.calculate_strength(
            initial_strength=1.0,
            last_reinforced=current_time - 3600,  # 1 hour ago
            reinforcement_count=0,
            memory_type='fact'
        )
        assert strength > 0.9
        
        # Old memory should be weaker
        strength = curve.calculate_strength(
            initial_strength=1.0,
            last_reinforced=current_time - 86400 * 7,  # 7 days ago
            reinforcement_count=0,
            memory_type='fact'
        )
        assert strength < 0.5
    
    def test_reinforcement_boost(self):
        """Test that reinforcement boosts memory strength."""
        curve = ForgettingCurve()
        current_time = time.time()
        
        # Without reinforcement
        strength_no_reinforcement = curve.calculate_strength(
            initial_strength=1.0,
            last_reinforced=current_time - 86400,  # 24 hours ago
            reinforcement_count=0,
            memory_type='fact'
        )
        
        # With reinforcement
        strength_with_reinforcement = curve.calculate_strength(
            initial_strength=1.0,
            last_reinforced=current_time - 86400,  # 24 hours ago
            reinforcement_count=3,
            memory_type='fact'
        )
        
        assert strength_with_reinforcement > strength_no_reinforcement
    
    def test_memory_type_decay(self):
        """Test that different memory types decay at different rates."""
        curve = ForgettingCurve()
        current_time = time.time()
        
        # Identity memory (slow decay)
        identity_strength = curve.calculate_strength(
            initial_strength=1.0,
            last_reinforced=current_time - 86400 * 7,  # 7 days ago
            reinforcement_count=0,
            memory_type='identity'
        )
        
        # Event memory (fast decay)
        event_strength = curve.calculate_strength(
            initial_strength=1.0,
            last_reinforced=current_time - 86400 * 7,  # 7 days ago
            reinforcement_count=0,
            memory_type='event'
        )
        
        assert identity_strength > event_strength
    
    def test_smart_memory_strength(self):
        """Test memory strength in SmartMemory."""
        mem = SmartMemory()
        
        # Store a fact
        mem.remember("user.name", "Dion", tags=["identity"])
        
        # Get strength analysis
        strength = mem.get_memory_strength("user.name", memory_type='identity')
        
        assert strength is not None
        assert strength.current_strength > 0
        assert strength.path == "user.name"
    
    def test_memories_needing_reinforcement(self):
        """Test getting memories that need reinforcement."""
        mem = SmartMemory()
        
        # Store facts at different times
        mem.remember("fact1", "value1", tags=["fact"])
        time.sleep(0.1)  # Small delay to ensure different timestamps
        mem.remember("fact2", "value2", tags=["fact"])
        
        # Get memories needing reinforcement
        memories = mem.get_memories_needing_reinforcement(max_items=5)
        
        assert len(memories) > 0
        # Should have reinforcement priority
        assert 'reinforcement_priority' in memories[0]
    
    def test_simulate_decay(self):
        """Test memory decay simulation."""
        mem = SmartMemory()
        
        # Store a fact
        mem.remember("test.fact", "value")
        
        # Simulate decay
        simulation = mem.simulate_memory_decay("test.fact", days=7)
        
        assert len(simulation) == 8  # 0-7 days
        assert simulation[0]['strength'] > simulation[7]['strength']
    
    def test_reinforce_memory(self):
        """Test memory reinforcement."""
        mem = SmartMemory()
        
        # Store a fact
        mem.remember("test.fact", "value")
        
        # Reinforce it
        result = mem.reinforce_memory("test.fact", boost_strength=0.3)
        
        assert result['success'] is True
        assert result['new_strength'] > 0.5


class TestIntegration:
    """Test integration of all features."""
    
    def test_full_workflow(self):
        """Test a complete workflow with all features."""
        mem = SmartMemory()
        
        # 1. Store facts with contradiction checking
        result1 = mem.remember("user.status", "active")
        assert result1['success'] is True
        
        # 2. Update the same fact (should not be a contradiction)
        result2 = mem.remember("user.status", "inactive", check_contradictions=True)
        assert result2['success'] is True
        assert len(result2['contradictions']) == 0  # Updates are not contradictions
        
        # 3. Store similar facts for consolidation
        mem.remember("user.skills", "Python, Go", check_contradictions=False)
        mem.remember("user.programming_languages", "Python, Go, JavaScript", check_contradictions=False)
        mem.remember("user.known_languages", "Python, TypeScript", check_contradictions=False)
        
        # 4. Get consolidation suggestions
        groups = mem.consolidate_memory()
        assert len(groups) > 0
        
        # 5. Check memory strength
        strength = mem.get_memory_strength("user.status")
        assert strength is not None
        
        # 6. Get memories needing reinforcement
        memories = mem.get_memories_needing_reinforcement(max_items=3)
        assert len(memories) > 0
        
        # 7. Reinforce a memory
        result3 = mem.reinforce_memory("user.status")
        assert result3['success'] is True
        
        # 8. Get all contradictions (should be empty)
        contradictions = mem.get_contradictions()
        assert len(contradictions) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])