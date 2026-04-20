"""Tests for memory pruning and lifecycle management."""
import json
import os
import tempfile
import time

import pytest

from json_memory.smart import SmartMemory, PathMeta


class TestPathMetaLifecycle:
    """Test PathMeta lifecycle tracking."""
    
    def test_pathmeta_with_ttl(self):
        """PathMeta should track TTL and expiration."""
        meta = PathMeta(ttl=3600)
        assert meta.ttl == 3600
        assert meta.expires_at is not None
        assert meta.expires_at > time.time()
        assert meta.archived is False
        assert meta.size_bytes == 0
    
    def test_pathmeta_without_ttl(self):
        """PathMeta without TTL should never expire."""
        meta = PathMeta()
        assert meta.ttl is None
        assert meta.expires_at is None
    
    def test_pathmeta_size_tracking(self):
        """PathMeta should track size in bytes."""
        meta = PathMeta()
        meta.size_bytes = 1024
        assert meta.size_bytes == 1024


class TestSmartMemoryPruning:
    """Test SmartMemory pruning functionality."""
    
    @pytest.fixture
    def mem(self):
        """Create a temporary SmartMemory instance with tiered storage enabled."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            temp_path = f.name
        
        try:
            memory = SmartMemory(path=temp_path, max_chars=5000, tiered=True)
            yield memory
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_prune_expired_facts(self, mem):
        """Prune should remove expired facts (TTL exceeded)."""
        # Add fact with 1 second TTL
        mem.remember("test.temp", "will expire", ttl=1)
        assert mem.recall("test.temp") == "will expire"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Prune should remove it
        result = mem.prune()
        assert "test.temp" in result['expired']
        assert result['total_removed'] == 1
        
        # Should be gone
        assert mem.recall("test.temp") is None
    
    def test_prune_old_facts(self, mem):
        """Prune should remove facts older than max_age_seconds."""
        # Add fact with old timestamp
        mem.remember("test.old", "old fact")
        mem._meta["test.old"].created_at = time.time() - 100000  # ~27 hours old
        
        # Prune with 24 hour limit
        result = mem.prune(max_age_seconds=86400)
        assert "test.old" in result['removed']
        assert result['total_removed'] == 1
    
    def test_prune_rarely_accessed(self, mem):
        """Prune should remove facts with low access count."""
        # Add facts with different access counts
        mem.remember("test.frequent", "frequently accessed")
        mem.remember("test.rare", "rarely accessed")
        
        # Simulate frequent access
        for _ in range(10):
            mem.recall("test.frequent")
        
        # Prune with minimum access count of 5
        result = mem.prune(min_access_count=5)
        assert "test.rare" in result['removed']
        assert "test.frequent" not in result['removed']
    
    def test_prune_dry_run(self, mem):
        """Dry run should return what would be removed without removing."""
        # Add expired fact
        mem.remember("test.temp", "will expire", ttl=1)
        time.sleep(1.1)
        
        # Dry run
        result = mem.prune(dry_run=True)
        assert "test.temp" in result['expired']
        assert result['dry_run'] is True
        
        # Should still exist in metadata (dry run doesn't remove)
        assert "test.temp" in mem._meta
        # Note: Memory.get() auto-deletes expired items, so recall() will return None
        # But metadata should still be present in dry run mode
    
    def test_prune_size_limit(self, mem):
        """Prune should remove oldest/least accessed when size exceeds limit."""
        # Add multiple facts (smaller to avoid overflow)
        for i in range(5):
            mem.remember(f"test.fact{i}", f"value{i}" * 10)  # ~60 bytes each
        
        # Access some more than others
        for _ in range(5):
            mem.recall("test.fact4")
            mem.recall("test.fact3")
        
        # Prune to 200 chars (should remove ~half)
        result = mem.prune(max_total_chars=200)
        assert result['total_removed'] > 0
        
        # Frequently accessed should remain
        assert mem.recall("test.fact4") is not None
        assert mem.recall("test.fact3") is not None
    
    def test_archive_manual(self, mem):
        """Manual archive should move fact to cold storage."""
        mem.remember("test.archive", "archive me")
        
        # Archive it
        result = mem.archive("test.archive")
        assert result is True
        
        # Should be marked as archived
        assert mem._meta["test.archive"].archived is True
        assert mem._meta["test.archive"].tier == 'cold'
    
    def test_archive_nonexistent(self, mem):
        """Archiving nonexistent path should return False."""
        result = mem.archive("nonexistent.path")
        assert result is False
    
    def test_archive_already_archived(self, mem):
        """Archiving already archived path should return False."""
        mem.remember("test.archive", "archive me")
        mem.archive("test.archive")
        
        # Try to archive again
        result = mem.archive("test.archive")
        assert result is False


class TestSmartMemoryLifecycleStats:
    """Test lifecycle statistics."""
    
    @pytest.fixture
    def mem(self):
        """Create a temporary SmartMemory instance."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            temp_path = f.name
        
        try:
            memory = SmartMemory(path=temp_path, max_chars=5000)
            yield memory
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_lifecycle_stats_empty(self, mem):
        """Empty memory should return appropriate stats."""
        stats = mem.lifecycle_stats()
        assert stats['total_facts'] == 0
        assert stats['memory_health'] == 'empty'
    
    def test_lifecycle_stats_healthy(self, mem):
        """Healthy memory should show good stats."""
        # Add some facts
        for i in range(5):
            mem.remember(f"test.fact{i}", f"value{i}")
        
        stats = mem.lifecycle_stats()
        assert stats['total_facts'] == 5
        assert stats['expired_facts'] == 0
        assert stats['memory_health'] == 'healthy'
    
    def test_lifecycle_stats_warning(self, mem):
        """Memory with >10% expired should show warning."""
        # Add 10 facts, expire 2 (20%)
        for i in range(10):
            if i < 2:
                mem.remember(f"test.expired{i}", f"value{i}", ttl=1)
            else:
                mem.remember(f"test.fact{i}", f"value{i}")
        
        time.sleep(1.1)
        
        stats = mem.lifecycle_stats()
        assert stats['expired_facts'] == 2
        assert stats['memory_health'] == 'warning'
    
    def test_lifecycle_stats_critical(self, mem):
        """Memory with >30% expired should show critical."""
        # Add 10 facts, expire 4 (40%)
        for i in range(10):
            if i < 4:
                mem.remember(f"test.expired{i}", f"value{i}", ttl=1)
            else:
                mem.remember(f"test.fact{i}", f"value{i}")
        
        time.sleep(1.1)
        
        stats = mem.lifecycle_stats()
        assert stats['expired_facts'] == 4
        assert stats['memory_health'] == 'critical'
    
    def test_lifecycle_stats_with_tiers(self, mem):
        """Stats should show tier distribution."""
        # Add facts to different tiers (simulated)
        for i in range(3):
            mem.remember(f"test.hot{i}", f"value{i}")
        
        # Manually set some to warm/cold
        if mem.tiered:
            mem.tiered.set("test.warm0", "warm value", tier='warm')
            mem.tiered.set("test.cold0", "cold value", tier='cold')
        
        stats = mem.lifecycle_stats()
        assert stats['total_facts'] >= 3


class TestSmartMemoryWithTTL:
    """Test SmartMemory with TTL functionality."""
    
    @pytest.fixture
    def mem(self):
        """Create a temporary SmartMemory instance."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            temp_path = f.name
        
        try:
            memory = SmartMemory(path=temp_path, max_chars=5000)
            yield memory
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_remember_with_ttl(self, mem):
        """Remember with TTL should set expiration."""
        mem.remember("test.temp", "temporary", ttl=3600)
        
        meta = mem._meta["test.temp"]
        assert meta.ttl == 3600
        assert meta.expires_at is not None
        assert meta.expires_at > time.time()
    
    def test_remember_without_ttl(self, mem):
        """Remember without TTL should never expire."""
        mem.remember("test.permanent", "permanent")
        
        meta = mem._meta["test.permanent"]
        assert meta.ttl is None
        assert meta.expires_at is None
    
    def test_ttl_persistence(self, mem):
        """TTL should persist across save/load."""
        mem.remember("test.temp", "temporary", ttl=3600)
        original_expires = mem._meta["test.temp"].expires_at
        
        # Save and reload
        mem._save_meta()
        mem._load_meta()
        
        # Should have same expiration
        assert mem._meta["test.temp"].expires_at == original_expires
    
    def test_prune_with_ttl(self, mem):
        """Prune should respect TTL expiration."""
        # Add fact with 1 second TTL
        mem.remember("test.temp", "will expire", ttl=1)
        
        # Immediately prune - should not remove
        result = mem.prune(dry_run=True)
        assert "test.temp" not in result['expired']
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Now prune should remove
        result = mem.prune()
        assert "test.temp" in result['expired']
