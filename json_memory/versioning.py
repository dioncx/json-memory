"""
Memory Versioning — Track history of all memory changes.

This module provides versioning capabilities for memory:
- Track all changes with timestamps
- Query memory state at any point in time
- Diff between versions
- Audit trail of who/what/when
"""

import time
import json
import copy
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class MemoryVersion:
    """Represents a single version of memory state."""
    version_id: str
    timestamp: float
    path: str
    old_value: Any
    new_value: Any
    operation: str  # 'set', 'delete', 'update'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryDiff:
    """Represents differences between two memory states."""
    added: Dict[str, Any]
    modified: Dict[str, Tuple[Any, Any]]  # path -> (old_value, new_value)
    deleted: Dict[str, Any]
    timestamp_old: float
    timestamp_new: float


class MemoryVersioning:
    """Track and query memory history."""
    
    def __init__(self, max_history: int = 10000):
        """Initialize versioning system.
        
        Args:
            max_history: Maximum number of versions to keep
        """
        self.max_history = max_history
        self.versions: List[MemoryVersion] = []
        self.path_index: Dict[str, List[int]] = defaultdict(list)  # path -> version indices
        self.version_counter = 0
    
    def record_change(self, path: str, old_value: Any, new_value: Any, 
                     operation: str = 'set', metadata: Dict[str, Any] = None) -> str:
        """Record a memory change.
        
        Args:
            path: Memory path that changed
            old_value: Previous value (None for new paths)
            new_value: New value (None for deletions)
            operation: Type of operation ('set', 'delete', 'update')
            metadata: Optional metadata (source, user, etc.)
            
        Returns:
            Version ID
        """
        version_id = f"v{self.version_counter}"
        self.version_counter += 1
        
        version = MemoryVersion(
            version_id=version_id,
            timestamp=time.time(),
            path=path,
            old_value=copy.deepcopy(old_value),
            new_value=copy.deepcopy(new_value),
            operation=operation,
            metadata=metadata or {}
        )
        
        self.versions.append(version)
        self.path_index[path].append(len(self.versions) - 1)
        
        # Trim if needed
        if len(self.versions) > self.max_history:
            self._trim_history()
        
        return version_id
    
    def get_history(self, path: str = None, limit: int = 100,
                   start_time: float = None, end_time: float = None) -> List[MemoryVersion]:
        """Get version history.
        
        Args:
            path: Optional path to filter by
            limit: Maximum versions to return
            start_time: Optional start timestamp
            end_time: Optional end timestamp
            
        Returns:
            List of MemoryVersion objects
        """
        if path:
            # Get versions for specific path
            indices = self.path_index.get(path, [])
            versions = [self.versions[i] for i in indices if i < len(self.versions)]
        else:
            # Get all versions
            versions = self.versions
        
        # Filter by time
        if start_time:
            versions = [v for v in versions if v.timestamp >= start_time]
        if end_time:
            versions = [v for v in versions if v.timestamp <= end_time]
        
        # Sort by timestamp (newest first)
        versions = sorted(versions, key=lambda v: v.timestamp, reverse=True)
        
        return versions[:limit]
    
    def get_value_at(self, path: str, timestamp: float) -> Tuple[Any, bool]:
        """Get the value of a path at a specific time.
        
        Args:
            path: Memory path
            timestamp: Target timestamp
            
        Returns:
            Tuple of (value, found)
        """
        # Get all versions for this path
        indices = self.path_index.get(path, [])
        
        if not indices:
            return None, False
        
        # Find the version just before or at timestamp
        for i in reversed(indices):
            version = self.versions[i]
            if version.timestamp <= timestamp:
                # If it was deleted at this point, return None
                if version.new_value is None:
                    return None, True
                return version.new_value, True
        
        # No version before timestamp
        return None, False
    
    def get_state_at(self, timestamp: float) -> Dict[str, Any]:
        """Get complete memory state at a specific time.
        
        Args:
            timestamp: Target timestamp
            
        Returns:
            Dict of path -> value
        """
        state = {}
        
        # For each path, find its value at timestamp
        for path in self.path_index.keys():
            value, found = self.get_value_at(path, timestamp)
            if found and value is not None:
                state[path] = value
        
        return state
    
    def diff(self, timestamp_old: float, timestamp_new: float) -> MemoryDiff:
        """Get differences between two points in time.
        
        Args:
            timestamp_old: Earlier timestamp
            timestamp_new: Later timestamp
            
        Returns:
            MemoryDiff object
        """
        state_old = self.get_state_at(timestamp_old)
        state_new = self.get_state_at(timestamp_new)
        
        added = {}
        modified = {}
        deleted = {}
        
        # Find added and modified
        for path, new_value in state_new.items():
            if path not in state_old:
                added[path] = new_value
            elif state_old[path] != new_value:
                modified[path] = (state_old[path], new_value)
        
        # Find deleted
        for path, old_value in state_old.items():
            if path not in state_new:
                deleted[path] = old_value
        
        return MemoryDiff(
            added=added,
            modified=modified,
            deleted=deleted,
            timestamp_old=timestamp_old,
            timestamp_new=timestamp_new
        )
    
    def get_recent_changes(self, seconds: float = 3600, limit: int = 100) -> List[MemoryVersion]:
        """Get recent changes within time window.
        
        Args:
            seconds: Time window in seconds
            limit: Maximum versions to return
            
        Returns:
            List of recent MemoryVersion objects
        """
        cutoff = time.time() - seconds
        return self.get_history(limit=limit, start_time=cutoff)
    
    def get_change_count(self, path: str = None, seconds: float = None) -> int:
        """Get count of changes.
        
        Args:
            path: Optional path filter
            seconds: Optional time window
            
        Returns:
            Number of changes
        """
        if path:
            versions = self.get_history(path=path, limit=float('inf'))
        else:
            versions = self.versions
        
        if seconds:
            cutoff = time.time() - seconds
            versions = [v for v in versions if v.timestamp >= cutoff]
        
        return len(versions)
    
    def get_most_changed(self, limit: int = 10, seconds: float = None) -> List[Tuple[str, int]]:
        """Get most frequently changed paths.
        
        Args:
            limit: Maximum paths to return
            seconds: Optional time window
            
        Returns:
            List of (path, change_count) tuples
        """
        counts = []
        
        for path, indices in self.path_index.items():
            if seconds:
                cutoff = time.time() - seconds
                count = sum(1 for i in indices if self.versions[i].timestamp >= cutoff)
            else:
                count = len(indices)
            
            if count > 0:
                counts.append((path, count))
        
        # Sort by count (descending)
        counts.sort(key=lambda x: x[1], reverse=True)
        
        return counts[:limit]
    
    def export_history(self, path: str = None, format: str = 'json') -> str:
        """Export version history.
        
        Args:
            path: Optional path filter
            format: Export format ('json', 'csv')
            
        Returns:
            Exported history as string
        """
        versions = self.get_history(path=path, limit=float('inf'))
        
        if format == 'json':
            data = []
            for v in versions:
                data.append({
                    'version_id': v.version_id,
                    'timestamp': v.timestamp,
                    'path': v.path,
                    'old_value': v.old_value,
                    'new_value': v.new_value,
                    'operation': v.operation,
                    'metadata': v.metadata
                })
            return json.dumps(data, indent=2, default=str)
        
        elif format == 'csv':
            lines = ['version_id,timestamp,path,operation,old_value,new_value']
            for v in versions:
                old_str = json.dumps(v.old_value) if v.old_value else ''
                new_str = json.dumps(v.new_value) if v.new_value else ''
                lines.append(f"{v.version_id},{v.timestamp},{v.path},{v.operation},{old_str},{new_str}")
            return '\n'.join(lines)
        
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _trim_history(self):
        """Trim history to max_history."""
        if len(self.versions) <= self.max_history:
            return
        
        # Remove oldest versions
        to_remove = len(self.versions) - self.max_history
        self.versions = self.versions[to_remove:]
        
        # Rebuild index
        self.path_index.clear()
        for i, version in enumerate(self.versions):
            self.path_index[version.path].append(i)
    
    def clear(self, before_timestamp: float = None):
        """Clear version history.
        
        Args:
            before_timestamp: Optional clear only before this timestamp
        """
        if before_timestamp:
            self.versions = [v for v in self.versions if v.timestamp >= before_timestamp]
        else:
            self.versions = []
        
        # Rebuild index
        self.path_index.clear()
        for i, version in enumerate(self.versions):
            self.path_index[version.path].append(i)
    
    def stats(self) -> Dict[str, Any]:
        """Get versioning statistics.
        
        Returns:
            Dict with statistics
        """
        if not self.versions:
            return {
                'total_versions': 0,
                'unique_paths': 0,
                'oldest_timestamp': None,
                'newest_timestamp': None
            }
        
        timestamps = [v.timestamp for v in self.versions]
        
        return {
            'total_versions': len(self.versions),
            'unique_paths': len(self.path_index),
            'oldest_timestamp': min(timestamps),
            'newest_timestamp': max(timestamps),
            'avg_versions_per_path': len(self.versions) / len(self.path_index) if self.path_index else 0
        }


def create_versioning(max_history: int = 10000) -> MemoryVersioning:
    """Convenience function to create MemoryVersioning."""
    return MemoryVersioning(max_history=max_history)


# Test the versioning system
if __name__ == "__main__":
    versioning = MemoryVersioning()
    
    # Record some changes
    print("Recording changes...")
    versioning.record_change("user.name", None, "Alice")
    time.sleep(0.1)
    versioning.record_change("user.name", "Alice", "Bob")
    time.sleep(0.1)
    versioning.record_change("user.age", None, 30)
    
    # Get history
    print("\nHistory for user.name:")
    for v in versioning.get_history("user.name"):
        print(f"  {v.version_id}: {v.old_value} -> {v.new_value}")
    
    # Get value at specific time
    now = time.time()
    value, found = versioning.get_value_at("user.name", now - 0.05)
    print(f"\nValue at {now - 0.05}: {value} (found: {found})")
    
    # Get stats
    print(f"\nStats: {versioning.stats()}")