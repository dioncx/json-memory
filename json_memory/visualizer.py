"""
Memory Visualization — Visualize memory structure and relationships.

This module provides functionality to visualize memory structure,
relationships, and statistics in various formats.
"""

import json
import time
from typing import Optional, List, Dict, Any
from collections import defaultdict


class MemoryVisualizer:
    """Visualize memory structure and relationships."""
    
    def __init__(self, memory):
        """Initialize with a SmartMemory instance."""
        self.memory = memory
    
    def tree_view(self, max_depth: int = 3, show_values: bool = True) -> str:
        """Generate a tree view of memory structure.
        
        Args:
            max_depth: Maximum depth to show
            show_values: Whether to show values
            
        Returns:
            String representation of memory tree
        """
        with self.memory._lock:
            lines = []
            lines.append("Memory Tree:")
            lines.append("=" * 40)
            
            # Get all paths
            paths = self.memory.mem.paths()
            
            # Build tree structure
            tree = {}
            for path in paths:
                parts = path.split('.')
                current = tree
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = self.memory.mem.get(path)
            
            # Print tree
            def print_tree(node, prefix="", depth=0):
                if depth > max_depth:
                    return
                
                if isinstance(node, dict):
                    items = sorted(node.items())
                    for i, (key, value) in enumerate(items):
                        is_last = i == len(items) - 1
                        connector = "└── " if is_last else "├── "
                        
                        if isinstance(value, dict):
                            lines.append(f"{prefix}{connector}{key}/")
                            new_prefix = prefix + ("    " if is_last else "│   ")
                            print_tree(value, new_prefix, depth + 1)
                        else:
                            value_str = ""
                            if show_values:
                                value_str = f" = {self._format_value(value)}"
                            lines.append(f"{prefix}{connector}{key}{value_str}")
                else:
                    lines.append(f"{prefix}└── {self._format_value(node)}")
            
            print_tree(tree)
            return "\n".join(lines)
    
    def stats_view(self) -> str:
        """Generate a statistics view of memory.
        
        Returns:
            String representation of memory statistics
        """
        with self.memory._lock:
            lines = []
            lines.append("Memory Statistics:")
            lines.append("=" * 40)
            
            # Basic stats
            stats = self.memory.stats()
            lines.append(f"Total facts: {stats['entries']}")
            lines.append(f"Memory usage: {stats['chars_used']}/{stats['chars_max']} chars")
            lines.append(f"Utilization: {stats['utilization']}")
            
            # Lifecycle stats
            lifecycle = self.memory.lifecycle_stats()
            lines.append(f"\nLifecycle:")
            lines.append(f"  Hot facts: {lifecycle['hot_facts']}")
            lines.append(f"  Warm facts: {lifecycle['warm_facts']}")
            lines.append(f"  Cold facts: {lifecycle['cold_facts']}")
            lines.append(f"  Expired facts: {lifecycle['expired_facts']}")
            lines.append(f"  Health: {lifecycle['memory_health']}")
            
            # Path statistics
            paths = self.memory.mem.paths()
            if paths:
                # Count by depth
                depth_counts = defaultdict(int)
                for path in paths:
                    depth = path.count('.') + 1
                    depth_counts[depth] += 1
                
                lines.append(f"\nPath depths:")
                for depth in sorted(depth_counts.keys()):
                    lines.append(f"  Depth {depth}: {depth_counts[depth]} facts")
                
                # Count by prefix
                prefix_counts = defaultdict(int)
                for path in paths:
                    prefix = path.split('.')[0]
                    prefix_counts[prefix] += 1
                
                lines.append(f"\nTop prefixes:")
                for prefix, count in sorted(prefix_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    lines.append(f"  {prefix}: {count} facts")
            
            return "\n".join(lines)
    
    def strength_view(self, top_n: int = 10) -> str:
        """Generate a view of memory strengths.
        
        Args:
            top_n: Number of top memories to show
            
        Returns:
            String representation of memory strengths
        """
        with self.memory._lock:
            lines = []
            lines.append("Memory Strengths:")
            lines.append("=" * 40)
            
            # Get all memories with strengths
            memories = []
            for path in self.memory.mem.paths():
                strength = self.memory.get_memory_strength(path)
                if strength:
                    memories.append({
                        'path': path,
                        'strength': strength.current_strength,
                        'last_accessed': strength.last_reinforced,
                        'reinforcement_count': strength.reinforcement_count,
                    })
            
            # Sort by strength (lowest first - need reinforcement)
            memories.sort(key=lambda x: x['strength'])
            
            # Show weakest memories
            lines.append(f"\nWeakest memories (need reinforcement):")
            for i, memory in enumerate(memories[:top_n]):
                lines.append(f"  {i+1}. {memory['path']}: {memory['strength']:.3f}")
            
            # Show strongest memories
            lines.append(f"\nStrongest memories:")
            for i, memory in enumerate(memories[-top_n:]):
                lines.append(f"  {i+1}. {memory['path']}: {memory['strength']:.3f}")
            
            return "\n".join(lines)
    
    def contradiction_view(self) -> str:
        """Generate a view of contradictions in memory.
        
        Returns:
            String representation of contradictions
        """
        with self.memory._lock:
            lines = []
            lines.append("Memory Contradictions:")
            lines.append("=" * 40)
            
            contradictions = self.memory.get_contradictions()
            
            if not contradictions:
                lines.append("No contradictions found!")
                return "\n".join(lines)
            
            lines.append(f"Found {len(contradictions)} contradictions:")
            for i, contradiction in enumerate(contradictions, 1):
                lines.append(f"\n{i}. {contradiction.explanation}")
                lines.append(f"   Confidence: {contradiction.confidence:.2f}")
                lines.append(f"   Type: {contradiction.contradiction_type}")
            
            return "\n".join(lines)
    
    def consolidation_view(self, max_groups: int = 5) -> str:
        """Generate a view of consolidation opportunities.
        
        Args:
            max_groups: Maximum groups to show
            
        Returns:
            String representation of consolidation opportunities
        """
        with self.memory._lock:
            lines = []
            lines.append("Memory Consolidation Opportunities:")
            lines.append("=" * 40)
            
            groups = self.memory.consolidate_memory(max_groups=max_groups)
            
            if not groups:
                lines.append("No consolidation opportunities found!")
                return "\n".join(lines)
            
            lines.append(f"Found {len(groups)} groups:")
            for i, group in enumerate(groups, 1):
                lines.append(f"\n{i}. Confidence: {group.confidence:.2f}")
                lines.append(f"   Paths: {group.paths}")
                lines.append(f"   Suggested: {group.suggested_path} = {group.suggested_value}")
                lines.append(f"   Reason: {group.reason}")
            
            return "\n".join(lines)
    
    def timeline_view(self, max_episodes: int = 10) -> str:
        """Generate a timeline view of conversation episodes.
        
        Args:
            max_episodes: Maximum episodes to show
            
        Returns:
            String representation of conversation timeline
        """
        with self.memory._lock:
            lines = []
            lines.append("Conversation Timeline:")
            lines.append("=" * 40)
            
            episodes = self.memory.recall_episodes(max_age_seconds=float('inf'), limit=max_episodes)
            
            if not episodes:
                lines.append("No episodes found!")
                return "\n".join(lines)
            
            for i, episode in enumerate(episodes, 1):
                timestamp = episode['timestamp']
                time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
                lines.append(f"\n{i}. [{time_str}] {episode['topic']}")
                lines.append(f"   {episode['summary']}")
                if episode.get('paths'):
                    lines.append(f"   Related: {', '.join(episode['paths'])}")
            
            return "\n".join(lines)
    
    def _format_value(self, value: Any, max_length: int = 50) -> str:
        """Format a value for display."""
        if isinstance(value, str):
            if len(value) > max_length:
                return value[:max_length-3] + "..."
            return value
        elif isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        else:
            return str(value)
    
    def full_report(self) -> str:
        """Generate a full memory report.
        
        Returns:
            String representation of complete memory report
        """
        sections = [
            self.tree_view(max_depth=2, show_values=False),
            self.stats_view(),
            self.strength_view(top_n=5),
            self.contradiction_view(),
            self.consolidation_view(max_groups=3),
            self.timeline_view(max_episodes=5),
        ]
        
        return "\n\n".join(sections)


def visualize_memory(memory, format: str = "full") -> str:
    """Convenience function to visualize memory.
    
    Args:
        memory: SmartMemory instance
        format: Format to use ('tree', 'stats', 'strength', 'contradictions', 
                'consolidation', 'timeline', 'full')
    
    Returns:
        String representation of memory visualization
    """
    visualizer = MemoryVisualizer(memory)
    
    if format == "tree":
        return visualizer.tree_view()
    elif format == "stats":
        return visualizer.stats_view()
    elif format == "strength":
        return visualizer.strength_view()
    elif format == "contradictions":
        return visualizer.contradiction_view()
    elif format == "consolidation":
        return visualizer.consolidation_view()
    elif format == "timeline":
        return visualizer.timeline_view()
    elif format == "full":
        return visualizer.full_report()
    else:
        raise ValueError(f"Unknown format: {format}")


# Test the memory visualizer
if __name__ == "__main__":
    from json_memory import SmartMemory
    
    # Create test memory
    mem = SmartMemory("test_viz.json", max_chars=5000)
    
    # Store some facts
    mem.remember("user.name", "Alice")
    mem.remember("user.timezone", "GMT+7")
    mem.remember("bot.status", "running")
    mem.remember("server.ip", "192.168.1.100")
    
    # Create visualizer
    viz = MemoryVisualizer(mem)
    
    # Test different views
    print("Tree view:")
    print(viz.tree_view())
    print("\nStats view:")
    print(viz.stats_view())
    
    # Clean up
    import os
    for f in ["test_viz.json", "test_viz.meta.json"]:
        if os.path.exists(f):
            os.remove(f)