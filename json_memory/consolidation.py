"""
Memory Consolidation — Merge related facts and remove redundancy.

This module provides functionality to consolidate memory by:
1. Merging facts that are about the same thing
2. Removing redundant information
3. Creating summaries of related facts
"""

import re
import time
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ConsolidationGroup:
    """Represents a group of related facts that can be consolidated."""
    paths: List[str]
    values: List[Any]
    suggested_path: str
    suggested_value: Any
    reason: str
    confidence: float  # 0.0-1.0


class MemoryConsolidator:
    """Consolidate related facts in memory."""
    
    def __init__(self):
        # Similarity thresholds
        self.path_similarity_threshold = 0.4  # Lowered from 0.6
        self.value_similarity_threshold = 0.7
        self.semantic_similarity_threshold = 0.8
        
        # Patterns for detecting similar facts
        self.similarity_patterns = [
            # Same subject, different attributes
            (r'^(\w+)\.(\w+)$', r'^(\1)\.(\w+)$'),  # user.name vs user.age
            # List-like values
            (r'^\[.*\]$', r'^\[.*\]$'),  # [a, b, c] vs [a, b, d]
            # Comma-separated values
            (r'^[\w\s,]+$', r'^[\w\s,]+$'),  # "Python, Go" vs "Python, JavaScript"
        ]
    
    def consolidate(self, facts: Dict[str, Any], max_groups: int = 10) -> List[ConsolidationGroup]:
        """Find groups of related facts that can be consolidated.
        
        Args:
            facts: Dict of facts {path: value}
            max_groups: Maximum number of groups to return
            
        Returns:
            List of ConsolidationGroup objects
        """
        groups = []
        
        # Group facts by subject (first path component)
        subject_groups = defaultdict(list)
        for path, value in facts.items():
            subject = path.split('.')[0] if '.' in path else path
            subject_groups[subject].append((path, value))
        
        # Process each subject group
        for subject, subject_facts in subject_groups.items():
            if len(subject_facts) < 2:
                continue
            
            # Find similar facts within subject
            similar_groups = self._find_similar_facts(subject_facts)
            groups.extend(similar_groups)
            
            if len(groups) >= max_groups:
                break
        
        # Sort by confidence (highest first)
        groups.sort(key=lambda g: g.confidence, reverse=True)
        return groups[:max_groups]
    
    def _find_similar_facts(self, facts: List[Tuple[str, Any]]) -> List[ConsolidationGroup]:
        """Find similar facts within a list of (path, value) tuples."""
        groups = []
        used_indices = set()
        
        for i, (path1, value1) in enumerate(facts):
            if i in used_indices:
                continue
            
            # Find similar facts
            similar_indices = [i]
            for j, (path2, value2) in enumerate(facts):
                if j <= i or j in used_indices:
                    continue
                
                similarity = self._calculate_similarity(path1, value1, path2, value2)
                if similarity >= self.path_similarity_threshold:
                    similar_indices.append(j)
            
            if len(similar_indices) >= 2:
                # Create consolidation group
                group_paths = [facts[i][0] for i in similar_indices]
                group_values = [facts[i][1] for i in similar_indices]
                
                # Suggest consolidated path and value
                suggested_path, suggested_value, reason = self._suggest_consolidation(
                    group_paths, group_values
                )
                
                # Calculate confidence
                confidence = self._calculate_group_confidence(group_paths, group_values)
                
                groups.append(ConsolidationGroup(
                    paths=group_paths,
                    values=group_values,
                    suggested_path=suggested_path,
                    suggested_value=suggested_value,
                    reason=reason,
                    confidence=confidence
                ))
                
                used_indices.update(similar_indices)
        
        return groups
    
    def _calculate_similarity(self, path1: str, value1: Any, path2: str, value2: Any) -> float:
        """Calculate similarity between two facts (0.0-1.0)."""
        # Path similarity
        path_sim = self._path_similarity(path1, path2)
        
        # Value similarity
        value_sim = self._value_similarity(value1, value2)
        
        # Weighted combination
        return 0.6 * path_sim + 0.4 * value_sim
    
    def _path_similarity(self, path1: str, path2: str) -> float:
        """Calculate similarity between two paths."""
        parts1 = path1.lower().split('.')
        parts2 = path2.lower().split('.')
        
        # Same first component (subject)
        if parts1[0] != parts2[0]:
            return 0.0
        
        # Calculate overlap
        set1 = set(parts1)
        set2 = set(parts2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _value_similarity(self, value1: Any, value2: Any) -> float:
        """Calculate similarity between two values."""
        str1 = str(value1).lower().strip()
        str2 = str(value2).lower().strip()
        
        # Exact match
        if str1 == str2:
            return 1.0
        
        # Check if one contains the other
        if str1 in str2 or str2 in str1:
            return 0.8
        
        # Check for list-like values
        if self._is_list_like(str1) and self._is_list_like(str2):
            return self._list_similarity(str1, str2)
        
        # Check for numeric values
        if self._is_numeric(str1) and self._is_numeric(str2):
            try:
                num1 = float(re.sub(r'[^\d.]', '', str1))
                num2 = float(re.sub(r'[^\d.]', '', str2))
                if num1 == num2:
                    return 1.0
                # Close numbers
                diff = abs(num1 - num2)
                avg = (num1 + num2) / 2
                if avg > 0:
                    return max(0, 1 - (diff / avg))
            except:
                pass
        
        # Token overlap
        tokens1 = set(re.findall(r'\w{2,}', str1))
        tokens2 = set(re.findall(r'\w{2,}', str2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def _is_list_like(self, text: str) -> bool:
        """Check if text looks like a list."""
        # Check for commas, semicolons, or "and"
        return bool(re.search(r'[,;]|\band\b', text))
    
    def _list_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two list-like texts."""
        # Extract items
        items1 = set(re.split(r'[,;]|\band\b', text1))
        items2 = set(re.split(r'[,;]|\band\b', text2))
        
        # Clean items
        items1 = {item.strip().lower() for item in items1 if item.strip()}
        items2 = {item.strip().lower() for item in items2 if item.strip()}
        
        if not items1 or not items2:
            return 0.0
        
        intersection = len(items1 & items2)
        union = len(items1 | items2)
        
        return intersection / union if union > 0 else 0.0
    
    def _is_numeric(self, text: str) -> bool:
        """Check if text contains numbers."""
        return bool(re.search(r'\d', text))
    
    def _suggest_consolidation(self, paths: List[str], values: List[Any]) -> Tuple[str, Any, str]:
        """Suggest a consolidated path and value for a group of facts."""
        # Find common path prefix
        common_prefix = self._find_common_path_prefix(paths)
        
        # Create suggested path
        if common_prefix:
            suggested_path = f"{common_prefix}.consolidated"
        else:
            # Use the shortest path
            suggested_path = min(paths, key=len)
        
        # Create suggested value
        if all(isinstance(v, str) for v in values):
            # String values - combine unique parts
            all_parts = set()
            for value in values:
                parts = re.split(r'[,;]|\band\b', str(value))
                all_parts.update(part.strip().lower() for part in parts if part.strip())
            
            if len(all_parts) <= 5:
                suggested_value = ', '.join(sorted(all_parts))
            else:
                suggested_value = f"{len(all_parts)} items"
        else:
            # Non-string values - use the most recent or most common
            suggested_value = values[0]  # Simplified - could be enhanced
        
        reason = f"Consolidated {len(paths)} related facts"
        return suggested_path, suggested_value, reason
    
    def _find_common_path_prefix(self, paths: List[str]) -> str:
        """Find common prefix among paths."""
        if not paths:
            return ""
        
        # Split all paths
        split_paths = [path.split('.') for path in paths]
        
        # Find common prefix
        prefix_parts = []
        for parts in zip(*split_paths):
            if len(set(parts)) == 1:
                prefix_parts.append(parts[0])
            else:
                break
        
        return '.'.join(prefix_parts)
    
    def _calculate_group_confidence(self, paths: List[str], values: List[Any]) -> float:
        """Calculate confidence for a consolidation group."""
        # Base confidence
        confidence = 0.5
        
        # More facts = higher confidence
        if len(paths) >= 3:
            confidence += 0.2
        elif len(paths) >= 2:
            confidence += 0.1
        
        # Path similarity
        path_similarities = []
        for i, path1 in enumerate(paths):
            for path2 in paths[i+1:]:
                path_similarities.append(self._path_similarity(path1, path2))
        
        if path_similarities:
            avg_path_sim = sum(path_similarities) / len(path_similarities)
            confidence += 0.3 * avg_path_sim
        
        # Value similarity
        value_similarities = []
        for i, value1 in enumerate(values):
            for value2 in values[i+1:]:
                value_similarities.append(self._value_similarity(value1, value2))
        
        if value_similarities:
            avg_value_sim = sum(value_similarities) / len(value_similarities)
            confidence += 0.2 * avg_value_sim
        
        return min(confidence, 1.0)


def consolidate_memory(facts: Dict[str, Any], max_groups: int = 10) -> List[ConsolidationGroup]:
    """Convenience function to consolidate memory."""
    consolidator = MemoryConsolidator()
    return consolidator.consolidate(facts, max_groups)


# Test the memory consolidator
if __name__ == "__main__":
    consolidator = MemoryConsolidator()
    
    # Test cases
    test_facts = {
        "user.skills": "Python, Go, JavaScript",
        "user.languages": "Python, JavaScript, Rust",
        "user.programming": "Python, Go, Rust, TypeScript",
        "project.name": "Trading Bot",
        "project.title": "Trading Bot Project",
        "server.ip": "192.168.1.100",
        "server.port": "8080",
    }
    
    groups = consolidator.consolidate(test_facts)
    
    print("Consolidation groups:")
    for i, group in enumerate(groups, 1):
        print(f"\nGroup {i}:")
        print(f"  Paths: {group.paths}")
        print(f"  Values: {group.values}")
        print(f"  Suggested: {group.suggested_path} = {group.suggested_value}")
        print(f"  Reason: {group.reason}")
        print(f"  Confidence: {group.confidence:.2f}")