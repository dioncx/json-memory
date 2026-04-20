"""
Contradiction Detection — Detect conflicting facts in memory.

This module provides functionality to detect when new information
contradicts existing facts in memory, helping maintain consistency.
"""

import re
import time
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class Contradiction:
    """Represents a detected contradiction between two facts."""
    existing_path: str
    existing_value: any
    new_path: str
    new_value: any
    confidence: float  # 0.0-1.0 confidence that this is a real contradiction
    contradiction_type: str  # 'direct', 'semantic', 'temporal'
    explanation: str


class ContradictionDetector:
    """Detect contradictions between facts in memory."""
    
    def __init__(self):
        # Direct contradiction patterns
        self.direct_patterns = [
            # "X is Y" vs "X is Z" where Y != Z
            (r'(\w+)\s+is\s+(\w+)', r'(\w+)\s+is\s+(\w+)'),
            # "X was Y" vs "X is Z" (tense change)
            (r'(\w+)\s+was\s+(\w+)', r'(\w+)\s+is\s+(\w+)'),
            # "X has Y" vs "X has Z"
            (r'(\w+)\s+has\s+(\w+)', r'(\w+)\s+has\s+(\w+)'),
            # "X can Y" vs "X cannot Y"
            (r'(\w+)\s+can\s+(\w+)', r'(\w+)\s+cannot\s+(\w+)'),
            # "X will Y" vs "X will not Y"
            (r'(\w+)\s+will\s+(\w+)', r'(\w+)\s+will\s+not\s+(\w+)'),
        ]
        
        # Semantic opposites
        self.semantic_opposites = {
            'yes': 'no', 'no': 'yes',
            'true': 'false', 'false': 'true',
            'on': 'off', 'off': 'on',
            'enabled': 'disabled', 'disabled': 'enabled',
            'active': 'inactive', 'inactive': 'active',
            'online': 'offline', 'offline': 'online',
            'connected': 'disconnected', 'disconnected': 'connected',
            'running': 'stopped', 'stopped': 'running',
            'up': 'down', 'down': 'up',
            'high': 'low', 'low': 'high',
            'fast': 'slow', 'slow': 'fast',
            'good': 'bad', 'bad': 'good',
            'success': 'failure', 'failure': 'success',
            'profit': 'loss', 'loss': 'profit',
            'buy': 'sell', 'sell': 'buy',
            'long': 'short', 'short': 'long',
            'open': 'closed', 'closed': 'open',
            'start': 'stop', 'stop': 'start',
            'begin': 'end', 'end': 'begin',
            'create': 'destroy', 'destroy': 'create',
            'add': 'remove', 'remove': 'add',
            'include': 'exclude', 'exclude': 'include',
            'allow': 'deny', 'deny': 'allow',
            'permit': 'forbid', 'forbid': 'permit',
            'accept': 'reject', 'reject': 'accept',
            'approve': 'disapprove', 'disapprove': 'approve',
            'like': 'dislike', 'dislike': 'like',
            'love': 'hate', 'hate': 'love',
            'prefer': 'avoid', 'avoid': 'prefer',
            'want': 'refuse', 'refuse': 'want',
            'need': 'don\'t need', 'don\'t need': 'need',
            'have': 'don\'t have', 'don\'t have': 'have',
            'can': 'can\'t', 'can\'t': 'can',
            'will': 'won\'t', 'won\'t': 'will',
            'should': 'shouldn\'t', 'shouldn\'t': 'should',
            'could': 'couldn\'t', 'couldn\'t': 'could',
            'would': 'wouldn\'t', 'wouldn\'t': 'would',
        }
        
        # Temporal contradictions
        self.temporal_patterns = [
            # "X before Y" vs "X after Y"
            (r'before\s+(\w+)', r'after\s+(\1)'),
            (r'after\s+(\w+)', r'before\s+(\1)'),
            # "X first" vs "X last"
            (r'(\w+)\s+first', r'(\1)\s+last'),
            (r'(\w+)\s+last', r'(\1)\s+first'),
        ]
    
    def detect(self, new_path: str, new_value: any, existing_facts: dict) -> list[Contradiction]:
        """Detect contradictions between new fact and existing facts.
        
        Args:
            new_path: Path of the new fact
            new_value: Value of the new fact
            existing_facts: Dict of existing facts {path: value}
            
        Returns:
            List of Contradiction objects
        """
        contradictions = []
        
        # Convert new value to string for comparison
        new_str = str(new_value).lower().strip()
        
        for existing_path, existing_value in existing_facts.items():
            # Skip if same path (update, not contradiction)
            if existing_path == new_path:
                continue
                
            existing_str = str(existing_value).lower().strip()
            
            # 1. Direct contradiction (same subject, different values)
            direct_contradiction = self._check_direct_contradiction(
                new_path, new_str, existing_path, existing_str
            )
            if direct_contradiction:
                contradictions.append(direct_contradiction)
                continue
            
            # 2. Semantic contradiction (opposite values)
            semantic_contradiction = self._check_semantic_contradiction(
                new_path, new_str, existing_path, existing_str
            )
            if semantic_contradiction:
                contradictions.append(semantic_contradiction)
                continue
            
            # 3. Temporal contradiction (time conflicts)
            temporal_contradiction = self._check_temporal_contradiction(
                new_path, new_str, existing_path, existing_str
            )
            if temporal_contradiction:
                contradictions.append(temporal_contradiction)
                continue
        
        return contradictions
    
    def _check_direct_contradiction(self, new_path: str, new_value: str, 
                                   existing_path: str, existing_value: str) -> Optional[Contradiction]:
        """Check for direct contradictions (same subject AND same attribute, different values)."""
        # Extract subject and attribute from paths
        new_parts = new_path.split('.')
        existing_parts = existing_path.split('.')
        
        # Must have same number of parts
        if len(new_parts) != len(existing_parts):
            return None
        
        # Check if it's the exact same path (update, not contradiction)
        if new_path == existing_path:
            return None
        
        # For direct contradiction, we need same subject but different attribute
        # OR same full path (which would be an update)
        if len(new_parts) >= 2 and len(existing_parts) >= 2:
            # Same subject (first part), different attribute (rest of path)
            if new_parts[0] == existing_parts[0]:
                # Check if attributes are different
                new_attr = '.'.join(new_parts[1:])
                existing_attr = '.'.join(existing_parts[1:])
                
                if new_attr != existing_attr:
                    # Different attributes of same subject - not a direct contradiction
                    return None
            else:
                # Different subjects - not a direct contradiction
                return None
        
        # Check if values are different
        if new_value == existing_value:
            return None
        
        # Check if it's a meaningful difference (not just formatting)
        if self._is_meaningful_difference(new_value, existing_value):
            confidence = 0.7  # Moderate confidence for direct contradiction
            
            # Boost confidence if paths are very similar
            if self._paths_are_similar(new_path, existing_path):
                confidence = 0.9
            
            return Contradiction(
                existing_path=existing_path,
                existing_value=existing_value,
                new_path=new_path,
                new_value=new_value,
                confidence=confidence,
                contradiction_type='direct',
                explanation=f"Direct contradiction: {existing_path}={existing_value} vs {new_path}={new_value}"
            )
        
        return None
    
    def _check_semantic_contradiction(self, new_path: str, new_value: str,
                                     existing_path: str, existing_value: str) -> Optional[Contradiction]:
        """Check for semantic contradictions (opposite values)."""
        # First, check if paths are related (same subject AND same attribute)
        new_parts = new_path.split('.')
        existing_parts = existing_path.split('.')
        
        # Must have same number of parts
        if len(new_parts) != len(existing_parts):
            return None
        
        # Must have same first component (subject)
        if len(new_parts) < 1 or len(existing_parts) < 1:
            return None
        
        if new_parts[0] != existing_parts[0]:
            # Different subjects - not a semantic contradiction
            return None
        
        # For semantic contradictions, we need same subject AND same attribute
        # (i.e., same full path)
        if new_path != existing_path:
            # Different attributes - not a semantic contradiction
            return None
        
        # Check if values are semantic opposites
        if new_value in self.semantic_opposites and existing_value == self.semantic_opposites[new_value]:
            return Contradiction(
                existing_path=existing_path,
                existing_value=existing_value,
                new_path=new_path,
                new_value=new_value,
                confidence=0.85,
                contradiction_type='semantic',
                explanation=f"Semantic contradiction: {existing_path}={existing_value} vs {new_path}={new_value}"
            )
        
        # Check for negation patterns
        if self._contains_negation(new_value) and not self._contains_negation(existing_value):
            # Check if it's negating the same thing
            base_new = re.sub(r'\b(not|never|no|none|nothing|don\'t|doesn\'t|didn\'t|won\'t|can\'t|cannot)\b', '', new_value).strip()
            if base_new == existing_value:
                return Contradiction(
                    existing_path=existing_path,
                    existing_value=existing_value,
                    new_path=new_path,
                    new_value=new_value,
                    confidence=0.9,
                    contradiction_type='semantic',
                    explanation=f"Negation contradiction: {existing_path}={existing_value} vs {new_path}={new_value}"
                )
        
        return None
    
    def _check_temporal_contradiction(self, new_path: str, new_value: str,
                                     existing_path: str, existing_value: str) -> Optional[Contradiction]:
        """Check for temporal contradictions (time conflicts)."""
        # This is more complex and would require understanding time relationships
        # For now, just check for basic temporal keywords
        temporal_keywords = ['before', 'after', 'first', 'last', 'earlier', 'later', 'previously', 'subsequently']
        
        new_has_temporal = any(keyword in new_value for keyword in temporal_keywords)
        existing_has_temporal = any(keyword in existing_value for keyword in temporal_keywords)
        
        if new_has_temporal and existing_has_temporal:
            # Check for conflicting temporal relationships
            for pattern1, pattern2 in self.temporal_patterns:
                if re.search(pattern1, new_value) and re.search(pattern2, existing_value):
                    return Contradiction(
                        existing_path=existing_path,
                        existing_value=existing_value,
                        new_path=new_path,
                        new_value=new_value,
                        confidence=0.75,
                        contradiction_type='temporal',
                        explanation=f"Temporal contradiction: {existing_path}={existing_value} vs {new_path}={new_value}"
                    )
        
        return None
    
    def _are_related_subjects(self, subject1: str, subject2: str) -> bool:
        """Check if two subjects are related (could be same entity)."""
        # Simple check: same first word or substring
        words1 = subject1.lower().split()
        words2 = subject2.lower().split()
        
        if words1[0] == words2[0]:
            return True
        
        # Check if one contains the other
        if subject1.lower() in subject2.lower() or subject2.lower() in subject1.lower():
            return True
        
        return False
    
    def _paths_are_similar(self, path1: str, path2: str) -> bool:
        """Check if two paths are similar (could be about same thing)."""
        parts1 = path1.lower().split('.')
        parts2 = path2.lower().split('.')
        
        # Same first component
        if parts1[0] == parts2[0]:
            return True
        
        # High overlap in components
        set1 = set(parts1)
        set2 = set(parts2)
        overlap = len(set1 & set2)
        total = len(set1 | set2)
        
        return overlap / total > 0.5 if total > 0 else False
    
    def _is_meaningful_difference(self, value1: str, value2: str) -> bool:
        """Check if the difference between values is meaningful."""
        # Skip if values are too similar (typos, formatting)
        if value1 == value2:
            return False
        
        # Check Levenshtein distance (simplified)
        if len(value1) > 3 and len(value2) > 3:
            # If values are very similar, probably not a contradiction
            if abs(len(value1) - len(value2)) <= 1:
                # Count different characters
                diff_count = sum(1 for a, b in zip(value1, value2) if a != b)
                if diff_count <= 1:
                    return False
        
        return True
    
    def _contains_negation(self, text: str) -> bool:
        """Check if text contains negation words."""
        negation_words = ['not', 'never', 'no', 'none', 'nothing', 'don\'t', 'doesn\'t', 
                         'didn\'t', 'won\'t', 'can\'t', 'cannot', 'shouldn\'t', 'wouldn\'t',
                         'couldn\'t', 'without', 'lack', 'absence', 'missing']
        
        text_lower = text.lower()
        return any(word in text_lower for word in negation_words)


def detect_contradictions(new_path: str, new_value: any, existing_facts: dict) -> list[Contradiction]:
    """Convenience function to detect contradictions."""
    detector = ContradictionDetector()
    return detector.detect(new_path, new_value, existing_facts)


# Test the contradiction detector
if __name__ == "__main__":
    detector = ContradictionDetector()
    
    # Test cases
    existing_facts = {
        "user.status": "active",
        "bot.running": "yes",
        "trading.position": "long",
        "server.ip": "192.168.1.100",
        "meeting.time": "3pm",
    }
    
    # Test direct contradiction
    contradictions = detector.detect("user.status", "inactive", existing_facts)
    print("Test 1 - Direct contradiction:")
    for c in contradictions:
        print(f"  {c.explanation} (confidence: {c.confidence})")
    
    # Test semantic contradiction
    contradictions = detector.detect("bot.running", "no", existing_facts)
    print("\nTest 2 - Semantic contradiction:")
    for c in contradictions:
        print(f"  {c.explanation} (confidence: {c.confidence})")
    
    # Test no contradiction
    contradictions = detector.detect("server.port", "8080", existing_facts)
    print(f"\nTest 3 - No contradiction: {len(contradictions)} contradictions found")
    
    # Test negation
    contradictions = detector.detect("user.status", "not active", existing_facts)
    print("\nTest 4 - Negation contradiction:")
    for c in contradictions:
        print(f"  {c.explanation} (confidence: {c.confidence})")