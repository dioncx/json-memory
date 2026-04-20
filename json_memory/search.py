"""
Advanced Search — Regex, fuzzy, and semantic search capabilities.

This module provides advanced search functionality for memory:
- Regex search
- Fuzzy search (approximate string matching)
- Semantic search (optional, requires FAISS)
"""

import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class SearchResult:
    """Represents a search result."""
    path: str
    value: Any
    score: float  # 0.0-1.0 relevance score
    match_type: str  # 'exact', 'regex', 'fuzzy', 'semantic'
    highlights: List[str] = None  # Matched portions


class AdvancedSearch:
    """Advanced search capabilities for memory."""
    
    def __init__(self, memory):
        """Initialize with a SmartMemory instance."""
        self.memory = memory
    
    def regex_search(self, pattern: str, field: str = 'both', 
                    case_sensitive: bool = False) -> List[SearchResult]:
        """Search using regular expressions.
        
        Args:
            pattern: Regex pattern to search for
            field: Where to search ('path', 'value', 'both')
            case_sensitive: Whether to use case-sensitive matching
            
        Returns:
            List of SearchResult objects
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
        
        results = []
        
        with self.memory._lock:
            for path in self.memory.mem.paths():
                value = self.memory.mem.get(path)
                if value is None:
                    continue
                
                score = 0.0
                highlights = []
                
                # Search in path
                if field in ('path', 'both'):
                    if regex.search(path):
                        score += 0.5
                        highlights.append(f"path: {path}")
                
                # Search in value
                if field in ('value', 'both'):
                    value_str = str(value)
                    match = regex.search(value_str)
                    if match:
                        score += 0.5
                        highlights.append(f"value: {match.group()}")
                
                if score > 0:
                    results.append(SearchResult(
                        path=path,
                        value=value,
                        score=score,
                        match_type='regex',
                        highlights=highlights
                    ))
        
        # Sort by score (descending)
        results.sort(key=lambda r: r.score, reverse=True)
        return results
    
    def fuzzy_search(self, query: str, threshold: float = 0.6,
                    field: str = 'both') -> List[SearchResult]:
        """Search using fuzzy string matching.
        
        Args:
            query: Search query
            threshold: Minimum similarity score (0.0-1.0)
            field: Where to search ('path', 'value', 'both')
            
        Returns:
            List of SearchResult objects
        """
        query_lower = query.lower()
        results = []
        
        with self.memory._lock:
            for path in self.memory.mem.paths():
                value = self.memory.mem.get(path)
                if value is None:
                    continue
                
                score = 0.0
                highlights = []
                
                # Fuzzy match in path
                if field in ('path', 'both'):
                    path_similarity = self._fuzzy_similarity(query_lower, path.lower())
                    if path_similarity >= threshold:
                        score += path_similarity * 0.5
                        highlights.append(f"path: {path} ({path_similarity:.2f})")
                
                # Fuzzy match in value
                if field in ('value', 'both'):
                    value_str = str(value).lower()
                    value_similarity = self._fuzzy_similarity(query_lower, value_str)
                    if value_similarity >= threshold:
                        score += value_similarity * 0.5
                        highlights.append(f"value: {value_str[:50]} ({value_similarity:.2f})")
                
                if score > 0:
                    results.append(SearchResult(
                        path=path,
                        value=value,
                        score=score,
                        match_type='fuzzy',
                        highlights=highlights
                    ))
        
        # Sort by score (descending)
        results.sort(key=lambda r: r.score, reverse=True)
        return results
    
    def _fuzzy_similarity(self, query: str, text: str) -> float:
        """Calculate fuzzy similarity between query and text."""
        # Check for exact substring
        if query in text:
            return 1.0
        
        # Check for word overlap
        query_words = set(re.findall(r'\w+', query))
        text_words = set(re.findall(r'\w+', text))
        
        if query_words and text_words:
            overlap = len(query_words & text_words)
            total = len(query_words | text_words)
            return overlap / total if total > 0 else 0.0
        
        # Use sequence similarity
        return SequenceMatcher(None, query, text).ratio()
    
    def full_text_search(self, query: str, case_sensitive: bool = False) -> List[SearchResult]:
        """Full-text search across all memory.
        
        Args:
            query: Search query (supports multiple words)
            case_sensitive: Whether to use case-sensitive matching
            
        Returns:
            List of SearchResult objects
        """
        # Split query into words
        words = query.split()
        
        if not words:
            return []
        
        results = []
        
        with self.memory._lock:
            for path in self.memory.mem.paths():
                value = self.memory.mem.get(path)
                if value is None:
                    continue
                
                value_str = str(value)
                if not case_sensitive:
                    value_str = value_str.lower()
                    path_lower = path.lower()
                    words_lower = [w.lower() for w in words]
                else:
                    path_lower = path
                    words_lower = words
                
                # Count word matches
                matches = 0
                highlights = []
                
                for word in words_lower:
                    # Check path
                    if word in path_lower:
                        matches += 1
                        highlights.append(f"path: {word}")
                    
                    # Check value
                    if word in value_str:
                        matches += 1
                        highlights.append(f"value: {word}")
                
                if matches > 0:
                    # Score based on proportion of words matched
                    score = matches / (len(words_lower) * 2)  # Max 2 matches per word (path + value)
                    
                    results.append(SearchResult(
                        path=path,
                        value=value,
                        score=min(score, 1.0),
                        match_type='full_text',
                        highlights=highlights
                    ))
        
        # Sort by score (descending)
        results.sort(key=lambda r: r.score, reverse=True)
        return results
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Semantic search using embeddings (requires FAISS).
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        # Check if semantic search is available
        try:
            from .semantic import get_semantic_index
            index = get_semantic_index()
            if index is None:
                return []
        except ImportError:
            return []
        
        # Use semantic index for search
        results = []
        
        with self.memory._lock:
            # Get all paths and values
            paths = []
            values = []
            for path in self.memory.mem.paths():
                value = self.memory.mem.get(path)
                if value is not None:
                    paths.append(path)
                    values.append(str(value))
            
            if not paths:
                return []
            
            # Search using semantic index
            try:
                search_results = index.search(query, top_k=top_k)
                
                for result in search_results:
                    path = result['path']
                    score = result['score']
                    
                    # Find the value
                    value = None
                    for i, p in enumerate(paths):
                        if p == path:
                            value = values[i]
                            break
                    
                    if value:
                        results.append(SearchResult(
                            path=path,
                            value=value,
                            score=score,
                            match_type='semantic',
                            highlights=[f"semantic match: {score:.3f}"]
                        ))
            except Exception as e:
                print(f"Semantic search error: {e}", flush=True)
                return []
        
        return results
    
    def search(self, query: str, search_type: str = 'auto', **kwargs) -> List[SearchResult]:
        """Unified search interface.
        
        Args:
            query: Search query
            search_type: Type of search ('auto', 'regex', 'fuzzy', 'full_text', 'semantic')
            **kwargs: Additional arguments for specific search types
            
        Returns:
            List of SearchResult objects
        """
        if search_type == 'auto':
            # Auto-detect search type
            if query.startswith('/') and query.endswith('/'):
                # Regex pattern
                return self.regex_search(query[1:-1], **kwargs)
            elif len(query.split()) > 1:
                # Multiple words - use full text search
                return self.full_text_search(query, **kwargs)
            else:
                # Single word - use fuzzy search
                return self.fuzzy_search(query, **kwargs)
        
        elif search_type == 'regex':
            return self.regex_search(query, **kwargs)
        
        elif search_type == 'fuzzy':
            return self.fuzzy_search(query, **kwargs)
        
        elif search_type == 'full_text':
            return self.full_text_search(query, **kwargs)
        
        elif search_type == 'semantic':
            return self.semantic_search(query, **kwargs)
        
        else:
            raise ValueError(f"Unknown search type: {search_type}")
    
    def suggest(self, partial: str, limit: int = 10) -> List[str]:
        """Suggest paths based on partial input.
        
        Args:
            partial: Partial path input
            limit: Maximum suggestions
            
        Returns:
            List of suggested paths
        """
        partial_lower = partial.lower()
        suggestions = []
        
        with self.memory._lock:
            for path in self.memory.mem.paths():
                if partial_lower in path.lower():
                    suggestions.append(path)
                    
                    if len(suggestions) >= limit:
                        break
        
        return suggestions


def create_search(memory) -> AdvancedSearch:
    """Convenience function to create AdvancedSearch."""
    return AdvancedSearch(memory)


# Test the advanced search
if __name__ == "__main__":
    from json_memory import SmartMemory
    
    # Create test memory
    mem = SmartMemory("test_search.json", max_chars=5000)
    
    # Store some facts
    mem.remember("user.name", "Alice")
    mem.remember("user.email", "alice@example.com")
    mem.remember("project.name", "Trading Bot")
    mem.remember("project.description", "A bot for trading cryptocurrencies")
    
    # Create search
    search = AdvancedSearch(mem)
    
    # Test regex search
    print("Regex search for 'alice':")
    results = search.regex_search('alice')
    for r in results:
        print(f"  {r.path}: {r.value} (score: {r.score})")
    
    # Test fuzzy search
    print("\nFuzzy search for 'tradng':")
    results = search.fuzzy_search('tradng')
    for r in results:
        print(f"  {r.path}: {r.value} (score: {r.score})")
    
    # Test full text search
    print("\nFull text search for 'bot trading':")
    results = search.full_text_search('bot trading')
    for r in results:
        print(f"  {r.path}: {r.value} (score: {r.score})")
    
    # Clean up
    import os
    for f in ["test_search.json", "test_search.meta.json"]:
        if os.path.exists(f):
            os.remove(f)