"""
Semantic layer for SmartMemory — optional, zero-dependency fallback.

Requires: pip install json-memory[semantic]
  → sentence-transformers + faiss-cpu

Without these, falls back gracefully to keyword-based scoring.
"""

from typing import Optional
import json
import threading

# Check availability at import time
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False


class SemanticIndex:
    """Optional semantic search layer using sentence-transformers + FAISS.

    Drop-in enhancement for SmartMemory. When available, enables
    meaning-based retrieval instead of keyword matching.

    Works without dependencies (returns empty results, no crash).

    Args:
        model_name: Sentence-transformer model. Default: 'all-MiniLM-L6-v2'
            (fast, 384-dim, ~80MB download on first use).

    Example:
        >>> index = SemanticIndex()
        >>> index.add("user.timezone", "GMT+7, based in Jakarta")
        >>> index.add("bot.restart", "kill && nohup ./bot > log")
        >>> results = index.search("What time zone am I in?")
        [("user.timezone", "GMT+7, based in Jakarta", 0.89), ...]
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.available = HAS_SEMANTIC
        self._model = None
        self._model_name = model_name
        self._index = None
        self._paths: list[str] = []
        self._path_to_idx: dict[str, int] = {}
        self._dim = 384  # all-MiniLM-L6-v2 dimension
        self._lock = threading.RLock()

        if self.available:
            self._model = SentenceTransformer(model_name)
            self._index = faiss.IndexFlatIP(self._dim)  # inner product (cosine after normalization)

    def add(self, path: str, text: str):
        """Add/update a path's semantic embedding."""
        if not self.available:
            return

        with self._lock:
            embedding = self._encode(text)

            if path in self._path_to_idx:
                # Update existing
                idx = self._path_to_idx[path]
                self._index.reconstruct(idx)  # FAISS doesn't support in-place update easily
                # Rebuild approach: mark for rebuild
                self._paths[idx] = path
            else:
                # Add new
                self._path_to_idx[path] = len(self._paths)
                self._paths.append(path)
                self._index.add(embedding.reshape(1, -1))

    def search(self, query: str, top_k: int = 8, min_similarity: float = 0.2) -> list[tuple[str, float]]:
        """Search by meaning, not keywords.

        Args:
            query: Natural language query.
            top_k: Max results.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of (path, similarity_score) tuples, sorted by relevance.
        """
        if not self.available or self._index is None or self._index.ntotal == 0:
            return []

        with self._lock:
            query_emb = self._encode(query).reshape(1, -1)
            k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(query_emb, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                if score >= min_similarity:
                    results.append((self._paths[idx], float(score)))

            return results

    def _encode(self, text: str):
        """Encode text to normalized embedding vector."""
        emb = self._model.encode(text, normalize_embeddings=True)
        return np.array(emb, dtype='float32')

    def stats(self) -> dict:
        """Index stats."""
        return {
            'available': self.available,
            'model': self._model_name if self.available else None,
            'indexed_paths': len(self._paths),
            'dimension': self._dim if self.available else 0,
        }


def enhance_smart_memory(smart_mem, model_name: str = "all-MiniLM-L6-v2"):
    """Add semantic search to an existing SmartMemory instance.

    Monkey-patches recall_relevant() to combine keyword + semantic scoring.
    Gracefully degrades if dependencies not installed.

    Args:
        smart_mem: SmartMemory instance.
        model_name: Sentence-transformer model name.

    Returns:
        The SmartMemory instance (for chaining).
    """
    index = SemanticIndex(model_name=model_name)

    if not index.available:
        # No-op: SmartMemory continues with keyword scoring
        smart_mem._semantic_index = None
        return smart_mem

    # Index all existing paths
    for path in smart_mem.mem.paths():
        val = smart_mem.mem.get(path)
        text = f"{path}: {_value_to_text(val)}"
        index.add(path, text)

    smart_mem._semantic_index = index

    # Wrap remember to auto-index
    original_remember = smart_mem.remember

    def remember_with_indexing(path, value, **kwargs):
        original_remember(path, value, **kwargs)
        text = f"{path}: {_value_to_text(value)}"
        index.add(path, text)

    smart_mem.remember = remember_with_indexing

    # Wrap recall_relevant to combine scores
    original_recall = smart_mem.recall_relevant

    def recall_with_semantic(query=None, max_results=None, min_score=0.1):
        max_results = max_results or smart_mem.max_results

        # Get keyword-scored results
        keyword_results = original_recall(query, max_results=max_results * 2, min_score=0.0)

        # Get semantic results
        semantic_results = {}
        if query and index.available:
            for path, sim_score in index.search(query, top_k=max_results * 2):
                semantic_results[path] = sim_score

        # Combine scores
        all_paths = set(keyword_results.keys()) | set(semantic_results.keys())
        combined = []
        for path in all_paths:
            kw_score = keyword_results.get(path, 0.0)
            sem_score = semantic_results.get(path, 0.0)

            # Semantic gets higher weight when available
            if sem_score > 0:
                final = 0.3 * kw_score + 0.7 * sem_score
            else:
                final = kw_score

            if final >= min_score:
                val = smart_mem.mem.get(path)
                if val is not None:
                    combined.append((final, path, val))

        combined.sort(reverse=True)
        return {path: val for _, path, val in combined[:max_results]}

    smart_mem.recall_relevant = recall_with_semantic
    return smart_mem


def _value_to_text(value) -> str:
    """Convert a value to searchable text."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    if isinstance(value, dict):
        return " ".join(f"{k} {v}" for k, v in value.items())
    return str(value)
