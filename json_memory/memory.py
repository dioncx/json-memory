"""
Memory — Core hierarchical JSON memory for AI agents.

Provides dotted-path access to nested JSON data with minified storage.
"""

import json
import time
import copy
from typing import Any, Optional, Union


class _Missing:
    """Sentinel for get() default — distinguishes 'not found' from 'found None'."""
    pass

_MISSING = _Missing()


class Memory:
    """Hierarchical JSON memory with dotted-path access.

    Args:
        max_chars: Maximum character budget for the memory string.
        data: Optional initial data dict.

    Example:
        >>> mem = Memory(max_chars=2000)
        >>> mem.set("user.name", "Alice")
        >>> mem.set("bot.restart", "kill && nohup ./bot > log")
        >>> mem.get("user.name")
        'Alice'
        >>> mem.export()
        '{"user":{"name":"Alice"},"bot":{"restart":"kill && nohup ./bot > log"}}'
    """

    def __init__(self, max_chars: int = 2200, data: Optional[dict] = None):
        self.max_chars = max_chars
        self._data: dict = data if data is not None else {}
        self._cache: Optional[str] = None

    def _invalidate(self):
        """Invalidate the export cache."""
        self._cache = None

    # ── Core Access ───────────────────────────────────────────────

    def get(self, path: str, default: Any = None) -> Any:
        """Get a value by dotted path. Example: ``mem.get("bot.binance.rst")``

        Returns default if path doesn't exist. To distinguish 'not found'
        from 'found None', pass a custom sentinel:

            _SENTINEL = object()
            result = mem.get("key", _SENTINEL)
            if result is _SENTINEL:
                ...  # path doesn't exist
        """
        keys = path.split(".")
        node = self._data
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    def get_or_set(self, path: str, default: Any) -> Any:
        """Get value at path, or set and return default if missing."""
        result = self.get(path, _MISSING)
        if result is _MISSING:
            self.set(path, default)
            return default
        return result

    def increment(self, path: str, delta: int = 1) -> int:
        """Atomically increment a numeric value at path (initializes to 0 if missing)."""
        current = self.get(path, 0)
        if not isinstance(current, (int, float)):
            current = 0
        new_val = current + delta
        self.set(path, new_val)
        return new_val

    def touch(self, path: str, timestamp: float = None) -> "Memory":
        """Set a timestamp at path. Defaults to time.time()."""
        ts = timestamp if timestamp is not None else time.time()
        return self.set(path, ts)

    def batch_get(self, paths: list[str], default: Any = None) -> dict:
        """Get multiple paths at once. Returns {path: value} dict."""
        return {p: self.get(p, default) for p in paths}

    def set(self, path: str, value: Any) -> "Memory":
        """Set a value by dotted path. Creates intermediate dicts as needed.

        Returns self to allow chaining.
        Raises ValueError if it would exceed max_chars (data unchanged).
        """
        keys = path.split(".")

        # Snapshot current state for rollback
        snapshot = copy.deepcopy(self._data)

        # Build path and set value
        node = self._data
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value

        # Invalidate cache before check
        self._invalidate()

        # Check budget
        if len(self.export()) > self.max_chars:
            # Rollback to snapshot
            self._data = snapshot
            self._invalidate()
            raise ValueError(
                f"Memory overflow: setting '{path}' would exceed "
                f"{self.max_chars} chars"
            )
        return self

    def delete(self, path: str, prune: bool = False) -> bool:
        """Delete a value by dotted path. Returns True if found and deleted.

        Args:
            path: Dotted path to delete.
            prune: If True, also remove empty parent dicts along the path.
        """
        keys = path.split(".")
        node = self._data
        stack = [(None, self._data)]  # (key_in_parent, node)
        
        for key in keys[:-1]:
            if isinstance(node, dict) and key in node:
                node = node[key]
                stack.append((key, node))
            else:
                return False
        
        if keys[-1] in node:
            del node[keys[-1]]
            
            if prune:
                # Bubble up and remove empty dicts
                for i in range(len(stack) - 1, 0, -1):
                    key_to_delete, current_node = stack[i]
                    parent_node = stack[i-1][1]
                    if isinstance(current_node, dict) and not current_node:
                        del parent_node[key_to_delete]
                    else:
                        break

            self._invalidate()
            return True
        return False

    def clear(self, path: str = "") -> "Memory":
        """Clear memory at a path (or everything if empty)."""
        if not path:
            self._data = {}
        else:
            keys = path.split(".")
            node = self._data
            for key in keys[:-1]:
                if isinstance(node, dict) and key in node:
                    node = node[key]
                else:
                    return self
            if keys[-1] in node:
                node[keys[-1]] = {}
        
        self._invalidate()
        return self

    def has(self, path: str) -> bool:
        """Check if a dotted path exists.

        Returns True even if the value is None — distinguishes
        'path exists with value None' from 'path doesn't exist'.
        """
        return self.get(path, _MISSING) is not _MISSING

    def keys(self, path: str = "") -> list:
        """List keys at a given path (or root if empty)."""
        node = self.get(path) if path else self._data
        if isinstance(node, dict):
            return list(node.keys())
        return []

    # ── Bulk Operations ───────────────────────────────────────────

    def merge(self, data: dict, prefix: str = "") -> int:
        """Merge a dict into memory at an optional prefix path. Alias: update().

        Atomic: either all keys merge or none do (rollback on overflow).
        Returns number of keys merged.
        Raises ValueError if the combined result would exceed max_chars.
        """
        # Snapshot before any changes
        snapshot = copy.deepcopy(self._data)

        try:
            # Apply all changes without individual overflow checks
            count = self._merge_apply(data, prefix)

            # Invalidate cache before check
            self._invalidate()

            # Check budget once for the whole batch
            if len(self.export()) > self.max_chars:
                raise ValueError(
                    f"Memory overflow: merging {count} keys would exceed "
                    f"{self.max_chars} chars"
                )
            return count
        except ValueError:
            # Rollback entire batch — all or nothing
            self._data = snapshot
            self._invalidate()
            raise

    def update(self, data: dict, prefix: str = "") -> int:
        """Alias for merge()."""
        return self.merge(data, prefix)

    def _merge_apply(self, data: dict, prefix: str = "") -> int:
        """Apply merge operations without overflow checks (internal)."""
        count = 0
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                count += self._merge_apply(value, prefix=path)
            else:
                # Direct write — no snapshot, no overflow check
                keys = path.split(".")
                node = self._data
                for k in keys[:-1]:
                    if k not in node or not isinstance(node[k], dict):
                        node[k] = {}
                    node = node[k]
                node[keys[-1]] = value
                count += 1
        return count

    def to_dict(self) -> dict:
        """Return the full memory as a plain dict."""
        return copy.deepcopy(self._data)

    def paths(self, prefix: str = "") -> list:
        """List all leaf paths in the memory tree."""
        results = []
        node = self.get(prefix) if prefix else self._data
        if isinstance(node, dict):
            for key, value in node.items():
                full_path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    results.extend(self.paths(full_path))
                else:
                    results.append(full_path)
        return sorted(results)

    # ── Serialization ─────────────────────────────────────────────

    def export(self) -> str:
        """Export as minified JSON string."""
        if self._cache is None:
            self._cache = json.dumps(self._data, separators=(",", ":"), ensure_ascii=False)
        return self._cache

    def export_pretty(self) -> str:
        """Export as pretty-printed JSON string."""
        return json.dumps(self._data, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str, max_chars: int = 2200) -> "Memory":
        """Create Memory from a JSON string."""
        data = json.loads(json_str)
        return cls(max_chars=max_chars, data=data)

    @classmethod
    def from_file(cls, path: str, max_chars: int = 2200) -> "Memory":
        """Create Memory from a JSON file."""
        with open(path, "r") as f:
            return cls.from_json(f.read(), max_chars=max_chars)

    def save(self, path: str) -> None:
        """Save memory to a JSON file."""
        with open(path, "w") as f:
            f.write(self.export())

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return memory statistics."""
        exported = self.export()
        return {
            "entries": len(self.paths()),
            "chars_used": len(exported),
            "chars_max": self.max_chars,
            "chars_free": self.max_chars - len(exported),
            "utilization": f"{len(exported) / self.max_chars * 100:.1f}%",
        }

    # ── Dunder ────────────────────────────────────────────────────

    def __getitem__(self, path: str) -> Any:
        result = self.get(path, _MISSING)
        if result is _MISSING:
            raise KeyError(path)
        return result

    def __setitem__(self, path: str, value: Any) -> None:
        self.set(path, value)

    def __contains__(self, path: str) -> bool:
        return self.has(path)

    def __len__(self) -> int:
        return len(self.export())

    def __repr__(self) -> str:
        return f"Memory(leafs={len(self.paths())}, chars={len(self.export())}/{self.max_chars})"
