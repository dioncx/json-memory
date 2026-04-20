"""
Memory — Core hierarchical JSON memory for AI agents.

Provides dotted-path access to nested JSON data with minified storage.
"""

import re
import threading
import json
import time
import copy
import sys
from typing import Any, Optional, Union, Callable, List

from .adapters import StorageAdapter, FileAdapter


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

    def __init__(self, data: Optional[dict] = None, max_chars: int = 2200, 
                 eviction_policy: str = "error", auto_flush_path: Optional[str] = None,
                 storage_adapter: Optional[StorageAdapter] = None,
                 track_history: bool = False, redact_keys: Optional[List[str]] = None,
                 on_evict: Optional[Callable[[str, Any], None]] = None,
                 cold_storage_path: Optional[str] = None):
        self._data = data or {}
        self.max_chars = max_chars
        self.eviction_policy = eviction_policy
        self.auto_flush_path = auto_flush_path
        self.storage_adapter = storage_adapter
        self.track_history = track_history
        self.redact_keys = redact_keys or []
        self.on_evict = on_evict
        self.cold_storage_path = cold_storage_path
        self._cold_store: Optional['Memory'] = None  # Lazy-loaded cold storage
        self._lock = threading.RLock()
        self._audit_log: list[dict] = []
        self._cache: Optional[str] = None
        self._watchers: dict[str, list[tuple[Callable, bool]]] = {}
        self._ttls: dict[str, float] = {}  # full_path -> expires_at
        self._snapshots: dict[str, dict] = {}
        self._access_times: dict[str, float] = {}
        
        # Auto-configure adapter if path is given but no adapter provided
        if self.auto_flush_path and not self.storage_adapter:
            self.storage_adapter = FileAdapter(self.auto_flush_path)

        # Load initial state from adapter if available
        if self.storage_adapter:
            saved_state = self.storage_adapter.load()
            if saved_state:
                # Merge saved data if self._data was empty, otherwise keep current
                if not self._data:
                    self.set_state(saved_state)

        # Initial access tracking for existing data
        if self._data:
            with self._lock:
                for path in self.paths():
                    self._track_access(path)

    def watch(self, path: str, callback: Callable[[str, Any], None], exact: bool = False) -> "Memory":
        """Register a callback triggered when path (or sub-path) is modified."""
        with self._lock:
            if path not in self._watchers:
                self._watchers[path] = []
            self._watchers[path].append((callback, exact))
            return self

    def unwatch(self, path: str, callback: Optional[Callable] = None) -> "Memory":
        """Unregister callback(s) for a path."""
        with self._lock:
            if path in self._watchers:
                if callback is None:
                    del self._watchers[path]
                else:
                    self._watchers[path] = [w for w in self._watchers[path] if w[0] != callback]
            return self

    def _trigger_watchers(self, mutated_path: str, value: Any):
        """Internal: dispatch events to watchers."""
        for watch_path, handlers in self._watchers.items():
            for callback, exact in handlers:
                is_match = (mutated_path == watch_path) or (
                    not exact and mutated_path.startswith(watch_path + ".")
                )
                if is_match:
                    try:
                        callback(mutated_path, value)
                    except Exception as e:
                        print(f"Memory watcher error on {watch_path}: {e}", file=sys.stderr)

    def _invalidate(self):
        """Invalidate the export cache."""
        self._cache = None

    # ── Core Access ───────────────────────────────────────────────

    def get(self, path: str, default: Any = None) -> Any:
        """Get a value by dotted path. Returns default if not found."""
        with self._lock:
            expired_path = self._is_expired(path)
            if expired_path:
                self.delete(expired_path)
                return default

            keys = path.split(".")
            node = self._data
            for key in keys:
                if isinstance(node, dict) and key in node:
                    node = node[key]
                else:
                    return default
            
            self._track_access(path)
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
        self._track_access(path)
        return new_val

    def touch(self, path: str, timestamp: float = None) -> "Memory":
        """Set a timestamp at path. Defaults to time.time()."""
        ts = timestamp if timestamp is not None else time.time()
        return self.set(path, ts)

    def batch_get(self, paths: list[str], default: Any = None) -> dict:
        """Get multiple paths at once. Returns {path: value} dict."""
        return {p: self.get(p, default) for p in paths}

    def find(self, pattern: str) -> dict:
        """Find paths matching a wildcard pattern.
        
        '*' matches one segment, '**' matches zero or more segments.
        Example: "users.*.status", "**.config"
        """
        # 1. Convert glob to regex
        # Escape all characters, then replace wildcard patterns
        regex_pat = re.escape(pattern)
        # ** at start: (?:.*?\.)?
        regex_pat = regex_pat.replace(r'\*\*\.', r'(?:.*\.)?')
        # ** at end: (?:\..*?)?
        regex_pat = regex_pat.replace(r'\.\*\*', r'(?:\..*)?')
        # ** internal: .*
        regex_pat = regex_pat.replace(r'\*\*', r'.*')
        # * internal: [^.]+
        regex_pat = regex_pat.replace(r'\*', r'[^.]+')
        
        regex = re.compile(f"^{regex_pat}$")
        
        matches = [p for p in self.paths() if regex.match(p)]
        for p in matches:
            self._track_access(p)
        return self.batch_get(matches)

    def set(self, path: str, value: Any, ttl: Optional[float] = None) -> "Memory":
        """Set a value by dotted path. Creates intermediate dicts as needed."""
        with self._lock:
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
                if self.eviction_policy in ("lru", "lru-archive"):
                    # Get all leaf paths, excluding the one we just set
                    all_paths = [p for p in self.paths() if p != path]
                    # Sort by access time (oldest first)
                    sorted_paths = sorted(all_paths, key=lambda p: self._access_times.get(p, 0))
                    
                    for p in sorted_paths:
                        if len(self.export()) <= self.max_chars:
                            break
                        evicted_value = self.get(p)
                        
                        # Archive to cold storage before deleting
                        if self.eviction_policy == "lru-archive":
                            self._archive_to_cold(p, evicted_value)
                        
                        # Fire on_evict callback
                        if self.on_evict:
                            try:
                                self.on_evict(p, evicted_value)
                            except Exception:
                                pass  # Don't let callback errors break eviction
                        
                        self.delete(p, prune=True)
                
                # Final check - if still too large (or policy is "error")
                if len(self.export()) > self.max_chars:
                    # Rollback to snapshot
                    self._data = snapshot
                    self._invalidate()
                    raise ValueError(
                        f"Memory overflow: setting '{path}' would exceed "
                        f"{self.max_chars} chars"
                    )
            
            if ttl is not None:
                self._ttls[path] = time.time() + ttl
            elif path in self._ttls:
                del self._ttls[path]
                
            if self.track_history:
                val_to_log = value
                if any(red_key in path.split(".") for red_key in self.redact_keys):
                    val_to_log = "***REDACTED***"
                
                self._audit_log.append({
                    "time": time.time(),
                    "action": "set",
                    "path": path,
                    "value": copy.deepcopy(val_to_log)
                })

            self._track_access(path)
            self._trigger_watchers(path, value)
            self._auto_flush()
            return self

    def _archive_to_cold(self, path: str, value: Any) -> None:
        """Archive a fact to cold storage file before eviction.
        
        Saves evicted facts to a .cold.json file alongside the main memory file,
        preserving them for later recovery.
        """
        cold_data = self._load_cold()
        if not cold_data and not self.cold_storage_path:
            # No path available, skip archival
            if not self.auto_flush_path:
                return
        
        cold_data[path] = {
            "value": value,
            "evicted_at": time.time(),
            "source": self.auto_flush_path or "unknown"
        }
        
        self._save_cold(cold_data)

    def recover_from_cold(self, path: str) -> bool:
        """Recover an archived fact from cold storage back to hot memory.
        
        Args:
            path: The dotted path of the fact to recover.
            
        Returns:
            True if recovered, False if not found in cold storage.
        """
        cold_data = self._load_cold()
        if path not in cold_data:
            return False
        
        entry = cold_data[path]
        value = entry.get("value") if isinstance(entry, dict) else entry
        
        # Restore to hot memory
        self.set(path, value)
        
        # Remove from cold storage
        del cold_data[path]
        self._save_cold(cold_data)
        
        return True

    def cold_stats(self) -> dict:
        """Get statistics about cold storage."""
        if not self.cold_storage_path:
            if self.auto_flush_path:
                from pathlib import Path
                self.cold_storage_path = str(Path(self.auto_flush_path).with_suffix('.cold.json'))
            else:
                return {"count": 0, "chars": 0, "path": None}
        
        try:
            from pathlib import Path
            cold_path = Path(self.cold_storage_path)
            if not cold_path.exists():
                return {"count": 0, "chars": 0, "path": str(cold_path)}
            
            raw = cold_path.read_text(encoding='utf-8')
            cold_data = json.loads(raw)
            
            paths = list(cold_data.keys())
            oldest = None
            newest = None
            if cold_data:
                timestamps = [
                    v.get("evicted_at", 0) for v in cold_data.values()
                    if isinstance(v, dict)
                ]
                if timestamps:
                    oldest = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(min(timestamps)))
                    newest = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(max(timestamps)))
            
            return {
                "count": len(paths),
                "chars": len(raw),
                "path": str(cold_path),
                "oldest": oldest,
                "newest": newest,
                "paths": paths
            }
        except Exception:
            return {"count": 0, "chars": 0, "path": self.cold_storage_path}

    def cold_search(self, query: str = None, path_pattern: str = None,
                    older_than: float = None, newer_than: float = None) -> list[dict]:
        """Search cold storage for archived facts.
        
        Args:
            query: Search in values (substring match, case-insensitive).
            path_pattern: Glob pattern for paths (e.g., "project.*").
            older_than: Only facts evicted before this timestamp.
            newer_than: Only facts evicted after this timestamp.
            
        Returns:
            List of matching entries with path, value, evicted_at.
        """
        cold_data = self._load_cold()
        if not cold_data:
            return []
        
        import fnmatch
        results = []
        
        for path, entry in cold_data.items():
            if isinstance(entry, dict):
                value = entry.get("value")
                evicted_at = entry.get("evicted_at", 0)
            else:
                value = entry
                evicted_at = 0
            
            # Filter by path pattern
            if path_pattern and not fnmatch.fnmatch(path, path_pattern):
                continue
            
            # Filter by age
            if older_than and evicted_at >= older_than:
                continue
            if newer_than and evicted_at <= newer_than:
                continue
            
            # Filter by value content
            if query:
                val_str = json.dumps(value, ensure_ascii=False).lower() if not isinstance(value, str) else value.lower()
                if query.lower() not in val_str:
                    continue
            
            results.append({
                "path": path,
                "value": value,
                "evicted_at": evicted_at,
                "evicted_ago": time.time() - evicted_at if evicted_at else None
            })
        
        return sorted(results, key=lambda r: r.get("evicted_at", 0))

    def recover_all(self) -> dict:
        """Recover all facts from cold storage back to hot memory.
        
        Returns:
            Dict with 'recovered' (list of paths), 'failed' (list of paths), 'count'.
        """
        cold_data = self._load_cold()
        if not cold_data:
            return {"recovered": [], "failed": [], "count": 0}
        
        result = {"recovered": [], "failed": [], "count": 0}
        paths = list(cold_data.keys())
        
        for path in paths:
            if self.recover_from_cold(path):
                result["recovered"].append(path)
            else:
                result["failed"].append(path)
        
        result["count"] = len(result["recovered"])
        return result

    def recover_matching(self, pattern: str) -> dict:
        """Recover facts from cold storage matching a glob pattern.
        
        Args:
            pattern: Glob pattern (e.g., "project.*", "user.**").
            
        Returns:
            Dict with 'recovered', 'failed', 'count'.
        """
        import fnmatch
        cold_data = self._load_cold()
        if not cold_data:
            return {"recovered": [], "failed": [], "count": 0}
        
        result = {"recovered": [], "failed": [], "count": 0}
        
        for path in cold_data:
            if fnmatch.fnmatch(path, pattern):
                if self.recover_from_cold(path):
                    result["recovered"].append(path)
                else:
                    result["failed"].append(path)
        
        result["count"] = len(result["recovered"])
        return result

    def purge_cold(self, older_than: float = None, keep_last: int = None) -> dict:
        """Permanently delete old facts from cold storage.
        
        Args:
            older_than: Delete facts evicted before this timestamp (epoch).
                        Use with time.time() - seconds for relative age.
            keep_last: Keep the N most recently evicted facts, delete the rest.
            
        Returns:
            Dict with 'purged' (list of paths), 'kept' (int), 'count'.
        """
        cold_data = self._load_cold()
        if not cold_data:
            return {"purged": [], "kept": 0, "count": 0}
        
        paths_by_age = sorted(
            cold_data.items(),
            key=lambda item: item[1].get("evicted_at", 0) if isinstance(item[1], dict) else 0
        )
        
        to_purge = []
        to_keep = []
        
        if keep_last is not None:
            # Keep the N most recent, purge the rest
            for i, (path, entry) in enumerate(paths_by_age):
                if i < len(paths_by_age) - keep_last:
                    to_purge.append(path)
                else:
                    to_keep.append(path)
        elif older_than is not None:
            for path, entry in paths_by_age:
                evicted_at = entry.get("evicted_at", 0) if isinstance(entry, dict) else 0
                if evicted_at < older_than:
                    to_purge.append(path)
                else:
                    to_keep.append(path)
        else:
            return {"purged": [], "kept": len(paths_by_age), "count": 0,
                    "error": "Specify older_than or keep_last"}
        
        # Write back only kept entries
        if to_purge:
            cold_data = {p: cold_data[p] for p in to_keep}
            self._save_cold(cold_data)
        
        return {"purged": to_purge, "kept": len(to_keep), "count": len(to_purge)}

    def _load_cold(self) -> dict:
        """Load cold storage data from disk."""
        if not self.cold_storage_path:
            if self.auto_flush_path:
                from pathlib import Path
                self.cold_storage_path = str(Path(self.auto_flush_path).with_suffix('.cold.json'))
            else:
                return {}
        
        try:
            from pathlib import Path
            cold_path = Path(self.cold_storage_path)
            if not cold_path.exists():
                return {}
            return json.loads(cold_path.read_text(encoding='utf-8'))
        except Exception:
            return {}

    def _save_cold(self, data: dict) -> None:
        """Save cold storage data to disk."""
        if not self.cold_storage_path:
            return
        try:
            from pathlib import Path
            cold_path = Path(self.cold_storage_path)
            cold_path.parent.mkdir(parents=True, exist_ok=True)
            cold_path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        except Exception:
            pass

    def delete(self, path: str, prune: bool = False) -> bool:
        """Delete a value by dotted path. Returns True if found and deleted."""
        with self._lock:
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
                if self.track_history:
                    self._audit_log.append({
                        "time": time.time(),
                        "action": "delete",
                        "path": path,
                        "value": None
                    })

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
                self._trigger_watchers(path, None)
                
                # Remove metadata for this path and all sub-paths
                pref = path + "."
                self._ttls = {k: v for k, v in self._ttls.items() 
                             if k != path and not k.startswith(pref)}
                self._access_times = {k: v for k, v in self._access_times.items()
                                     if k != path and not k.startswith(pref)}
                
                self._auto_flush()
                return True
            return False

    def clear(self, path: str = "") -> "Memory":
        """Clear memory at a path (or everything if empty)."""
        with self._lock:
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
            self._trigger_watchers(path, None)
            
            if not path:
                self._ttls = {}
                self._access_times = {}
            else:
                pref = path + "."
                self._ttls = {k: v for k, v in self._ttls.items() 
                             if k != path and not k.startswith(pref)}
                self._access_times = {k: v for k, v in self._access_times.items()
                                     if k != path and not k.startswith(pref)}
                
            self._auto_flush()
            return self

    def has(self, path: str) -> bool:
        """Check if a dotted path exists.

        Returns True even if the value is None — distinguishes
        'path exists with value None' from 'path doesn't exist'.
        """
        with self._lock:
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
            
            # Trigger watchers for each merged key
            self._trigger_merge_watchers(data, prefix)
            return count
        except ValueError:
            # Rollback entire batch — all or nothing
            self._data = snapshot
            self._invalidate()
            raise

    def purge_expired(self) -> int:
        """Explicitly remove all expired keys. Returns count of removed items."""
        with self._lock:
            now = time.time()
            # Sort by path length so parents are deleted before children
            expired_paths = sorted(
                [p for p, exp in self._ttls.items() if exp <= now],
                key=len
            )
            count = 0
            for path in expired_paths:
                # Check if still there (might have been deleted by a parent already)
                if self._is_expired(path):
                    self.delete(path)
                    count += 1
            
            if count > 0:
                self._auto_flush()
            return count

    def _is_expired(self, path: str) -> Optional[str]:
        """Internal: Check if path or any parent path has expired.
        
        Returns the path of the first expired element found, or None.
        """
        if not path:
            return None
            
        now = time.time()
        keys = path.split(".")
        current = ""
        for i, key in enumerate(keys):
            current = f"{current}.{key}" if current else key
            if current in self._ttls and self._ttls[current] <= now:
                return current
        return None

    def _trigger_merge_watchers(self, data: dict, prefix: str = ""):
        """Internal: trigger watchers for a merge operation."""
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self._trigger_merge_watchers(value, path)
            else:
                self._trigger_watchers(path, value)

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

    def to_dict(self, redact: bool = False) -> dict:
        """Return the full memory as a plain dict (excludes metadata like TTLs)."""
        with self._lock:
            data = copy.deepcopy(self._data)
            if redact and self.redact_keys:
                data = self._redact_data(data)
            return data

    def get_state(self) -> dict:
        """Return the complete state including TTL metadata."""
        with self._lock:
            return {
                "data": copy.deepcopy(self._data),
                "ttls": copy.deepcopy(self._ttls)
            }

    def set_state(self, state: dict) -> "Memory":
        """Restore memory from a state dict produced by get_state()."""
        with self._lock:
            self._data = copy.deepcopy(state.get("data", {}))
            self._ttls = copy.deepcopy(state.get("ttls", {}))
            self._invalidate()
            self._auto_flush()
            return self

    def snapshot(self, name: str) -> None:
        """Create a named checkpoint of the current memory state."""
        with self._lock:
            self._snapshots[name] = self.get_state()

    def rollback(self, name: str) -> bool:
        """Rollback to a previously saved checkpoint. Returns True if successful."""
        with self._lock:
            if name not in self._snapshots:
                return False
            self.set_state(self._snapshots[name])
            self._auto_flush()
            return True

    def flush(self) -> None:
        """Manually flush current state to disk if an adapter is configured."""
        with self._lock:
            if not self.storage_adapter:
                return
            self.storage_adapter.save(self.get_state())

    def _auto_flush(self) -> None:
        """Internal: flush if adapter is configured."""
        if self.storage_adapter:
            self.flush()

    def _redact_data(self, data: Any) -> Any:
        """Internal: recursively redact keys in a dict."""
        if not isinstance(data, dict):
            return data
        
        redacted = {}
        for k, v in data.items():
            if k in self.redact_keys:
                redacted[k] = "***REDACTED***"
            elif isinstance(v, dict):
                redacted[k] = self._redact_data(v)
            else:
                redacted[k] = v
        return redacted

    def _track_access(self, path: str):
        """Internal: record access time for a path and its parents."""
        now = time.time()
        keys = path.split(".")
        current = ""
        for key in keys:
            current = f"{current}.{key}" if current else key
            self._access_times[current] = now

    def paths(self, prefix: str = "") -> list:
        """List all leaf paths in the memory tree."""
        with self._lock:
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

    def history(self) -> list[dict]:
        """Return a copy of the mutation audit log."""
        with self._lock:
            log = copy.deepcopy(self._audit_log)
            if self.redact_keys:
                for entry in log:
                    if "value" in entry:
                        entry["value"] = self._redact_data(entry["value"])
            return log

    # ── Serialization ─────────────────────────────────────────────

    def export(self, redact: bool = False) -> str:
        """Export serialized memory as a minified JSON string."""
        if not redact:
            self.purge_expired()
            
        with self._lock:
            if not redact and self._cache is not None:
                return self._cache
            
            data = self._data
            if redact and self.redact_keys:
                data = self._redact_data(copy.deepcopy(self._data))
                
            res = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
            if not redact:
                self._cache = res
            return res

    def export_pretty(self, redact: bool = False) -> str:
        """Export serialized memory as an indented JSON string."""
        with self._lock:
            data = self._data
            if redact and self.redact_keys:
                data = self._redact_data(copy.deepcopy(self._data))
            return json.dumps(data, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str, max_chars: int = 2200) -> "Memory":
        """Create Memory from a JSON string."""
        data = json.loads(json_str)
        if isinstance(data, dict) and "data" in data and "ttls" in data:
            # Full state load
            mem = cls(max_chars=max_chars)
            mem.set_state(data)
            return mem
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

    def estimate_size(self, value: Any) -> int:
        """Estimate the JSON character size of a value.
        
        Args:
            value: Any JSON-serializable value.
            
        Returns:
            Estimated character count when serialized.
        """
        try:
            return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')))
        except (TypeError, ValueError):
            return len(str(value))

    def available_budget(self) -> int:
        """Return how many characters can still be written before overflow.
        
        Returns:
            Number of characters remaining in the budget.
        """
        return self.max_chars - len(self.export())

    def will_fit(self, path: str, value: Any) -> dict:
        """Check if a value will fit at the given path without overflow.
        
        Simulates the write without committing. Accounts for:
        - Current memory usage
        - New value size
        - Overwritten value size (if path exists)
        - Intermediate dict overhead for new paths
        
        Args:
            path: Dotted path where the value would be stored.
            value: The value to check.
            
        Returns:
            Dict with:
            - 'fits' (bool): Whether the value fits in budget
            - 'current_chars' (int): Current memory size
            - 'new_chars' (int): Size after write
            - 'delta' (int): Net change (+/-)
            - 'available' (int): Current free space
            - 'overflow_by' (int): How much over budget (0 if fits)
            - 'eviction_needed' (int): How many chars need to be evicted (0 if fits)
        """
        current = self.export()
        current_chars = len(current)
        
        # Simulate the write
        snapshot = copy.deepcopy(self._data)
        keys = path.split(".")
        node = self._data
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value
        self._invalidate()
        
        new_export = self.export()
        new_chars = len(new_export)
        
        # Restore
        self._data = snapshot
        self._invalidate()
        
        delta = new_chars - current_chars
        available = self.max_chars - current_chars
        fits = new_chars <= self.max_chars
        overflow_by = max(0, new_chars - self.max_chars)
        
        return {
            'fits': fits,
            'current_chars': current_chars,
            'new_chars': new_chars,
            'delta': delta,
            'available': available,
            'overflow_by': overflow_by,
            'eviction_needed': overflow_by if not fits else 0,
        }

    def suggest_budget(self, target_facts: int = 50, avg_value_size: int = 80) -> dict:
        """Suggest a max_chars budget based on desired capacity.
        
        Args:
            target_facts: How many facts you want to store.
            avg_value_size: Average value size in characters.
            
        Returns:
            Dict with 'suggested_max_chars', 'estimated_facts', 'overhead'.
        """
        # Overhead: JSON structure, path separators, quotes, commas
        overhead_per_fact = 30  # Conservative estimate for JSON structure
        total = target_facts * (avg_value_size + overhead_per_fact)
        
        return {
            'suggested_max_chars': total,
            'target_facts': target_facts,
            'avg_value_size': avg_value_size,
            'overhead_per_fact': overhead_per_fact,
            'total_estimated': total,
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
