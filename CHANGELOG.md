# Changelog

All notable changes to this project will be documented in this file.

## [0.1.6] - 2024-04-19

### Performance
- **Memory Optimization**: Added `export()` caching with a dirty-flag. Avoids redundant JSON serialization on repeated exports.
- **Faster Rollbacks**: Switched from JSON round-trip to `copy.deepcopy` for memory snapshots.
- **Graph Optimization**: Switched BFS queue in `Synapse.find_path()` to `collections.deque` for O(1) pops.
- **WeightGate Optimization**: Added concept-level guards to skip processing for unmentioned concepts.
- **Caching**: Added tokenization cache for concept names in `WeightGate`.

### Added
- **API Chaining**: `Memory.set()` now returns `self` for fluent chaining.
- **Memory.delete(prune=True)**: Optional pruning of empty parent dicts after deletion.
- **Memory.clear(path)**: Easily wipe a specific path or the entire memory.
- **Memory.update()**: Added as a descriptive alias for `merge()`.
- **Synapse.merge()**: Ability to combine two independent graphs.
- **Required Fields**: Schema now supports required keys using the `!` prefix (e.g., `{"!user": "str"}`).
- **HAS_SNOWBALL**: Exported in `__init__.py` to check for optional stemming support.

### Changed
- Improved `README.md` examples for consistency.
- Standardized `stats()` output with `entries` and `chars_free`.

### Fixed
- Corrected various small bugs in graph traversal and state management.
- Fixed mismatched keys in documentation examples.

### Developer Experience
- Added `dev` optional dependencies in `pyproject.toml` (pytest, black, mypy, etc.).
- Created this `CHANGELOG.md`.
