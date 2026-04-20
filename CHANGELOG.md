# Changelog

## v1.2.0 — Memory Visualization & Performance

### Added
- **Memory visualization** — Visualize memory structure and statistics:
  - `visualize()` method on SmartMemory
  - Multiple formats: tree, stats, strength, contradictions, consolidation, timeline, full
  - `MemoryVisualizer` class for standalone visualization
  - Tree view shows hierarchical structure
  - Stats view shows memory usage and health
  - Strength view shows memory decay and reinforcement needs
  - Contradiction view shows conflicting facts
  - Consolidation view shows related facts
  - Timeline view shows conversation episodes

### Fixed
- **Version mismatch** — Fixed pyproject.toml version to match __init__.py (1.1.0)
- **Syntax error** — Fixed unterminated string literal in visualizer.py

### Performance
- **Storage**: ~6.75ms per fact (good)
- **Retrieval**: ~0.01ms per fact (excellent)
- **Smart retrieval**: ~6ms per query (good)

### Tests
- **243 tests passing** (no new tests for visualization)
- All existing tests continue to pass

### Usage
```python
mem = SmartMemory("agent.json")

# Store facts
mem.remember("user.name", "Alice")
mem.remember("bot.status", "running")

# Visualize memory
print(mem.visualize("tree"))  # Tree view
print(mem.visualize("stats"))  # Statistics
print(mem.visualize("strength"))  # Memory strengths
print(mem.visualize("full"))  # Full report
```

## v1.1.0 — Medium Priority Gaps Complete!

### 🎉 Milestone Release
All MEDIUM priority gaps now complete! The memory system now has:
- ✓ **Contradiction detection** — Detects conflicting facts in memory
- ✓ **Memory consolidation** — Merges related facts and removes redundancy
- ✓ **Forgetting curve** — Models natural memory decay with Ebbinghaus curve
- ✓ **Memory reinforcement** — Strengthens memories against forgetting

### Added
- **Contradiction detection** — Detects when new facts contradict existing ones:
  - Direct contradictions (same subject, same attribute, different values)
  - Semantic contradictions (opposite values like "yes"/"no")
  - Temporal contradictions (time conflicts)
  - `check_contradictions` parameter on `remember()` to warn about contradictions
  - `get_contradictions()` method to find all contradictions in memory
  - `ContradictionDetector` class for standalone contradiction detection

- **Memory consolidation** — Finds and merges related facts:
  - `consolidate_memory()` method to find groups of related facts
  - `auto_consolidate()` method to automatically merge high-confidence groups
  - `ConsolidationGroup` class for consolidation suggestions
  - Path and value similarity detection
  - Automatic tagging of consolidated facts

- **Forgetting curve** — Models natural memory decay:
  - Ebbinghaus forgetting curve implementation
  - Memory strength calculation based on time and reinforcement
  - Different decay rates for different memory types (identity, skill, fact, event, temporary)
  - `get_memory_strength()` method to analyze memory strength
  - `get_memories_needing_reinforcement()` method to prioritize reinforcement
  - `simulate_memory_decay()` method to predict future strength
  - `reinforce_memory()` method to strengthen memories against forgetting
  - `ForgettingCurve` class for standalone memory decay modeling

### Implementation Details
- **Contradiction detection** requires same subject AND same attribute for direct contradictions
- **Semantic contradictions** only detected for same path (updates, not contradictions)
- **Memory consolidation** uses path and value similarity scoring
- **Forgetting curve** uses exponential decay with reinforcement boosting
- **Memory types** have different decay multipliers (identity decays slowest)

### New Modules
- `json_memory.contradiction` — Contradiction detection module
- `json_memory.consolidation` — Memory consolidation module
- `json_memory.forgetting` — Forgetting curve module

### Tests
- **19 new tests** for medium priority features (all passing)
- **Total tests**: 243 passing (was 224)
- **Test coverage**: All new features fully tested

### Performance
- **Contradiction detection**: O(n) per fact check, where n = number of existing facts
- **Memory consolidation**: O(n²) for group detection, optimized for small to medium memories
- **Forgetting curve**: O(1) per memory strength calculation
- **Memory reinforcement**: O(1) per reinforcement operation

### Usage Examples

#### Contradiction Detection
```python
mem = SmartMemory()

# Store a fact
mem.remember("user.status", "active")

# Try to store contradictory fact (same path)
result = mem.remember("user.status", "inactive", check_contradictions=True)
# No contradiction detected (same path is an update)

# Get all contradictions
contradictions = mem.get_contradictions()
```

#### Memory Consolidation
```python
# Store similar facts
mem.remember("user.skills", "Python, Go, JavaScript")
mem.remember("user.programming_languages", "Python, Go, JavaScript, Rust")

# Get consolidation suggestions
groups = mem.consolidate_memory()
for group in groups:
    print(f"Consolidate: {group.paths} → {group.suggested_path}")

# Auto-consolidate high-confidence groups
result = mem.auto_consolidate(min_confidence=0.7)
```

#### Forgetting Curve
```python
# Analyze memory strength
strength = mem.get_memory_strength("user.name", memory_type='identity')
print(f"Current strength: {strength.current_strength:.3f}")
print(f"Predicted forget time: {strength.predicted_forget_time}")

# Get memories needing reinforcement
memories = mem.get_memories_needing_reinforcement(max_items=5)
for memory in memories:
    print(f"{memory['path']}: priority={memory['reinforcement_priority']:.3f}")

# Reinforce a memory
result = mem.reinforce_memory("user.name", boost_strength=0.3)
```

### What's Next (LOW Priority)
- **Conversation context** — Multi-turn understanding and context carryover
- **Enhanced auto-extraction** — More patterns and better confidence scoring
- **Memory visualization** — Graph-based memory visualization
- **Performance optimization** — Caching and indexing for large memories