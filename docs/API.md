# json-memory API Reference

Full API documentation for all classes and methods.

---

## Table of Contents

- [SmartMemory](#smartmemory) — Intelligent agent memory (recommended)
- [Memory](#memory) — Low-level structured storage
- [Synapse](#synapse) — Associative concept graph
- [WeightGate](#weightgate) — Passive learning middleware
- [Schema](#schema) — Validation and OpenAI tool export
- [TieredMemory](#tieredmemory) — Hot/warm/cold storage tiers
- [Compress](#compress) — Key abbreviation and minification
- [Semantic Search](#semantic-search) — Optional embedding-based retrieval
- [Concept Map](#concept-map) — Zero-dep semantic expansion

---

## SmartMemory

**The recommended class for AI agent memory.** Combines Memory + Synapse with intelligent retrieval, auto-extraction, and episodic memory.

```python
from json_memory import SmartMemory

mem = SmartMemory(
    path="agent.json",          # File path for persistence
    max_chars=5000,             # Character budget
    max_results=8,              # Max facts returned per query
    extract_confidence=0.6,     # Min confidence for auto-extraction (0.0-1.0)
    tiered=False,               # Enable hot/warm/cold tiers
    recency_half_life=3600,     # Half-life for recency scoring (seconds)
)
```

### Core Operations

#### `remember(path, value, ttl=None, tags=None)`

Store a fact with optional TTL and tags.

```python
mem.remember("user.name", "Alice")
mem.remember("session.token", "xyz", ttl=300)  # expires in 5 min
mem.remember("bot.symbol", "BNB", tags=["trading", "active"])
```

- **path** (str): Dotted path key (e.g. `"user.name"`, `"bot.config.api_key"`)
- **value**: Any JSON-serializable value (str, int, float, list, dict, bool, None)
- **ttl** (int, optional): Time-to-live in seconds. Auto-expires after this.
- **tags** (list[str], optional): Tags for categorization and Synapse linking.

#### `recall(path, default=None)`

Retrieve a fact by exact dotted path. Records access (boosts frequency score).

```python
mem.recall("user.name")          # → "Alice"
mem.recall("missing", "N/A")     # → "N/A"
```

#### `forget(path)`

Delete a fact and its metadata.

```python
mem.forget("session.token")
```

#### `search(pattern)`

Find facts matching a glob pattern.

```python
mem.search("user.*")      # → {"user.name": "Alice", "user.tz": "GMT+7"}
mem.search("**.api_key")  # → deep search at any depth
```

#### `context()`

Export ALL memory as compact JSON string. For full-context injection.

```python
json_str = mem.context()  # → '{"user":{"name":"Alice"},"bot":{"symbol":"BNB"}}'
```

### Smart Retrieval

#### `recall_relevant(query=None, max_results=None, min_score=0.1, fallback=True)`

Retrieve only facts relevant to the query. **This is the core improvement over injecting everything.**

```python
mem.recall_relevant("What's my timezone?")
# → {"user.timezone": "GMT+7"}  (only relevant, not everything)

mem.recall_relevant()  # no query → top-scored by recency × frequency
```

- **query** (str, optional): User's message. If provided, keyword relevance dominates scoring.
- **max_results** (int, optional): Override default max results.
- **min_score** (float): Minimum score threshold (0.0-1.0).
- **fallback** (bool): If True and no strong keyword match, boost by active topics.

**Scoring formula (with query):**
```
score = 0.1 × recency + 0.05 × frequency + 0.85 × keyword_relevance
```

**Scoring formula (without query):**
```
score = 0.6 × recency + 0.4 × frequency
```

#### `prompt_context(query=None, max_results=None, format_fn=None)`

Generate lean prompt context from relevant facts only.

```python
ctx = mem.prompt_context("What's my timezone?")
# → "## Memory\n- user.timezone: GMT+7"  (32 chars, not 5000)
```

- **format_fn** (callable, optional): Custom formatter `(path, value) → str`.

#### `build_context(query=None, max_chars=2000, include_episodes=True)`

All-in-one context builder: relevant facts + recent episodes + active topics, within budget.

```python
ctx = mem.build_context("How do I deploy?", max_chars=500)
# → "## Memory\n- bot.build: go build -o bot\n\n## Recent Topics\n- [5m ago] Discussed deploy\n\n## Active: deployment, trading"
```

#### `explain_score(path, query=None)`

Debug why a fact was or wasn't returned.

```python
mem.explain_score("user.timezone", "What's my timezone?")
# → {"path": "user.timezone", "recency": 0.95, "frequency": 0.8,
#    "keyword_relevance": 0.43, "final_score": 0.45, "access_count": 12,
#    "age_seconds": 42.3, "tier": "hot"}
```

#### `score(path, query_tokens=None, now=None)`

Score a path's relevance. Returns float 0.0-1.0.

### Auto-Extraction

#### `process_conversation(user_msg, agent_msg=None)`

Passively extract and store facts from conversation. Also auto-detects topics and logs episodes.

```python
extracted = mem.process_conversation("My name is Bob and I live in Tokyo")
# → [{"path": "user.name", "value": "Bob", "confidence": 0.8, "source": "user"},
#    {"path": "user.location", "value": "Tokyo", "confidence": 0.85, "source": "user"}]
```

Patterns detected:
- Names: "My name is X", "I'm called X", "Call me X"
- Locations: "I live in X", "I'm from X", "Based in X"
- Timezones: "My timezone is UTC+7"
- Preferences: "I prefer X", "I like X", "I use X"
- Platforms: "on Telegram", "from Discord"
- Remember requests: "Remember that X"
- Projects: "Project is X", "Working on X"

### Episodic Memory

#### `log_episode(topic, summary=None, paths=None)`

Log a conversation episode for timeline-based recall.

```python
mem.log_episode("deployment", "Discussed bot restart process", ["bot.build", "server.restart"])
```

#### `recall_episodes(topic=None, max_age_seconds=86400, limit=5)`

Find past conversation episodes by topic or recency.

```python
mem.recall_episodes("bot")  # → [{"topic": "trading", "summary": "...", "timestamp": ...}]
mem.recall_episodes()       # → most recent episodes (last 24h)
```

#### `active_topics` (property)

Topics discussed in current session. Persisted to disk.

```python
mem.active_topics  # → ["trading", "deployment", "time"]
```

#### `advance_turn()`

Call once per conversation turn. Tracks session context. Called automatically by `process_conversation()`.

### Associative Memory

#### `link(concept, associations, weights=None)`

Create concept associations for associative recall.

```python
mem.link("debugging", ["check_logs", "reproduce", "git_bisect"],
         weights={"check_logs": 0.9, "reproduce": 0.7})
```

#### `associate(concept, depth=1)`

Recall associated concepts via graph traversal.

```python
mem.associate("debugging")         # → ["check_logs", "reproduce", "git_bisect"]
mem.associate("debugging", depth=2)  # → deeper traversal
```

### Snapshots

#### `snapshot(label)`

Save state before risky operations.

```python
mem.snapshot("before_experiment")
```

#### `rollback(label)`

Restore state from a snapshot.

```python
mem.rollback("before_experiment")
```

### Stats

#### `stats()`

Memory statistics including scoring metadata.

```python
mem.stats()
# → {"entries": 16, "chars_used": 486, "chars_max": 5000, "paths": 18,
#    "tiers": "disabled", "top_scored": [{"path": "bot.exchange", "score": 1.0}]}
```

---

## Memory

Low-level hierarchical JSON storage with dotted-path access.

```python
from json_memory import Memory

mem = Memory(
    data=None,                  # Optional initial dict
    max_chars=2200,             # Character budget
    eviction_policy="error",    # "error", "lru", or "warn"
    auto_flush_path=None,       # Auto-save to this file on changes
    storage_adapter=None,       # Custom StorageAdapter
    track_history=False,        # Enable mutation audit log
    redact_keys=None,           # Keys to redact in exports (e.g. ["secret", "password"])
)
```

### Core Operations

| Method | Description |
|--------|-------------|
| `set(path, value, ttl=None)` | Set value with optional TTL |
| `get(path, default=None)` | Get value by dotted path |
| `delete(path, prune=False)` | Delete path; prune=True removes empty parents |
| `has(path)` | Check if path exists |
| `keys(path="")` | List keys at path (or root) |
| `paths(prefix="")` | List all leaf paths |
| `merge(data, prefix="")` | Bulk merge dict into memory |
| `clear(path="")` | Clear subtree or everything |

### Convenience

| Method | Description |
|--------|-------------|
| `get_or_set(path, default)` | Get or set+return default |
| `increment(path, delta=1)` | Atomically increment numeric value |
| `touch(path, timestamp=None)` | Set timestamp at path |
| `batch_get(paths, default=None)` | Get multiple paths at once |
| `find(pattern)` | Glob search: `"user.*"`, `"**.api_key"` |

### TTL (Time-To-Live)

```python
mem.set("session.token", "xyz", ttl=300)  # expires in 5 min
mem.get("session.token")  # → "xyz" (for 5 min), then → None
mem.purge_expired()       # manually clear all expired keys
```

Parent expiry: if a parent key expires, all children are implicitly expired.

### Persistence

```python
# Auto-flush (saves on every mutation)
mem = Memory(auto_flush_path="data.json")

# Manual save/load
mem.save("data.json")
mem2 = Memory.from_file("data.json")

# Export/import
json_str = mem.export()              # minified
json_str = mem.export_pretty()       # indented
mem2 = Memory.from_json(json_str)
```

### Snapshots & Rollback

```python
mem.snapshot("before_change")
mem.set("key", "new_value")
mem.rollback("before_change")  # restores full state including TTLs
```

### State Management

```python
state = mem.get_state()       # includes TTL metadata
mem.set_state(state)          # restore full state

data = mem.to_dict()          # raw data only (no metadata)
```

### Audit History

```python
mem = Memory(track_history=True)
mem.set("key", "value")
for event in mem.history():
    print(f"[{event['time']}] {event['action']} {event['path']} -> {event['value']}")
```

### Watchers

```python
def on_change(path, value):
    print(f"{path} changed to {value}")

mem.watch("user", on_change)        # fires on user.* changes
mem.watch("user.name", on_change, exact=True)  # fires only on exact path
mem.unwatch("user", on_change)
```

### LRU Eviction

```python
mem = Memory(max_chars=2000, eviction_policy="lru")
# When max_chars is exceeded, least-recently-used keys are evicted automatically
```

### Stats

```python
mem.stats()
# → {"entries": 4, "chars_used": 146, "chars_max": 2000,
#    "chars_free": 1854, "utilization": "7.3%"}
```

### Dict-like Access

```python
mem["user.name"] = "Alice"     # __setitem__
print(mem["user.name"])        # __getitem__
"user.name" in mem             # __contains__
len(mem)                       # __len__
```

---

## Synapse

Associative concept graph. Models how thinking of "coffee" activates "morning", "energy", "routine".

```python
from json_memory import Synapse

brain = Synapse()
```

### Building the Graph

#### `link(concept, associations, bidirectional=True, weights=None)`

Create weighted links between concepts.

```python
brain.link("trading", ["binance", "strategy", "risk"])

# With custom weights (higher = stronger association)
brain.link("coffee", ["cappuccino", "americano", "espresso"],
           weights={"cappuccino": 0.95, "americano": 0.2})
```

### Traversal

#### `activate(concept, depth=1, weighted=True)`

Activate a concept and traverse associations.

```python
brain.activate("trading")          # → ["binance", "strategy", "risk"]
brain.activate("trading", depth=2) # → includes 2nd-level associations
```

#### `find_path(start, end, max_depth=5)`

Shortest path between concepts (BFS).

```python
brain.find_path("trading", "cloudflare")
# → ["trading", "server", "nginx", "cloudflare"]
```

#### `find_strongest_path(start, end)`

Highest-weight path (Dijkstra).

```python
brain.find_strongest_path("coffee", "energy")
# → ["coffee", "morning", "energy"] (strongest conceptual chain)
```

### Learning & Decay

#### `strengthen(concept, assoc, boost=0.05)`

Increase association weight. Capped at 1.0.

#### `weaken(concept, assoc, decay=0.03)`

Decrease association weight. Floored at 0.0.

```python
for _ in range(10):
    brain.strengthen("coffee", "cappuccino")  # user always picks this
    brain.weaken("coffee", "americano")       # user never picks this

brain.top_associations("coffee")
# → [("cappuccino", 1.0), ("espresso", 0.5), ("americano", 0.2)]
```

### Querying

| Method | Description |
|--------|-------------|
| `concepts()` | List all concept names |
| `has_concept(concept)` | Check if concept exists |
| `get_associations(concept)` | Get all associations with weights |
| `get_weight(concept, assoc)` | Get weight of specific association |
| `set_weight(concept, assoc, weight)` | Set weight (0.0-1.0) |
| `top_associations(concept, limit=5)` | Top associations by weight |
| `get_frequency(concept, assoc)` | Activation count |
| `connections(concept)` | Structured connection info |
| `hubs(min_connections=3)` | Most connected concepts |

### Graph Operations

```python
sub = brain.subgraph(["trading", "binance", "strategy"])  # extract subset
brain.merge(other_brain)                                   # combine graphs
brain.rename_concept("old_name", "new_name")               # rename node
brain.remove_concept("concept")                            # remove node
```

### Persistence

```python
brain.save("synapse.json")
brain = Synapse.load("synapse.json")

json_str = brain.export()
brain = Synapse.from_json(json_str)

data = brain.to_dict()
brain = Synapse.from_dict(data)
```

---

## WeightGate

Passive learning middleware. Automatically updates concept weights as conversation flows through.

```python
from json_memory import WeightGate

gate = WeightGate(
    path="synapse.json",        # Persistence file
    synapse=None,               # Or wrap an existing Synapse
    decay_rate=0.01,            # Decay for unused associations
    boost_rate=0.05,            # Boost for mentioned concepts
    min_weight=0.1,             # Floor for decay
    enabled=False,              # Disabled by default (opt-in)
    ngram_size=1,               # N-gram size for multi-word concepts
)
```

### Usage

```python
# Enable and use
gate = WeightGate("synapse.json", enabled=True)
gate.add_concept("debug", {"check_logs": 0.9, "ask_user": 0.2})

gate.process_input("How do I restart the bot?")
# → concepts detected, weights updated

gate.process_output("Run: kill && nohup ./bot > log")
# → agent's response also updates weights

gate.top_associations("debug")
# → [("check_logs", 0.95), ("ask_user", 0.18)]  ← learned your pattern
```

### Context Manager

```python
with WeightGate("synapse.json") as gate:
    gate.process_conversation(user_msg, agent_response)
# Auto-saved and disabled on exit
```

### Manual Operations

| Method | Description |
|--------|-------------|
| `add_concept(concept, associations)` | Add concept with weights |
| `remove_concept(concept)` | Remove concept and links |
| `strengthen(concept, assoc, boost)` | Manual boost |
| `weaken(concept, assoc, decay)` | Manual decay |
| `set_weight(concept, assoc, weight)` | Set exact weight |
| `top_associations(concept, limit)` | Top associations |
| `get_weights(concept)` | Get all weights |
| `export_compact()` | Compact JSON export |

### Enable/Disable

```python
gate.enable()     # ON
gate.disable()    # OFF (becomes no-op)
gate.toggle()     # Toggle
gate.enabled      # Current state (property)
```

---

## Schema

Validation and OpenAI tool schema generation.

```python
from json_memory import Schema

schema = Schema({
    "!name": "str",           # ! prefix = required
    "!email": "str",
    "preferences": ["str"],   # list of strings
    "config": {               # nested object
        "theme": "str",
        "notifications": "bool",
    },
})
```

### Validation

```python
# Validate data
result = schema.validate({"name": "Alice", "email": "a@b.com"})
# → {"valid": True, "errors": []}

result = schema.validate({"name": "Alice"}, strict=True)
# → {"valid": False, "errors": ["Missing required: email"]}

# Validate Memory instance
result = schema.validate_memory(mem)

# Generate default skeleton
defaults = schema.defaults()
# → {"name": None, "email": None, "preferences": [], "config": {"theme": None, ...}}

# Diff: what's missing or extra
diff = schema.diff({"name": "Alice", "extra_field": 1})
# → {"missing": ["email", ...], "extra": ["extra_field"]}
```

### OpenAI Tool Export

```python
tools = schema.to_openai_tools("update_profile", "Update user profile")
# → [{"type": "function", "function": {"name": "update_profile", ...}}]
```

---

## TieredMemory

Manages hot/warm/cold memory tiers with automatic promotion/demotion.

```python
from json_memory import TieredMemory

tiered = TieredMemory(
    path="memory.json",        # Base path (creates .hot.json, .warm.json, .cold.json)
    max_hot_chars=2000,        # Hot tier budget
    max_warm_chars=5000,       # Warm tier budget
)
```

### Operations

```python
tiered.set("key", "value", tier="hot")     # Store in specific tier
tiered.get("key")                          # Get from any tier (hot → warm → cold)
tiered.get_tier("key")                     # → "hot", "warm", "cold", or None

tiered.promote("key")                      # Move to hot tier
tiered.demote("key", target="cold")        # Move to lower tier

tiered.all_paths()                         # All paths across all tiers
tiered.stats()                             # Stats for each tier
```

---

## Compress

Key abbreviation and JSON minification.

```python
from json_memory import compress, decompress, savings_report, minify

data = {"user": {"email": "alice@example.com", "timezone": "UTC+1"}}

# Compress keys
compressed = compress(data)
# → {"u": {"em": "alice@example.com", "tz": "UTC+1"}}

# Decompress (round-trip)
original = decompress(compressed)

# Minify (remove whitespace)
minified = minify(data)

# Measure savings
report = savings_report(json.dumps(data), json.dumps(compressed))
# → {"savings_pct": 8.3, "ratio": 0.917}
```

---

## Semantic Search

Optional embedding-based retrieval. Requires `pip install json-memory[semantic]`.

```python
from json_memory import SmartMemory
from json_memory.semantic import enhance_smart_memory, SemanticIndex

# Enhance existing SmartMemory
mem = SmartMemory("agent.json")
enhance_smart_memory(mem)  # monkey-patches with semantic scoring

# Now "When do I wake up?" finds user.timezone
mem.recall_relevant("When do I wake up?")
```

### Standalone SemanticIndex

```python
index = SemanticIndex(model_name="all-MiniLM-L6-v2")
index.add("user.timezone", "GMT+7 based in Jakarta")
index.add("bot.restart", "kill and restart the bot")

results = index.search("What time zone am I in?")
# → [("user.timezone", 0.89), ("bot.restart", 0.12)]
```

Gracefully degrades: if `sentence-transformers` or `faiss` not installed, returns empty results without crashing.

---

## Concept Map

Zero-dep semantic expansion for queries.

```python
from json_memory.concept_map import expand_query_semantic, get_concept_category

# Expand query tokens with semantic neighbors
tokens = expand_query_semantic({"when", "wake"})
# → {"when", "wake", "timezone", "time", "morning", "clock", "alarm", ...}

# Get concept category
get_concept_category("trading")  # → "trading"
get_concept_category("restart")  # → "action"
get_concept_category("server")   # → "location"
```

100+ concept mappings covering: identity, time, location, actions, trading, project, system, communication.

Used internally by `_normalize_tokens()` in SmartMemory for all retrieval operations.
