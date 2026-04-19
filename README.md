# json-memory

[![PyPI](https://img.shields.io/pypi/v/json-memory)](https://pypi.org/project/json-memory/)
[![Python](https://img.shields.io/pypi/pyversions/json-memory)](https://pypi.org/project/json-memory/)
[![License](https://img.shields.io/pypi/l/json-memory)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/json-memory)](https://pypi.org/project/json-memory/)

**Structured memory for AI agents** — organize, access, and navigate agent memory like a human brain.

```bash
pip install json-memory
```

**Performance** (991-char memory, commodity hardware):
- Parse: **0.014ms** (72K parses/sec)
- Access: **0.60μs** per dotted-path lookup
- Zero dependencies, pure Python

## The Problem

AI agents have limited memory windows. Storing facts as verbose prose wastes tokens and makes retrieval slow:

```
"User: Alice (@alice on Telegram). Prefers to be called Alice.
 Uses they/them pronouns. Timezone is UTC. Platform is Telegram. Prefers
 technical precision, especially in coding contexts. Wants a direct, warm..."
```
**~300 chars** for basic user info. No structured access — you scan the entire text every time.

## The Solution

Store memory as nested JSON with short keys — like synapses in a brain:

```json
{"u":{"n":"Alice","c":"@alice","p":"Alice","g":"they/them","tz":"UTC","plat":"Telegram"}}
```
**~95 chars** for the same data. But the real win isn't size — it's **O(1) access via dotted paths**: `memory.u.n` → `"Alice"`. No scanning. No parsing prose. Just keys.

## Why Structured Memory?

| | Prose | JSON Memory |
|---|---|---|
| Access pattern | Scan entire text | `memory.u.n` → instant |
| Nested hierarchy | ❌ Flat | ✅ Unlimited depth |
| Schema validation | ❌ No | ✅ Yes |
| Merge/upsert | ❌ Rewrite everything | ✅ Per-key updates |
| Human readable | ✅ Yes | ❌ Compact (but AI reads it) |

The trade-off: JSON is less human-readable but **machine-optimized**. For LLM agents with token budgets, that's the right call.

## Key Features

- 🧠 **Hierarchical nesting** — organize memory like a semantic tree
- 🗜️ **Key abbreviation** — ~25% size reduction on JSON keys
- 📦 **JSON minification** — ~30% savings removing whitespace
- ⚡ **Sub-millisecond parsing** — 0.05ms for 2KB of memory
- 🔗 **Synapse-like linking** — concepts connect to related concepts with weighted traversal
- 🐕 **WeightGate middleware** — passive learning from conversation flow
- 📐 **Schema validation** — define your memory structure once
- 🐍 **Zero dependencies** — pure Python, stdlib only

## Installation

```bash
git clone https://github.com/dioncx/json-memory.git
cd json-memory
pip install -e .
```

## Quick Start

```python
from json_memory import Memory

# Create a memory instance
mem = Memory(max_chars=2000)

# Set nested values
mem.set("u.name", "Alice")
mem.set("u.tz", "UTC")
mem.set("bot.binance.restart", "kill && nohup ./bot > log 2>&1")
mem.set("bot.binance.watchlist", ["BNB", "KITE", "AGLD"])

# Get by dotted path
print(mem.get("u.name"))           # "Alice"
print(mem.get("bot.binance.restart"))  # "kill && nohup ./bot > log 2>&1"

# Export/import
json_str = mem.export()            # minified JSON string
mem2 = Memory.from_json(json_str)  # reconstruct

# Stats
print(mem.stats())
# {"entries": 4, "chars_used": 146, "chars_max": 2000, "chars_free": 1854, "utilization": "7.3%"}
```

## Synapse Mode (Associative Memory)

Like how thinking of "coffee" activates "morning", "energy", "routine":

```python
from json_memory import Synapse

brain = Synapse()

# Define associations
brain.link("trading", ["binance", "strategy", "risk", "signals"])
brain.link("binance", ["api", "demo", "watchlist", "orders"])
brain.link("strategy", ["entry", "exit", "stoploss", "take_profit"])

# Traverse like a brain
results = brain.activate("trading")
# → ["binance", "strategy", "risk", "signals"]

results = brain.activate("trading", depth=2)
# → ["binance", "api", "demo", "watchlist", "orders", "strategy", "entry", "exit", ...]

# Find connections
brain.connections("binance")
# → {"parent": "trading", "children": ["api", "demo", "watchlist", "orders"]}
```

### Personalized Weights

Everyone's brain works differently. Set weights to customize recall order:

```python
# Person A: loves cappuccino
person_a = Synapse()
person_a.link("coffee", ["cappuccino", "americano", "espresso"],
              weights={"cappuccino": 0.95, "americano": 0.2, "espresso": 0.5})

# Person B: loves americano
person_b = Synapse()
person_b.link("coffee", ["cappuccino", "americano", "espresso"],
              weights={"cappuccino": 0.2, "americano": 0.9, "espresso": 0.4})

person_a.activate("coffee")  # → ["cappuccino", "espresso", "americano"]
person_b.activate("coffee")  # → ["americano", "espresso", "cappuccino"]
```

### Learning & Decay

Mimic how human memory strengthens with use and decays without:

```python
brain = Synapse()
brain.link("coffee", ["cappuccino", "americano"],
           weights={"cappuccino": 0.5, "americano": 0.5})

# User always picks cappuccino → connection strengthens
for _ in range(10):
    brain.strengthen("coffee", "cappuccino", boost=0.05)

# User never picks americano → connection decays
for _ in range(10):
    brain.weaken("coffee", "americano", decay=0.03)

brain.top_associations("coffee")
# → [("cappuccino", 1.0), ("americano", 0.2)]

brain.get_frequency("coffee", "cappuccino")  # → 10 (activation count)
```

## WeightGate — Passive Learning Middleware

Update weights **automatically** as messages flow through. No tool calls needed.

```python
from json_memory import WeightGate

# Create a gate (disabled by default — opt-in)
gate = WeightGate("synapse.json", enabled=True)

# Set up your concepts
gate.add_concept("coffee", {"cappuccino": 0.9, "americano": 0.3})
gate.add_concept("debug", {"check_logs": 0.9, "ask_user": 0.2})

# Process messages — weights update automatically
gate.process_input("How do I restart the bot?")
# → bot.restart strengthened, unused associations decay

gate.process_output("Run: kill && nohup ./bot > log")
# → Agent's response also updates weights

# After 20 interactions:
gate.top_associations("debug")
# → [("check_logs", 0.95), ("ask_user", 0.18)]  ← learned your pattern
```

### Enable/Disable

```python
# Disabled by default (opt-in)
gate = WeightGate("synapse.json")          # OFF
gate.enable()                               # ON
gate.disable()                              # OFF
gate.toggle()                               # Toggle

# Context manager (auto-enable, auto-save)
with WeightGate("synapse.json") as gate:
    gate.process_conversation(user_msg, agent_response)
# Gate disabled and saved on exit
```

### How It Works

```
User msg ──→ process_input() ──→ detect concepts ──→ weights ↑/↓
                                               ↓
                                    Agent processes
                                               ↓
Agent msg ──→ process_output() ──→ detect usage ──→ weights ↑
                                               ↓
                                    Response to user
```

- **Mentioned concepts** → associations strengthen (+0.05)
- **Unused associations** → decay (-0.01)
- **Agent's response** → further strengthens used concepts (+0.025)
- **Disabled gate** → returns empty dict, no side effects

## Compression Reality

The `compress()` module abbreviates JSON keys (e.g., `email` → `em`). Here's what it actually saves:

| Technique | Savings | What it does |
|-----------|---------|--------------|
| Key abbreviation | ~25% | `email` → `em`, `configuration` → `cfg` |
| JSON minification | ~30% | Removes whitespace from pretty-printed JSON |
| Combined | ~45-50% | Abbreviation + minification applied together |

**What it does NOT do:** compress values, deduplicate data, or apply general-purpose compression (gzip, zstd, etc.).

```python
from json_memory import compress, minify, savings_report

data = {"user": {"email": "alice@example.com", "timezone": "UTC+1"}}
compressed = compress(data)  # {"u": {"em": "alice@example.com", "tz": "UTC+1"}}

# Measure real savings (JSON vs JSON, not prose vs JSON)
report = savings_report(
    json.dumps(data),
    json.dumps(compressed)
)
# {"savings_pct": 8.3, "ratio": 0.917}  ← honest numbers
```

Parse speed: **0.05ms** for 2KB (tested on commodity hardware)

## Comparison

| Feature | Prose Memory | JSON Memory |
|---------|-------------|-------------|
| Human readable | ✅ Yes | ❌ Compact (but AI reads it) |
| Structured access | ❌ Scan entire text | ✅ Dotted path lookup |
| Nested hierarchy | ❌ Flat | ✅ Unlimited depth |
| Merge/upsert | ❌ Rewrite everything | ✅ Per-key updates |
| Parse speed | N/A | ✅ 0.05ms |
| Schema validation | ❌ No | ✅ Yes |

## Why Not Just Use [MemGPT/Letta]?

Those are full agent memory frameworks. This is a **building block** — a lightweight, zero-dependency library for structuring agent memory as JSON. Use it inside your agent, your RAG pipeline, your CLI tool, or your trading bot.

## Use Cases

- 🤖 **AI Agent memory** — compress context windows for LLMs
- 📊 **Trading bot state** — structured config and position tracking
- 🔧 **CLI tools** — compact persistent state
- 🎮 **Game state** — nested world/player/inventory data
- 📱 **IoT/Edge** — memory-constrained devices

## Contributing

PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
