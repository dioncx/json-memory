# json-memory

**Structured memory for AI agents** — store, retrieve, and navigate agent memory without wasting tokens.

```bash
pip install json-memory
```

Zero dependencies. Pure Python. Works with any LLM.

---

## What problem does this solve?

AI agents remember things as prose — a wall of text injected into every prompt:

```
"User: Alice (@alice on Telegram). Prefers to be called Alice.
 Uses they/them pronouns. Timezone is UTC. Platform is Telegram. Prefers
 technical precision, especially in coding contexts. Wants a direct, warm..."
```

**~500 tokens** for basic user info. Injected every single turn. Even when the user only asked about their timezone.

json-memory fixes this.

## Show me it works

### Basic: structured storage

```python
from json_memory import Memory

mem = Memory(auto_flush_path="agent.json", max_chars=5000)

mem.set("user.name", "Alice")
mem.set("user.timezone", "GMT+7")
mem.set("bot.restart", "kill && nohup ./bot > log")

print(mem.get("user.timezone"))  # → "GMT+7"
print(mem.export())              # → compact JSON, persists to disk
```

### Smart: only relevant facts per query

```python
from json_memory import SmartMemory

mem = SmartMemory("agent.json", max_chars=5000)

mem.remember("user.name", "Alice")
mem.remember("user.timezone", "GMT+7")
mem.remember("bot.restart_cmd", "kill && nohup ./bot > log")
mem.remember("server.ip", "10.0.0.1")

# "What's my timezone?" → only timezone, not everything
mem.recall_relevant("What's my timezone?")
# → {"user.timezone": "GMT+7"}

# Inject into prompt — 32 chars, not 5000
mem.prompt_context("What's my timezone?")
# → "## Memory\n- user.timezone: GMT+7"
```

**Measured result: 92% token savings** (564 → 47 tokens per turn on real agent memory).

### Passive: learns from conversation

```python
# No explicit remember() calls needed
mem.process_conversation("My name is Bob and I live in Tokyo")
# → auto-stores user.name: Bob, user.location: Tokyo

mem.process_conversation("Remember that the deploy command is go build")
# → auto-stores user.notes
```

Also auto-detects conversation topics and logs searchable episodes:

```python
mem.recall_episodes("bot")
# → [{"topic": "trading", "summary": "Discussed bot restart", ...}]
```

---

## How it works

Three layers, each optional:

### Layer 1: Memory (structured storage)
- Dotted-path access: `user.name`, `bot.config.api_key`
- TTL (auto-expiring keys), LRU eviction, snapshots
- Auto-flush to disk, thread-safe

### Layer 2: SmartMemory (intelligent retrieval)
- **Weighted scoring**: keyword relevance (85%) × recency (10%) × frequency (5%)
- **Smart filtering**: "What's my timezone?" returns 1 result, not 8
- **Semantic expansion**: "When do I wake up?" finds `user.timezone`
- **Auto-extraction**: learns facts from conversation passively
- **Episodic memory**: searchable timeline of what was discussed
- **Hybrid fallback**: when keywords fail, uses active topic context

### Layer 3: Synapse (associative memory)

```python
from json_memory import Synapse

brain = Synapse()
brain.link("trading", ["binance", "strategy", "risk"])
brain.link("binance", ["api", "demo", "watchlist"])

brain.activate("trading", depth=2)
# → ["binance", "api", "demo", "watchlist", "strategy", "risk"]
```

Like how thinking of "coffee" activates "morning", "energy". Weighted, learnable, decays over time.

### Layer 4: Advanced Features (v1.1.0+)

#### Contradiction Detection
```python
mem = SmartMemory("agent.json")

# Store a fact
mem.remember("user.status", "active")

# Try to store contradictory fact
result = mem.remember("user.status", "inactive", check_contradictions=True)
# No contradiction detected (same path is an update)

# Get all contradictions in memory
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

#### Forgetting Curve & Reinforcement
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

# Simulate memory decay
simulation = mem.simulate_memory_decay("project.deadline", days=30)
```

---

## Benchmarks

| Metric | Prose memory | json-memory | Savings |
|--------|-------------|-------------|---------|
| Tokens per turn | ~564 | ~47 | **92%** |
| Access speed | Scan text | 3.7μs | **150,000×** |
| Retrieval accuracy | N/A (dumps all) | 100% on structured queries | ✅ |
| Dependencies | 0 | 0 | Same |

Tested against real agent memory (38 facts, 7 query types). See `examples/smart_memory_demo.py` to reproduce.

---

## vs Alternatives

| | Prose | ChatGPT Memory | MemGPT | json-memory |
|---|---|---|---|---|
| Dependencies | 0 | SaaS | 15+ pkgs | **0** |
| Self-hosted | ✅ | ❌ | ✅ | **✅** |
| Smart retrieval | ❌ | Partial | ✅ | **✅** |
| Auto-extraction | ❌ | Partial | ✅ | **✅** |
| Associative memory | ❌ | ❌ | ❌ | **✅** |
| Token efficiency | 0% | ~30% | ~60% | **92%** |
| Setup | None | None | Complex | **4 lines** |

json-memory is the **SQLite of agent memory** — not the most powerful, but the lightest, simplest, and cheapest to run.

---

## Who is this for?

- **AI agent builders** — drop-in memory for Claude, GPT, or custom agents
- **Bot developers** — structured state for trading bots, chatbots, automation
- **CLI tool makers** — compact persistent state without a database
- **Anyone** who needs to store structured data with fast access and zero dependencies

---

## Installation

```bash
pip install json-memory
```

Optional extras:
```bash
pip install json-memory[semantic]   # FAISS + sentence-transformers for embedding search
pip install json-memory[stem]       # Snowball stemmer for better word matching
```

---

## Works with

- **Claude** — inject `mem.prompt_context()` into system prompt
- **OpenAI GPT** — inject into system message, or use `Schema.to_openai_tools()`
- **LangChain** — wrap `SmartMemory` as a custom memory class
- **Any agent loop** — call `remember()` / `recall_relevant()` / `prompt_context()`

No lock-in. Just Python.

---

## Examples

| File | What it shows |
|------|--------------|
| `examples/basic_usage.py` | Memory + Synapse basics |
| `examples/agent_memory.py` | Drop-in AgentMemory class |
| `examples/smart_memory_demo.py` | SmartMemory full demo with benchmarks |
| `examples/advanced_features_demo.py` | v1.1.0 features: contradiction detection, consolidation, forgetting curve |

---

## API (Quick Reference)

**SmartMemory** (recommended for agents):
```python
mem = SmartMemory("agent.json", max_chars=5000)
mem.remember("user.name", "Alice")          # store
mem.recall("user.name")                     # exact lookup
mem.recall_relevant("What's my name?")      # smart retrieval
mem.prompt_context("What's my name?")       # lean prompt injection
mem.process_conversation("My name is Bob")  # auto-extract + auto-log
mem.recall_episodes("bot")                  # episodic timeline
mem.link("debug", ["logs", "reproduce"])    # associative memory
mem.associate("debug")                      # concept activation
mem.snapshot("before_change")               # save state
mem.rollback("before_change")               # restore state
mem.explain_score("user.name", "Who am I?") # debug scoring

# v1.1.0+ Advanced features
mem.get_contradictions()                    # find conflicting facts
mem.consolidate_memory()                    # find related facts
mem.auto_consolidate()                      # merge related facts
mem.get_memory_strength("user.name")        # analyze memory strength
mem.get_memories_needing_reinforcement()    # prioritize reinforcement
mem.reinforce_memory("user.name")           # strengthen against forgetting
mem.simulate_memory_decay("project.deadline", days=30)  # predict decay
```

**Memory** (low-level storage):
```python
mem = Memory(max_chars=2000, auto_flush_path="data.json")
mem.set("path.to.key", "value")
mem.get("path.to.key")
mem.delete("path.to.key")
mem.find("user.*")           # glob search
mem.merge({"a": {"b": 1}})   # bulk update
mem.export()                 # minified JSON
mem.stats()                  # size/utilization
```

Full API docs: [docs/API.md](docs/API.md)

---

## Contributing

PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
<b>json-memory</b> — the lightest agent memory that actually works.<br>
<a href="https://pypi.org/project/json-memory/">PyPI</a> ·
<a href="https://github.com/dioncx/json-memory">GitHub</a> ·
<a href="examples/">Examples</a>
</p>
