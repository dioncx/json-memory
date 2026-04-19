# Changelog

## v0.2.0 — SmartMemory

### Added
- **SmartMemory** class: intelligent agent memory with weighted retrieval scoring
  - `remember()`, `recall()`, `forget()`, `search()` — structured storage
  - `recall_relevant(query)` — returns only facts relevant to the query (not everything)
  - `prompt_context(query)` — lean prompt injection (78% token savings vs full memory)
  - `process_conversation()` — auto-extraction of facts from conversation
  - `explain_score()` — debug why a fact was/wasn't returned
  - `link()` / `associate()` — associative concept memory
  - `snapshot()` / `rollback()` — state safety
- **Weighted retrieval scoring**: keyword relevance (85%) × recency decay (10%) × frequency (5%)
- **Smart filtering**: adaptive threshold suppresses noise when strong keyword matches exist
- **Synonym expansion**: "who am I?" → searches for `name`, `user`, `identity`
- **Auto-extraction patterns**: names, locations, timezones, preferences, platforms, "remember" requests
- **TieredMemory**: hot/warm/cold tiers with automatic promotion/demotion
- **Semantic layer** (`json_memory/semantic.py`): optional FAISS + sentence-transformers for embedding-based retrieval
  - `pip install json-memory[semantic]` — graceful fallback without deps
  - `enhance_smart_memory()` — monkey-patches SmartMemory with semantic scoring
- **144 tests** (41 new for SmartMemory)
- **Examples**: `examples/smart_memory_demo.py` — full end-to-end demo

### Changed
- Version bumped to 0.2.0
- README: added "SmartMemory — Intelligent Retrieval" section with scoring table, auto-extraction docs, semantic search guide

## v0.1.6

- Fixed sdist naming conventions
- Added TTL, LRU eviction, thread safety, audit logging, OpenAI tool schema generation

## v0.1.5

- WeightGate passive learning middleware
- Snowball stemmer integration

## v0.1.0

- Initial release: Memory, Synapse, Schema, Compress
