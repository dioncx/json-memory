# Changelog

## v0.6.0 — Memory Pruning & Lifecycle Management

### Added
- **Memory pruning system** — prevents unbounded growth in long-running agents:
  - `prune()` method with multiple pruning strategies:
    - TTL-based expiration (time-to-live per fact)
    - Age-based pruning (remove facts older than X seconds)
    - Frequency-based pruning (remove rarely accessed facts)
    - Size-based pruning (enforce total memory size limit)
    - Dry run mode for safe testing
  - `archive()` method for manual archival to cold storage
  - `lifecycle_stats()` method for memory health monitoring:
    - Total facts, size, and average age
    - Expired and archived fact counts
    - Tier distribution (hot/warm/cold)
    - Health assessment (healthy/warning/critical)

- **Enhanced PathMeta** for lifecycle tracking:
  - TTL (time-to-live) support per fact
  - Expiration timestamp tracking
  - Archive status flag
  - Size tracking in bytes

- **20 new tests** for pruning functionality (all passing)

### Fixed
- **TieredMemory deletion** — now properly deletes from all three tiers (hot/warm/cold)
- **Size calculation** — uses paths() instead of non-existent values() method

### Performance
- **Total tests**: 164 passing (was 144)
- **Prevents memory bloat** in production agents running 24/7
- **Configurable retention policies** for different use cases


## v0.5.0 — Enhanced Concept Map & Path Token Fix

### Added
- **Enhanced concept map** for better semantic matching:
  - `mistakes` ↔ `lesson` — "What mistakes?" now finds lesson facts
  - `projects` ↔ `project` — "What projects?" now finds project facts
  - `talk` ↔ `communication` — "How should I talk?" now finds user.communication
  - `auto` ↔ `systemd`/`restart` — "How does it restart automatically?" finds bot status
  - Added: `speak`, `chat`, `conversation`, `learning`, `remember`, `forget`, `systemd`, `daemon`

### Fixed
- **Path token extraction** in `score()` method:
  - Now splits paths on dots AND underscores (e.g., `mem.lesson_gateway` → `["lesson", "gateway"]`)
  - Previously: `mem.lesson_gateway` → `["lesson_gateway"]` (single token)
  - Path token boost now works correctly for compound path names
- **Complex query retrieval** — 100% success rate on 7 targeted test queries:
  - "What mistakes have been made?" → finds lesson facts ✓
  - "What projects are being worked on?" → finds project facts ✓
  - "How should I talk to the user?" → finds user.communication ✓
  - "How does the bot restart automatically?" → finds bot status ✓
  - "What runs on a schedule?" → finds cron config ✓

### Performance
- **Stress test results**: 10/10 complex queries successful
- **Cross-domain connections**: 5/10 queries connect multiple domains
- **Average injection**: ~252 chars per query (97.6% more efficient than legacy)
- **All 144 tests passing**


## v0.4.0 — Stem Expansion & Retrieval Quality

### Added
- **Stem expansion** in `_normalize_tokens()` — strip common suffixes (ial, ion, ing, ed, ly, ment, ness, able, ive, al, ic, ty) for broader token matching
- Handles word variant matching: "professional" ↔ "profession", "trading" ↔ "trade"
- Part of memory-maintenance skill patches for retrieval quality

### Fixed
- **_keyword_relevance()** now returns 1.0 for perfect token matches (was returning 0.8)
- Fixes `test_keyword_perfect_match` test failure
- All 144 tests now pass

### Changed
- **All 7 patches from memory-maintenance skill complete:**
  1. ✓ Stem expansion in `_normalize_tokens()`
  2. ✓ Containment scoring in `_keyword_relevance()`
  3. ✓ Zero keyword match suppression in `score()`
  4. ✓ Pass path tokens to `_keyword_relevance()`
  5. ✓ Adaptive threshold (0.4) in `recall_relevant()`
  6. ✓ Career/professional concept mappings
  7. ✓ Expanded 'who' mapping with background/profession
- **SmartMemory benchmark: 97.6% efficiency gain vs legacy system**
  - ~120 chars contextual injection vs ~4,918 chars legacy full dump
  - Constant scaling regardless of memory size
  - Targeted relevance per query


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
