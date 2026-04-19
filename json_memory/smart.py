"""
SmartMemory — Intelligent memory layer for AI agents.

Wraps Memory + Synapse with:
- Weighted retrieval scoring (recency + frequency + keyword relevance)
- Auto-extraction from conversations
- Tiered memory (hot/warm/cold)
- Smart prompt injection (only relevant context)
- Optional semantic search (pip install json-memory[semantic])

Zero dependencies for core functionality.
"""

import re
import time
import math
import json
import threading
from typing import Any, Optional
from pathlib import Path
from collections import Counter

from .memory import Memory
from .synapse import Synapse
from .concept_map import expand_query_semantic, get_concept_category


# ── Auto-Extractor Patterns ───────────────────────────────────────────

EXTRACTION_PATTERNS = [
    # Name patterns (use lookahead to stop at delimiters)
    (r"(?:my name is|i'?m called|call me|I am) (\w+)(?:\s+and|\s*,|\s*\.|$)", "user.name", 0.8),
    (r"(?:they/them|he/him|she/her) pronouns?", "user.pronouns", 0.9),
    # Location/timezone
    (r"(?:i live in|i'?m (?:from|in)|based in) ([A-Z][\w\s,]+?)(?:\s*\.|\s*,|\s+(?:and|but|so)\s)", "user.location", 0.85),
    (r"(?:timezone is|my timezone|I'?m in) ((?:GMT|UTC|CST|EST|PST|IST)[+-]?\d*)", "user.timezone", 0.95),
    # Platform
    (r"(?:on |from )?(telegram|discord|slack|whatsapp|twitter|threads)", "user.platform", 0.7),
    # Preferences — "I prefer to use X" or "I prefer X"
    (r"(?:i prefer to use|i prefer|i like|i use) (\w+)(?:\s+for\s|\s*\.|\s*,|\s+(?:and|but|so)\s|$)", "user.preferences", 0.6),
    # Technical
    (r"(?:my (?:stack|tech|lang)) (?:is|includes?) ([\w\s/\-+,.]+?)(?:\s*\.|\s*,|\s+(?:and|but))", "user.tech_stack", 0.7),
    # Explicit remember requests
    (r"(?:remember|note|save) (?:that )?(.+)", "user.requested", 0.95),
    # Project names
    (r"(?:project (?:is|called|named)|working on) ([\w\-_]+)", "project.name", 0.7),
]


def _normalize_tokens(text: str) -> set[str]:
    """Split text into lowercase tokens for matching. Includes synonym + semantic expansion."""
    tokens = set(re.findall(r'\b\w{3,}\b', text.lower()))

    # Basic synonym expansion
    synonyms = {
        'who': {'name', 'user', 'identity'},
        'me': {'user', 'name'},
        'my': {'user'},
        'restart': {'restart_cmd', 'restart', 'start', 'boot'},
        'ip': {'ip', 'address'},
        'server': {'server', 'host', 'machine'},
        'project': {'project', 'repo', 'repository'},
        'trading': {'bot', 'exchange', 'strategy', 'trade'},
        'trades': {'bot', 'exchange', 'strategy'},
        'exchange': {'exchange', 'bot', 'binance'},
        'timezone': {'timezone', 'gmt', 'utc'},
        'time': {'timezone', 'gmt', 'utc'},
        'location': {'location', 'city', 'country'},
    }

    expanded = set(tokens)
    for token in tokens:
        if token in synonyms:
            expanded.update(synonyms[token])

    # Semantic expansion (concept map)
    expanded = expand_query_semantic(expanded)

    return expanded


def _recency_score(last_accessed: float, now: float, half_life: float = 3600) -> float:
    """Exponential decay based on time since last access. Half-life in seconds."""
    if last_accessed <= 0:
        return 0.1
    age = now - last_accessed
    return math.exp(-0.693 * age / half_life)


def _frequency_score(access_count: int, max_count: int) -> float:
    """Log-scaled frequency. Prevents domination by one over-accessed key."""
    if max_count <= 0:
        return 0.5
    return math.log1p(access_count) / math.log1p(max_count)


def _keyword_relevance(fact_tokens: set[str], query_tokens: set[str]) -> float:
    """Jaccard-like overlap between fact and query tokens."""
    if not fact_tokens or not query_tokens:
        return 0.0
    intersection = fact_tokens & query_tokens
    union = fact_tokens | query_tokens
    return len(intersection) / len(union) if union else 0.0


# ── Path Metadata ─────────────────────────────────────────────────────

class PathMeta:
    """Tracks per-path metadata for scoring."""
    __slots__ = ['last_accessed', 'access_count', 'created_at', 'tier', 'tokens']

    def __init__(self):
        now = time.time()
        self.last_accessed = now
        self.access_count = 1
        self.created_at = now
        self.tier = 'hot'
        self.tokens: set[str] = set()


class TieredMemory:
    """Manages hot/warm/cold memory tiers with automatic promotion/demotion.

    Tiers:
        hot   — recently/frequently accessed. Injected into prompts.
        warm  — older but still reachable. Available via recall_relevant().
        cold  — archived to disk. Loaded on-demand only.
    """

    def __init__(self, path: str, max_hot_chars: int = 2000, max_warm_chars: int = 5000):
        self.path = Path(path)
        self.max_hot_chars = max_hot_chars
        self.max_warm_chars = max_warm_chars

        # Three Memory instances
        self.hot = Memory(max_chars=max_hot_chars, auto_flush_path=str(self.path.with_suffix('.hot.json')))
        self.warm = Memory(max_chars=max_warm_chars, auto_flush_path=str(self.path.with_suffix('.warm.json')))
        self.cold_path = self.path.with_suffix('.cold.json')
        self.cold = Memory(max_chars=50000)

        # Load cold from disk if exists
        if self.cold_path.exists():
            try:
                self.cold = Memory.from_json(self.cold_path.read_text(encoding='utf-8'), max_chars=50000)
            except Exception:
                pass

    def set(self, path: str, value, tier: str = 'hot', ttl: int = None):
        """Store value in specified tier."""
        target = {'hot': self.hot, 'warm': self.warm, 'cold': self.cold}[tier]
        target.set(path, value, ttl=ttl)
        if tier == 'cold':
            self._flush_cold()

    def get(self, path: str, default=None):
        """Get from any tier (hot → warm → cold)."""
        val = self.hot.get(path, default=_MISSING)
        if val is not _MISSING:
            return val
        val = self.warm.get(path, default=_MISSING)
        if val is not _MISSING:
            return val
        return self.cold.get(path, default=default)

    def get_tier(self, path: str) -> Optional[str]:
        """Find which tier a path lives in."""
        if self.hot.has(path):
            return 'hot'
        if self.warm.has(path):
            return 'warm'
        if self.cold.has(path):
            return 'cold'
        return None

    def promote(self, path: str):
        """Move path to hot tier."""
        val = self.get(path)
        if val is not None:
            self.warm.delete(path, prune=True)
            self.cold.delete(path, prune=True)
            self.hot.set(path, val)
            self._flush_cold()

    def demote(self, path: str, target: str = 'cold'):
        """Move path to a lower tier."""
        val = self.hot.get(path, default=_MISSING)
        if val is _MISSING:
            val = self.warm.get(path, default=_MISSING)
        if val is _MISSING:
            return
        self.hot.delete(path, prune=True)
        self.warm.delete(path, prune=True)
        self.set(path, val, tier=target)

    def all_paths(self) -> list[str]:
        """All paths across all tiers."""
        seen = set()
        paths = []
        for mem in [self.hot, self.warm, self.cold]:
            for p in mem.paths():
                if p not in seen:
                    seen.add(p)
                    paths.append(p)
        return paths

    def stats(self) -> dict:
        """Stats for all tiers."""
        return {
            'hot': self.hot.stats(),
            'warm': self.warm.stats(),
            'cold': self.cold.stats(),
        }

    def _flush_cold(self):
        """Persist cold tier to disk."""
        self.cold_path.parent.mkdir(parents=True, exist_ok=True)
        self.cold_path.write_text(self.cold.export(), encoding='utf-8')


# Sentinel for tiered get
class _Missing:
    pass
_MISSING = _Missing()


class SmartMemory:
    """Intelligent agent memory with weighted retrieval and auto-extraction.

    Combines Memory (structured storage) + Synapse (associative links) with:
    - Weighted scoring: recency × frequency × keyword relevance
    - Auto-extraction: passive fact detection from conversation
    - Smart recall: only relevant facts injected into prompts
    - Tiered storage: hot/warm/cold with automatic promotion/demotion

    Args:
        path: File path for persistence.
        max_chars: Total character budget.
        max_results: Max facts to return from recall_relevant().
        extract_confidence: Minimum confidence for auto-extraction (0.0-1.0).
        tiered: Enable tiered memory (hot/warm/cold).
        recency_half_life: Half-life in seconds for recency scoring.

    Example:
        >>> mem = SmartMemory("agent.json", max_chars=5000)
        >>> mem.remember("user.name", "Alice")
        >>> mem.remember("bot.restart", "kill && nohup ./bot")
        >>> mem.recall_relevant("How do I restart?")
        {"bot.restart": "kill && nohup ./bot"}
        >>> mem.prompt_context("How do I restart?")
        '## Memory\\n- bot.restart: kill && nohup ./bot'
    """

    def __init__(self, path: str = "smart_memory.json", max_chars: int = 5000,
                 max_results: int = 8, extract_confidence: float = 0.6,
                 tiered: bool = False, recency_half_life: float = 3600):
        self.path = Path(path)
        self.max_chars = max_chars
        self.max_results = max_results
        self.extract_confidence = extract_confidence
        self.recency_half_life = recency_half_life

        # Core storage
        self.mem = Memory(max_chars=max_chars, auto_flush_path=str(self.path))
        self.brain = Synapse()

        # Tiered storage (optional)
        self.tiered = TieredMemory(path, max_hot_chars=max_chars) if tiered else None

        # Per-path metadata for scoring
        self._meta: dict[str, PathMeta] = {}
        self._meta_path = self.path.with_suffix('.meta.json')
        self._lock = threading.RLock()

        # Episodic memory: timeline of conversation topics
        self._episodes: list[dict] = []
        self._episodes_path = self.path.with_suffix('.episodes.json')
        self._max_episodes = 100

        # Conversation context: tracks active topics in current session
        self._active_topics: list[str] = []
        self._turn_count = 0

        # Load metadata
        self._load_meta()
        self._load_episodes()

        # Initialize meta for existing data
        for p in self.mem.paths():
            if p not in self._meta:
                self._init_meta(p, self.mem.get(p))

    # ── Core Operations ──────────────────────────────────────────────

    def remember(self, path: str, value, ttl: int = None, tags: list[str] = None):
        """Store a fact. Use dotted paths: 'user.name', 'project.status'

        Args:
            path: Dotted path key.
            value: Any JSON-serializable value.
            ttl: Optional time-to-live in seconds.
            tags: Optional list of tags for categorization.
        """
        with self._lock:
            self.mem.set(path, value, ttl=ttl)
            self._init_meta(path, value)

            if self.tiered:
                self.tiered.set(path, value, tier='hot', ttl=ttl)

            # Auto-link to tags via Synapse
            if tags:
                for tag in tags:
                    self.brain.link(tag, [path])

            self._save_meta()

    def recall(self, path: str, default=None):
        """Retrieve a fact by exact dotted path."""
        with self._lock:
            self._touch(path)
            if self.tiered:
                return self.tiered.get(path, default=default)
            return self.mem.get(path, default=default)

    def forget(self, path: str):
        """Delete a fact and its metadata."""
        with self._lock:
            self.mem.delete(path, prune=True)
            self._meta.pop(path, None)
            if self.tiered:
                self.tiered.hot.delete(path, prune=True)
                self.tiered.warm.delete(path, prune=True)
                self.tiered.cold.delete(path, prune=True)
            self._save_meta()

    def search(self, pattern: str) -> dict:
        """Find facts matching a glob pattern."""
        return self.mem.find(pattern)

    def context(self) -> str:
        """Export ALL memory as compact JSON for injection."""
        return self.mem.export()

    # ── Smart Retrieval ──────────────────────────────────────────────

    def score(self, path: str, query_tokens: set[str] = None, now: float = None) -> float:
        """Score a path's relevance. Combines recency × frequency × keyword match.

        Args:
            path: The dotted path to score.
            query_tokens: Token set from user's message (for keyword matching).
            now: Current timestamp (defaults to time.time()).

        Returns:
            Float 0.0-1.0 relevance score.
        """
        now = now or time.time()
        meta = self._meta.get(path)
        if not meta:
            return 0.0

        # Recency: exponential decay
        recency = _recency_score(meta.last_accessed, now, self.recency_half_life)

        # Frequency: log-scaled
        max_count = max((m.access_count for m in self._meta.values()), default=1)
        frequency = _frequency_score(meta.access_count, max_count)

        # Keyword relevance
        keyword = 0.5  # default (neutral) when no query
        if query_tokens:
            keyword = _keyword_relevance(meta.tokens, query_tokens)

        # Weighted combination
        if query_tokens:
            # With a query: keyword dominates, recency/frequency are tiebreakers
            if keyword > 0:
                return 0.1 * recency + 0.05 * frequency + 0.85 * keyword
            else:
                # No keyword match at all: very low base score
                return 0.03 * recency + 0.02 * frequency
        else:
            # Without a query: recency + frequency only
            return 0.6 * recency + 0.4 * frequency

    def prompt_context(self, query: str = None, max_results: int = None,
                       format_fn=None) -> str:
        """Generate lean prompt context from relevant facts only.

        Instead of injecting the entire memory (wasting tokens), returns
        only the facts relevant to the current query.

        Args:
            query: User's current message.
            max_results: Override default max results.
            format_fn: Custom formatter (path, value) -> str.

        Returns:
            Formatted string ready for prompt injection.
        """
        relevant = self.recall_relevant(query, max_results)
        if not relevant:
            return ""

        lines = []
        formatter = format_fn or (lambda p, v: f"- {p}: {v}" if not isinstance(v, (list, dict)) else f"- {p}: {json.dumps(v, ensure_ascii=False)}")

        for path, value in relevant.items():
            lines.append(formatter(path, value))

        return "## Memory\n" + "\n".join(lines)

    # ── Auto-Extraction ──────────────────────────────────────────────

    def process_conversation(self, user_msg: str, agent_msg: str = None) -> list[dict]:
        """Passively extract and store facts from conversation.

        Detects factual statements without explicit remember() calls.
        Auto-logs conversation episodes for timeline recall.
        Returns list of extracted facts for review.

        Args:
            user_msg: User's message.
            agent_msg: Optional agent's response.

        Returns:
            List of dicts with 'path', 'value', 'confidence', 'source'.
        """
        extracted = []
        text = user_msg
        if agent_msg:
            text += " " + agent_msg

        for pattern, default_path, base_confidence in EXTRACTION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip() if match.lastindex else match.group(0).strip()
                if len(value) < 2 or len(value) > 200:
                    continue

                # Boost confidence if user explicitly asks to remember
                confidence = base_confidence
                if 'remember' in user_msg.lower() or 'note' in user_msg.lower():
                    confidence = min(confidence + 0.15, 1.0)

                if confidence >= self.extract_confidence:
                    # Determine path
                    path = default_path
                    if 'remember' in user_msg.lower() and path == 'user.requested':
                        # Try to infer a better path
                        path = self._infer_path(value)

                    entry = {
                        'path': path,
                        'value': value,
                        'confidence': confidence,
                        'source': 'user' if user_msg else 'agent',
                    }
                    extracted.append(entry)

                    # Store if not duplicate
                    existing = self.mem.get(path)
                    if existing != value:
                        self.remember(path, value, tags=['auto_extracted'])

        # Auto-detect topic and log episode (passive — no manual call needed)
        topic = self._detect_topic(text)
        if topic:
            related_paths = [e['path'] for e in extracted] if extracted else []
            summary = self._summarize_turn(user_msg, agent_msg)
            self.log_episode(topic, summary=summary, paths=related_paths)

        # Advance turn counter
        self.advance_turn()

        return extracted

    def _detect_topic(self, text: str) -> str | None:
        """Detect the dominant topic from conversation text.

        Uses token matching against known topic categories.
        Returns the topic string or None if unclear.
        """
        tokens = set(re.findall(r'\w{3,}', text.lower()))

        # Score each known topic category
        topic_scores = {}
        for token in tokens:
            category = get_concept_category(token)
            if category:
                topic_scores[category] = topic_scores.get(category, 0) + 1

        if not topic_scores:
            # Fallback: check against existing memory paths
            for token in tokens:
                for path in self.mem.paths():
                    if token in path.lower():
                        topic_scores[token] = topic_scores.get(token, 0) + 1

        if topic_scores:
            best_topic = max(topic_scores, key=topic_scores.get)
            if topic_scores[best_topic] >= 2:  # at least 2 matches
                return best_topic

        return None

    def _summarize_turn(self, user_msg: str, agent_msg: str = None) -> str:
        """Create a brief summary of the conversation turn for episode logging."""
        # Truncate to first meaningful sentence
        msg = user_msg.strip().split('.')[0].split('\n')[0]
        if len(msg) > 80:
            msg = msg[:77] + "..."
        return msg

    def _infer_path(self, value: str) -> str:
        """Try to infer a good dotted path from the value content."""
        v_lower = value.lower()
        if any(w in v_lower for w in ['name is', 'call me', "i'm "]):
            return 'user.name'
        if any(w in v_lower for w in ['timezone', 'utc', 'gmt', 'est', 'pst']):
            return 'user.timezone'
        if any(w in v_lower for w in ['prefer', 'like', 'use']):
            return 'user.preferences'
        return 'user.notes'

    # ── Episodic Memory ──────────────────────────────────────────────

    def log_episode(self, topic: str, summary: str = None, paths: list[str] = None):
        """Log a conversation episode for timeline-based recall.

        Use this to track what was discussed, so later queries like
        "what did we talk about regarding the bot?" can find it.

        Args:
            topic: Topic keyword (e.g. "bot", "deployment", "PyPI").
            summary: Brief description of what was discussed.
            paths: Memory paths relevant to this episode.
        """
        with self._lock:
            episode = {
                'topic': topic.lower(),
                'summary': summary or topic,
                'paths': paths or [],
                'timestamp': time.time(),
                'turn': self._turn_count,
            }
            self._episodes.append(episode)

            # Trim old episodes
            if len(self._episodes) > self._max_episodes:
                self._episodes = self._episodes[-self._max_episodes:]

            # Track active topic
            if topic.lower() not in self._active_topics:
                self._active_topics.insert(0, topic.lower())
                self._active_topics = self._active_topics[:10]  # keep last 10

            self._save_episodes()

    def recall_episodes(self, topic: str = None, max_age_seconds: float = 86400,
                        limit: int = 5) -> list[dict]:
        """Find past conversation episodes by topic or recency.

        Solves: "what did we discuss about the bot yesterday?"

        Args:
            topic: Filter by topic keyword. None = most recent episodes.
            max_age_seconds: Only episodes within this window (default: 24h).
            limit: Max episodes to return.

        Returns:
            List of episode dicts with topic, summary, timestamp, paths.
        """
        now = time.time()
        cutoff = now - max_age_seconds

        candidates = [ep for ep in self._episodes if ep['timestamp'] >= cutoff]

        if topic:
            topic_lower = topic.lower()
            # Token-overlap scoring for topic matching
            topic_tokens = set(re.findall(r'\w{2,}', topic_lower))
            scored = []
            for ep in candidates:
                ep_tokens = set(re.findall(r'\w{2,}', ep['topic']))
                overlap = len(topic_tokens & ep_tokens) / max(len(topic_tokens), 1)
                # Also check summary
                summary_tokens = set(re.findall(r'\w{2,}', ep.get('summary', '').lower()))
                summary_overlap = len(topic_tokens & summary_tokens) / max(len(topic_tokens), 1)
                score = max(overlap, summary_overlap * 0.5)
                if score > 0:
                    scored.append((score, ep['timestamp'], ep))
            scored.sort(reverse=True)
            return [ep for _, _, ep in scored[:limit]]

        # No topic filter: return most recent
        candidates.sort(key=lambda ep: ep['timestamp'], reverse=True)
        return candidates[:limit]

    @property
    def active_topics(self) -> list[str]:
        """Topics discussed in current session."""
        return list(self._active_topics)

    def advance_turn(self):
        """Call once per conversation turn. Tracks session context."""
        self._turn_count += 1

    # ── Hybrid Fallback ──────────────────────────────────────────────

    def recall_relevant(self, query: str = None, max_results: int = None,
                        min_score: float = 0.1, fallback: bool = True) -> dict:
        """Retrieve only facts relevant to the current context.

        Scores all paths, returns top-N by relevance. This is the core
        improvement over injecting the entire memory into prompts.

        Args:
            query: User's message or query string.
            max_results: Override default max results.
            min_score: Minimum score threshold.
            fallback: If True and no strong matches, boost recent/active topics.

        Returns:
            Dict of {path: value} for relevant facts, ordered by score.
        """
        max_results = max_results or self.max_results
        now = time.time()
        query_tokens = _normalize_tokens(query) if query else set()

        scored = []
        for path in self.mem.paths():
            s = self.score(path, query_tokens, now)
            scored.append((s, path, self._meta.get(path)))

        # Smart filtering: when query has strong matches, suppress noise
        if query_tokens:
            # Find the max keyword score among all paths
            max_keyword = 0.0
            for _, path, meta in scored:
                if meta:
                    kw = _keyword_relevance(meta.tokens, query_tokens)
                    max_keyword = max(max_keyword, kw)

            # If there are strong keyword matches, apply adaptive threshold
            if max_keyword > 0.15:
                # Only keep items with meaningful keyword overlap or high recency
                adaptive_threshold = max(min_score, max_keyword * 0.3)
                scored = [(s, p, m) for s, p, m in scored if s >= adaptive_threshold]
            elif fallback and self._active_topics:
                # No strong matches — boost paths related to active topics
                scored = self._boost_by_active_topics(scored, query_tokens, now)
                scored = [(s, p, m) for s, p, m in scored if s >= min_score * 0.5]
            else:
                scored = [(s, p, m) for s, p, m in scored if s >= min_score]
        else:
            scored = [(s, p, m) for s, p, m in scored if s >= min_score]

        scored.sort(reverse=True)
        result = {}
        for score, path, _ in scored[:max_results]:
            val = self.mem.get(path)
            if val is not None:
                result[path] = val
                self._touch(path)

        return result

    def _boost_by_active_topics(self, scored, query_tokens, now):
        """When keyword match is weak, boost paths related to active session topics."""
        # Get tokens from active topics
        topic_tokens = set()
        for topic in self._active_topics[:5]:
            topic_tokens.update(re.findall(r'\w{2,}', topic.lower()))

        boosted = []
        for score, path, meta in scored:
            if meta and topic_tokens:
                topic_overlap = _keyword_relevance(meta.tokens, topic_tokens)
                if topic_overlap > 0:
                    score = score + 0.15 * topic_overlap  # moderate boost
            boosted.append((score, path, meta))
        return boosted

    # ── Context Window Manager ───────────────────────────────────────

    def build_context(self, query: str = None, max_chars: int = 2000,
                      include_episodes: bool = True) -> str:
        """Build complete agent context: relevant facts + recent episodes.

        This is the "all-in-one" method. Call this instead of prompt_context()
        when you want maximum recall with minimum tokens.

        Args:
            query: User's current message.
            max_chars: Character budget for the entire context block.
            include_episodes: Include recent conversation episodes.

        Returns:
            Formatted context string ready for prompt injection.
        """
        parts = []
        budget_remaining = max_chars

        # 1. Relevant facts (primary)
        relevant = self.recall_relevant(query, max_results=6, fallback=True)
        if relevant:
            fact_lines = []
            for path, value in relevant.items():
                line = f"- {path}: {_value_to_str(value)}"
                if len('\n'.join(fact_lines + [line])) <= budget_remaining * 0.7:
                    fact_lines.append(line)
            if fact_lines:
                parts.append("## Memory\n" + "\n".join(fact_lines))
                budget_remaining -= sum(len(l) for l in fact_lines)

        # 2. Recent episodes (secondary, if space allows)
        if include_episodes and budget_remaining > 100:
            episodes = self.recall_episodes(
                topic=query if query else None,
                max_age_seconds=3600,  # last hour
                limit=3
            )
            if episodes:
                ep_lines = []
                for ep in episodes:
                    line = f"- [{_time_ago(ep['timestamp'])}] {ep['summary']}"
                    if len('\n'.join(ep_lines + [line])) <= budget_remaining:
                        ep_lines.append(line)
                if ep_lines:
                    parts.append("## Recent Topics\n" + "\n".join(ep_lines))

        # 3. Active topics hint (tiny, always included)
        if self._active_topics and budget_remaining > 50:
            topics = ", ".join(self._active_topics[:5])
            parts.append(f"## Active: {topics}")

        return "\n\n".join(parts)

    # ── Associative Memory ───────────────────────────────────────────

    def link(self, concept: str, associations: list[str], weights: dict = None):
        """Create concept associations for associative recall."""
        self.brain.link(concept, associations, weights=weights)

    def associate(self, concept: str, depth: int = 1) -> list[str]:
        """Recall associated concepts."""
        return self.brain.activate(concept, depth=depth)

    # ── Snapshots ────────────────────────────────────────────────────

    def snapshot(self, label: str):
        """Save state before risky operations."""
        self.mem.snapshot(label)

    def rollback(self, label: str):
        """Restore state from snapshot."""
        self.mem.rollback(label)
        self._save_meta()

    # ── Stats & Debug ────────────────────────────────────────────────

    def stats(self) -> dict:
        """Memory stats with scoring metadata."""
        base = self.mem.stats()
        base['paths'] = len(self._meta)
        base['tiers'] = 'enabled' if self.tiered else 'disabled'

        # Top scored paths (without query)
        top = self._top_scored(n=5)
        base['top_scored'] = top

        if self.tiered:
            base['tier_stats'] = self.tiered.stats()

        return base

    def explain_score(self, path: str, query: str = None) -> dict:
        """Debug: show how a path's score is calculated."""
        meta = self._meta.get(path)
        if not meta:
            return {'error': f'No metadata for path: {path}'}

        now = time.time()
        query_tokens = _normalize_tokens(query) if query else set()
        max_count = max((m.access_count for m in self._meta.values()), default=1)

        recency = _recency_score(meta.last_accessed, now, self.recency_half_life)
        frequency = _frequency_score(meta.access_count, max_count)
        keyword = _keyword_relevance(meta.tokens, query_tokens) if query_tokens else 0.5

        if query_tokens:
            if keyword > 0:
                final = 0.1 * recency + 0.05 * frequency + 0.85 * keyword
            else:
                final = 0.03 * recency + 0.02 * frequency
        else:
            final = 0.6 * recency + 0.4 * frequency

        return {
            'path': path,
            'recency': round(recency, 3),
            'frequency': round(frequency, 3),
            'keyword_relevance': round(keyword, 3),
            'final_score': round(final, 3),
            'access_count': meta.access_count,
            'age_seconds': round(now - meta.last_accessed, 1),
            'tier': meta.tier,
        }

    # ── Internal ─────────────────────────────────────────────────────

    def _init_meta(self, path: str, value):
        """Initialize or update metadata for a path."""
        if path not in self._meta:
            self._meta[path] = PathMeta()
        meta = self._meta[path]
        meta.last_accessed = time.time()
        meta.tokens = self._extract_tokens(path, value)

    def _touch(self, path: str):
        """Record access to a path."""
        with self._lock:
            if path in self._meta:
                meta = self._meta[path]
                meta.last_accessed = time.time()
                meta.access_count += 1

                # Promotion: if accessed enough, move to hot
                if self.tiered and meta.tier != 'hot' and meta.access_count >= 3:
                    self.tiered.promote(path)
                    meta.tier = 'hot'

    def _extract_tokens(self, path: str, value) -> set[str]:
        """Extract searchable tokens from path and value."""
        tokens = set(re.findall(r'\w{2,}', path.lower()))
        if isinstance(value, str):
            tokens.update(re.findall(r'\w{2,}', value.lower()))
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str):
                    tokens.update(re.findall(r'\w{2,}', item.lower()))
        elif isinstance(value, dict):
            for k, v in value.items():
                tokens.update(re.findall(r'\w{2,}', str(k).lower()))
                tokens.update(re.findall(r'\w{2,}', str(v).lower()))
        return tokens

    def _top_scored(self, n: int = 5) -> list[dict]:
        """Return top-N scored paths (no query)."""
        now = time.time()
        scored = []
        for path in self._meta:
            s = self.score(path, now=now)
            scored.append((s, path))
        scored.sort(reverse=True)
        return [{'path': p, 'score': round(s, 3)} for s, p in scored[:n]]

    def _load_meta(self):
        """Load metadata from disk."""
        if self._meta_path.exists():
            try:
                raw = json.loads(self._meta_path.read_text(encoding='utf-8'))
                for path, data in raw.items():
                    meta = PathMeta()
                    meta.last_accessed = data.get('last_accessed', 0)
                    meta.access_count = data.get('access_count', 1)
                    meta.created_at = data.get('created_at', 0)
                    meta.tier = data.get('tier', 'hot')
                    meta.tokens = set(data.get('tokens', []))
                    self._meta[path] = meta
            except Exception:
                pass

    def _save_meta(self):
        """Persist metadata to disk."""
        try:
            raw = {}
            for path, meta in self._meta.items():
                raw[path] = {
                    'last_accessed': meta.last_accessed,
                    'access_count': meta.access_count,
                    'created_at': meta.created_at,
                    'tier': meta.tier,
                    'tokens': sorted(meta.tokens),
                }
            self._meta_path.parent.mkdir(parents=True, exist_ok=True)
            self._meta_path.write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
        except Exception:
            pass

    def _load_episodes(self):
        """Load episodic memory from disk."""
        if self._episodes_path.exists():
            try:
                data = json.loads(self._episodes_path.read_text(encoding='utf-8'))
                self._episodes = data.get('episodes', data) if isinstance(data, dict) else data

                # Load persistent active topics
                if isinstance(data, dict) and 'active_topics' in data:
                    self._active_topics = data['active_topics'][:10]
                else:
                    # Rebuild from recent episodes
                    recent = sorted(self._episodes, key=lambda e: e.get('timestamp', 0), reverse=True)
                    seen = set()
                    for ep in recent[:10]:
                        topic = ep.get('topic', '').lower()
                        if topic and topic not in seen:
                            seen.add(topic)
                            self._active_topics.append(topic)
            except Exception:
                self._episodes = []

    def _save_episodes(self):
        """Persist episodic memory and active topics to disk."""
        try:
            self._episodes_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'episodes': self._episodes,
                'active_topics': self._active_topics[:10],
            }
            self._episodes_path.write_text(
                json.dumps(data, ensure_ascii=False), encoding='utf-8'
            )
        except Exception:
            pass


def _value_to_str(value) -> str:
    """Convert value to readable string."""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _time_ago(timestamp: float) -> str:
    """Human-readable time ago."""
    seconds = time.time() - timestamp
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds / 3600)}h ago"
    return f"{int(seconds / 86400)}d ago"
