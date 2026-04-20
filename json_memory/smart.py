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
from typing import Any, Optional, List, Dict, Tuple
from pathlib import Path
from collections import Counter

from .memory import Memory
from .synapse import Synapse
from .concept_map import expand_query_semantic, get_concept_category
from .contradiction import detect_contradictions, Contradiction, ContradictionDetector
from .consolidation import consolidate_memory, ConsolidationGroup
from .forgetting import ForgettingCurve, MemoryStrength
from .visualizer import MemoryVisualizer, visualize_memory
from .versioning import MemoryVersioning, MemoryDiff
from .encryption import MemoryEncryption, EncryptedValue
from .search import AdvancedSearch, SearchResult


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
    # Email addresses
    (r"[\w.+-]+@[\w-]+\.[\w.-]+", "user.email", 0.9),
    # Phone numbers
    (r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "user.phone", 0.85),
    # Social media handles
    (r"@(\w+)", "user.social", 0.75),
    # Job titles/companies
    (r"(?:i(?:'m| am) (?:a|an) )([a-z]+(?:\s[a-z]+)*?)(?:\s+(?:at|for|with)\s+([A-Z][\w\s]+?))?(?:\s*\.|\s*,|\s+(?:and|but|so)\s|$)", "user.profession", 0.8),
    # Skills/technologies
    (r"(?:i(?:'m| am) (?:good at|skilled in|experienced with)|my skills (?:include|are)) ([\w\s,]+?)(?:\s*\.|\s*,|\s+(?:and|but|so)\s|$)", "user.skills", 0.75),
    # Goals/objectives
    (r"(?:my goal is|i(?:'m| am) trying to|i want to) (.+?)(?:\s*\.|\s*,|\s+(?:and|but|so)\s|$)", "user.goals", 0.8),
    # Detailed preferences
    (r"(?:i prefer|i like|i enjoy|i love) (?:to )?(.+?)(?:\s*\.|\s*,|\s+(?:and|but|so)\s|$)", "user.likes", 0.65),
    (r"(?:i dislike|i hate|i don't like|i avoid) (?:to )?(.+?)(?:\s*\.|\s*,|\s+(?:and|but|so)\s|$)", "user.dislikes", 0.65),
    # Dates and times
    (r"(?:my birthday is|i was born on|born on) (\w+ \d{1,2}(?:st|nd|rd|th)?(?:,? \d{4})?)", "user.birthday", 0.9),
    (r"(?:i(?:'m| am) available (?:at|on|from)) (.+?)(?:\s*\.|\s*,|\s+(?:and|but|so)\s|$)", "user.availability", 0.7),
    # Numbers and measurements
    (r"(?:i(?:'m| am) (\d+) years old)", "user.age", 0.85),
    (r"(?:my (?:height|weight) is) (\d+(?:\.\d+)?(?:\s*(?:cm|kg|lbs?|feet|ft|inches?|in))?)", "user.physical", 0.8),
    # Locations (more detailed)
    (r"(?:my (?:address|office|home) is) (.+?)(?:\s*\.|\s*,|\s+(?:and|but|so)\s|$)", "user.address", 0.85),
    (r"(?:i work (?:at|in|from)) (.+?)(?:\s*\.|\s*,|\s+(?:and|but|so)\s|$)", "user.work_location", 0.75),
    # Education
    (r"(?:i (?:studied|graduated|major) (?:at|in|from)) (.+?)(?:\s*\.|\s*,|\s+(?:and|but|so)\s|$)", "user.education", 0.8),
    # Interests/hobbies
    (r"(?:my (?:hobbies|interests) (?:include|are)) (.+?)(?:\s*\.|\s*,|\s+(?:and|but|so)\s|$)", "user.hobbies", 0.7),
    # Family/relationships
    (r"(?:my (?:wife|husband|partner|spouse|kid|child|son|daughter) (?:is|are)) (.+?)(?:\s*\.|\s*,|\s+(?:and|but|so)\s|$)", "user.family", 0.75),
    # Pets
    (r"(?:my (?:pet|cat|dog) (?:is|are) (?:named|called)) (\w+)", "user.pets", 0.8),
    # Health/medical
    (r"(?:i have|i(?:'m| am) (?:allergic to|diagnosed with)) (.+?)(?:\s*\.|\s*,|\s+(?:and|but|so)\s|$)", "user.health", 0.7),
    # Finance
    (r"(?:my (?:salary|income|budget) is) (\$?\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:per\s+)?(?:year|month|week|hour|annually))?)", "user.financial", 0.75),
    # Travel
    (r"(?:i(?:'ve| have) (?:been to|visited|traveled to)) (.+?)(?:\s*\.|\s*,|\s+(?:and|but|so)\s|$)", "user.travel", 0.7),
    # Food preferences
    (r"(?:my favorite (?:food|cuisine|restaurant|meal) is) (.+?)(?:\s*\.|\s*,|\s+(?:and|but|so)\s|$)", "user.food", 0.7),
    # Entertainment
    (r"(?:my favorite (?:movie|show|book|music|artist|band|song) is) (.+?)(?:\s*\.|\s*,|\s+(?:and|but|so)\s|$)", "user.entertainment", 0.7),
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

    # Stem expansion — strip common suffixes for broader matching
    # Handles cases like "professional" ↔ "profession", "trading" ↔ "trade"
    stems = set()
    suffixes = ['ial', 'ion', 'ing', 'ed', 'ly', 'ment', 'ness', 'able', 'ive', 'al', 'ic', 'ty']
    for token in list(expanded):
        for suffix in suffixes:
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                stem = token[:-len(suffix)]
                if len(stem) >= 3:
                    stems.add(stem)
    expanded.update(stems)

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


def _keyword_relevance(fact_tokens: set[str], query_tokens: set[str], 
                          path_tokens: set[str] = None) -> float:
    """Weighted overlap between fact and query tokens.
    
    Uses containment-based scoring instead of pure Jaccard to avoid
    penalizing long fact values. Path tokens get 3x weight boost
    since they carry semantic meaning about what the fact IS.
    """
    if not fact_tokens or not query_tokens:
        return 0.0
    
    # Basic overlap
    intersection = fact_tokens & query_tokens
    
    # Special case: perfect match (all tokens match)
    if intersection == query_tokens and intersection == fact_tokens:
        return 1.0
    
    # Path token boost: matching path tokens are more meaningful
    # e.g., matching 'user' in 'user.profession' when query says 'who'
    path_boost = 0.0
    if path_tokens:
        path_matches = path_tokens & query_tokens
        path_boost = len(path_matches) * 0.15  # 15% per path token match
    
    # Containment: what fraction of query tokens appear in fact?
    # Better than Jaccard for long facts
    query_coverage = len(intersection) / len(query_tokens) if query_tokens else 0.0
    
    # Fact precision: what fraction of fact tokens match query?
    # Use Jaccard as a precision component
    union = fact_tokens | query_tokens
    precision = len(intersection) / len(union) if union else 0.0
    
    # Combined: favor coverage (did we find what user asked about?)
    # with precision as tiebreaker and path boost as bonus
    score = 0.5 * query_coverage + 0.3 * precision + path_boost
    return min(score, 1.0)


# ── Negation Patterns ─────────────────────────────────────────────────

NEGATION_PATTERNS = [
    r'\bnot\b',
    r"\bdon'?t\b",
    r"\bdoesn'?t\b",
    r"\bdidn'?t\b",
    r"\bwon'?t\b",
    r"\bcan'?t\b",
    r"\bcannot\b",
    r"\bnever\b",
    r"\bavoid\b",
    r"\bno\b",
    r"\bnone\b",
    r"\bnothing\b",
    r"\bneither\b",
    r"\bnor\b",
    r'\bwithout\b',
    r'\bwarning\b',
    r'\bmistake[s]?\b',  # Handle plural
    r'\berror[s]?\b',
    r'\bproblem[s]?\b',
    r'\bissue[s]?\b',
    r'\bfail[s]?\b',
    r'\bwrong\b',
    r'\bbad\b',
    r'\bdanger\b',
    r'\brisk[s]?\b',
    r"\bshouldn'?t\b",  # Add shouldn't
    r"\bwouldn'?t\b",  # Add wouldn't
    r"\bcouldn'?t\b",  # Add couldn't
]

def _detect_negation(query: str) -> dict:
    """Detect negation intent in a query.
    
    Returns:
        Dict with negation info: is_negated, negation_type, negation_keyword
    """
    if not query:
        return {'is_negated': False, 'negation_type': None, 'negation_keyword': None}
    
    query_lower = query.lower()
    
    # Check for negation patterns
    import re
    for pattern in NEGATION_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            keyword = match.group(0)
            
            # Determine negation type (normalize keyword to handle plurals)
            keyword_base = keyword.rstrip('s')  # Remove plural 's'
            
            if keyword_base in ['not', "don't", "doesn't", "didn't", "won't", "can't", 'cannot', 'never', 'without',
                               "shouldn't", "wouldn't", "couldn't"]:
                negation_type = 'exclusion'
            elif keyword_base in ['avoid', 'warning', 'mistake', 'error', 'problem', 'issue', 'fail', 'wrong', 'bad', 'danger', 'risk']:
                negation_type = 'warning'
            elif keyword_base in ['no', 'none', 'nothing', 'neither', 'nor']:
                negation_type = 'absence'
            else:
                negation_type = 'general'
            
            return {
                'is_negated': True,
                'negation_type': negation_type,
                'negation_keyword': keyword,
            }
    
    return {'is_negated': False, 'negation_type': None, 'negation_keyword': None}

# ── Temporal Patterns ─────────────────────────────────────────────────

TEMPORAL_PATTERNS = {
    # Recent patterns (last X time units)
    'recent': [
        r'recent(?:ly)?',
        r'lately',
        r'just now',
        r'new(?:ly)?',
        r'fresh',
        r'latest',
        r'last (?:few |several )?(?:days?|weeks?|months?|hours?|minutes?)',
    ],
    # Past patterns (X ago, in the past)
    'past': [
        r'ago',
        r'past',
        r'before',
        r'earlier',
        r'previous',
        r'last (?:week|month|year|time)',
        r'(?:a |several |few )?(?:days?|weeks?|months?|years?) ago',
    ],
    # Old patterns (old, ancient, ancient)
    'old': [
        r'old(?:er)?',
        r'ancient',
        r'outdated',
        r'stale',
        r'forgotten',
        r'long.?ago',
        r'long.?time',
    ],
    # Future patterns (upcoming, soon, next)
    'future': [
        r'upcoming',
        r'soon',
        r'next',
        r'future',
        r'planned',
        r'scheduled',
    ],
}

# Time unit conversions (to seconds)
TIME_UNITS = {
    'second': 1,
    'minute': 60,
    'hour': 3600,
    'day': 86400,
    'week': 604800,
    'month': 2592000,  # 30 days
    'year': 31536000,  # 365 days
}

def _detect_temporal_intent(query: str) -> dict:
    """Detect temporal intent in a query.
    
    Returns:
        Dict with temporal intent: recent, past, old, future, or None
        Also extracts time ranges if specified (e.g., "last 7 days")
    """
    if not query:
        return {'intent': None, 'range_seconds': None}
    
    query_lower = query.lower()
    
    # Check for explicit time ranges first (e.g., "last 7 days", "past week")
    import re
    
    # Pattern: "last/past X days/weeks/months"
    range_patterns = [
        (r'(?:last|past|previous)\s+(\d+)\s+(second|minute|hour|day|week|month|year)s?', 'past'),
        (r'(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago', 'past'),
        (r'(?:last|past|previous)\s+(second|minute|hour|day|week|month|year)', 'past'),
        (r'in the (?:last|past|previous)\s+(\d+)\s+(second|minute|hour|day|week|month|year)s?', 'past'),
        (r'in the (?:last|past|previous)\s+(second|minute|hour|day|week|month|year)', 'past'),
    ]
    
    for pattern, intent in range_patterns:
        match = re.search(pattern, query_lower)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                try:
                    count = int(groups[0])
                    unit = groups[1].rstrip('s')  # Remove plural
                    if unit in TIME_UNITS:
                        range_seconds = count * TIME_UNITS[unit]
                        return {'intent': intent, 'range_seconds': range_seconds}
                except (ValueError, KeyError):
                    pass
            elif len(groups) == 1:
                # Single group (e.g., "last week")
                unit = groups[0].rstrip('s')
                if unit in TIME_UNITS:
                    range_seconds = TIME_UNITS[unit]
                    return {'intent': intent, 'range_seconds': range_seconds}
    
    # Check for temporal keywords (but skip if we already matched a range pattern)
    for intent, patterns in TEMPORAL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                # Default ranges for common patterns
                if intent == 'recent':
                    return {'intent': 'recent', 'range_seconds': 604800}  # Last 7 days
                elif intent == 'past':
                    return {'intent': 'past', 'range_seconds': 2592000}  # Last 30 days
                elif intent == 'old':
                    return {'intent': 'old', 'range_seconds': 7776000}  # Older than 90 days
                elif intent == 'future':
                    return {'intent': 'future', 'range_seconds': 2592000}  # Next 30 days
    
    return {'intent': None, 'range_seconds': None}

def _temporal_score(meta, temporal_intent: dict, now: float) -> float:
    """Calculate temporal relevance score based on intent.
    
    Args:
        meta: PathMeta object with timestamps
        temporal_intent: Dict from _detect_temporal_intent()
        now: Current timestamp
        
    Returns:
        Temporal score (0.0-1.0)
    """
    if not temporal_intent or temporal_intent['intent'] is None:
        return 0.5  # Neutral score when no temporal intent
    
    intent = temporal_intent['intent']
    range_seconds = temporal_intent['range_seconds']
    
    age_seconds = now - meta.created_at
    recency_seconds = now - meta.last_accessed
    
    if intent == 'recent':
        # Recent intent: prefer recently created or accessed facts
        if range_seconds:
            # Within specified range
            if age_seconds <= range_seconds or recency_seconds <= range_seconds:
                return 1.0
            else:
                return 0.0
        else:
            # General recent: exponential decay
            return math.exp(-0.693 * age_seconds / 604800)  # 7-day half-life
    
    elif intent == 'past':
        # Past intent: prefer facts from the specified time range
        if range_seconds:
            if age_seconds <= range_seconds:
                return 1.0
            else:
                return 0.0
        else:
            # General past: prefer older facts (inverse of recent)
            return 1.0 - math.exp(-0.693 * age_seconds / 2592000)  # 30-day half-life
    
    elif intent == 'old':
        # Old intent: prefer older facts
        if range_seconds:
            if age_seconds >= range_seconds:
                return 1.0
            else:
                return 0.0
        else:
            # General old: older = higher score
            return min(age_seconds / 7776000, 1.0)  # Cap at 90 days
    
    elif intent == 'future':
        # Future intent: neutral (can't predict future)
        return 0.5
    
    return 0.5

def _negation_score(meta, negation_info: dict, query_tokens: set[str] = None) -> float:
    """Calculate negation relevance score.
    
    For negated queries (e.g., "What should I NOT do?"), we want to:
    1. Boost facts tagged as warnings/mistakes
    2. Find facts about things to avoid
    
    Args:
        meta: PathMeta object with metadata
        negation_info: Dict from _detect_negation()
        query_tokens: Token set from query
        
    Returns:
        Negation score (0.0-1.0)
    """
    if not negation_info or not negation_info['is_negated']:
        return 0.5  # Neutral
    
    negation_type = negation_info['negation_type']
    
    # Check if fact is tagged as warning/mistake
    tags = meta.tags or []
    has_warning_tag = any(tag in ['warning', 'mistake', 'error', 'problem', 'issue', 'avoid'] 
                         for tag in tags)
    
    # Check if path suggests warning/mistake
    path_lower = meta.path.lower() if hasattr(meta, 'path') else ''
    has_warning_path = any(word in path_lower for word in 
                          ['warning', 'mistake', 'error', 'problem', 'issue', 'lesson', 'avoid'])
    
    # Check if tokens suggest warning/mistake
    warning_tokens = {'warning', 'mistake', 'error', 'problem', 'issue', 'avoid', 'fail', 'wrong', 'bad', 'danger', 'risk'}
    has_warning_tokens = bool(meta.tokens & warning_tokens) if meta.tokens else False
    
    is_warning = has_warning_tag or has_warning_path or has_warning_tokens
    
    if negation_type == 'warning':
        # Query is asking for warnings/mistakes
        if is_warning:
            return 1.0  # Perfect match - boost warning facts
        else:
            return 0.3  # Not a warning, but still include with low score
    
    elif negation_type == 'exclusion':
        # Query is asking to exclude something (e.g., "don't use X")
        # Boost facts that mention the excluded thing
        if query_tokens and meta.tokens:
            if query_tokens & meta.tokens:
                return 0.9  # Mentions excluded thing - high relevance
        return 0.5  # Doesn't mention excluded thing - neutral
    
    elif negation_type == 'absence':
        # Query is asking about absence (e.g., "what's missing?")
        # Neutral - can't determine absence from facts
        return 0.5
    
    else:
        # General negation
        if is_warning:
            return 0.8  # Likely relevant
        else:
            return 0.3  # Less likely, but still include

# ── Procedural Memory ─────────────────────────────────────────────────

class Skill:
    """Represents a transferable skill or principle extracted from experience."""
    __slots__ = ['name', 'principle', 'domains', 'strength', 'last_used', 
                 'created_at', 'examples', 'transfer_count']
    
    def __init__(self, name: str, principle: str, domains: list[str] = None):
        now = time.time()
        self.name = name  # e.g., "balance", "momentum_stability"
        self.principle = principle  # Abstract principle: "Forward momentum stabilizes lateral movement"
        self.domains = domains or []  # Domains where skill applies: ["cycling", "motorcycling", "skiing"]
        self.strength = 1.0  # Skill strength (0.0-1.0)
        self.last_used = now  # When skill was last applied
        self.created_at = now  # When skill was first extracted
        self.examples = []  # List of specific experiences that contributed
        self.transfer_count = 0  # How many times skill transferred to new domain


class ProceduralMemory:
    """Manages skills and principles extracted from experiences."""
    
    def __init__(self, path: str = None):
        self.path = Path(path) if path else None
        self.skills: dict[str, Skill] = {}  # name → Skill
        self.domain_index: dict[str, set[str]] = {}  # domain → set of skill names
        self._lock = threading.RLock()
        
        if self.path and self.path.exists():
            self._load()
    
    def add_skill(self, name: str, principle: str, domains: list[str] = None,
                  examples: list[str] = None) -> Skill:
        """Add a new skill or strengthen existing one."""
        with self._lock:
            if name in self.skills:
                # Strengthen existing skill
                skill = self.skills[name]
                skill.strength = min(skill.strength + 0.1, 1.0)
                if examples:
                    skill.examples.extend(examples[:3])  # Keep last 3 examples
                if domains:
                    for domain in domains:
                        if domain not in skill.domains:
                            skill.domains.append(domain)
            else:
                # Create new skill
                skill = Skill(name, principle, domains or [])
                if examples:
                    skill.examples = examples[:3]
                self.skills[name] = skill
            
            # Update domain index
            for domain in skill.domains:
                if domain not in self.domain_index:
                    self.domain_index[domain] = set()
                self.domain_index[domain].add(name)
            
            return skill
    
    def get_skills_for_domain(self, domain: str) -> list[Skill]:
        """Get all skills applicable to a domain."""
        with self._lock:
            skill_names = self.domain_index.get(domain, set())
            return [self.skills[name] for name in skill_names if name in self.skills]
    
    def find_transferable_skills(self, new_domain: str, 
                                 context_keywords: set[str] = None) -> list[Skill]:
        """Find skills that might transfer to a new domain."""
        with self._lock:
            transferable = []
            
            for skill in self.skills.values():
                # Direct domain match
                if new_domain in skill.domains:
                    transferable.append(skill)
                    continue
                
                # Keyword overlap with principle
                if context_keywords:
                    principle_words = set(skill.principle.lower().split())
                    if context_keywords & principle_words:
                        transferable.append(skill)
                        continue
                
                # Similar domains (fuzzy matching)
                for domain in skill.domains:
                    if self._domains_similar(domain, new_domain):
                        transferable.append(skill)
                        break
            
            # Sort by strength and transfer count
            transferable.sort(key=lambda s: (s.strength, s.transfer_count), reverse=True)
            return transferable
    
    def apply_skill(self, skill_name: str, new_domain: str = None) -> bool:
        """Record that a skill was applied (strengthens it)."""
        with self._lock:
            if skill_name not in self.skills:
                return False
            
            skill = self.skills[skill_name]
            skill.strength = min(skill.strength + 0.05, 1.0)
            skill.last_used = time.time()
            skill.transfer_count += 1
            
            if new_domain and new_domain not in skill.domains:
                skill.domains.append(new_domain)
                if new_domain not in self.domain_index:
                    self.domain_index[new_domain] = set()
                self.domain_index[new_domain].add(skill_name)
            
            return True
    
    def extract_principles(self, experience: str, domain: str = None) -> list[dict]:
        """Extract principles from an experience (simplified pattern matching)."""
        # This is a simplified version - in production, use LLM or NLP
        principles = []
        
        # Pattern: "learned that X causes Y"
        patterns = [
            (r'learned that (.+?) causes? (.+)', 'causal'),
            (r'discovered that (.+?) leads? to (.+)', 'causal'),
            (r'found that (.+?) requires? (.+)', 'requirement'),
            (r'realized that (.+?) needs? (.+)', 'requirement'),
            (r'noticed that (.+?) improves? (.+)', 'improvement'),
            (r'understood that (.+?) stabilizes? (.+)', 'stabilization'),
        ]
        
        import re
        for pattern, principle_type in patterns:
            matches = re.findall(pattern, experience.lower())
            for match in matches:
                if isinstance(match, tuple) and len(match) == 2:
                    cause, effect = match
                    principle = f"{cause.strip()} → {effect.strip()}"
                    principles.append({
                        'principle': principle,
                        'type': principle_type,
                        'domains': [domain] if domain else []
                    })
        
        return principles
    
    def _domains_similar(self, domain1: str, domain2: str) -> bool:
        """Check if two domains are similar."""
        # Normalize to lowercase
        d1 = domain1.lower()
        d2 = domain2.lower()
        
        # Direct substring match
        if d1 in d2 or d2 in d1:
            return True
        
        # Shared words (split by underscore or space)
        import re
        words1 = set(re.split(r'[_\s]+', d1))
        words2 = set(re.split(r'[_\s]+', d2))
        
        if words1 & words2:
            return True
        
        # Check for common prefixes (at least 3 chars)
        if len(d1) >= 3 and len(d2) >= 3:
            if d1[:3] == d2[:3]:
                return True
        
        # Check for common suffixes
        if len(d1) >= 3 and len(d2) >= 3:
            if d1[-3:] == d2[-3:]:
                return True
        
        return False
    
    def competence_map(self) -> dict:
        """Get overview of all skills and their domains."""
        with self._lock:
            return {
                'total_skills': len(self.skills),
                'domains': list(self.domain_index.keys()),
                'strongest': sorted(
                    [(s.name, s.strength, s.domains) for s in self.skills.values()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
                'most_transferred': sorted(
                    [(s.name, s.transfer_count, s.domains) for s in self.skills.values()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
            }
    
    def _save(self):
        """Persist skills to disk."""
        if not self.path:
            return
        
        try:
            data = {}
            for name, skill in self.skills.items():
                data[name] = {
                    'principle': skill.principle,
                    'domains': skill.domains,
                    'strength': skill.strength,
                    'last_used': skill.last_used,
                    'created_at': skill.created_at,
                    'examples': skill.examples,
                    'transfer_count': skill.transfer_count,
                }
            
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        except Exception:
            pass
    
    def _load(self):
        """Load skills from disk."""
        if not self.path or not self.path.exists():
            return
        
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            for name, skill_data in data.items():
                skill = Skill(
                    name=name,
                    principle=skill_data['principle'],
                    domains=skill_data.get('domains', [])
                )
                skill.strength = skill_data.get('strength', 1.0)
                skill.last_used = skill_data.get('last_used', 0)
                skill.created_at = skill_data.get('created_at', 0)
                skill.examples = skill_data.get('examples', [])
                skill.transfer_count = skill_data.get('transfer_count', 0)
                self.skills[name] = skill
                
                # Rebuild domain index
                for domain in skill.domains:
                    if domain not in self.domain_index:
                        self.domain_index[domain] = set()
                    self.domain_index[domain].add(name)
        except Exception:
            pass


# ── Path Metadata ─────────────────────────────────────────────────────

class PathMeta:
    """Tracks per-path metadata for scoring and lifecycle management."""
    __slots__ = ['last_accessed', 'access_count', 'created_at', 'tier', 'tokens', 
                 'ttl', 'expires_at', 'archived', 'size_bytes', 'protected', 'tags',
                 'confidence', 'overwrite_count']

    def __init__(self, ttl: int = None, protected: bool = False, tags: list[str] = None,
                 confidence: float = 1.0):
        now = time.time()
        self.last_accessed = now
        self.access_count = 1
        self.created_at = now
        self.tier = 'hot'
        self.tokens: set[str] = set()
        self.ttl = ttl  # Time-to-live in seconds (None = never expires)
        self.expires_at = (now + ttl) if ttl else None
        self.archived = False
        self.size_bytes = 0
        self.protected = protected  # Protected facts are immune to pruning
        self.tags = tags or []
        self.confidence = confidence  # 0.0-1.0: how confident is this fact?
        self.overwrite_count = 0  # How many times this path was overwritten


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
                 tiered: bool = False, recency_half_life: float = 3600,
                 procedural: bool = False, eviction_policy: str = "lru-archive"):
        self.path = Path(path)
        self.max_chars = max_chars
        self.max_results = max_results
        self.extract_confidence = extract_confidence
        self.recency_half_life = recency_half_life

        # Core storage
        self.mem = Memory(max_chars=max_chars, auto_flush_path=str(self.path),
                          eviction_policy=eviction_policy)
        self.brain = Synapse()

        # Tiered storage (optional)
        self.tiered = TieredMemory(path, max_hot_chars=max_chars) if tiered else None

        # Procedural memory (optional)
        self.procedural = ProceduralMemory(
            path=str(self.path.with_suffix('.skills.json'))
        ) if procedural else None
        
        # Forgetting curve for memory decay modeling
        self.forgetting_curve = ForgettingCurve()
        
        # Versioning for tracking memory changes
        self.versioning = MemoryVersioning()
        
        # Event callbacks for memory changes
        self._event_callbacks: Dict[str, List[callable]] = {
            'on_set': [],
            'on_delete': [],
            'on_update': [],
            'on_change': [],  # Any change
        }
        
        # Encryption for sensitive data
        self.encryption: Optional[MemoryEncryption] = None
        
        # Advanced search capabilities
        self.search_engine = AdvancedSearch(self)

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

    def remember(self, path: str, value, ttl: int = None, tags: list[str] = None, 
                 protected: bool = False, check_contradictions: bool = True,
                 confidence: float = 1.0) -> dict:
        """Store a fact. Use dotted paths: 'user.name', 'project.status'

        Args:
            path: Dotted path key.
            value: Any JSON-serializable value.
            ttl: Optional time-to-live in seconds.
            tags: Optional list of tags for categorization.
            protected: If True, fact is immune to pruning (for critical identity facts).
            check_contradictions: If True, check for contradictions before storing.
            confidence: Confidence in this fact (0.0-1.0). Default 1.0 for explicit facts.
                       Use lower values for auto-extracted or uncertain facts.
                       Affects recall scoring — lower confidence = lower relevance.

        Returns:
            Dict with 'success' (bool), 'contradictions' (list), 'warnings' (list),
            'overwritten' (bool), 'old_value' (any), 'is_new' (bool)
        """
        with self._lock:
            result = {
                'success': True,
                'contradictions': [],
                'warnings': [],
                'overwritten': False,
                'old_value': None,
                'is_new': True,
            }
            
            # Check for contradictions if requested
            if check_contradictions:
                existing_facts = {}
                for existing_path in self.mem.paths():
                    existing_value = self.mem.get(existing_path)
                    if existing_value is not None:
                        existing_facts[existing_path] = existing_value
                
                contradictions = detect_contradictions(path, value, existing_facts)
                if contradictions:
                    result['contradictions'] = contradictions
                    result['warnings'].append(f"Found {len(contradictions)} contradiction(s)")
                    
                    # Log contradictions but don't block storage
                    for c in contradictions:
                        print(f"⚠️  Contradiction detected: {c.explanation}", flush=True)
            
            # Store the fact
            old_value = self.mem.get(path)  # Get old value for versioning
            is_new = old_value is None
            
            # Track overwrite
            result['is_new'] = is_new
            if not is_new:
                result['old_value'] = old_value
                if old_value != value:
                    result['overwritten'] = True
                    result['warnings'].append(
                        f"Overwrote '{path}': {old_value!r} → {value!r}"
                    )
                    # Increment overwrite count in meta
                    if path in self._meta:
                        self._meta[path].overwrite_count += 1
            
            self.mem.set(path, value, ttl=ttl)
            self._init_meta(path, value, ttl=ttl, protected=protected, tags=tags,
                            confidence=confidence)

            if self.tiered:
                self.tiered.set(path, value, tier='hot', ttl=ttl)

            # Auto-link to tags via Synapse
            if tags:
                for tag in tags:
                    self.brain.link(tag, [path])

            self._save_meta()
            
            # Record version
            operation = 'set' if is_new else 'update'
            self.versioning.record_change(
                path=path,
                old_value=old_value,
                new_value=value,
                operation=operation,
                metadata={'tags': tags, 'protected': protected, 'ttl': ttl}
            )
            
            # Trigger events
            self._trigger_event('on_set' if is_new else 'on_update', path, old_value, value)
            self._trigger_event('on_change', path, old_value, value)
            
            return result

    def get_contradictions(self) -> list[Contradiction]:
        """Get all contradictions in memory.
        
        Returns:
            List of Contradiction objects found in memory.
        """
        with self._lock:
            all_facts = {}
            for path in self.mem.paths():
                value = self.mem.get(path)
                if value is not None:
                    all_facts[path] = value
            
            contradictions = []
            detector = ContradictionDetector()
            
            # Check each fact against all others
            paths = list(all_facts.keys())
            for i, path1 in enumerate(paths):
                for path2 in paths[i+1:]:
                    value1 = all_facts[path1]
                    value2 = all_facts[path2]
                    
                    # Check both directions
                    contradictions.extend(detector.detect(path1, value1, {path2: value2}))
                    contradictions.extend(detector.detect(path2, value2, {path1: value1}))
            
            # Remove duplicates
            seen = set()
            unique_contradictions = []
            for c in contradictions:
                key = (c.existing_path, c.new_path, c.contradiction_type)
                if key not in seen:
                    seen.add(key)
                    unique_contradictions.append(c)
            
            return unique_contradictions

    def consolidate_memory(self, max_groups: int = 10) -> List[ConsolidationGroup]:
        """Find groups of related facts that can be consolidated.
        
        Args:
            max_groups: Maximum number of groups to return
            
        Returns:
            List of ConsolidationGroup objects with consolidation suggestions
        """
        with self._lock:
            all_facts = {}
            for path in self.mem.paths():
                value = self.mem.get(path)
                if value is not None:
                    all_facts[path] = value
            
            return consolidate_memory(all_facts, max_groups)
    
    def auto_consolidate(self, min_confidence: float = 0.7) -> dict:
        """Automatically consolidate high-confidence groups.
        
        Args:
            min_confidence: Minimum confidence threshold for auto-consolidation
            
        Returns:
            Dict with 'consolidated' (list), 'skipped' (list), 'warnings' (list)
        """
        with self._lock:
            result = {
                'consolidated': [],
                'skipped': [],
                'warnings': []
            }
            
            # Get consolidation suggestions
            groups = self.consolidate_memory(max_groups=20)
            
            for group in groups:
                if group.confidence >= min_confidence:
                    # Auto-consolidate
                    self.remember(group.suggested_path, group.suggested_value, 
                                tags=['consolidated'])
                    
                    # Mark original facts for review (don't delete automatically)
                    for path in group.paths:
                        if path != group.suggested_path:
                            # Add a tag to indicate it's been consolidated
                            if path in self._meta:
                                if 'consolidated' not in self._meta[path].tags:
                                    self._meta[path].tags.append('consolidated')
                    
                    result['consolidated'].append({
                        'paths': group.paths,
                        'suggested_path': group.suggested_path,
                        'suggested_value': group.suggested_value,
                        'reason': group.reason
                    })
                else:
                    result['skipped'].append({
                        'paths': group.paths,
                        'confidence': group.confidence,
                        'reason': group.reason
                    })
            
            if result['consolidated']:
                result['warnings'].append(f"Auto-consolidated {len(result['consolidated'])} groups")
                self._save_meta()
            
            return result

    def get_memory_strength(self, path: str, memory_type: str = 'fact') -> Optional[MemoryStrength]:
        """Get the strength analysis for a memory.
        
        Args:
            path: Memory path
            memory_type: Type of memory ('identity', 'skill', 'fact', 'event', 'temporary')
            
        Returns:
            MemoryStrength object or None if path not found
        """
        with self._lock:
            if path not in self._meta:
                return None
            
            meta = self._meta[path]
            value = self.mem.get(path)
            
            if value is None:
                return None
            
            # Get reinforcement count from tags or default to 0
            reinforcement_count = 0
            if 'reinforced' in meta.tags:
                reinforcement_count = meta.tags.count('reinforced')
            
            return self.forgetting_curve.analyze_memory(
                path=path,
                value=value,
                initial_strength=1.0,  # Assume full strength when first stored
                last_reinforced=meta.last_accessed,
                reinforcement_count=reinforcement_count,
                memory_type=memory_type
            )
    
    def get_memories_needing_reinforcement(self, max_items: int = 10,
                                         memory_type: str = None) -> List[Dict[str, Any]]:
        """Get memories that need reinforcement based on forgetting curve.
        
        Args:
            max_items: Maximum items to return
            memory_type: Optional filter by memory type
            
        Returns:
            List of memory dicts with strength analysis
        """
        with self._lock:
            memories = []
            
            for path in self.mem.paths():
                value = self.mem.get(path)
                if value is None:
                    continue
                
                meta = self._meta.get(path)
                if not meta:
                    continue
                
                # Get reinforcement count
                reinforcement_count = 0
                if 'reinforced' in meta.tags:
                    reinforcement_count = meta.tags.count('reinforced')
                
                # Determine memory type from path or tags
                detected_type = memory_type or 'fact'
                if 'identity' in meta.tags or 'user' in path:
                    detected_type = 'identity'
                elif 'skill' in meta.tags:
                    detected_type = 'skill'
                elif 'event' in meta.tags or 'meeting' in path:
                    detected_type = 'event'
                elif 'temporary' in meta.tags or 'temp' in path:
                    detected_type = 'temporary'
                
                memories.append({
                    'path': path,
                    'value': value,
                    'initial_strength': 1.0,
                    'last_reinforced': meta.last_accessed,
                    'reinforcement_count': reinforcement_count,
                    'memory_type': detected_type,
                    'tags': meta.tags,
                    'protected': meta.protected,
                })
            
            # Filter by memory type if specified
            if memory_type:
                memories = [m for m in memories if m['memory_type'] == memory_type]
            
            return self.forgetting_curve.prioritize_for_reinforcement(
                memories, max_items=max_items
            )
    
    def simulate_memory_decay(self, path: str, days: int = 30,
                            memory_type: str = 'fact') -> List[Dict[str, Any]]:
        """Simulate how a memory will decay over time.
        
        Args:
            path: Memory path
            days: Number of days to simulate
            memory_type: Type of memory
            
        Returns:
            List of daily strength values
        """
        with self._lock:
            if path not in self._meta:
                return []
            
            meta = self._meta[path]
            value = self.mem.get(path)
            
            if value is None:
                return []
            
            # Get reinforcement count
            reinforcement_count = 0
            if 'reinforced' in meta.tags:
                reinforcement_count = meta.tags.count('reinforced')
            
            return self.forgetting_curve.simulate_decay(
                initial_strength=1.0,
                reinforcement_count=reinforcement_count,
                memory_type=memory_type,
                days=days
            )
    
    def reinforce_memory(self, path: str, boost_strength: float = 0.2) -> dict:
        """Reinforce a memory (strengthen it against forgetting).
        
        Args:
            path: Memory path to reinforce
            boost_strength: How much to boost strength (0.0-1.0)
            
        Returns:
            Dict with 'success' (bool), 'new_strength' (float), 'message' (str)
        """
        with self._lock:
            if path not in self._meta:
                return {'success': False, 'new_strength': 0, 'message': 'Path not found'}
            
            meta = self._meta[path]
            
            # Add reinforcement tag
            if 'reinforced' not in meta.tags:
                meta.tags.append('reinforced')
            
            # Update last accessed time (acts as reinforcement)
            meta.last_accessed = time.time()
            
            # Calculate new strength
            strength_analysis = self.get_memory_strength(path)
            new_strength = strength_analysis.current_strength + boost_strength if strength_analysis else 1.0
            new_strength = min(1.0, new_strength)
            
            self._save_meta()
            
            return {
                'success': True,
                'new_strength': new_strength,
                'message': f'Reinforced {path} (strength: {new_strength:.3f})'
            }

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
            old_value = self.mem.get(path)  # Get old value for versioning
            
            self.mem.delete(path, prune=True)
            self._meta.pop(path, None)
            if self.tiered:
                self.tiered.hot.delete(path, prune=True)
                self.tiered.warm.delete(path, prune=True)
                self.tiered.cold.delete(path, prune=True)
            self._save_meta()
            
            # Record version
            self.versioning.record_change(
                path=path,
                old_value=old_value,
                new_value=None,
                operation='delete'
            )
            
            # Trigger events
            self._trigger_event('on_delete', path, old_value, None)
            self._trigger_event('on_change', path, old_value, None)

    def search(self, pattern: str) -> dict:
        """Find facts matching a glob pattern."""
        return self.mem.find(pattern)

    def search_value(self, query: str, case_sensitive: bool = False,
                     field: str = "value") -> dict:
        """Search memory by value content (substring match).

        Args:
            query: Search string.
            case_sensitive: Case-sensitive matching.
            field: 'value', 'path', or 'both'.

        Returns:
            Dict of {path: value} for matching entries.
        """
        return self.mem.search_value(query, case_sensitive=case_sensitive, field=field)

    def context(self) -> str:
        """Export ALL memory as compact JSON for injection."""
        return self.mem.export()

    # ── Smart Retrieval ──────────────────────────────────────────────

    def score(self, path: str, query_tokens: set[str] = None, now: float = None,
              temporal_intent: dict = None, negation_info: dict = None) -> float:
        """Score a path's relevance. Combines recency × frequency × keyword match
           × temporal × negation × forgetting strength × confidence.

        Args:
            path: The dotted path to score.
            query_tokens: Token set from user's message (for keyword matching).
            now: Current timestamp (defaults to time.time()).
            temporal_intent: Dict from _detect_temporal_intent() for time-based scoring.
            negation_info: Dict from _detect_negation() for negation handling.

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
            # Extract path tokens for boosting
                # Split on dots and underscores to get individual words
                path_tokens = set()
                if path:
                    # Split path into components, then tokenize each component
                    for component in re.split(r'[._]', path.lower()):
                        if component:
                            tokens = set(re.findall(r'\w{2,}', component))
                            path_tokens.update(tokens)
                keyword = _keyword_relevance(meta.tokens, query_tokens, path_tokens)

        # Temporal score
        temporal = _temporal_score(meta, temporal_intent, now)
        
        # Negation score
        negation = 0.5  # default (neutral)
        if negation_info and negation_info['is_negated']:
            negation = _negation_score(meta, negation_info, query_tokens)

        # Forgetting curve strength: decays over time unless reinforced
        reinforcement_count = 0
        if meta.tags:
            reinforcement_count = meta.tags.count('reinforced')
        strength = self.forgetting_curve.calculate_strength(
            initial_strength=1.0,
            last_reinforced=meta.last_accessed,
            reinforcement_count=reinforcement_count,
            memory_type='fact',
            current_time=now
        )

        # Confidence: from meta (default 1.0 for explicit facts, <1.0 for auto-extracted)
        confidence = meta.confidence

        # Weighted combination
        if query_tokens:
            # With a query: keyword dominates, recency/frequency are tiebreakers
            if keyword > 0:
                # Adjust weights based on temporal and negation intent
                temporal_weight = 0.20 if temporal_intent and temporal_intent['intent'] else 0.0
                negation_weight = 0.20 if negation_info and negation_info['is_negated'] else 0.0
                keyword_weight = 0.85 - temporal_weight - negation_weight
                
                base_score = (0.1 * recency + 0.05 * frequency + 
                              keyword_weight * keyword + 
                              temporal_weight * temporal +
                              negation_weight * negation)
            else:
                # No keyword match at all
                if negation_info and negation_info['is_negated']:
                    base_score = 0.1 * recency + 0.05 * frequency + 0.85 * negation
                else:
                    return 0.0
        else:
            # Without a query: recency + frequency + temporal + negation
            temporal_weight = 0.5 if temporal_intent and temporal_intent['intent'] else 0.0
            negation_weight = 0.5 if negation_info and negation_info['is_negated'] else 0.0
            remaining = 1.0 - temporal_weight - negation_weight
            
            base_score = (remaining * 0.6 * recency + remaining * 0.4 * frequency + 
                          temporal_weight * temporal +
                          negation_weight * negation)

        # Apply forgetting curve and confidence as multiplicative factors
        return base_score * strength * confidence

    def prompt_context(self, query: str = None, max_results: int = None,
                      max_tokens: int = None, chars_per_token: float = 4.0,
                      format_fn=None) -> str:
        """Generate lean prompt context from relevant facts only.

        Instead of injecting the entire memory (wasting tokens), returns
        only the facts relevant to the current query.

        Args:
            query: User's current message.
            max_results: Override default max results.
            max_tokens: Token budget (overrides max_results if set).
                        Converts to chars using chars_per_token ratio.
            chars_per_token: Characters per token ratio. Default 4.0 (English).
                             Use 2.5-3.0 for CJK, 3.5 for mixed.
            format_fn: Custom formatter (path, value) -> str.

        Returns:
            Formatted string ready for prompt injection.
        """
        # Convert token budget to char budget
        if max_tokens is not None:
            char_budget = _tokens_to_chars(max_tokens, chars_per_token)
            # Get more results than needed, then trim by budget
            relevant = self.recall_relevant(query, max_results=max_results or 20)
        else:
            relevant = self.recall_relevant(query, max_results=max_results)
        
        if not relevant:
            return ""

        lines = []
        formatter = format_fn or (lambda p, v: f"- {p}: {v}" if not isinstance(v, (list, dict)) else f"- {p}: {json.dumps(v, ensure_ascii=False)}")
        
        total_chars = len("## Memory\n")
        for path, value in relevant.items():
            line = formatter(path, value)
            if max_tokens is not None:
                if total_chars + len(line) + 1 > char_budget:
                    break
                total_chars += len(line) + 1
            lines.append(line)

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
                        self.remember(path, value, tags=['auto_extracted'],
                                      confidence=confidence)

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
        
        # Detect temporal intent in query
        temporal_intent = _detect_temporal_intent(query) if query else {'intent': None, 'range_seconds': None}
        
        # Detect negation in query
        negation_info = _detect_negation(query) if query else {'is_negated': False, 'negation_type': None}

        scored = []
        for path in self.mem.paths():
            s = self.score(path, query_tokens, now, temporal_intent, negation_info)
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
                adaptive_threshold = max(min_score, max_keyword * 0.4)
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
                      max_tokens: int = None, chars_per_token: float = 4.0,
                      include_episodes: bool = True) -> str:
        """Build complete agent context: relevant facts + recent episodes.

        This is the "all-in-one" method. Call this instead of prompt_context()
        when you want maximum recall with minimum tokens.

        Args:
            query: User's current message.
            max_chars: Character budget for the entire context block.
                       Ignored if max_tokens is set.
            max_tokens: Token budget (overrides max_chars).
                        Converts to chars using chars_per_token ratio.
            chars_per_token: Characters per token ratio. Default 4.0 (English).
                             Use 2.5-3.0 for CJK, 3.5 for mixed.
            include_episodes: Include recent conversation episodes.

        Returns:
            Formatted context string ready for prompt injection.
        """
        # Convert token budget to char budget
        if max_tokens is not None:
            budget = _tokens_to_chars(max_tokens, chars_per_token)
        else:
            budget = max_chars
        
        parts = []
        budget_remaining = budget

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

    def save_snapshot(self, name: str, description: str = "") -> bool:
        """Save current state as a persistent snapshot to disk.

        Unlike snapshot() which is in-memory only, this survives process restarts.

        Args:
            name: Snapshot name.
            description: Optional human-readable description.

        Returns:
            True if saved.
        """
        return self.mem.save_snapshot(name, description)

    def load_snapshot(self, name: str) -> bool:
        """Load a persistent snapshot from disk and restore it.

        Args:
            name: Snapshot name to restore.

        Returns:
            True if loaded and restored.
        """
        result = self.mem.load_snapshot(name)
        if result:
            self._save_meta()
        return result

    def list_snapshots(self) -> list[dict]:
        """List all persistent snapshots on disk."""
        return self.mem.list_snapshots()

    def delete_snapshot(self, name: str) -> bool:
        """Delete a persistent snapshot from disk."""
        return self.mem.delete_snapshot(name)

    def diff_snapshots(self, name_a: str, name_b: str) -> dict:
        """Compare two snapshots and return the differences."""
        return self.mem.diff_snapshots(name_a, name_b)

    def move(self, src_prefix: str, dst_prefix: str, overwrite: bool = False) -> dict:
        """Move/rename all paths under a prefix to a new prefix.

        Args:
            src_prefix: Source prefix (e.g., "trading").
            dst_prefix: Destination prefix (e.g., "crypto.trading").
            overwrite: Overwrite existing destination paths.

        Returns:
            Dict with 'moved', 'count', 'skipped'.
        """
        result = self.mem.move(src_prefix, dst_prefix, overwrite=overwrite)
        # Transfer metadata for moved paths
        for m in result.get("moved", []):
            from_path = m["from"]
            to_path = m["to"]
            if from_path in self._meta:
                self._meta[to_path] = self._meta.pop(from_path)
        self._save_meta()
        return result

    def merge_from_file(self, path: str, prefix: str = "",
                        conflict: str = "overwrite") -> dict:
        """Merge data from a JSON file into memory.

        Args:
            path: Path to JSON file.
            prefix: Optional prefix to merge under.
            conflict: 'overwrite', 'skip'.

        Returns:
            Dict with 'imported', 'skipped', 'total'.
        """
        return self.mem.merge_from_file(path, prefix=prefix, conflict=conflict)

    def merge_from(self, other, conflict_strategy: str = "keep_newer") -> dict:
        """Merge another SmartMemory instance into this one.
        
        Transfers facts, metadata, and handles path conflicts.
        Use this for combining agent memories across sessions or agents.
        
        Args:
            other: Another SmartMemory instance (or dict of {path: value}).
            conflict_strategy: What to do when both memories have the same path:
                - "keep_newer": Keep the one with more recent last_accessed (default)
                - "keep_other": Always use the other memory's version
                - "keep_self": Skip conflicting paths, keep existing
                - "keep_higher_confidence": Keep the one with higher confidence
                - "merge_both": Store both as path and path._merged (tagged)
                
        Returns:
            Dict with 'merged' (int), 'skipped' (int), 'conflicts' (list of paths),
            'errors' (list).
        """
        result = {'merged': 0, 'skipped': 0, 'conflicts': [], 'errors': []}
        
        # Handle raw dict input
        if isinstance(other, dict):
            for path, value in other.items():
                try:
                    existing = self.mem.get(path)
                    if existing is None:
                        self.remember(path, value, tags=['merged'])
                        result['merged'] += 1
                    else:
                        result['conflicts'].append(path)
                        if conflict_strategy == "keep_other":
                            self.remember(path, value, tags=['merged'])
                            result['merged'] += 1
                        elif conflict_strategy == "keep_self":
                            result['skipped'] += 1
                        else:
                            # keep_newer / keep_higher_confidence: same as keep_other for raw dict
                            self.remember(path, value, tags=['merged'])
                            result['merged'] += 1
                except Exception as e:
                    result['errors'].append(f"{path}: {e}")
            return result
        
        # Handle SmartMemory input
        if not hasattr(other, 'mem') or not hasattr(other, '_meta'):
            result['errors'].append("Input must be SmartMemory or dict")
            return result
        
        with self._lock:
            other_paths = other.mem.paths()
            
            for path in other_paths:
                try:
                    other_value = other.mem.get(path)
                    if other_value is None:
                        continue
                    
                    other_meta = other._meta.get(path)
                    other_confidence = other_meta.confidence if other_meta else 1.0
                    other_tags = (other_meta.tags if other_meta else []) + ['merged']
                    other_protected = other_meta.protected if other_meta else False
                    
                    existing_value = self.mem.get(path)
                    
                    if existing_value is None:
                        # No conflict — direct merge
                        self.remember(path, other_value, 
                                      tags=other_tags,
                                      protected=other_protected,
                                      confidence=other_confidence,
                                      check_contradictions=False)
                        result['merged'] += 1
                    else:
                        # Conflict
                        result['conflicts'].append(path)
                        self_meta = self._meta.get(path)
                        
                        if conflict_strategy == "keep_other":
                            self.remember(path, other_value, tags=other_tags,
                                          protected=other_protected,
                                          confidence=other_confidence,
                                          check_contradictions=False)
                            result['merged'] += 1
                        elif conflict_strategy == "keep_self":
                            result['skipped'] += 1
                        elif conflict_strategy == "keep_newer":
                            other_time = other_meta.last_accessed if other_meta else 0
                            self_time = self_meta.last_accessed if self_meta else 0
                            if other_time > self_time:
                                self.remember(path, other_value, tags=other_tags,
                                              confidence=other_confidence,
                                              check_contradictions=False)
                                result['merged'] += 1
                            else:
                                result['skipped'] += 1
                        elif conflict_strategy == "keep_higher_confidence":
                            if other_confidence > (self_meta.confidence if self_meta else 1.0):
                                self.remember(path, other_value, tags=other_tags,
                                              confidence=other_confidence,
                                              check_contradictions=False)
                                result['merged'] += 1
                            else:
                                result['skipped'] += 1
                        elif conflict_strategy == "merge_both":
                            # Store other's version under a _merged suffix
                            merged_path = f"{path}._merged"
                            self.remember(merged_path, other_value, tags=other_tags,
                                          confidence=other_confidence,
                                          check_contradictions=False)
                            result['merged'] += 1
                        else:
                            result['skipped'] += 1
                except Exception as e:
                    result['errors'].append(f"{path}: {e}")
            
            self._save_meta()
        
        return result

    # ── Stats & Debug ────────────────────────────────────────────────

    def knowledge_summary(self, topic: str = None, group_by: str = "prefix") -> dict:
        """Get a structured overview of what's stored in memory.
        
        Solves: "what do I know about X?" — returns facts grouped by topic/prefix.
        
        Args:
            topic: Filter by topic keyword (matches against path and value).
                   If None, returns overview of all knowledge.
            group_by: How to group facts:
                - "prefix": Group by first path segment (e.g., "user", "project")
                - "tag": Group by tags
                - "confidence": Group by confidence level (high/medium/low)
                
        Returns:
            Dict with:
            - 'total_facts': Number of facts
            - 'groups': Dict of {group_name: [facts]}
            - 'topics': List of detected topic names
            - 'summary': Human-readable summary string
        """
        with self._lock:
            all_paths = self.mem.paths()
            
            # Filter by topic if specified
            if topic:
                topic_lower = topic.lower()
                filtered = []
                for path in all_paths:
                    value = self.mem.get(path)
                    # Match against path segments and value content
                    path_match = topic_lower in path.lower()
                    value_match = False
                    if value is not None:
                        val_str = json.dumps(value, ensure_ascii=False).lower() if not isinstance(value, str) else value.lower()
                        value_match = topic_lower in val_str
                    if path_match or value_match:
                        filtered.append(path)
                all_paths = filtered
            
            # Group facts
            groups = {}
            
            if group_by == "prefix":
                for path in all_paths:
                    prefix = path.split('.')[0] if '.' in path else "root"
                    if prefix not in groups:
                        groups[prefix] = []
                    value = self.mem.get(path)
                    meta = self._meta.get(path)
                    groups[prefix].append({
                        'path': path,
                        'value': value,
                        'confidence': meta.confidence if meta else 1.0,
                        'age_days': round((time.time() - (meta.created_at if meta else time.time())) / 86400, 1),
                    })
            
            elif group_by == "tag":
                # Group by tags, untagged goes to "untagged"
                for path in all_paths:
                    meta = self._meta.get(path)
                    tags = meta.tags if meta else []
                    value = self.mem.get(path)
                    entry = {
                        'path': path,
                        'value': value,
                        'confidence': meta.confidence if meta else 1.0,
                    }
                    if tags:
                        for tag in tags:
                            if tag not in groups:
                                groups[tag] = []
                            groups[tag].append(entry)
                    else:
                        if "untagged" not in groups:
                            groups["untagged"] = []
                        groups["untagged"].append(entry)
            
            elif group_by == "confidence":
                for path in all_paths:
                    meta = self._meta.get(path)
                    conf = meta.confidence if meta else 1.0
                    value = self.mem.get(path)
                    entry = {'path': path, 'value': value, 'confidence': conf}
                    
                    if conf >= 0.8:
                        bucket = "high_confidence"
                    elif conf >= 0.5:
                        bucket = "medium_confidence"
                    else:
                        bucket = "low_confidence"
                    
                    if bucket not in groups:
                        groups[bucket] = []
                    groups[bucket].append(entry)
            
            # Detect unique topics (first path segments)
            topics = sorted(set(p.split('.')[0] for p in all_paths if '.' in p))
            
            # Build human-readable summary
            summary_lines = []
            if topic:
                summary_lines.append(f"Knowledge about '{topic}': {len(all_paths)} facts")
            else:
                summary_lines.append(f"Memory overview: {len(all_paths)} facts across {len(topics)} topics")
            
            for group_name, facts in sorted(groups.items()):
                summary_lines.append(f"\n  {group_name} ({len(facts)} facts):")
                for f in facts[:5]:  # Show max 5 per group
                    val_str = str(f['value'])[:50]
                    conf_str = f" [conf:{f['confidence']:.1f}]" if f['confidence'] < 1.0 else ""
                    summary_lines.append(f"    - {f['path']}: {val_str}{conf_str}")
                if len(facts) > 5:
                    summary_lines.append(f"    ... and {len(facts) - 5} more")
            
            return {
                'total_facts': len(all_paths),
                'groups': groups,
                'topics': topics,
                'summary': '\n'.join(summary_lines),
            }

    def stats(self) -> dict:
        """Memory stats with scoring metadata."""
        base = self.mem.stats()
        base['paths'] = len(self._meta)
        base['tiers'] = 'enabled' if self.tiered else 'disabled'

        # Top scored paths (without query)
        top = self._top_scored(n=5)
        base['top_scored'] = top

        return base

    def estimate_size(self, value) -> int:
        """Estimate the JSON character size of a value.
        
        Args:
            value: Any JSON-serializable value.
            
        Returns:
            Estimated character count when serialized.
        """
        return self.mem.estimate_size(value)

    def available_budget(self) -> int:
        """Return how many characters can still be written before overflow.
        
        Returns:
            Number of characters remaining in the budget.
        """
        return self.mem.available_budget()

    def will_fit(self, path: str, value) -> dict:
        """Check if a value will fit at the given path without overflow.
        
        Simulates the write without committing. Use this before remember()
        to know if eviction will happen.
        
        Args:
            path: Dotted path where the value would be stored.
            value: The value to check.
            
        Returns:
            Dict with 'fits', 'current_chars', 'new_chars', 'delta',
            'available', 'overflow_by', 'eviction_needed'.
        """
        return self.mem.will_fit(path, value)

    def suggest_budget(self, target_facts: int = 50, avg_value_size: int = 80) -> dict:
        """Suggest a max_chars budget based on desired capacity.
        
        Args:
            target_facts: How many facts you want to store.
            avg_value_size: Average value size in characters.
            
        Returns:
            Dict with 'suggested_max_chars', 'estimated_facts', 'overhead'.
        """
        return self.mem.suggest_budget(target_facts, avg_value_size)

    def estimate_tokens(self, text: str = None, chars_per_token: float = 4.0) -> dict:
        """Estimate token count for text or entire memory.
        
        Args:
            text: Specific text to estimate. If None, estimates for entire memory.
            chars_per_token: Characters per token ratio. Default 4.0 (English).
            
        Returns:
            Dict with 'chars', 'estimated_tokens', 'chars_per_token'.
        """
        if text is not None:
            chars = len(text)
        else:
            chars = len(self.mem.export())
        
        return {
            'chars': chars,
            'estimated_tokens': _chars_to_tokens(chars, chars_per_token),
            'chars_per_token': chars_per_token,
            'context_tokens': _chars_to_tokens(chars, chars_per_token),
        }

    def visualize(self, format: str = "full") -> str:
        """Visualize memory structure and statistics.
        
        Args:
            format: Format to use ('tree', 'stats', 'strength', 'contradictions', 
                    'consolidation', 'timeline', 'full')
        
        Returns:
            String representation of memory visualization
        """
        return visualize_memory(self, format)

    # ── Event System ──────────────────────────────────────────────
    
    def on(self, event: str, callback: callable) -> "SmartMemory":
        """Register an event callback.
        
        Args:
            event: Event name ('on_set', 'on_delete', 'on_update', 'on_change')
            callback: Function to call when event occurs
            
        Returns:
            Self for chaining
        """
        if event not in self._event_callbacks:
            raise ValueError(f"Unknown event: {event}. Valid events: {list(self._event_callbacks.keys())}")
        
        self._event_callbacks[event].append(callback)
        return self
    
    def off(self, event: str, callback: callable = None) -> "SmartMemory":
        """Unregister an event callback.
        
        Args:
            event: Event name
            callback: Optional specific callback to remove (None = remove all)
            
        Returns:
            Self for chaining
        """
        if event not in self._event_callbacks:
            return self
        
        if callback is None:
            self._event_callbacks[event].clear()
        else:
            self._event_callbacks[event] = [cb for cb in self._event_callbacks[event] if cb != callback]
        
        return self
    
    def _trigger_event(self, event: str, path: str, old_value: Any, new_value: Any):
        """Internal: trigger event callbacks."""
        for callback in self._event_callbacks.get(event, []):
            try:
                callback(path, old_value, new_value)
            except Exception as e:
                print(f"Event callback error ({event}): {e}", flush=True)
    
    # ── Versioning ────────────────────────────────────────────────
    
    def get_history(self, path: str = None, limit: int = 100) -> list:
        """Get version history for a path or all paths.
        
        Args:
            path: Optional path to filter by
            limit: Maximum versions to return
            
        Returns:
            List of version dicts
        """
        versions = self.versioning.get_history(path=path, limit=limit)
        return [{
            'version_id': v.version_id,
            'timestamp': v.timestamp,
            'path': v.path,
            'old_value': v.old_value,
            'new_value': v.new_value,
            'operation': v.operation,
            'metadata': v.metadata
        } for v in versions]
    
    def get_value_at(self, path: str, timestamp: float) -> Tuple[Any, bool]:
        """Get the value of a path at a specific time.
        
        Args:
            path: Memory path
            timestamp: Target timestamp
            
        Returns:
            Tuple of (value, found)
        """
        return self.versioning.get_value_at(path, timestamp)
    
    def get_state_at(self, timestamp: float) -> Dict[str, Any]:
        """Get complete memory state at a specific time.
        
        Args:
            timestamp: Target timestamp
            
        Returns:
            Dict of path -> value
        """
        return self.versioning.get_state_at(timestamp)
    
    def diff(self, timestamp_old: float, timestamp_new: float) -> Dict[str, Any]:
        """Get differences between two points in time.
        
        Args:
            timestamp_old: Earlier timestamp
            timestamp_new: Later timestamp
            
        Returns:
            Dict with 'added', 'modified', 'deleted'
        """
        diff = self.versioning.diff(timestamp_old, timestamp_new)
        return {
            'added': diff.added,
            'modified': diff.modified,
            'deleted': diff.deleted,
            'timestamp_old': diff.timestamp_old,
            'timestamp_new': diff.timestamp_new
        }
    
    def get_recent_changes(self, seconds: float = 3600, limit: int = 100) -> list:
        """Get recent changes within time window.
        
        Args:
            seconds: Time window in seconds
            limit: Maximum versions to return
            
        Returns:
            List of version dicts
        """
        return self.get_history(limit=limit)
    
    def get_most_changed(self, limit: int = 10, seconds: float = None) -> List[Tuple[str, int]]:
        """Get most frequently changed paths.
        
        Args:
            limit: Maximum paths to return
            seconds: Optional time window
            
        Returns:
            List of (path, change_count) tuples
        """
        return self.versioning.get_most_changed(limit=limit, seconds=seconds)

    # ── Encryption ────────────────────────────────────────────────
    
    def enable_encryption(self, master_key: str = None) -> "SmartMemory":
        """Enable encryption for sensitive data.
        
        Args:
            master_key: Optional master key (generated if not provided)
            
        Returns:
            Self for chaining
        """
        self.encryption = MemoryEncryption(master_key)
        return self
    
    def disable_encryption(self) -> "SmartMemory":
        """Disable encryption."""
        self.encryption = None
        return self
    
    def remember_encrypted(self, path: str, value: Any, **kwargs) -> dict:
        """Store a fact with encryption.
        
        Args:
            path: Dotted path key
            value: Value to encrypt and store
            **kwargs: Additional arguments for remember()
            
        Returns:
            Dict with 'success' (bool), 'contradictions' (list), 'warnings' (list)
        """
        if not self.encryption:
            raise ValueError("Encryption not enabled. Call enable_encryption() first.")
        
        # Encrypt the value
        encrypted = self.encryption.encrypt(value)
        encrypted_dict = self.encryption.to_dict(encrypted)
        
        # Store encrypted value
        result = self.remember(path, encrypted_dict, **kwargs)
        
        # Mark as encrypted in metadata
        if path in self._meta:
            self._meta[path].tags.append('encrypted')
        
        return result
    
    def recall_decrypted(self, path: str, default: Any = None) -> Any:
        """Retrieve and decrypt a fact.
        
        Args:
            path: Dotted path key
            default: Default value if not found
            
        Returns:
            Decrypted value or default
        """
        if not self.encryption:
            raise ValueError("Encryption not enabled. Call enable_encryption() first.")
        
        value = self.recall(path)
        
        if value is None:
            return default
        
        # Check if encrypted
        if self.encryption.is_encrypted(value):
            encrypted = self.encryption.from_dict(value)
            return self.encryption.decrypt(encrypted)
        
        return value
    
    def is_encrypted(self, path: str) -> bool:
        """Check if a path contains encrypted data.
        
        Args:
            path: Dotted path key
            
        Returns:
            True if encrypted
        """
        if not self.encryption:
            return False
        
        value = self.recall(path)
        if value is None:
            return False
        
        return self.encryption.is_encrypted(value)

    # ── Advanced Search ───────────────────────────────────────────
    
    def search_regex(self, pattern: str, field: str = 'both', 
                    case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """Search using regular expressions.
        
        Args:
            pattern: Regex pattern to search for
            field: Where to search ('path', 'value', 'both')
            case_sensitive: Whether to use case-sensitive matching
            
        Returns:
            List of result dicts
        """
        results = self.search_engine.regex_search(pattern, field, case_sensitive)
        return [{
            'path': r.path,
            'value': r.value,
            'score': r.score,
            'match_type': r.match_type,
            'highlights': r.highlights
        } for r in results]
    
    def search_fuzzy(self, query: str, threshold: float = 0.6,
                    field: str = 'both') -> List[Dict[str, Any]]:
        """Search using fuzzy string matching.
        
        Args:
            query: Search query
            threshold: Minimum similarity score (0.0-1.0)
            field: Where to search ('path', 'value', 'both')
            
        Returns:
            List of result dicts
        """
        results = self.search_engine.fuzzy_search(query, threshold, field)
        return [{
            'path': r.path,
            'value': r.value,
            'score': r.score,
            'match_type': r.match_type,
            'highlights': r.highlights
        } for r in results]
    
    def search_full_text(self, query: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """Full-text search across all memory.
        
        Args:
            query: Search query (supports multiple words)
            case_sensitive: Whether to use case-sensitive matching
            
        Returns:
            List of result dicts
        """
        results = self.search_engine.full_text_search(query, case_sensitive)
        return [{
            'path': r.path,
            'value': r.value,
            'score': r.score,
            'match_type': r.match_type,
            'highlights': r.highlights
        } for r in results]
    
    def search_advanced(self, query: str, search_type: str = 'auto', **kwargs) -> List[Dict[str, Any]]:
        """Unified advanced search interface.
        
        Args:
            query: Search query
            search_type: Type of search ('auto', 'regex', 'fuzzy', 'full_text', 'semantic')
            **kwargs: Additional arguments for specific search types
            
        Returns:
            List of result dicts
        """
        results = self.search_engine.search(query, search_type, **kwargs)
        return [{
            'path': r.path,
            'value': r.value,
            'score': r.score,
            'match_type': r.match_type,
            'highlights': r.highlights
        } for r in results]
    
    def suggest_paths(self, partial: str, limit: int = 10) -> List[str]:
        """Suggest paths based on partial input.
        
        Args:
            partial: Partial path input
            limit: Maximum suggestions
            
        Returns:
            List of suggested paths
        """
        return self.search_engine.suggest(partial, limit)

    # ── Private Helpers ────────────────────────────────────────────

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
        # Extract path tokens for boosting
        path_tokens = set(re.findall(r'\w{2,}', path.lower())) if path else set()
        keyword = _keyword_relevance(meta.tokens, query_tokens, path_tokens) if query_tokens else 0.5

        if query_tokens:
            if keyword > 0:
                final = 0.1 * recency + 0.05 * frequency + 0.85 * keyword
            else:
                final = 0.0  # No keyword match = suppress completely
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

    # ── Procedural Memory Operations ─────────────────────────────────

    def learn(self, experience: str, domain: str = None, extract_principles: bool = True) -> dict:
        """Extract principles and skills from an experience.
        
        Args:
            experience: Description of what was learned
            domain: Domain/context of the experience (e.g., "cycling", "programming")
            extract_principles: Whether to automatically extract principles
            
        Returns:
            Dict with extracted principles and created/strengthened skills
        """
        if not self.procedural:
            return {'error': 'Procedural memory not enabled. Initialize with procedural=True'}
        
        with self._lock:
            result = {
                'principles_extracted': [],
                'skills_created': [],
                'skills_strengthened': [],
            }
            
            # Extract principles from experience
            if extract_principles:
                principles = self.procedural.extract_principles(experience, domain)
                for p in principles:
                    skill_name = f"principle_{len(self.procedural.skills)}"
                    skill = self.procedural.add_skill(
                        name=skill_name,
                        principle=p['principle'],
                        domains=p.get('domains', [domain] if domain else []),
                        examples=[experience[:100]]
                    )
                    result['principles_extracted'].append(p['principle'])
                    result['skills_created'].append(skill_name)
            
            # Also store the experience as a fact for reference
            fact_path = f"mem.experience.{domain or 'general'}.{int(time.time())}"
            self.remember(fact_path, experience, tags=['experience', 'learning'])
            
            # Save procedural memory
            self.procedural._save()
            
            return result

    def transfer(self, new_situation: str, domain: str = None) -> dict:
        """Find skills that might transfer to a new situation.
        
        Args:
            new_situation: Description of the new situation
            domain: Domain of the new situation (e.g., "motorcycling")
            
        Returns:
            Dict with transferable skills and how they apply
        """
        if not self.procedural:
            return {'error': 'Procedural memory not enabled. Initialize with procedural=True'}
        
        with self._lock:
            # Extract keywords from situation
            keywords = set(re.findall(r'\w{3,}', new_situation.lower()))
            
            # Find transferable skills
            skills = self.procedural.find_transferable_skills(
                new_domain=domain,
                context_keywords=keywords
            )
            
            result = {
                'situation': new_situation,
                'domain': domain,
                'transferable_skills': [],
                'total_found': len(skills),
            }
            
            for skill in skills[:5]:  # Top 5 most relevant
                result['transferable_skills'].append({
                    'name': skill.name,
                    'principle': skill.principle,
                    'strength': round(skill.strength, 2),
                    'domains': skill.domains,
                    'transfer_count': skill.transfer_count,
                    'how_it_applies': self._explain_transfer(skill, new_situation),
                })
            
            return result

    def apply_skill(self, skill_name: str, new_domain: str = None, 
                    outcome: str = None) -> bool:
        """Record that a skill was applied (strengthens it).
        
        Args:
            skill_name: Name of the skill to apply
            new_domain: Domain where skill was applied
            outcome: Description of how it was applied
            
        Returns:
            True if skill was found and applied
        """
        if not self.procedural:
            return False
        
        with self._lock:
            success = self.procedural.apply_skill(skill_name, new_domain)
            
            if success and outcome:
                # Store the application as an experience
                fact_path = f"mem.skill_applied.{skill_name}.{int(time.time())}"
                self.remember(fact_path, {
                    'skill': skill_name,
                    'domain': new_domain,
                    'outcome': outcome,
                }, tags=['skill_application', 'learning'])
                
                # Save procedural memory
                self.procedural._save()
            
            return success

    def competence_map(self) -> dict:
        """Get overview of all transferable skills and their domains.
        
        Returns:
            Dict with skill statistics and strongest skills
        """
        if not self.procedural:
            return {'error': 'Procedural memory not enabled. Initialize with procedural=True'}
        
        return self.procedural.competence_map()

    def _explain_transfer(self, skill, new_situation: str) -> str:
        """Explain how a skill might transfer to a new situation."""
        # Simplified explanation - in production, use LLM
        return f"The principle '{skill.principle}' applies because it addresses the core concept of {skill.name.replace('_', ' ')}."

    # ── Memory Lifecycle & Pruning ────────────────────────────────────

    def prune(self, max_age_seconds: int = None, min_access_count: int = None,
              max_total_chars: int = None, dry_run: bool = False) -> dict:
        """Remove expired, unused, or oversized memory entries.
        
        Args:
            max_age_seconds: Remove facts older than this (None = no age limit)
            min_access_count: Remove facts accessed fewer than this (None = no limit)
            max_total_chars: If total memory exceeds this, remove oldest/least accessed
            dry_run: If True, return what would be removed without actually removing
            
        Returns:
            Dict with pruning statistics and removed paths
            
        Note:
            Protected facts (protected=True) are immune to all pruning except TTL expiration.
            Identity facts (user.name, user.profession, etc.) should be marked protected.
        """
        with self._lock:
            now = time.time()
            removed = []
            expired = []
            archived = []
            skipped_protected = []
            
            # 1. Remove expired facts (TTL exceeded) — even protected facts can expire
            for path in list(self._meta.keys()):
                meta = self._meta[path]
                if meta.expires_at and now > meta.expires_at:
                    if not dry_run:
                        self.mem.delete(path, prune=True)
                        if self.tiered:
                            self.tiered.hot.delete(path, prune=True)
                            self.tiered.warm.delete(path, prune=True)
                            self.tiered.cold.delete(path, prune=True)
                        del self._meta[path]
                    expired.append(path)
            
            # 2. Remove old facts (age-based pruning) — skip protected
            if max_age_seconds:
                for path in list(self._meta.keys()):
                    if path in expired:  # Skip already expired
                        continue
                    meta = self._meta[path]
                    
                    # Skip protected facts
                    if meta.protected:
                        skipped_protected.append(path)
                        continue
                    
                    age = now - meta.created_at
                    if age > max_age_seconds:
                        if not dry_run:
                            self.mem.delete(path, prune=True)
                            if self.tiered:
                                self.tiered.hot.delete(path, prune=True)
                                self.tiered.warm.delete(path, prune=True)
                                self.tiered.cold.delete(path, prune=True)
                            del self._meta[path]
                        removed.append(path)
            
            # 3. Remove rarely accessed facts (frequency-based pruning) — skip protected
            if min_access_count:
                for path in list(self._meta.keys()):
                    if path in expired or path in removed:  # Skip already processed
                        continue
                    meta = self._meta[path]
                    
                    # Skip protected facts
                    if meta.protected:
                        if path not in skipped_protected:
                            skipped_protected.append(path)
                        continue
                    
                    if meta.access_count < min_access_count:
                        if not dry_run:
                            self.mem.delete(path, prune=True)
                            if self.tiered:
                                self.tiered.hot.delete(path, prune=True)
                                self.tiered.warm.delete(path, prune=True)
                                self.tiered.cold.delete(path, prune=True)
                            del self._meta[path]
                        removed.append(path)
            
            # 4. Size-based pruning (if total exceeds limit) — skip protected
            if max_total_chars:
                total_chars = 0
                for path in self.mem.paths():
                    value = self.mem.get(path)
                    if value:
                        total_chars += len(str(value))
                
                if total_chars > max_total_chars:
                    # Sort by score (oldest, least accessed first), but exclude protected
                    paths_by_score = sorted(
                        [(p, m) for p, m in self._meta.items() if not m.protected],
                        key=lambda x: (x[1].access_count, x[1].last_accessed)
                    )
                    
                    chars_to_remove = total_chars - max_total_chars
                    chars_removed = 0
                    
                    for path, meta in paths_by_score:
                        if chars_removed >= chars_to_remove:
                            break
                        if path in expired or path in removed:
                            continue
                        
                        value = self.mem.get(path)
                        if value:
                            value_chars = len(str(value))
                            if not dry_run:
                                self.mem.delete(path, prune=True)
                                if self.tiered:
                                    self.tiered.hot.delete(path, prune=True)
                                    self.tiered.warm.delete(path, prune=True)
                                    self.tiered.cold.delete(path, prune=True)
                                del self._meta[path]
                            removed.append(path)
                            chars_removed += value_chars
            
            # 5. Archive old facts to cold storage (if tiered enabled) — skip protected
            if self.tiered:
                for path in list(self._meta.keys()):
                    if path in expired or path in removed:
                        continue
                    meta = self._meta[path]
                    
                    # Skip protected facts
                    if meta.protected:
                        continue
                    
                    age = now - meta.last_accessed
                    
                    # Archive if not accessed in 7 days
                    if age > 604800 and meta.tier != 'cold':
                        if not dry_run:
                            value = self.mem.get(path)
                            if value:
                                self.tiered.set(path, value, tier='cold')
                                self.mem.delete(path, prune=True)
                                self.tiered.hot.delete(path, prune=True)
                                self.tiered.warm.delete(path, prune=True)
                                meta.tier = 'cold'
                                meta.archived = True
                        archived.append(path)
            
            # Save metadata if changes were made
            if not dry_run and (removed or expired or archived):
                self._save_meta()
            
            return {
                'removed': removed,
                'expired': expired,
                'archived': archived,
                'skipped_protected': skipped_protected,
                'total_removed': len(removed) + len(expired),
                'total_archived': len(archived),
                'total_protected': len(skipped_protected),
                'dry_run': dry_run,
            }

    def archive(self, path: str) -> bool:
        """Manually archive a fact to cold storage.
        
        Args:
            path: Path to archive
            
        Returns:
            True if archived, False if not found or already archived
        """
        with self._lock:
            if path not in self._meta:
                return False
            
            meta = self._meta[path]
            if meta.archived:
                return False
            
            value = self.mem.get(path)
            if not value:
                return False
            
            # Move to cold storage if tiered enabled
            if self.tiered:
                self.tiered.set(path, value, tier='cold')
                self.mem.delete(path, prune=True)
                meta.tier = 'cold'
                meta.archived = True
                self._save_meta()
                return True
            
            # Fallback: use Memory-level cold storage (.cold.json)
            self.mem._archive_to_cold(path, value)
            self.mem.delete(path, prune=True)
            meta.tier = 'cold'
            meta.archived = True
            self._save_meta()
            return True

    def cold_stats(self) -> dict:
        """Get statistics about cold (archived) storage.
        
        Returns:
            Dict with cold storage metrics: count, chars, path, oldest, paths
        """
        return self.mem.cold_stats()

    def recover_from_cold(self, path: str) -> bool:
        """Recover an archived fact from cold storage back to hot memory.
        
        Args:
            path: The dotted path of the fact to recover.
            
        Returns:
            True if recovered, False if not found in cold storage.
        """
        with self._lock:
            success = self.mem.recover_from_cold(path)
            if success:
                # Re-initialize metadata for recovered fact
                value = self.mem.get(path)
                if path not in self._meta and value is not None:
                    self._init_meta(path, value)
                self._save_meta()
            return success

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
        return self.mem.cold_search(query=query, path_pattern=path_pattern,
                                    older_than=older_than, newer_than=newer_than)

    def recover_all(self) -> dict:
        """Recover all facts from cold storage back to hot memory.
        
        Returns:
            Dict with 'recovered' (list of paths), 'failed' (list of paths), 'count'.
        """
        with self._lock:
            result = self.mem.recover_all()
            # Re-initialize metadata for recovered facts
            for path in result["recovered"]:
                value = self.mem.get(path)
                if path not in self._meta and value is not None:
                    self._init_meta(path, value)
            if result["recovered"]:
                self._save_meta()
            return result

    def recover_matching(self, pattern: str) -> dict:
        """Recover facts from cold storage matching a glob pattern.
        
        Args:
            pattern: Glob pattern (e.g., "project.*", "user.**").
            
        Returns:
            Dict with 'recovered', 'failed', 'count'.
        """
        with self._lock:
            result = self.mem.recover_matching(pattern)
            for path in result["recovered"]:
                value = self.mem.get(path)
                if path not in self._meta and value is not None:
                    self._init_meta(path, value)
            if result["recovered"]:
                self._save_meta()
            return result

    def purge_cold(self, older_than: float = None, keep_last: int = None) -> dict:
        """Permanently delete old facts from cold storage.
        
        Args:
            older_than: Delete facts evicted before this timestamp (epoch).
            keep_last: Keep the N most recently evicted facts, delete the rest.
            
        Returns:
            Dict with 'purged' (list of paths), 'kept' (int), 'count'.
        """
        return self.mem.purge_cold(older_than=older_than, keep_last=keep_last)

    def lifecycle_stats(self) -> dict:
        """Get memory lifecycle statistics.
        
        Returns:
            Dict with memory health metrics
        """
        with self._lock:
            now = time.time()
            
            if not self._meta:
                return {
                    'total_facts': 0,
                    'total_chars': 0,
                    'avg_age_seconds': 0,
                    'expired_facts': 0,
                    'archived_facts': 0,
                    'hot_facts': 0,
                    'warm_facts': 0,
                    'cold_facts': 0,
                    'memory_health': 'empty',
                    'cold_storage': self.cold_stats(),
                }
            
            total_facts = len(self._meta)
            total_chars = sum(m.size_bytes for m in self._meta.values())
            ages = [now - m.created_at for m in self._meta.values()]
            avg_age = sum(ages) / len(ages) if ages else 0
            
            expired = sum(1 for m in self._meta.values() if m.expires_at and now > m.expires_at)
            archived = sum(1 for m in self._meta.values() if m.archived)
            
            tier_counts = {'hot': 0, 'warm': 0, 'cold': 0}
            for m in self._meta.values():
                tier_counts[m.tier] = tier_counts.get(m.tier, 0) + 1
            
            # Memory health assessment
            if expired > total_facts * 0.3:
                health = 'critical'  # >30% expired
            elif expired > total_facts * 0.1:
                health = 'warning'   # >10% expired
            elif avg_age > 2592000:  # >30 days average age
                health = 'aging'
            else:
                health = 'healthy'
            
            # Actionable health warnings
            warnings = []
            
            # Capacity warning
            capacity = self.mem.stats()
            utilization_pct = float(capacity.get('utilization', '0%').replace('%', ''))
            if utilization_pct > 90:
                warnings.append(f"CRITICAL: Memory {utilization_pct:.0f}% full - eviction imminent. Consider increasing max_chars or pruning.")
            elif utilization_pct > 70:
                warnings.append(f"WARNING: Memory {utilization_pct:.0f}% full - plan for growth.")
            
            # Expired facts warning
            if expired > 0:
                warnings.append(f"{expired} expired fact(s) still in memory - run purge_expired() to reclaim budget.")
            
            # Never-accessed facts (created but never recalled)
            never_accessed = sum(1 for m in self._meta.values() 
                               if m.access_count <= 1 and (now - m.created_at) > 3600)
            if never_accessed > 0:
                pct = never_accessed / total_facts * 100
                if pct > 30:
                    warnings.append(f"{never_accessed} facts ({pct:.0f}%) never accessed after creation - consider removing unused data.")
            
            # High overwrite count
            high_overwrite = [(p, m.overwrite_count) for p, m in self._meta.items() 
                             if m.overwrite_count > 3]
            if high_overwrite:
                top = sorted(high_overwrite, key=lambda x: x[1], reverse=True)[:3]
                paths_str = ", ".join(f"{p} ({c}x)" for p, c in top)
                warnings.append(f"Frequently overwritten paths: {paths_str} - consider if these need versioning.")
            
            # Low confidence facts
            low_conf = sum(1 for m in self._meta.values() if m.confidence < 0.5)
            if low_conf > 0:
                warnings.append(f"{low_conf} fact(s) with confidence < 0.5 - may pollute recall. Review or remove.")
            
            # Contradictions
            contradictions = self.get_contradictions()
            if contradictions:
                warnings.append(f"{len(contradictions)} contradiction(s) detected - memory may be inconsistent.")
            
            # Cold storage bloat
            cold = self.cold_stats()
            if cold['count'] > 100:
                warnings.append(f"Cold storage has {cold['count']} archived facts - consider purge_cold(keep_last=50).")
            
            return {
                'total_facts': total_facts,
                'total_chars': total_chars,
                'avg_age_seconds': round(avg_age, 1),
                'avg_age_days': round(avg_age / 86400, 1),
                'expired_facts': expired,
                'archived_facts': archived,
                'hot_facts': tier_counts['hot'],
                'warm_facts': tier_counts['warm'],
                'cold_facts': tier_counts['cold'],
                'memory_health': health,
                'cold_storage': cold,
                'never_accessed': never_accessed,
                'low_confidence_facts': low_conf,
                'high_overwrite_paths': len(high_overwrite),
                'capacity_utilization': utilization_pct,
                'warnings': warnings,
                'warning_count': len(warnings),
            }

    # ── Internal ─────────────────────────────────────────────────────

    def _init_meta(self, path: str, value, ttl: int = None, protected: bool = False, 
                   tags: list[str] = None, confidence: float = 1.0):
        """Initialize or update metadata for a path."""
        if path not in self._meta:
            self._meta[path] = PathMeta(ttl=ttl, protected=protected, tags=tags,
                                         confidence=confidence)
        meta = self._meta[path]
        meta.last_accessed = time.time()
        meta.tokens = self._extract_tokens(path, value)
        
        # Update size tracking
        if isinstance(value, str):
            meta.size_bytes = len(value.encode('utf-8'))
        else:
            meta.size_bytes = len(json.dumps(value, ensure_ascii=False).encode('utf-8'))

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
                    meta.ttl = data.get('ttl', None)
                    meta.expires_at = data.get('expires_at', None)
                    meta.archived = data.get('archived', False)
                    meta.size_bytes = data.get('size_bytes', 0)
                    meta.protected = data.get('protected', False)
                    meta.tags = data.get('tags', [])
                    meta.confidence = data.get('confidence', 1.0)
                    meta.overwrite_count = data.get('overwrite_count', 0)
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
                    'ttl': meta.ttl,
                    'expires_at': meta.expires_at,
                    'archived': meta.archived,
                    'size_bytes': meta.size_bytes,
                    'protected': meta.protected,
                    'tags': meta.tags,
                    'confidence': meta.confidence,
                    'overwrite_count': meta.overwrite_count,
                },
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


def _chars_to_tokens(chars: int, ratio: float = 4.0) -> int:
    """Estimate token count from character count.
    
    Args:
        chars: Number of characters.
        ratio: Characters per token. Default 4.0 (English average).
               Use 2.5-3.0 for CJK, 3.5 for mixed content.
    
    Returns:
        Estimated token count.
    """
    return int(chars / ratio)


def _tokens_to_chars(tokens: int, ratio: float = 4.0) -> int:
    """Convert token budget to character budget.
    
    Args:
        tokens: Token budget.
        ratio: Characters per token. Default 4.0 (English average).
    
    Returns:
        Equivalent character count.
    """
    return int(tokens * ratio)
