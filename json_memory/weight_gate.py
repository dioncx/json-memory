"""
WeightGate — Passive middleware for automatic synapse weight updates.

Wraps a Synapse instance. Sits between input/output and updates weights
transparently through the shared Synapse graph. No explicit tool calls needed.
Single source of truth — no parallel data structures.

Usage:
    from json_memory import Synapse, WeightGate

    # Option A: WeightGate creates and owns the Synapse
    gate = WeightGate("/path/to/synapse.json", enabled=True)

    # Option B: WeightGate wraps an existing Synapse (shared state)
    brain = Synapse()
    brain.link("coffee", ["cappuccino", "espresso"])
    gate = WeightGate(synapse=brain, enabled=True)

    # Process messages (weights auto-update in the shared graph)
    gate.process_input("I love cappuccino")
    gate.process_output("Here's your cappuccino")

    # Both see the same state
    brain.top_associations("coffee")
    gate.top_associations("coffee")  # same result

    # Or use as context manager (auto-saves on exit)
    with WeightGate("/path/to/synapse.json") as gate:
        gate.process_conversation(user_msg, agent_response)

Stemming:
    Three layers (union of all candidates):
    1. Dictionary — 150+ root families for reliable matching
    2. snowballstemmer — optional dep, Porter algorithm for unknown words
    3. Suffix rules — built-in fallback when snowball unavailable

    pip install json-memory[stem]   # adds snowballstemmer (~200KB)
"""

import json
import re
import time
from pathlib import Path
from typing import Optional, Union

from .synapse import Synapse

# ── Optional: snowballstemmer (Porter2 algorithm, ~200KB) ───────────
try:
    import snowballstemmer as _snowball
    _SNOWBALL = _snowball.stemmer("english")
    HAS_SNOWBALL = True
except ImportError:
    _SNOWBALL = None
    HAS_SNOWBALL = False

# ── Layer 1: Dictionary — known variant → root mappings ─────────────
# Covers irregular verbs, technical terms, and common derivations
# where Porter/suffix rules produce unusable stems.
# e.g., "configuration" → "configur" (Porter) vs "config" (dict)

_COMMON_ROOTS: dict[str, str] = {}


def _build_root_map() -> dict[str, str]:
    """Build reverse lookup: word_variant → canonical root."""
    root_families = [
        # -tion family (worst offenders with suffix rules)
        ("config", "configuration", "configurations", "configuring", "configured"),
        ("predict", "prediction", "predictions", "predicting", "predicted", "predictive"),
        ("compress", "compression", "compressions", "compressing", "compressed"),
        ("decompress", "decompression", "decompressing", "decompressed"),
        ("connect", "connection", "connections", "connecting", "connected", "connective"),
        ("construct", "construction", "constructing", "constructed", "constructive"),
        ("detect", "detection", "detections", "detecting", "detected"),
        ("direct", "direction", "directions", "directing", "directed", "directive"),
        ("execut", "execution", "executions", "executing", "executed", "executive"),
        ("extract", "extraction", "extracting", "extracted"),
        ("inject", "injection", "injecting", "injected"),
        ("inspect", "inspection", "inspecting", "inspected"),
        ("interact", "interaction", "interactions", "interacting", "interactive"),
        ("interrupt", "interruption", "interrupting", "interrupted"),
        ("operat", "operation", "operations", "operating", "operated", "operative"),
        ("process", "processing", "processed", "processor", "processes"),
        ("produc", "production", "producing", "produced", "productive"),
        ("reduc", "reduction", "reducing", "reduced"),
        ("select", "selection", "selecting", "selected", "selective"),
        ("solut", "solution", "solutions"),
        ("transact", "transaction", "transactions"),
        ("transform", "transformation", "transforming", "transformed"),

        # -ment family
        ("deploy", "deployment", "deployments", "deploying", "deployed"),
        ("develop", "development", "developing", "developed"),
        ("environ", "environment", "environments"),
        ("manage", "management", "managing", "managed", "manager"),
        ("require", "requirement", "requirements", "requiring", "required"),
        ("replace", "replacement", "replacing", "replaced"),
        ("state", "statement", "statements"),

        # -ness / -ity family
        ("ready", "readiness"),
        ("happi", "happiness", "happy"),
        ("effectiv", "effectiveness", "effective", "effectively"),
        ("activ", "activity", "activities", "active", "actively"),
        ("respons", "responsibility", "responsible", "response", "responses"),

        # -ly family
        ("quick", "quickly"),
        ("basic", "basically"),
        ("technic", "technical", "technically", "technique", "techniques"),
        ("specif", "specific", "specifically", "specification", "specifications"),
        ("automat", "automatic", "automatically", "automation", "automate", "automated"),
        ("probabl", "probably", "probability"),
        ("gener", "general", "generally", "generate", "generating", "generated",
         "generator", "generators"),

        # -ing / -ed family (irregular cases)
        ("debug", "debugging", "debugged", "debugger"),
        ("run", "running", "runner", "ran", "runs"),
        ("log", "logging", "logged", "logger", "logs"),
        ("test", "testing", "tested", "tester", "tests"),
        ("build", "building", "builder", "builds"),
        ("monitor", "monitoring", "monitored"),
        ("trigger", "triggering", "triggered", "triggers"),
        ("stream", "streaming", "streamed", "streams"),
        ("parse", "parsing", "parsed", "parser"),
        ("search", "searching", "searched", "searches"),
        ("match", "matching", "matched", "matches", "matcher"),
        ("filter", "filtering", "filtered", "filters"),
        ("transform", "transforming", "transformed", "transformation"),
        ("convert", "converting", "converted", "conversion"),
        ("insert", "inserting", "inserted"),
        ("delet", "deleting", "deleted", "deletion"),
        ("updat", "updating", "updated", "updates"),
        ("creati", "creating", "created", "creation", "creative"),

        # Irregular verbs (Porter can't handle these)
        ("run", "running", "runner", "ran", "runs"),
        ("break", "breaking", "broken", "breaks", "broke"),
        ("mak", "making", "made", "makes", "maker"),
        ("tak", "taking", "taken", "takes", "took"),
        ("giv", "giving", "given", "gives", "gave"),
        ("find", "finding", "found", "finds"),
        ("get", "getting", "gets", "got"),
        ("set", "setting", "sets"),
        ("put", "putting", "puts"),
        ("go", "going", "gone", "went", "goes"),
        ("do", "doing", "done", "did", "does"),
        ("see", "seeing", "seen", "saw", "sees"),
        ("know", "knowing", "known", "knew", "knows"),
        ("think", "thinking", "thought", "thinks"),
        ("write", "writing", "written", "wrote", "writes"),
        ("speak", "speaking", "spoken", "spoke", "speaks"),
        ("drive", "driving", "driven", "drove", "drives"),
        ("read", "reading", "reads"),
        ("com", "coming", "came", "comes"),

        # -s / plural / common technical terms
        ("bot", "bots"),
        ("token", "tokens"),
        ("signal", "signals"),
        ("strateg", "strategy", "strategies"),
        ("analys", "analysis", "analyses", "analyze", "analyzing", "analyzed"),
        ("memor", "memory", "memories"),
        ("synaps", "synapse", "synapses"),
        ("concept", "concepts"),
        ("weight", "weights", "weighted", "weighting"),
        ("model", "models", "modeling", "modeled"),
        ("trade", "trades", "trading", "trader", "traders", "traded"),
        ("error", "errors", "erroring", "errored"),
        ("function", "functions", "functional", "functionality"),
        ("class", "classes"),
        ("data", "dataset", "datasets", "database"),
        ("code", "codes", "coding", "coded", "coder"),
        ("server", "servers"),
        ("client", "clients"),
        ("request", "requests", "requesting", "requested"),
        ("response", "responses", "responding", "responded"),
        ("messag", "message", "messages", "messaging", "messaged"),
        ("command", "commands", "commanding", "commanded"),
        ("script", "scripts", "scripting", "scripted"),
        ("instal", "install", "installation", "installing", "installed"),
        ("depend", "dependency", "dependencies", "dependent", "depending"),
        ("file", "files", "filing", "filed"),
        ("path", "paths"),
        ("direct", "directory", "directories", "direction", "directions"),
        ("import", "imports", "importing", "imported"),
        ("export", "exports", "exporting", "exported"),
        ("valid", "validate", "validation", "validating", "validated", "validator"),
        ("serial", "serialize", "serialization", "serializing", "serialized"),
        ("pars", "parse", "parsing", "parsed", "parser"),
        ("stor", "store", "storage", "storing", "stored"),
        ("load", "loading", "loaded", "loader", "loads"),
        ("sav", "save", "saving", "saved", "saves"),
        ("write", "writing", "written", "wrote", "writes", "writer"),
    ]

    mapping = {}
    for group in root_families:
        root = group[0]
        mapping[root] = root  # identity
        for variant in group[1:]:
            mapping[variant] = root
    return mapping


_COMMON_ROOTS = _build_root_map()

# ── Layer 2: Suffix rules — fallback when snowball unavailable ──────
# Only clearly productive suffixes (≥4 chars) that rarely false-match.
# Short suffixes (-ing, -ed, -er, -es, -al, -en, -s, -ent, -ant, -ive)
# are deliberately excluded — they match too many base words and produce
# garbage stems: "current"→"curr", "important"→"import" (keyword!),
# "power"→"pow", "signal"→"sign".
#
# Rule: if a suffix would fire on "signal", "power", "current", "important",
# "happen", "cases" — it's too short. Remove it.

_SUFFIX_RULES = [
    # ≥6 chars — very safe, almost never false-match
    ("ational", "ate"),    # operational → operate
    ("fulness", "ful"),    # helpfulness → helpful
    ("ousness", "ous"),    # dangerousness → dangerous
    ("iveness", "ive"),    # effectiveness → effective
    ("lessly", "less"),    # endlessly → endless
    ("ically", "ic"),      # technically → technic
    # 5 chars — safe
    ("tional", "te"),      # functional → functe
    ("ously", "ous"),      # dangerously → dangerous
    ("ively", "ive"),      # effectively → effective
    ("ality", "al"),       # functionality → functional
    ("ities", "ity"),      # activities → activity
    ("ments", "ment"),     # deployments → deployment
    ("ally", "al"),        # technically → technical
    # 4 chars — productive derivational suffixes
    ("ness", ""),          # readiness → ready
    ("tion", ""),          # configuration → configura
    ("sion", ""),          # compression → compres
    ("ment", ""),          # deployment → deploy
    ("ies", "y"),          # strategies → strategy
    ("ied", "y"),          # modified → modify
    # Deliberately REMOVED (too short, high false-positive rate):
    #   ing, ed, er, es, al, en, s, ent, ant, ive, ly, ful, est, ers
    # Dictionary + snowball handle these cases.
]


def _candidates(word: str) -> set[str]:
    """Generate all plausible root forms of a word.

    Three-layer approach (all unioned):
    1. Dictionary lookup for known words (covers irregular + common cases)
    2. snowballstemmer if installed (Porter2 — good general coverage)
    3. Suffix rules as fallback (built-in, no deps)

    Returns original word + all plausible stems. Multiple candidates
    increase match probability without false positives.
    """
    word = word.lower()
    result = {word}

    # Layer 1: Dictionary — known roots (check even for short words)
    if word in _COMMON_ROOTS:
        result.add(_COMMON_ROOTS[word])

    if len(word) <= 3:
        return result

    # Layer 2: snowballstemmer (Porter2) — optional
    if _SNOWBALL is not None:
        stemmed = _SNOWBALL.stemWord(word)
        if len(stemmed) >= 3 and stemmed != word:
            result.add(stemmed)

    # Layer 3: Suffix rules — built-in fallback
    for suffix, replacement in _SUFFIX_RULES:
        if word.endswith(suffix):
            stem = word[:-len(suffix)] + replacement
            if len(stem) >= 3:
                result.add(stem)
                # Handle consonant doubling: debugg → debug, runn → run
                if (len(stem) >= 4
                        and stem[-1] == stem[-2]
                        and stem[-1] not in 'aeiou'):
                    result.add(stem[:-1])

    return result


def _tokenize(text: str) -> set[str]:
    """Extract all word forms from text as a flat set of candidates."""
    raw_tokens = re.findall(r'\b\w+\b', text.lower())
    all_forms = set()
    for token in raw_tokens:
        all_forms.update(_candidates(token))
    return all_forms


def _matches_term(tokens: set[str], term: str, cache: dict[str, set[str]] = None) -> bool:
    """Check if a term (concept or association) appears in the token set.

    Compares candidate sets for overlap — if any root form of the term
    matches any root form found in text, it's a hit.
    """
    if cache is not None and term in cache:
        term_candidates = cache[term]
    else:
        term_lower = term.lower().replace("_", " ")
        parts = term_lower.split()

        term_candidates = set()
        for part in parts:
            term_candidates.update(_candidates(part))
        
        if cache is not None:
            cache[term] = term_candidates

    return bool(term_candidates & tokens)


class WeightGate:
    """Passive weight update middleware — wraps a Synapse graph.

    Intercepts messages and updates concept weights automatically.
    Can be enabled/disabled at runtime. Single source of truth:
    all graph operations delegate to the underlying Synapse.

    Args:
        path: Path to the synapse JSON file (creates Synapse from this).
              Mutually exclusive with `synapse` param.
        synapse: An existing Synapse instance to wrap (shared state).
                 Mutually exclusive with `path` param.
        decay_rate: How much unused associations decay per interaction (default 0.01).
        boost_rate: How much mentioned associations strengthen (default 0.05).
        min_weight: Floor for association weights — decay stops here (default 0.1).
        enabled: Start enabled or disabled (default False — opt-in).

    Example:
        >>> gate = WeightGate("synapse.json", enabled=True)
        >>> gate.add_concept("coffee", {"cappuccino": 0.9, "americano": 0.3})
        >>> gate.process_input("I love cappuccino")
        >>> gate.top_associations("coffee")
        [('cappuccino', 0.95), ('americano', 0.29)]
    """

    def __init__(self, path: str = None, synapse: Synapse = None,
                 decay_rate: float = 0.01, boost_rate: float = 0.05,
                 min_weight: float = 0.1, enabled: bool = False):
        if path is not None and synapse is not None:
            raise ValueError("Provide either 'path' or 'synapse', not both")
        if path is None and synapse is None:
            raise ValueError("Provide either 'path' or 'synapse'")

        self._path = Path(path) if path else None
        self._synapse = synapse if synapse is not None else Synapse.load(str(self._path))
        self.decay_rate = decay_rate
        self.boost_rate = boost_rate
        self.min_weight = min_weight
        self._enabled = enabled
        self._interactions = 0
        self._created = time.time()
        self._term_token_cache: dict[str, set[str]] = {}

    @property
    def synapse(self) -> Synapse:
        """The underlying Synapse graph. Direct access for graph operations."""
        return self._synapse

    @property
    def path(self) -> Optional[Path]:
        """File path for persistence (None if wrapping an external Synapse)."""
        return self._path

    def _save(self):
        """Persist to file if we own the path."""
        if self._path:
            self._synapse.save(str(self._path))

    # ── Enable/Disable ───────────────────────────────────────────

    def enable(self) -> "WeightGate":
        """Enable automatic weight updates."""
        self._enabled = True
        return self

    def disable(self) -> "WeightGate":
        """Disable automatic weight updates (gate becomes a no-op)."""
        self._enabled = False
        return self

    @property
    def enabled(self) -> bool:
        return self._enabled

    def toggle(self) -> bool:
        """Toggle enabled state. Returns new state."""
        self._enabled = not self._enabled
        return self._enabled

    # ── Context Manager ──────────────────────────────────────────

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *args):
        self.disable()
        self._save()

    # ── Core Processing ──────────────────────────────────────────

    def process_input(self, text: str) -> dict:
        """Process user input — detect concepts and update weights.

        Uses word-boundary tokenization + multi-layer stemming to detect
        concepts and associations. Avoids false positives from substring
        matching (e.g., "debug" won't match "debugging" as substring —
        it matches via shared root "debug").

        Decay behavior: only fires when a concept AND at least one of its
        associations are both mentioned — meaning there was a competition
        and the unmentioned associations lost. Unmentioned concepts and
        their associations are left untouched.

        If gate is disabled, returns empty dict (no-op).
        """
        if not self._enabled:
            return {}

        tokens = _tokenize(text)
        self._interactions += 1
        detected = {}

        for concept in self._synapse.concepts():
            if not _matches_term(tokens, concept, self._term_token_cache):
                continue  # Concept not mentioned — skip entirely

            assocs = self._synapse.get_associations(concept)
            mentioned_assocs = []
            unmentioned_assocs = []

            for assoc, weight in assocs.items():
                if assoc.startswith("_"):
                    continue
                if _matches_term(tokens, assoc, self._term_token_cache):
                    mentioned_assocs.append(assoc)
                else:
                    unmentioned_assocs.append(assoc)

            # Boost mentioned associations
            if mentioned_assocs:
                print(f"DEBUG: Concept {concept} has mentioned_assocs: {mentioned_assocs}")
            for assoc in mentioned_assocs:
                self._synapse.strengthen(concept, assoc, self.boost_rate)
                detected.setdefault(concept, []).append(f"{assoc}↑")

            # Decay unmentioned ONLY if competition occurred
            # (concept mentioned + at least one association mentioned)
            if mentioned_assocs:
                for assoc in unmentioned_assocs:
                    current = self._synapse.get_weight(concept, assoc)
                    if current > self.min_weight:
                        new_w = max(self.min_weight, current - self.decay_rate)
                        self._synapse.set_weight(concept, assoc, new_w)
                        detected.setdefault(concept, []).append(f"{assoc}↓")

        self._save()
        return detected

    def process_output(self, text: str) -> dict:
        """Process agent output — strengthen concepts that were actually used.

        Uses word-boundary tokenization + multi-layer stemming for detection.
        If gate is disabled, returns empty dict (no-op).
        """
        if not self._enabled:
            return {}

        tokens = _tokenize(text)
        strengthened = {}

        for concept in self._synapse.concepts():
            for assoc, weight in self._synapse.get_associations(concept).items():
                if assoc.startswith("_"):
                    continue
                if _matches_term(tokens, assoc, self._term_token_cache):
                    new_w = self._synapse.strengthen(
                        concept, assoc, self.boost_rate * 0.5
                    )
                    strengthened.setdefault(concept, []).append(
                        f"{assoc}→{new_w:.2f}"
                    )

        self._save()
        return strengthened

    def process_conversation(self, user_msg: str, agent_response: str) -> dict:
        """Full cycle: process input + output in one call.

        If gate is disabled, returns empty dict (no-op).
        """
        if not self._enabled:
            return {}

        input_changes = self.process_input(user_msg)
        output_changes = self.process_output(agent_response)

        return {
            "input": input_changes,
            "output": output_changes,
            "total_interactions": self._interactions,
        }

    # ── Delegated Graph Operations ───────────────────────────────
    # All delegate to the underlying Synapse — single source of truth.

    def add_concept(self, concept: str, associations: dict[str, float]) -> None:
        """Add a new concept with weighted associations."""
        self._synapse.link(concept, list(associations.keys()), weights=associations)
        self._term_token_cache.clear()
        self._save()

    def remove_concept(self, concept: str) -> bool:
        """Remove a concept and all its links. Returns True if found."""
        removed = self._synapse.remove_concept(concept)
        if removed:
            self._term_token_cache.clear()
            self._save()
        return removed

    def get_weights(self, concept: str) -> dict:
        """Get current weights for a concept."""
        return self._synapse.get_associations(concept)

    def top_associations(self, concept: str, limit: int = 5) -> list:
        """Get top associations by weight (descending)."""
        return self._synapse.top_associations(concept, limit)

    def set_weight(self, concept: str, assoc: str, weight: float) -> None:
        """Manually set a weight."""
        self._synapse.set_weight(concept, assoc, weight)
        self._save()

    def strengthen(self, concept: str, assoc: str, boost: float = None) -> float:
        """Manually strengthen an association."""
        boost = boost if boost is not None else self.boost_rate
        return self._synapse.strengthen(concept, assoc, boost)

    def weaken(self, concept: str, assoc: str, decay: float = None) -> float:
        """Manually weaken an association."""
        decay = decay if decay is not None else self.decay_rate
        return self._synapse.weaken(concept, assoc, decay)

    # ── Import/Export ─────────────────────────────────────────────

    def export_compact(self) -> str:
        """Export synapse weights in compact JSON format."""
        compact = {}
        for concept in self._synapse.concepts():
            assocs = self._synapse.get_associations(concept)
            items = {k: round(v, 2) for k, v in assocs.items() if not k.startswith("_")}
            if items:
                compact[concept] = items
        return json.dumps(compact, separators=(",", ":"))

    # ── Stats ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get gate statistics."""
        concepts = self._synapse.concepts()
        total_assocs = sum(
            len([k for k in self._synapse.get_associations(c) if not k.startswith("_")])
            for c in concepts
        )
        return {
            "enabled": self._enabled,
            "concepts": len(concepts),
            "total_associations": total_assocs,
            "interactions": self._interactions,
            "file": str(self._path) if self._path else "<shared synapse>",
        }

    def __repr__(self):
        state = "ON" if self._enabled else "OFF"
        stats = self.get_stats()
        return (f"WeightGate([{state}] concepts={stats['concepts']}, "
                f"interactions={stats['interactions']})")
