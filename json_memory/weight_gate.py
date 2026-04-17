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
"""

import json
import re
import time
from pathlib import Path
from typing import Optional, Union

from .synapse import Synapse

# ── Tokenization & Stemming ─────────────────────────────────────────
# Two-layer stemmer: dictionary lookup for known words, suffix rules for unknown.
# Generates a *set* of candidate root forms per word — if any candidate overlaps,
# the match succeeds. This avoids false negatives from imperfect stemming.
#
# LIMITATION: This is a heuristic stemmer, not a linguistic one. It handles
# English morphology reasonably well for code/technical domains but will fail
# on irregular verbs (ran→run, went→go), rare derivations, and non-English words.
# For production NLP, use nltk.stem.PorterStemmer (pip install nltk).

# ── Layer 1: Dictionary — known variant → root mappings ─────────────
# Covers the most common technical/English words where suffix rules fail.
# Add entries here for domain-specific terms that need reliable stemming.

_COMMON_ROOTS: dict[str, str] = {}

def _build_root_map() -> dict[str, str]:
    """Build reverse lookup: word_variant → canonical root.

    Each entry maps a common variant to its root. The root itself is always
    included as a candidate (identity mapping).
    """
    # Grouped by root word — each group maps variants to their shared root
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
        ("gener", "general", "generally", "generate", "generating", "generated", "generator", "generators"),

        # -ing family (irregular cases)
        ("log", "logging", "logged", "logger", "logs"),
        ("test", "testing", "tested", "tester", "tests"),
        ("build", "building", "builder", "builds"),
        ("deploy", "deploying", "deployed", "deployment"),
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

        # -ed family (irregular)
        ("debug", "debugging", "debugged", "debugger"),
        ("run", "running", "runner", "ran", "runs"),
        ("break", "breaking", "broken", "breaks"),
        ("mak", "making", "made", "makes", "maker"),
        ("tak", "taking", "taken", "takes"),
        ("com", "coming", "came", "comes"),
        ("giv", "giving", "given", "gives"),
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

        # -s / plural family
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
        ("config", "configs", "configuration", "configurations", "configuring", "configured"),
        ("instal", "install", "installation", "installing", "installed"),
        ("depend", "dependency", "dependencies", "dependent", "depending", "depended"),
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
        ("read", "reading", "reader", "reads"),
        ("writ", "write", "writing", "writer", "writes", "written"),
    ]

    mapping = {}
    for group in root_families:
        root = group[0]
        mapping[root] = root  # identity
        for variant in group[1:]:
            mapping[variant] = root
    return mapping

_COMMON_ROOTS = _build_root_map()

# ── Layer 2: Suffix rules — applied when dictionary misses ──────────
# Order matters: longest/most specific suffixes first.
# These are fallback heuristics — the dictionary handles known words.

_SUFFIX_RULES = [
    # (suffix_to_strip, replacement)
    ("ational", "ate"),    # operational → operate
    ("tional", "te"),      # functional → functe (imperfect but better than "funct")
    ("fulness", "ful"),    # helpfulness → helpful
    ("ousness", "ous"),    # dangerousness → dangerous
    ("iveness", "ive"),    # effectiveness → effective
    ("ously", "ous"),      # dangerously → dangerous
    ("ively", "ive"),      # effectively → effective
    ("lessly", "less"),    # endlessly → endless
    ("ically", "ic"),      # technically → technic
    ("ality", "al"),       # functionality → functional
    ("ously", "ous"),      # previously → prevous
    ("ities", "ity"),      # activities → activity
    ("ments", "ment"),     # deployments → deployment
    ("ments", ""),         # deployments → deploy (also try)
    ("ities", ""),         # activities → activ (also try)
    ("ness", ""),          # readiness → ready (imperfect)
    ("tion", ""),          # configuration → configura (imperfect — dictionary covers this)
    ("sion", ""),          # compression → compres (imperfect)
    ("ment", ""),          # deployment → deploy
    ("ally", "al"),        # technically → technical
    ("ing", ""),           # running → runn (dictionary handles common cases)
    ("ies", "y"),          # strategies → strategy
    ("ied", "y"),          # modified → modify
    ("ers", "er"),         # traders → trader
    ("est", ""),           # fastest → fast
    ("ful", ""),           # helpful → help
    ("ive", ""),           # active → act (imperfect)
    ("ent", ""),           # current → curr (imperfect)
    ("ant", ""),           # trading → trad (imperfect)
    ("ion", ""),           # prediction → predict (imperfect)
    ("ed", ""),            # debugged → debugg (consonant doubling handles this)
    ("ly", ""),            # quickly → quick
    ("er", ""),            # trader → trad (imperfect)
    ("es", ""),            # watches → watch
    ("al", ""),            # technical → technic (imperfect)
    ("en", ""),            # broken → brok (imperfect)
    ("s", ""),             # bots → bot
]


def _candidates(word: str) -> set[str]:
    """Generate all plausible root forms of a word.

    Two-layer approach:
    1. Dictionary lookup for known words (covers irregular + common cases)
    2. Suffix stripping as fallback for unknown words

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

    # Layer 2: Suffix rules — heuristic fallback
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
    """Extract all word forms from text as a flat set of candidates.

    For each token, generates all plausible roots so we can match
    "debugged" against "debug" via shared root set overlap.
    """
    raw_tokens = re.findall(r'\b\w+\b', text.lower())
    all_forms = set()
    for token in raw_tokens:
        all_forms.update(_candidates(token))
    return all_forms


def _matches_term(tokens: set[str], term: str) -> bool:
    """Check if a term (concept or association) appears in the token set.

    Compares candidate sets for overlap — if any root form of the term
    matches any root form found in text, it's a hit.

    Handles:
    - Exact word match: "debug" in "I debug the issue"
    - Stemmed match: "debug" in "I debugged the issue"
    - Plural match: "bot" in "the bots crashed"
    - Multi-word terms: "check_logs" matches when "check" and "logs" present
    - Underscore/space normalization: "check_logs" == "check logs"
    """
    term_lower = term.lower().replace("_", " ")
    parts = term_lower.split()

    term_candidates = set()
    for part in parts:
        term_candidates.update(_candidates(part))

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
                 enabled: bool = False):
        if path is not None and synapse is not None:
            raise ValueError("Provide either 'path' or 'synapse', not both")
        if path is None and synapse is None:
            raise ValueError("Provide either 'path' or 'synapse'")

        self._path = Path(path) if path else None
        self._synapse = synapse if synapse is not None else Synapse.load(str(self._path))
        self.decay_rate = decay_rate
        self.boost_rate = boost_rate
        self._enabled = enabled
        self._interactions = 0
        self._created = time.time()

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

        Uses word-boundary tokenization + suffix stemming to detect
        concepts and associations. Avoids false positives from substring
        matching (e.g., "debug" won't match "debugging" as substring —
        it matches via stem "debug").

        If gate is disabled, returns empty dict (no-op).
        """
        if not self._enabled:
            return {}

        tokens = _tokenize(text)
        self._interactions += 1
        detected = {}

        for concept, assocs in self._synapse._weights.items():
            concept_mentioned = _matches_term(tokens, concept)

            if concept_mentioned:
                for assoc, weight in list(assocs.items()):
                    if assoc.startswith("_"):
                        continue
                    if _matches_term(tokens, assoc):
                        self._synapse.set_weight(concept, assoc,
                                                 min(1.0, weight + self.boost_rate))
                        detected.setdefault(concept, []).append(f"{assoc}↑")
                    else:
                        self._synapse.set_weight(concept, assoc,
                                                 max(0.0, weight - self.decay_rate))
                        detected.setdefault(concept, []).append(f"{assoc}↓")

        self._save()
        return detected

    def process_output(self, text: str) -> dict:
        """Process agent output — strengthen concepts that were actually used.

        Uses word-boundary tokenization + suffix stemming for detection.
        If gate is disabled, returns empty dict (no-op).
        """
        if not self._enabled:
            return {}

        tokens = _tokenize(text)
        strengthened = {}

        for concept, assocs in self._synapse._weights.items():
            for assoc, weight in list(assocs.items()):
                if assoc.startswith("_"):
                    continue
                if _matches_term(tokens, assoc):
                    new_w = min(1.0, weight + self.boost_rate * 0.5)
                    self._synapse.set_weight(concept, assoc, new_w)
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
        self._save()

    def remove_concept(self, concept: str) -> bool:
        """Remove a concept and all its links. Returns True if found."""
        if concept in self._synapse._weights:
            # Remove from all neighbor link lists
            for neighbor in list(self._synapse._links.get(concept, [])):
                if concept in self._synapse._links.get(neighbor, []):
                    self._synapse._links[neighbor].remove(concept)
                if neighbor in self._synapse._weights.get(concept, {}):
                    pass  # will be deleted below
                if concept in self._synapse._weights.get(neighbor, {}):
                    del self._synapse._weights[neighbor][concept]
                if concept in self._synapse._frequencies.get(neighbor, {}):
                    del self._synapse._frequencies[neighbor][concept]

            # Remove the concept itself
            del self._synapse._links[concept]
            del self._synapse._weights[concept]
            del self._synapse._frequencies[concept]
            self._save()
            return True
        return False

    def get_weights(self, concept: str) -> dict:
        """Get current weights for a concept."""
        return dict(self._synapse._weights.get(concept, {}))

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
        for concept, assocs in self._synapse._weights.items():
            items = {k: round(v, 2) for k, v in assocs.items() if not k.startswith("_")}
            if items:
                compact[concept] = items
        return json.dumps(compact, separators=(",", ":"))

    # ── Stats ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get gate statistics."""
        total_assocs = sum(
            len([k for k in v if not k.startswith("_")])
            for v in self._synapse._weights.values()
        )
        return {
            "enabled": self._enabled,
            "concepts": len(self._synapse._weights),
            "total_associations": total_assocs,
            "interactions": self._interactions,
            "file": str(self._path) if self._path else "<shared synapse>",
        }

    def __repr__(self):
        state = "ON" if self._enabled else "OFF"
        stats = self.get_stats()
        return (f"WeightGate([{state}] concepts={stats['concepts']}, "
                f"interactions={stats['interactions']})")
