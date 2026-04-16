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
import time
from pathlib import Path
from typing import Optional, Union

from .synapse import Synapse


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

        If gate is disabled, returns empty dict (no-op).
        """
        if not self._enabled:
            return {}

        text_lower = text.lower()
        self._interactions += 1
        detected = {}

        for concept, assocs in self._synapse._weights.items():
            concept_mentioned = concept.lower() in text_lower

            if concept_mentioned:
                for assoc, weight in list(assocs.items()):
                    if assoc.startswith("_"):
                        continue
                    if assoc.lower() in text_lower:
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

        If gate is disabled, returns empty dict (no-op).
        """
        if not self._enabled:
            return {}

        text_lower = text.lower()
        strengthened = {}

        for concept, assocs in self._synapse._weights.items():
            for assoc, weight in list(assocs.items()):
                if assoc.startswith("_"):
                    continue
                if assoc.lower().replace("_", " ") in text_lower:
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
