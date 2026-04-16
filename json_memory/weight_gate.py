"""
WeightGate — Passive middleware for automatic synapse weight updates.

Sits between input/output and updates weights transparently.
No explicit tool calls needed.

Usage:
    from json_memory import WeightGate

    # Enable the gate
    gate = WeightGate("/path/to/synapse.json")
    gate.enable()

    # Process messages (weights auto-update)
    gate.process_input("How do I restart the bot?")
    gate.process_output("Run: kill && nohup ./bot > log")

    # Disable when not needed
    gate.disable()

    # Or use as context manager
    with WeightGate("/path/to/synapse.json") as gate:
        gate.process_conversation(user_msg, agent_response)
"""

import json
import os
import time
from pathlib import Path
from typing import Optional


class WeightGate:
    """Passive weight update middleware for synapse memory.

    Intercepts messages and updates concept weights automatically.
    Can be enabled/disabled at runtime.

    Args:
        path: Path to the synapse JSON file.
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

    def __init__(self, path: str, decay_rate: float = 0.01,
                 boost_rate: float = 0.05, enabled: bool = False):
        self.path = Path(path)
        self.decay_rate = decay_rate
        self.boost_rate = boost_rate
        self._enabled = enabled
        self.synapse = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {"_meta": {"created": time.time(), "interactions": 0}, "concepts": {}}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.synapse, f, separators=(",", ":"))

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
        self.synapse["_meta"]["interactions"] += 1
        detected = {}

        for concept, assocs in self.synapse.get("concepts", {}).items():
            concept_mentioned = concept.lower() in text_lower

            if concept_mentioned:
                for assoc, weight in list(assocs.items()):
                    if assoc.startswith("_"):
                        continue
                    if assoc.lower() in text_lower:
                        assocs[assoc] = min(1.0, weight + self.boost_rate)
                        detected.setdefault(concept, []).append(f"{assoc}↑")
                    else:
                        assocs[assoc] = max(0.0, weight - self.decay_rate)
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

        for concept, assocs in self.synapse.get("concepts", {}).items():
            for assoc, weight in list(assocs.items()):
                if assoc.startswith("_"):
                    continue
                if assoc.lower().replace("_", " ") in text_lower:
                    assocs[assoc] = min(1.0, weight + self.boost_rate * 0.5)
                    strengthened.setdefault(concept, []).append(
                        f"{assoc}→{assocs[assoc]:.2f}"
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
            "total_interactions": self.synapse["_meta"]["interactions"],
        }

    # ── Management ────────────────────────────────────────────────

    def add_concept(self, concept: str, associations: dict[str, float]) -> None:
        """Add a new concept with weighted associations."""
        if "concepts" not in self.synapse:
            self.synapse["concepts"] = {}
        self.synapse["concepts"][concept] = associations
        self._save()

    def remove_concept(self, concept: str) -> bool:
        """Remove a concept. Returns True if found and removed."""
        if concept in self.synapse.get("concepts", {}):
            del self.synapse["concepts"][concept]
            self._save()
            return True
        return False

    def get_weights(self, concept: str) -> dict:
        """Get current weights for a concept."""
        return self.synapse.get("concepts", {}).get(concept, {})

    def top_associations(self, concept: str, limit: int = 5) -> list:
        """Get top associations by weight (descending)."""
        assocs = self.get_weights(concept)
        ranked = sorted(
            [(k, v) for k, v in assocs.items() if not k.startswith("_")],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:limit]

    def set_weight(self, concept: str, assoc: str, weight: float) -> None:
        """Manually set a weight."""
        if concept not in self.synapse.get("concepts", {}):
            self.synapse["concepts"][concept] = {}
        self.synapse["concepts"][concept][assoc] = max(0.0, min(1.0, weight))
        self._save()

    def strengthen(self, concept: str, assoc: str, boost: float = None) -> float:
        """Manually strengthen an association."""
        boost = boost or self.boost_rate
        current = self.get_weights(concept).get(assoc, 0.5)
        new = min(1.0, current + boost)
        self.set_weight(concept, assoc, new)
        return new

    def weaken(self, concept: str, assoc: str, decay: float = None) -> float:
        """Manually weaken an association."""
        decay = decay or self.decay_rate
        current = self.get_weights(concept).get(assoc, 0.5)
        new = max(0.0, current - decay)
        self.set_weight(concept, assoc, new)
        return new

    # ── Import/Export ─────────────────────────────────────────────

    def export_compact(self) -> str:
        """Export synapse data in compact JSON format."""
        compact = {}
        for concept, assocs in self.synapse.get("concepts", {}).items():
            compact[concept] = {
                k: round(v, 2)
                for k, v in assocs.items()
                if not k.startswith("_")
            }
        return json.dumps(compact, separators=(",", ":"))

    def import_from_synapse(self, synapse_obj) -> int:
        """Import concepts from a json_memory.Synapse instance.

        Copies all weighted associations into the gate.
        Returns number of concepts imported.
        """
        count = 0
        for concept, weights in synapse_obj._weights.items():
            self.add_concept(concept, dict(weights))
            count += 1
        return count

    # ── Stats ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get gate statistics."""
        concepts = self.synapse.get("concepts", {})
        total_assocs = sum(
            len([k for k in v if not k.startswith("_")])
            for v in concepts.values()
        )
        return {
            "enabled": self._enabled,
            "concepts": len(concepts),
            "total_associations": total_assocs,
            "interactions": self.synapse["_meta"]["interactions"],
            "file": str(self.path),
        }

    def __repr__(self):
        state = "ON" if self._enabled else "OFF"
        stats = self.get_stats()
        return (f"WeightGate([{state}] concepts={stats['concepts']}, "
                f"interactions={stats['interactions']})")
