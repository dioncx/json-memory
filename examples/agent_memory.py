"""
Agent Memory Layer — Drop-in memory system for any LLM agent.

Copy this into your project. Works with Claude, GPT, OpenClaw, or any agent loop.

Usage:
    memory = AgentMemory("my_agent_memory.json")
    memory.remember("user.timezone", "GMT+7")
    print(memory.recall("user.timezone"))  # → "GMT+7"
"""

from json_memory import Memory, Synapse
import json
import os
from pathlib import Path


class AgentMemory:
    """Structured memory layer for AI agents. Persists to disk, survives restarts."""

    def __init__(self, path: str = "agent_memory.json", max_chars: int = 5000):
        self.path = Path(path)
        self.max_chars = max_chars

        # Load existing memory or start fresh
        if self.path.exists():
            raw = self.path.read_text(encoding="utf-8")
            self.mem = Memory.from_json(raw, max_chars=max_chars)
        else:
            self.mem = Memory(max_chars=max_chars)

        # Associative memory for concept linking
        self.brain = Synapse()

    def remember(self, path: str, value, ttl: int = None):
        """Store a fact. Use dotted paths: 'user.name', 'project.status'"""
        self.mem.set(path, value, ttl=ttl)
        self._flush()

    def recall(self, path: str, default=None):
        """Retrieve a fact by dotted path."""
        return self.mem.get(path, default=default)

    def forget(self, path: str):
        """Delete a fact."""
        self.mem.delete(path, prune=True)
        self._flush()

    def search(self, pattern: str) -> dict:
        """Find facts matching a glob pattern: 'user.*', '**.status'"""
        return self.mem.find(pattern)

    def context(self) -> str:
        """Export memory as compact JSON string for injection into LLM prompts."""
        return self.mem.export()

    def link(self, concept: str, associations: list[str], weights: dict = None):
        """Create concept associations for associative recall."""
        self.brain.link(concept, associations, weights=weights)

    def associate(self, concept: str, depth: int = 1) -> list[str]:
        """Recall associated concepts (like thinking 'coffee' → 'morning', 'energy')."""
        return self.brain.activate(concept, depth=depth)

    def snapshot(self, label: str):
        """Save state before risky operations. Roll back with rollback()."""
        self.mem.snapshot(label)

    def rollback(self, label: str):
        """Restore state from a snapshot."""
        self.mem.rollback(label)
        self._flush()

    def stats(self) -> dict:
        """Memory usage stats."""
        return self.mem.stats()

    def _flush(self):
        """Persist to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.mem.export(), encoding="utf-8")


# ─── Example: Basic Agent Loop Integration ─────────────────────────────

if __name__ == "__main__":
    # Initialize memory (persists across runs)
    memory = AgentMemory("my_agent.json", max_chars=3000)

    # ── User Profile ──
    memory.remember("user.name", "Alice")
    memory.remember("user.handle", "@alice")
    memory.remember("user.timezone", "UTC+7")
    memory.remember("user.preferences.style", "direct and technical")
    memory.remember("user.preferences.language", "Python")

    # ── Session State ──
    memory.remember("session.current_task", "reviewing PR #42")
    memory.remember("session.mood", "focused")

    # ── Skills & Knowledge ──
    memory.remember("skills.learned", ["git-rebase", "docker-compose", "sql-joins"])
    memory.remember("project.repo", "github.com/alice/awesome-app")
    memory.remember("project.stack", ["FastAPI", "PostgreSQL", "Redis"])

    # ── TTL: Ephemeral data that auto-expires ──
    memory.remember("session.temp_token", "xyz_123", ttl=300)  # expires in 5 min

    # ── Associative Memory ──
    memory.link("debugging", ["check_logs", "reproduce", "git_bisect", "ask_user"])
    memory.link("deploy", ["test", "build", "push", "verify"])
    memory.link("debugging", ["check_logs", "reproduce"],
                weights={"check_logs": 0.9, "reproduce": 0.7})

    # ── Recall ──
    print(f"User: {memory.recall('user.name')}")          # → "Alice"
    print(f"Task: {memory.recall('session.current_task')}")  # → "reviewing PR #42"
    print(f"Token: {memory.recall('session.temp_token')}")   # → "xyz_123" (for 5 min)

    # ── Associative Recall ──
    print(f"When debugging, first: {memory.associate('debugging')}")
    # → ["check_logs", "reproduce", "git_bisect", "ask_user"]

    # ── Search ──
    user_facts = memory.search("user.*")
    print(f"All user facts: {user_facts}")
    # → {"user.name": "Alice", "user.handle": "@alice", "user.timezone": "UTC+7", ...}

    # ── Inject into LLM Prompt ──
    system_prompt = f"""You are a helpful assistant.

## Memory
{memory.context()}

Use the memory above to personalize your responses."""

    print(f"\n--- Prompt-ready context ({len(memory.context())} chars) ---")
    print(memory.context())

    # ── Snapshot/Rollback ──
    memory.snapshot("before_experiment")
    memory.remember("session.current_task", "testing something risky")
    # ... something goes wrong ...
    memory.rollback("before_experiment")
    print(f"After rollback: {memory.recall('session.current_task')}")
    # → "reviewing PR #42"

    # ── Stats ──
    print(f"\nMemory stats: {memory.stats()}")
