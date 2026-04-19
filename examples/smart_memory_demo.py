"""
SmartMemory — Complete example showing intelligent agent memory.

Demonstrates:
1. Weighted retrieval (only relevant facts, not everything)
2. Auto-extraction from conversation
3. Associative memory
4. Prompt injection (lean, targeted)
5. Score explanation (debug)
"""

from json_memory import SmartMemory
import time


def main():
    # Initialize with persistence
    mem = SmartMemory("demo_smart.json", max_chars=5000)

    print("=" * 60)
    print("🧠 SmartMemory Demo — Intelligent Agent Memory")
    print("=" * 60)

    # ── 1. Store facts ────────────────────────────────────────────
    print("\n📦 Storing facts...")

    mem.remember("user.name", "Alice")
    mem.remember("user.handle", "@alice")
    mem.remember("user.timezone", "GMT+7")
    mem.remember("user.location", "Jakarta, Indonesia")
    mem.remember("user.preferences.style", "direct and technical")
    mem.remember("user.preferences.language", "Python")
    mem.remember("project.name", "json-memory")
    mem.remember("project.repo", "github.com/dioncx/json-memory")
    mem.remember("project.stack", ["Python", "FAISS", "sentence-transformers"])
    mem.remember("bot.name", "Veludra")
    mem.remember("bot.exchange", "Binance")
    mem.remember("bot.strategy", "RSI + MACD + Smart Money")
    mem.remember("bot.watchlist", ["BNB", "KITE", "AGLD"])
    mem.remember("server.ip", "192.168.1.100")
    mem.remember("server.os", "Ubuntu 24.04")
    mem.remember("server.restart_cmd", "systemctl restart nginx")

    print(f"  Stored {len(mem.mem.paths())} facts")

    # ── 2. Smart retrieval (the key feature) ──────────────────────
    print("\n🔍 Smart Retrieval — only relevant facts per query:")
    print("-" * 50)

    queries = [
        "What's my timezone?",
        "How do I restart the server?",
        "What's the bot trading?",
        "Who am I?",
        "What's the project about?",
    ]

    for query in queries:
        relevant = mem.recall_relevant(query)
        print(f"\n  Q: {query}")
        for path, value in relevant.items():
            print(f"     → {path}: {value}")

    # ── 3. Prompt injection comparison ────────────────────────────
    print("\n\n📝 Prompt Context — what goes into the LLM:")
    print("-" * 50)

    query = "How do I restart the server?"

    # BAD: inject everything (what most agents do)
    full_context = mem.context()
    print(f"\n  ❌ Full memory injection: {len(full_context)} chars")
    print(f"     {full_context[:120]}...")

    # GOOD: only relevant facts
    smart_context = mem.prompt_context(query)
    print(f"\n  ✅ Smart injection: {len(smart_context)} chars")
    print(f"     {smart_context}")

    savings = 1 - len(smart_context) / len(full_context)
    print(f"\n  💰 Token savings: {savings:.0%}")

    # ── 4. Auto-extraction ───────────────────────────────────────
    print("\n🤖 Auto-Extraction — facts detected from conversation:")
    print("-" * 50)

    conversations = [
        ("My name is Bob and I live in Tokyo", None),
        ("I prefer to use Go for backend services", None),
        ("Remember that the deployment key is sk-abc123", None),
        ("My timezone is UTC+9", None),
    ]

    for user_msg, agent_msg in conversations:
        extracted = mem.process_conversation(user_msg, agent_msg)
        print(f"\n  User: \"{user_msg}\"")
        for ext in extracted:
            print(f"     → {ext['path']}: {ext['value']} (confidence: {ext['confidence']:.0%})")

    # ── 5. Score explanation ──────────────────────────────────────
    print("\n🔬 Score Explanation — how relevance is calculated:")
    print("-" * 50)

    # Access bot.exchange a few times to boost its frequency
    for _ in range(5):
        mem.recall("bot.exchange")

    explanation = mem.explain_score("bot.exchange", query="What exchange does the bot use?")
    print(f"\n  Path: {explanation['path']}")
    print(f"  Recency:    {explanation['recency']}")
    print(f"  Frequency:  {explanation['frequency']}")
    print(f"  Keyword:    {explanation['keyword_relevance']}")
    print(f"  Final:      {explanation['final_score']}")
    print(f"  Accesses:   {explanation['access_count']}")

    # ── 6. Associative memory ────────────────────────────────────
    print("\n🧠 Associative Memory — concept linking:")
    print("-" * 50)

    mem.link("debugging", ["check_logs", "reproduce", "git_bisect", "ask_user"])
    mem.link("check_logs", ["tail_log", "grep_errors", "check_journal"])
    mem.link("deploy", ["test", "build", "push", "verify"])

    print(f"\n  Think 'debugging': {mem.associate('debugging')}")
    print(f"  Think 'debugging' (depth=2): {mem.associate('debugging', depth=2)}")
    print(f"  Think 'deploy': {mem.associate('deploy')}")

    # ── 7. Stats ─────────────────────────────────────────────────
    print("\n📊 Memory Stats:")
    print("-" * 50)
    stats = mem.stats()
    print(f"  Entries:     {stats['entries']}")
    print(f"  Chars used:  {stats['chars_used']}/{stats['chars_max']}")
    print(f"  Paths:       {stats['paths']}")
    print(f"  Top scored:  {stats['top_scored']}")

    # ── 8. End-to-end agent loop simulation ──────────────────────
    print("\n\n🔄 End-to-End Agent Loop Simulation:")
    print("=" * 60)

    # Fresh memory for clean demo
    agent = SmartMemory("demo_agent.json", max_chars=3000)
    agent.remember("user.name", "Alice")
    agent.remember("user.timezone", "GMT+7")
    agent.remember("bot.restart_cmd", "kill && nohup ./bot > log 2>&1")
    agent.remember("bot.symbol", "BNBUSDT")
    agent.remember("server.ip", "10.0.0.1")

    # Simulate user messages
    messages = [
        "Hi, what timezone am I in?",
        "How do I restart the bot?",
        "I prefer to use Docker btw",
        "What's my server IP?",
    ]

    for msg in messages:
        # Auto-extract
        extracted = agent.process_conversation(msg)
        if extracted:
            print(f"\n  📝 Extracted: {extracted}")

        # Smart recall
        relevant = agent.recall_relevant(msg)
        context = agent.prompt_context(msg)

        print(f"\n  👤 User: {msg}")
        print(f"  🧠 Relevant facts: {list(relevant.keys())}")
        print(f"  📋 Prompt context ({len(context)} chars):")
        print(f"     {context}")

    print("\n\n✅ Done! SmartMemory demo complete.")
    print(f"   Core: 0 dependencies")
    print(f"   Upgrade: pip install json-memory[semantic]")


if __name__ == "__main__":
    main()
