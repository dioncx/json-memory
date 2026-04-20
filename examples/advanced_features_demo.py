"""
SmartMemory — Advanced features demo.

Demonstrates:
1. Contradiction detection
2. Memory consolidation
3. Forgetting curve & reinforcement
4. Memory strength analysis
"""

from json_memory import SmartMemory
import time


def main():
    # Initialize with persistence
    mem = SmartMemory("demo_advanced.json", max_chars=5000)

    print("=" * 60)
    print("🧠 SmartMemory Advanced Features Demo")
    print("=" * 60)

    # ── 1. Contradiction Detection ─────────────────────────────────
    print("\n⚠️  Contradiction Detection:")
    print("-" * 50)

    # Store initial facts
    mem.remember("user.status", "active")
    mem.remember("bot.running", "yes")
    mem.remember("server.power", "on")

    print("  Stored initial facts:")
    print("    user.status = 'active'")
    print("    bot.running = 'yes'")
    print("    server.power = 'on'")

    # Try to store contradictory facts
    print("\n  Attempting to store contradictory facts:")

    # Same path - this is an update, not a contradiction
    result1 = mem.remember("user.status", "inactive", check_contradictions=True)
    print(f"    user.status = 'inactive': {len(result1['contradictions'])} contradictions (update)")

    # Different attributes - no contradiction
    result2 = mem.remember("user.state", "inactive", check_contradictions=True)
    print(f"    user.state = 'inactive': {len(result2['contradictions'])} contradictions (different attribute)")

    # Get all contradictions
    contradictions = mem.get_contradictions()
    print(f"\n  Total contradictions in memory: {len(contradictions)}")

    # ── 2. Memory Consolidation ────────────────────────────────────
    print("\n\n🔄 Memory Consolidation:")
    print("-" * 50)

    # Store similar facts
    mem.remember("user.skills", "Python, Go, JavaScript", check_contradictions=False)
    mem.remember("user.programming_languages", "Python, Go, JavaScript, Rust", check_contradictions=False)
    mem.remember("user.known_languages", "Python, TypeScript, Go", check_contradictions=False)

    print("  Stored similar facts:")
    print("    user.skills = 'Python, Go, JavaScript'")
    print("    user.programming_languages = 'Python, Go, JavaScript, Rust'")
    print("    user.known_languages = 'Python, TypeScript, Go'")

    # Get consolidation suggestions
    groups = mem.consolidate_memory()
    print(f"\n  Found {len(groups)} consolidation groups:")

    for i, group in enumerate(groups, 1):
        print(f"\n    Group {i}:")
        print(f"      Paths: {group.paths}")
        print(f"      Suggested: {group.suggested_path} = {group.suggested_value}")
        print(f"      Confidence: {group.confidence:.2f}")
        print(f"      Reason: {group.reason}")

    # Auto-consolidate high-confidence groups
    if groups:
        result = mem.auto_consolidate(min_confidence=0.5)
        print(f"\n  Auto-consolidated {len(result['consolidated'])} groups")

    # ── 3. Forgetting Curve & Reinforcement ────────────────────────
    print("\n\n📉 Forgetting Curve & Reinforcement:")
    print("-" * 50)

    # Store facts with different types
    mem.remember("user.name", "Alice", tags=["identity"])
    mem.remember("meeting.time", "3pm", tags=["event"])
    mem.remember("project.deadline", "next week", tags=["fact"])
    mem.remember("temp.cache", "some data", tags=["temporary"])

    print("  Stored facts with different memory types:")
    print("    user.name (identity) - slow decay")
    print("    meeting.time (event) - fast decay")
    print("    project.deadline (fact) - medium decay")
    print("    temp.cache (temporary) - very fast decay")

    # Analyze memory strength
    print("\n  Memory strength analysis:")
    for path in ["user.name", "meeting.time", "project.deadline", "temp.cache"]:
        strength = mem.get_memory_strength(path)
        if strength:
            print(f"    {path}: {strength.current_strength:.3f} strength")

    # Get memories needing reinforcement
    print("\n  Memories needing reinforcement:")
    memories = mem.get_memories_needing_reinforcement(max_items=3)
    for memory in memories:
        print(f"    {memory['path']}: priority={memory['reinforcement_priority']:.3f}")

    # Reinforce a memory
    print("\n  Reinforcing user.name...")
    result = mem.reinforce_memory("user.name", boost_strength=0.3)
    print(f"    Success: {result['success']}, new strength: {result['new_strength']:.3f}")

    # Simulate memory decay
    print("\n  Simulating memory decay for project.deadline (30 days):")
    simulation = mem.simulate_memory_decay("project.deadline", days=30)
    for day_data in [simulation[0], simulation[7], simulation[14], simulation[29]]:
        print(f"    Day {day_data['day']}: {day_data['strength']:.3f} ({day_data['percent_remaining']:.1f}%)")

    # ── 4. Memory Health Check ─────────────────────────────────────
    print("\n\n🏥 Memory Health Check:")
    print("-" * 50)

    # Get lifecycle stats
    stats = mem.lifecycle_stats()
    print(f"  Total facts: {stats['total_facts']}")
    print(f"  Memory size: {stats['total_chars']} chars")
    print(f"  Average age: {stats['avg_age_seconds']:.1f} seconds")
    print(f"  Health: {stats['memory_health']}")

    # Get memory stats
    mem_stats = mem.stats()
    print(f"\n  Current usage: {mem_stats['chars_used']}/{mem_stats['chars_max']} chars")
    print(f"  Utilization: {mem_stats['utilization']}")

    # ── 5. End-to-End Workflow ─────────────────────────────────────
    print("\n\n🔄 End-to-End Workflow:")
    print("=" * 60)

    # Simulate a real agent workflow
    agent = SmartMemory("demo_workflow.json", max_chars=3000)

    # Store user identity (protected)
    agent.remember("user.name", "Bob", protected=True, tags=["identity"])
    agent.remember("user.timezone", "UTC+8", protected=True, tags=["identity"])

    # Store project info
    agent.remember("project.name", "Trading Bot")
    agent.remember("project.status", "active")

    # Store temporary info
    agent.remember("session.last_query", "What's the market price?")
    agent.remember("cache.price_btc", "$45,000", ttl=3600)  # Expires in 1 hour

    print("  Agent memory initialized with:")
    print("    - Protected identity facts")
    print("    - Project information")
    print("    - Temporary session data")

    # Simulate conversation with auto-extraction
    conversations = [
        "My name is Bob and I live in Singapore",
        "I prefer to use Python for trading bots",
        "Remember that the API key is sk-xyz789",
        "What's my timezone?",
    ]

    print("\n  Simulating conversation:")
    for msg in conversations:
        extracted = agent.process_conversation(msg)
        if extracted:
            print(f"    User: '{msg}'")
            print(f"    Extracted: {[e['path'] for e in extracted]}")

    # Check memory health
    print("\n  Final memory state:")
    print(f"    Facts: {len(agent.mem.paths())}")
    print(f"    Usage: {agent.stats()['utilization']}")

    print("\n\n✅ Advanced features demo complete!")
    print("   Features: contradiction detection, consolidation, forgetting curve")
    print("   All working together for intelligent memory management")


if __name__ == "__main__":
    main()