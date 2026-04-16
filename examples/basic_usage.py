"""Example: Using json-memory for AI agent memory."""

from json_memory import Memory, Synapse


def example_basic():
    """Basic memory operations."""
    print("=== Basic Memory ===")

    mem = Memory(max_chars=2000)

    # Set nested values
    mem.set("u.name", "Alice")
    mem.set("u.handle", "@alice")
    mem.set("u.tz", "UTC")
    mem.set("u.style", "direct,honest,no_hedge")

    mem.set("bot.binance.rst", "kill && nohup ./bot > log 2>&1")
    mem.set("bot.binance.wl", "BNB,KITE,AGLD,BEL,ENSO,CRV")
    mem.set("bot.binance.bal", "$12.6K")

    mem.set("srv.ip", "192.168.1.100")
    mem.set("srv.os", "Ubuntu")

    # Access by dotted path
    print(f"  User: {mem.get('u.name')}")
    print(f"  Timezone: {mem.get('u.tz')}")
    print(f"  Bot restart: {mem.get('bot.binance.rst')}")
    print(f"  Server: {mem.get('srv.ip')}")

    # Stats
    print(f"  {mem.stats()}")
    print(f"  Exported: {mem.export()}")
    print()


def example_synapse():
    """Associative memory like a brain."""
    print("=== Synapse Memory ===")

    brain = Synapse()

    # Build knowledge graph
    brain.link("trading", ["binance", "strategy", "risk", "signals"])
    brain.link("binance", ["api", "demo", "watchlist", "orders"])
    brain.link("strategy", ["entry", "exit", "stoploss", "take_profit"])
    brain.link("risk", ["position_size", "max_loss", "drawdown"])
    brain.link("server", ["nginx", "ssl", "domains", "monitoring"])
    brain.link("nginx", ["config", "reload", "cloudflare"])

    # Activate — like thinking of a concept
    print(f"  Think 'trading' → {brain.activate('trading')}")
    print(f"  Think 'trading' (depth=2) → {brain.activate('trading', depth=2)}")

    # Find connections
    print(f"  Binance connections: {brain.connections('binance')}")

    # Find path between concepts
    print(f"  Path: trading → cloudflare: {brain.find_path('trading', 'cloudflare')}")

    # Find hubs
    print(f"  Hub concepts: {brain.hubs(min_connections=3)}")
    print()


def example_migration():
    """Migrate from prose to JSON."""
    print("=== Migration Example ===")

    # Old prose memory
    prose = """Bot: Go trade_binance @ demo-api.binance.com. Restart: kill && nohup
    ./my_bot > bot.log 2>&1. Watchlist: BNB,KITE,AGLD,BEL,ENSO,CRV.
    Balance: ~$12.6K. Cron: d604 smart_money 30min."""

    print(f"  Prose: {len(prose)} chars")

    # New JSON memory
    mem = Memory(max_chars=2200)
    mem.merge({
        "bot": {
            "lang": "Go",
            "ex": "demo-api.binance.com",
            "rst": "kill&&nohup ./my_bot>bot.log 2>&1",
            "wl": "BNB,KITE,AGLD,BEL,ENSO,CRV",
            "bal": "$12.6K",
            "cron": "d604:smart_money:30m"
        }
    })

    json_str = mem.export()
    print(f"  JSON: {len(json_str)} chars")
    print(f"  Savings: {len(prose) - len(json_str)} chars ({(1 - len(json_str)/len(prose))*100:.0f}%)")
    print(f"  Data: {json_str}")
    print()


if __name__ == "__main__":
    example_basic()
    example_synapse()
    example_migration()
