"""
Concrete test: Can SmartMemory replace the prose MEMORY block in an agent like Hermes?

Simulate the actual memory that gets injected into my (Hermes) system prompt every turn.
"""

import json
import sys
sys.path.insert(0, '/root/json-memory')

from json_memory import SmartMemory, Memory

# ── Current Hermes MEMORY block (what actually gets injected) ─────────
# This is roughly what my system prompt looks like right now.
# Taken from the actual MEMORY.md format Hermes uses.

PROSE_MEMORY = """
## Memory
- json-memory repo: github.com/dioncx/json-memory, v0.1.6, live on PyPI
- Build: python3 setup.py sdist bdist_wheel
- Upload: twine upload dist/*
- pypirc: ~/.pypirc, needs sdist fix (rename tar.gz)
- License: MIT

- trade_bot_binance: Go Binance demo bot at /www/wwwroot/trade_bot_binance/
- Strategy: SuperStrategy (RSI+MACD+Trend+Vol), BNBUSDT, $5/trade
- Fixed 2026-04-18: smart_money_confidence.py — count ALL buy signals, confidence 0.12→0.67
- All min_score thresholds lowered to 3. Bot trades now.
- PnL tracker: pnl_tracker.py
- Goal: 3 profitable demo days → live $10
- Binary: ./trade_bot_binance, Build: go build -o trade_bot_binance ./cmd/bot

- GATEWAY LIMITATION: Telegram gateway kills foreground terminal commands with SIGINT after ~10s
- For long ops (Go builds, server restarts), prefer execute_code with long timeout or delegate_task

- Server has python3.12 at /usr/bin/python3.12 (default python3 is 3.11)
- Use python3.12 explicitly for json-memory and snowballstemmer work
- json-memory repo cloned at /root/json-memory, pushed via git@github.com:dioncx/json-memory.git
- Terminal tool gets killed by gateway on long foreground processes

- json-memory: repo git@github.com:dioncx/json-memory.git, local /root/json-memory
- Python3.12. Values: zero/optional deps, strict encapsulation, atomicity
- Stemmer: 3-layer (dictionary->snowballstemmer->suffix)
- Fixed merge() atomic overflow, WeightGate public API, decay on competition only w/ 0.1 floor

- Heartbeat cron: 56ffe0e2244b (30 min, delivers to origin)
- Crons use openrouter/google/gemini-2.0-flash-001 as fallback when Nous exhausts
- Nous auth: oauth 15-min access tokens + agent key (24h), auto-refresh
- Main session uses Nous.

- TG signals group "Crypto Trading Signal" chat_id=-1003914507160
- Bot @veludra_hermesbot 8738619118
- Auto-post every 2h via cron 7d8167287883
- Formatter: trade_bot_binance/signal_formatter.py
- LS store 349236 approved, products via dashboard only
"""

# ── Convert to SmartMemory ───────────────────────────────────────────

smart = SmartMemory("/tmp/hermes_sim.json", max_chars=5000)

# Parse the prose into structured facts (what a human does when maintaining memory)
FACTS = {
    # json-memory project
    "jm.repo": "github.com/dioncx/json-memory",
    "jm.version": "0.2.0",
    "jm.status": "live_pypi",
    "jm.build": "python3 setup.py sdist bdist_wheel",
    "jm.upload": "twine upload dist/*",
    "jm.pypirc": "~/.pypirc",
    "jm.license": "MIT",
    "jm.local_path": "/root/json-memory",
    "jm.lang": "Python3.12",
    "jm.values": "zero deps, strict encapsulation, atomicity",
    "jm.stemmer": "3-layer dictionary→snowball→suffix",
    
    # Trading bot
    "bot.path": "/www/wwwroot/trade_bot_binance",
    "bot.lang": "Go",
    "bot.strategy": "SuperStrategy (RSI+MACD+Trend+Vol)",
    "bot.symbol": "BNBUSDT",
    "bot.size": "$5/trade",
    "bot.fix_date": "2026-04-18",
    "bot.confidence_fix": "count ALL buy signals, 0.12→0.67",
    "bot.min_score": "3 (lowered)",
    "bot.tracker": "pnl_tracker.py",
    "bot.goal": "3 profitable demo days → live $10",
    "bot.binary": "./trade_bot_binance",
    "bot.build": "go build -o trade_bot_binance ./cmd/bot",
    
    # Server
    "srv.python": "/usr/bin/python3.12",
    "srv.default_python": "3.11",
    
    # Gateway
    "gw.limitation": "Telegram kills foreground cmds after ~10s",
    "gw.workaround": "use execute_code with long timeout or delegate_task",
    
    # Crons
    "cron.heartbeat": "56ffe0e2244b, 30min, origin",
    "cron.fallback": "openrouter/google/gemini-2.0-flash-001",
    "cron.nous_auth": "oauth 15min + agent key 24h, auto-refresh",
    "cron.session": "Nous",
    
    # Telegram
    "tg.signals_group": "Crypto Trading Signal",
    "tg.signals_id": "-1003914507160",
    "tg.bot": "@veludra_hermesbot",
    "tg.bot_id": "8738619118",
    "tg.auto_post": "every 2h, cron 7d8167287883",
    "tg.formatter": "trade_bot_binance/signal_formatter.py",
    "tg.ls_store": "349236 approved",
}

for path, val in FACTS.items():
    smart.remember(path, val)

# ── Simulate Real Queries ────────────────────────────────────────────

queries = [
    ("How do I build the bot?", "Bot build query"),
    ("What's the PyPI upload command?", "PyPI query"),
    ("Why isn't the bot trading?", "Bot troubleshooting"),
    ("How do I restart the heartbeat cron?", "Cron query"),
    ("What's the Telegram signals setup?", "TG query"),
    ("What Python version should I use?", "Server query"),
    ("What's the gateway workaround?", "Gateway query"),
]

print("=" * 65)
print("🧠 ACTUAL HERMES MEMORY — Token Compression Test")
print("=" * 65)

prose_chars = len(PROSE_MEMORY.strip())
prose_tokens_est = prose_chars / 3.5  # rough estimate: ~3.5 chars per token

print(f"\n📊 Current Prose Memory:")
print(f"   Characters: {prose_chars}")
print(f"   Est. tokens: ~{prose_tokens_est:.0f}")
print(f"   Injected: EVERY turn (even when irrelevant)")

print(f"\n📊 SmartMemory (per-query):")
print(f"   Facts stored: {len(FACTS)}")

total_smart = 0
for query, label in queries:
    ctx = smart.prompt_context(query)
    chars = len(ctx)
    tokens_est = chars / 3.5
    total_smart += chars
    print(f"\n   {label}:")
    print(f"     Q: \"{query}\"")
    print(f"     → {chars} chars (~{tokens_est:.0f} tokens)")
    # Show what's actually returned
    lines = ctx.strip().split('\n')
    for line in lines[:4]:
        print(f"       {line}")
    if len(lines) > 4:
        print(f"       ... ({len(lines)-4} more)")

avg_smart = total_smart / len(queries)
avg_tokens = avg_smart / 3.5

print(f"\n{'=' * 65}")
print(f"📊 COMPARISON")
print(f"{'=' * 65}")
print(f"")
print(f"  Current (prose):    {prose_chars:>5} chars (~{prose_tokens_est:.0f} tokens) EVERY turn")
print(f"  SmartMemory avg:    {avg_smart:>5.0f} chars (~{avg_tokens:.0f} tokens) per query")
print(f"")
savings_pct = (1 - avg_smart / prose_chars) * 100
tokens_saved = prose_tokens_est - avg_tokens
print(f"  💰 Savings per turn: {savings_pct:.0f}% (~{tokens_saved:.0f} tokens)")
print(f"")
print(f"  Over 100 turns (typical session):")
print(f"    Current:  ~{prose_tokens_est * 100:.0f} tokens spent on memory")
print(f"    SmartMem: ~{avg_tokens * 100:.0f} tokens spent on memory")
print(f"    Saved:    ~{(prose_tokens_est - avg_tokens) * 100:.0f} tokens")
print(f"")
print(f"  Over 1000 turns (busy day):")
print(f"    Current:  ~{prose_tokens_est * 1000:.0f} tokens")
print(f"    SmartMem: ~{avg_tokens * 1000:.0f} tokens")
print(f"    Saved:    ~{(prose_tokens_est - avg_tokens) * 1000:.0f} tokens")

# Cleanup
import os
for f in ["/tmp/hermes_sim.json", "/tmp/hermes_sim.json.meta.json"]:
    if os.path.exists(f):
        os.remove(f)
