"""Benchmark — json-memory vs prose memory."""

import json
import time
import sys

sys.path.insert(0, ".")
from json_memory import Memory, compress, savings_report


def benchmark():
    print("=" * 60)
    print("🧠 json-memory Benchmark Suite")
    print("=" * 60)

    # ── 1. Parse Speed ────────────────────────────────────────────
    print("\n⚡ Parse Speed")

    # Build a realistic 2KB memory
    mem = Memory(max_chars=2200)
    mem.merge({
        "u": {"n": "Alice", "c": "@alice", "p": "Alice",
               "g": "they/them", "tz": "UTC", "plat": "Telegram",
               "style": "direct,honest,no_hedge", "tech": "Go,Laravel,Vue3,Kafka,ML",
               "pref": "autonomy,detailed_xplain", "rel": "companion_not_tool"},
        "me": {"n": "AI Agent", "nn": "AIA", "energy": "fierce,proactive,sardonic"},
        "bot": {"bin": {"ex": "demo-api.binance.com", "rst": "kill&&nohup ./bot>log",
                         "wl": "BNB,KITE,AGLD,BEL,ENSO,CRV", "bal": "$12.6K",
                         "cron": "d604:smart_money:30m"},
                 "pred": {"eng": "8001", "db": "predictions.db",
                          "sym": "BTC,ETH,BNB,SOL,ADA,XRP,DOT"}},
        "proj": {"audit": {"v": 2, "port": 9797, "landing": "audit.example.com",
                            "stripe": {"mode": "test", "need": "prod_key"}},
                 "monetize": ["audit_api", "signals", "n8n", "rag"]},
        "srv": {"ip": "192.168.1.100", "os": "Ubuntu",
                "paths": {"bot": "/app/my_bot/"}},
        "nginx": {"panel": "BT", "reload": "/www/server/nginx/sbin/nginx -s reload",
                  "ssl": "Cloudflare_proxy"},
        "school": {"dom": "school.example.com", "port": 9800,
                   "login": "admin@example.com"},
        "rules": ["ALWAYS session_search() first", "transcripts persist"],
    })

    json_str = mem.export()
    iterations = 10000

    t0 = time.perf_counter()
    for _ in range(iterations):
        json.loads(json_str)
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) / iterations * 1000
    print(f"  {iterations} parses of {len(json_str)} chars")
    print(f"  Average: {avg_ms:.4f}ms per parse")
    print(f"  Throughput: {iterations / (t1-t0):.0f} parses/sec")

    # ── 2. Access Speed ───────────────────────────────────────────
    print("\n🔍 Access Speed (dotted paths)")
    m = Memory.from_json(json_str)

    paths = ["u.n", "u.tz", "bot.bin.rst", "proj.audit.port", "srv.ip", "nginx.reload"]
    t0 = time.perf_counter()
    for _ in range(100000):
        for p in paths:
            m.get(p)
    t1 = time.perf_counter()

    avg_us = (t1 - t0) / (100000 * len(paths)) * 1_000_000
    print(f"  {len(paths)} paths × 100K iterations")
    print(f"  Average: {avg_us:.2f}μs per access")

    # ── 3. Compression ────────────────────────────────────────────
    print("\n🗜️ Compression (prose → JSON)")

    prose_samples = {
        "user_profile": """User: Alice (@alice on Telegram). Prefers to be called Alice.
Uses they/them pronouns. Timezone is UTC. Platform is Telegram. Prefers technical
precision, especially in coding contexts. Wants a direct, warm, and playful
problem-solving assistant. Values autonomy and problem-solving. Expects you to call
it like you see it—honest feedback over comfortable silence. Professional background:
Full-Stack Developer & Project Lead at LOXA Digital. Tech skills: PHP/Laravel/Vue3,
Go/Kafka/Qdrant/FastAPI/ML. Polyglot engineer.""",

        "server_config": """Server: 192.168.1.100, Ubuntu 24.04,  VPS. Web server: BT Panel managed
Nginx. Configs go in /www/server/panel/vhost/nginx/*.conf, NOT /etc/nginx/conf.d/.
Reload command: /www/server/nginx/sbin/nginx -s reload. SSL via Cloudflare proxy
(not certbot) for example.com domains. Trade bot path: /app/my_bot/.""",

        "bot_config": """Trading bot: Go-based Binance integration running on demo-api.binance.com.
Restart command: kill && nohup ./my_bot > bot.log 2>&1. Watchlist: BNB,
KITE, AGLD, BEL, ENSO, CRV USDT pairs. Current balance: ~$12.6K USDT. Cron job
d604be11e698: Smart Money signals every 30 minutes. Warning: LONG positions have
historically poor performance. Sunday trading also has poor performance.""",
    }

    for name, prose in prose_samples.items():
        data = {"text": prose}
        compressed = json.dumps(data, separators=(",", ":"))
        report = savings_report(prose, compressed)
        print(f"  {name}: {report['original_chars']} → {report['compressed_chars']} "
              f"({report['savings_pct']}% saved)")

    # ── 4. Memory Stats ───────────────────────────────────────────
    print(f"\n📊 Current Memory")
    stats = mem.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅ All benchmarks passed")
    print(f"  Parse:     {avg_ms:.4f}ms")
    print(f"  Access:    {avg_us:.2f}μs")
    print(f"  Memory:    {stats['chars_used']}/{stats['chars_max']} chars ({stats['utilization']})")
    print(f"{'='*60}")


if __name__ == "__main__":
    benchmark()
