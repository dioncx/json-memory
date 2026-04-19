"""
Lightweight semantic layer — no ML dependencies.

Approach: hand-built concept similarity using curated knowledge graphs.
Not as powerful as embeddings, but covers 80% of cases with zero deps.
"""

# Concept similarity mapping: word → related concepts
# This is the "cheat sheet" for semantic matching without ML
CONCEPT_MAP = {
    # Identity
    "who": ["name", "identity", "user", "person", "called", "background", "profession"],
    "about": ["identity", "background", "info", "information", "detail"],
    "name": ["who", "identity", "user", "called", "person"],
    "identity": ["who", "name", "user"],
    
    # Time
    "time": ["timezone", "clock", "hour", "schedule", "gmt", "utc"],
    "timezone": ["time", "gmt", "utc", "clock", "hour", "offset"],
    "when": ["time", "timezone", "schedule", "date"],
    "wake": ["timezone", "time", "morning", "clock", "alarm"],
    "sleep": ["timezone", "time", "night", "clock"],
    "morning": ["timezone", "time", "wake", "clock"],
    "late": ["timezone", "time", "night", "clock"],
    
    # Location
    "where": ["location", "place", "city", "country", "address", "server", "ip"],
    "location": ["where", "place", "city", "country", "address"],
    "live": ["location", "city", "country", "place"],
    "from": ["location", "city", "country", "origin"],
    "server": ["ip", "host", "machine", "address", "ssh"],
    "ip": ["server", "host", "address", "network"],
    
    # Actions
    "restart": ["start", "stop", "reboot", "reload", "reset", "run"],
    "start": ["restart", "run", "launch", "begin", "boot"],
    "stop": ["kill", "halt", "end", "terminate", "shutdown"],
    "build": ["compile", "make", "construct", "create"],
    "deploy": ["push", "release", "ship", "publish", "upload", "launch"],
    "upload": ["push", "publish", "deploy", "send", "submit"],
    "install": ["setup", "download", "pip", "npm"],
    
    # Trading
    "trading": ["trade", "bot", "exchange", "strategy", "signal", "order", "position"],
    "trade": ["trading", "bot", "exchange", "order", "buy", "sell"],
    "bot": ["trading", "trade", "exchange", "strategy", "automated"],
    "strategy": ["trading", "signal", "indicator", "rsi", "macd"],
    "exchange": ["trading", "binance", "exchange", "market"],
    "signal": ["trading", "alert", "notification", "indicator"],
    "profit": ["trading", "pnl", "gain", "earnings", "return"],
    "loss": ["trading", "pnl", "drawdown", "risk"],
    "price": ["trading", "market", "value", "cost", "rate"],
    "buy": ["trading", "order", "long", "entry"],
    "sell": ["trading", "order", "short", "exit"],
    
    # Project
    "project": ["repo", "repository", "code", "source", "library", "package"],
    "repo": ["project", "repository", "github", "git", "code"],
    "library": ["project", "package", "module", "pip", "pypi"],
    "version": ["release", "tag", "semver", "update"],
    
    # System
    "error": ["bug", "issue", "problem", "fail", "broken", "crash"],
    "fix": ["repair", "patch", "solve", "debug", "resolve"],
    "log": ["debug", "trace", "output", "print"],
    # Career / Professional
    "career": ["job", "work", "profession", "professional", "employment", "role", "position", "occupation"],
    "job": ["career", "work", "profession", "professional", "employment", "role", "position"],
    "work": ["career", "job", "profession", "professional", "employment", "company", "office"],
    "professional": ["career", "job", "work", "profession", "employment", "background", "experience"],
    "profession": ["career", "job", "work", "professional", "employment", "background", "field"],
    "background": ["history", "experience", "past", "career", "profession", "education", "qualification"],
    "experience": ["background", "history", "skill", "expertise", "career"],
    "employment": ["career", "job", "work", "company", "position"],
    "education": ["degree", "university", "school", "qualification", "background", "study"],
    "degree": ["education", "university", "qualification", "gpa", "bachelor", "master"],
    "skill": ["ability", "expertise", "talent", "competency", "experience", "tech"],
    "skills": ["ability", "expertise", "talent", "competency", "experience", "tech"],
    "config": ["settings", "options", "parameters", "env"],
    "api": ["endpoint", "request", "response", "key"],
    "key": ["api", "token", "secret", "credential", "auth"],
    "database": ["db", "sql", "postgres", "mysql", "store"],
    
    # Communication
    "message": ["chat", "send", "notify", "alert", "telegram"],
    "telegram": ["message", "chat", "bot", "notify", "channel"],
    "notify": ["alert", "message", "send", "telegram"],
    "cron": ["schedule", "timer", "interval", "periodic", "job"],
    "schedule": ["cron", "timer", "interval", "time"],
}


def expand_query_semantic(tokens: set[str]) -> set[str]:
    """Expand query tokens with semantically related concepts.

    Uses a curated concept map — not as smart as embeddings, but zero deps
    and covers the most common agent memory queries.

    Args:
        tokens: Set of lowercase tokens from the query.

    Returns:
        Expanded token set including semantic neighbors.
    """
    expanded = set(tokens)
    for token in tokens:
        if token in CONCEPT_MAP:
            expanded.update(CONCEPT_MAP[token])
    return expanded


def get_concept_category(token: str) -> str | None:
    """Return the semantic category of a token for grouping."""
    categories = {
        "identity": {"who", "name", "user", "person", "identity"},
        "time": {"time", "timezone", "when", "clock", "gmt", "utc", "hour", "schedule"},
        "location": {"where", "location", "server", "ip", "host", "address"},
        "action": {"restart", "start", "stop", "build", "deploy", "run", "kill"},
        "trading": {"trading", "trade", "bot", "strategy", "exchange", "signal", "price"},
        "project": {"project", "repo", "library", "package", "version", "code"},
        "system": {"error", "fix", "log", "config", "api", "key", "database"},
        "communication": {"message", "telegram", "notify", "chat", "cron", "schedule"},
    }
    for category, words in categories.items():
        if token in words:
            return category
    return None
