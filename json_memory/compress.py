"""
Compress — Utilities for measuring and maximizing memory compression.
"""

import json
import re
from typing import Any


# ── Short Key Dictionary ─────────────────────────────────────────

ABBREVIATIONS = {
    "user": "u", "name": "n", "nickname": "nn", "preferred": "p",
    "timezone": "tz", "platform": "plat", "language": "lang",
    "exchange": "ex", "restart": "rst", "watchlist": "wl",
    "balance": "bal", "password": "pw", "email": "em",
    "server": "srv", "database": "db", "configuration": "cfg",
    "project": "proj", "description": "desc", "version": "ver",
    "endpoint": "ep", "webhook": "wh", "domain": "dom",
    "directory": "dir", "command": "cmd", "interval": "intv",
    "status": "st", "prediction": "pred", "emergency": "emerg",
    "technical": "tech", "sentiment": "sent", "confidence": "conf",
    "position": "pos", "stop_loss": "sl", "take_profit": "tp",
    "environment": "env", "authentication": "auth", "token": "tk",
    "api_key": "ak", "secret": "sec", "public": "pub",
}


def compress(data: dict, abbreviations: dict = None) -> dict:
    """Compress a dict by abbreviating keys.

    Args:
        data: The dict to compress.
        abbreviations: Custom abbreviation mapping (merged with defaults).

    Returns:
        New dict with abbreviated keys.
    """
    abbr = {**ABBREVIATIONS}
    if abbreviations:
        abbr.update(abbreviations)

    return _compress_node(data, abbr)


def _compress_node(node: Any, abbr: dict) -> Any:
    if isinstance(node, dict):
        return {
            abbr.get(k, k): _compress_node(v, abbr)
            for k, v in node.items()
        }
    elif isinstance(node, list):
        return [_compress_node(item, abbr) for item in node]
    return node


def decompress(data: dict, abbreviations: dict = None) -> dict:
    """Decompress a dict by expanding abbreviated keys.

    Args:
        data: The compressed dict.
        abbreviations: Custom abbreviation mapping (merged with defaults).

    Returns:
        New dict with expanded keys.
    """
    abbr = {**ABBREVIATIONS}
    if abbreviations:
        abbr.update(abbreviations)

    # Reverse the mapping
    expand = {v: k for k, v in abbr.items()}
    return _compress_node(data, expand)


def savings_report(original: str, compressed: str) -> dict:
    """Calculate compression savings.

    Args:
        original: The original text (prose or expanded JSON).
        compressed: The compressed text (minified JSON).

    Returns:
        Dict with size metrics and savings percentage.
    """
    orig_len = len(original)
    comp_len = len(compressed)
    saved = orig_len - comp_len
    pct = (saved / orig_len * 100) if orig_len > 0 else 0

    return {
        "original_chars": orig_len,
        "compressed_chars": comp_len,
        "chars_saved": saved,
        "savings_pct": round(pct, 1),
        "ratio": round(comp_len / orig_len, 3) if orig_len > 0 else 0,
    }


def prose_to_json(prose: str, field_pattern: str = r"(\w+):\s*(.+?)(?:\n|$)") -> dict:
    """Convert simple prose key-value pairs to a dict.

    Example:
        >>> prose = "name: Alice\\ntz: UTC\\nplatform: Telegram"
        >>> prose_to_json(prose)
        {'name': 'Alice', 'tz': 'UTC', 'platform': 'Telegram'}
    """
    result = {}
    for match in re.finditer(field_pattern, prose):
        key, value = match.group(1), match.group(2).strip()
        result[key] = value
    return result


def minify(json_str: str) -> str:
    """Minify a JSON string (remove whitespace)."""
    return json.dumps(json.loads(json_str), separators=(",", ":"), ensure_ascii=False)
