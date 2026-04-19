"""json-memory — Hierarchical associative memory for AI agents."""

from .memory import Memory
from .synapse import Synapse
from .schema import Schema
from .compress import compress, decompress, savings_report
from .weight_gate import WeightGate, HAS_SNOWBALL
from .smart import SmartMemory, TieredMemory

__version__ = "0.2.0"
__all__ = [
    "Memory", "Synapse", "Schema", "WeightGate",
    "SmartMemory", "TieredMemory",
    "compress", "decompress", "savings_report",
    "HAS_SNOWBALL",
]
