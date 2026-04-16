"""json-memory — Hierarchical associative memory for AI agents."""

from .memory import Memory
from .synapse import Synapse
from .schema import Schema
from .compress import compress, decompress, savings_report
from .weight_gate import WeightGate

__version__ = "0.1.1"
__all__ = ["Memory", "Synapse", "Schema", "WeightGate", "compress", "decompress", "savings_report"]
