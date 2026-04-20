"""json-memory — Hierarchical associative memory for AI agents."""

from .memory import Memory
from .synapse import Synapse
from .schema import Schema
from .compress import compress, decompress, savings_report
from .weight_gate import WeightGate, HAS_SNOWBALL
from .smart import SmartMemory, TieredMemory
from .contradiction import detect_contradictions, Contradiction, ContradictionDetector
from .consolidation import consolidate_memory, ConsolidationGroup
from .forgetting import ForgettingCurve, MemoryStrength
from .visualizer import MemoryVisualizer, visualize_memory
from .versioning import MemoryVersioning, MemoryDiff
from .encryption import MemoryEncryption, EncryptedValue
from .search import AdvancedSearch, SearchResult

__version__ = "1.4.0"
__all__ = [
    "Memory", "Synapse", "Schema", "WeightGate",
    "SmartMemory", "TieredMemory",
    "compress", "decompress", "savings_report",
    "HAS_SNOWBALL",
    "detect_contradictions", "Contradiction", "ContradictionDetector",
    "consolidate_memory", "ConsolidationGroup",
    "ForgettingCurve", "MemoryStrength",
    "MemoryVisualizer", "visualize_memory",
    "MemoryVersioning", "MemoryDiff",
    "MemoryEncryption", "EncryptedValue",
    "AdvancedSearch", "SearchResult",
]
