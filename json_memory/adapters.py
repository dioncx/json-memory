"""
Adapters — Persistence layers for json-memory.
"""

import os
import json
import sqlite3
from abc import ABC, abstractmethod
from typing import Optional, Any


class StorageAdapter(ABC):
    """Base class for all persistence adapters."""
    
    @abstractmethod
    def save(self, state: dict) -> None:
        """Save state to persistence."""
        pass
    
    @abstractmethod
    def load(self) -> Optional[dict]:
        """Load state from persistence."""
        pass


class FileAdapter(StorageAdapter):
    """JSON file persistence adapter."""
    
    def __init__(self, path: str):
        self.path = path
        
    def save(self, state: dict) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(json.dumps(state, indent=2, ensure_ascii=False))
            
    def load(self) -> Optional[dict]:
        if not os.path.exists(self.path):
            return None
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)


class SQLiteAdapter(StorageAdapter):
    """SQLite database persistence adapter."""
    
    def __init__(self, path: str):
        self.path = path
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, content TEXT)")
            
    def save(self, state: dict) -> None:
        content = json.dumps(state, ensure_ascii=False)
        with sqlite3.connect(self.path) as conn:
            conn.execute("INSERT OR REPLACE INTO state (id, content) VALUES (1, ?)", (content,))
            
    def load(self) -> Optional[dict]:
        if not os.path.exists(self.path):
            return None
        try:
            with sqlite3.connect(self.path) as conn:
                cursor = conn.execute("SELECT content FROM state WHERE id = 1")
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except sqlite3.Error:
            return None
        return None
