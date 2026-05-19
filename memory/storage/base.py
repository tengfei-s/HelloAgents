from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from memory.base import MemoryItem, MemoryConfig


class StorageBackend(ABC):
    @abstractmethod
    def save(self, item: MemoryItem) -> str: pass

    @abstractmethod
    def load(self, memory_id: str) -> Optional[MemoryItem]: pass

    @abstractmethod
    def load_all(self) -> List[MemoryItem]: pass

    @abstractmethod
    def delete(self, memory_id: str) -> bool: pass

    @abstractmethod
    def update(self, memory_id: str, fields: Dict[str, Any]) -> bool: pass

    @abstractmethod
    def exists(self, memory_id: str) -> bool: pass

    @abstractmethod
    def clear(self): pass