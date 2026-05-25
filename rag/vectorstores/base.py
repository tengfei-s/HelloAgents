from abc import ABC, abstractmethod
from typing import List, Optional
from rag.base import Chunk


class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, chunks: List[Chunk]) -> None:
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[Chunk]:
        pass

    @abstractmethod
    def delete(self, document_id: Optional[str] = None, chunk_id: Optional[str] = None) -> int:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def count(self) -> int:
        pass