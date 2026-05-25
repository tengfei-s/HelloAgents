from abc import ABC, abstractmethod
from typing import List

from rag.base import Document, Chunk


class BaseChunker(ABC):
    @abstractmethod
    def split(self, document: Document) -> List[Chunk]:
        pass