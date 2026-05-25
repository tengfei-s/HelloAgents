import uuid
from datetime import datetime

from rag.base import Chunk, Document
from rag.chunkers.base import BaseChunker
from rag.chunkers.simple import SimpleChunker
from rag.vectorstores.base import BaseVectorStore
from rag.vectorstores.memory import MemoryVectorStore



class RagManager:
    def __init__(self, chunker: BaseChunker = None, vector_store: BaseVectorStore = None):
        self.chunker = chunker or (SimpleChunker())
        self.vector_store = vector_store or (MemoryVectorStore())

    def add_text(self, text: str, metadata=None) -> str:
        metadata = metadata or {}
        doc = Document(
            id=str(uuid.uuid4()),
            content=text,
            metadata=metadata,
            created_at=datetime.now(),
        )
        chunks = self.chunker.split(doc)
        self.vector_store.add(chunks)
        return doc.id

    def add_file(self, path: str, metadata=None) -> str:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        metadata = metadata or {}
        metadata.setdefault("source", path)
        return self.add_text(text, metadata)

    def search(self, query: str, limit: int = 5) -> list[Chunk]:
        return self.vector_store.search(query, limit)

    def build_context(self, query: str, limit: int = 5) -> str:
        chunks = self.search(query, limit)
        if not chunks:
            return ""

        context_parts = []
        for index, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "unknown")
            score = chunk.score if chunk.score is not None else 0.0
            context_parts.append(
                f"[{index}] source={source}, score={score:.4f}\n{chunk.content}"
            )
        return "\n\n".join(context_parts)

    def clear(self):
        self.vector_store.clear()

    def get_stats(self) -> dict:
        return {
            "chunk_count": self.vector_store.count(),
            "chunker": self.chunker.__class__.__name__,
            "vector_store": self.vector_store.__class__.__name__,
        }
