import re
from typing import List, Optional

from rag.base import Chunk
from rag.vectorstores.base import BaseVectorStore


class MemoryVectorStore(BaseVectorStore):
    def __init__(self):
        self.chunks: list[Chunk] = []

    def add(self, chunks: List[Chunk]) -> None:
        self.chunks.extend(chunks)

    def search(self, query: str, limit: int = 5) -> List[Chunk]:
        scored = []

        for chunk in self.chunks:
            score = self._score(query, chunk.content)
            if score > 0:
                scored.append(chunk.model_copy(update={"score": score}))

        scored.sort(key=lambda c: c.score or 0, reverse=True)
        return scored[:limit]

    def delete(self, document_id: Optional[str] = None, chunk_id: Optional[str] = None) -> int:
        before = len(self.chunks)

        if chunk_id:
            self.chunks = [c for c in self.chunks if c.id != chunk_id]
        elif document_id:
            self.chunks = [c for c in self.chunks if c.document_id != document_id]

        return before - len(self.chunks)

    def clear(self) -> None:
        self.chunks.clear()

    def count(self) -> int:
        return len(self.chunks)

    def _score(self, query: str, content: str) -> float:
        q = query.lower().strip()
        c = content.lower().strip()

        if not q or not c:
            return 0.0

        if q in c:
            return 1.0

        token_score = self._token_overlap_score(q, c)
        char_score = self._char_overlap_score(q, c)
        phrase_bonus = self._phrase_bonus(q, c)

        return min(1.0, token_score * 0.45 + char_score * 0.45 + phrase_bonus)

    @staticmethod
    def _token_overlap_score(query: str, content: str) -> float:
        query_tokens = MemoryVectorStore._tokenize(query)
        content_tokens = MemoryVectorStore._tokenize(content)
        if not query_tokens or not content_tokens:
            return 0.0

        overlap = query_tokens.intersection(content_tokens)
        union = query_tokens.union(content_tokens)
        return len(overlap) / len(union)

    @staticmethod
    def _char_overlap_score(query: str, content: str) -> float:
        query_chars = set(query.replace(" ", ""))
        content_chars = set(content.replace(" ", ""))
        if not query_chars or not content_chars:
            return 0.0

        overlap = query_chars.intersection(content_chars)
        union = query_chars.union(content_chars)
        return len(overlap) / len(union)

    @staticmethod
    def _phrase_bonus(query: str, content: str) -> float:
        bonus = 0.0
        for token in MemoryVectorStore._tokenize(query):
            if len(token) >= 2 and token in content:
                bonus += 0.08
        return min(0.25, bonus)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = re.split(r"[\s，。！？、；：,.!?;:（）()\[\]{}\"']+", text)
        return {token for token in tokens if token}
