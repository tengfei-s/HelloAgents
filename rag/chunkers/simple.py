import re
import uuid
from typing import List

from rag.base import Chunk, Document
from rag.chunkers.base import BaseChunker


class SimpleChunker(BaseChunker):
    def __init__(self, chunk_size: int = 120, overlap: int = 20):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: Document) -> List[Chunk]:
        text = document.content.strip()
        if not text:
            return []

        paragraphs = self._split_paragraphs(text)
        sentences = []
        for paragraph in paragraphs:
            sentences.extend(self._split_sentences(paragraph))

        raw_chunks = self._merge_units(sentences)

        chunks = []
        for index, content in enumerate(raw_chunks):
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    document_id=document.id,
                    content=content,
                    metadata={
                        **document.metadata,
                        "chunk_index": index,
                        "chunk_size": len(content),
                    },
                )
            )

        return chunks

    def _split_paragraphs(self, text: str) -> List[str]:
        paragraphs = []
        current = []

        for line in text.splitlines():
            line = line.strip()
            if not line:
                if current:
                    paragraphs.append("\n".join(current))
                    current = []
            else:
                current.append(line)

        if current:
            paragraphs.append("\n".join(current))

        return paragraphs or [text]

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r"(?<=[。！？!?；;])", text)
        sentences = [part.strip() for part in parts if part.strip()]

        result = []
        for sentence in sentences or [text]:
            if len(sentence) <= self.chunk_size:
                result.append(sentence)
            else:
                result.extend(self._split_long_text(sentence))
        return result

    def _merge_units(self, units: List[str]) -> List[str]:
        chunks = []
        buffer = ""

        for unit in units:
            candidate = unit if not buffer else buffer + unit
            if len(candidate) <= self.chunk_size:
                buffer = candidate
                continue

            if buffer:
                chunks.append(buffer)
            buffer = unit

        if buffer:
            chunks.append(buffer)

        return chunks
