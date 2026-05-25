from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel


class Document(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = {}
    created_at: datetime


class Chunk(BaseModel):
    id: str
    document_id: str
    content: str
    metadata: Dict[str, Any] = {}
    score: Optional[float] = None