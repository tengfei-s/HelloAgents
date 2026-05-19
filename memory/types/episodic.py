from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from memory.base import BaseMemory, MemoryConfig, MemoryItem
from memory.storage.base import StorageBackend
from memory.storage.local import LocalFileStorage
from memory.storage.sqlite import SQLiteStorage


class EpisodicMemory(BaseMemory):
    """
    EpisodicMemory（事件记忆）

    - 目标：跨会话持久化（默认 SQLite），并支持按关键词/时间/重要性等条件检索。
    - 说明：当前实现优先走 StorageBackend；若后端支持 `query()`（SQLiteStorage 有），则使用后端过滤，
      否则退化为 load_all + Python 侧过滤。
    """

    def __init__(self, config: MemoryConfig, storage_backend: Optional[StorageBackend] = None):
        if config is None:
            config = MemoryConfig()
        if storage_backend is None:
            storage_backend = SQLiteStorage(db_path=f"{config.storage_path}/memories.db")
        super().__init__(config, storage_backend)

        self.memories_by_id: Dict[str, MemoryItem] = {}
        self._load_cache()

    # ------------------------------------------------------------------
    # BaseMemory 接口
    # ------------------------------------------------------------------

    def add(self, memory_item: MemoryItem) -> str:
        if not memory_item.id:
            memory_item.id = self._generate_id()

        if memory_item.timestamp is None:
            memory_item.timestamp = datetime.now()

        if not memory_item.memory_type:
            memory_item.memory_type = self.memory_type

        # 重要性阈值过滤：太低的不进入 episodic（避免污染长期库）
        if memory_item.importance is not None and memory_item.importance < self.config.importance_threshold:
            return memory_item.id

        if self.storage is None:
            raise RuntimeError("EpisodicMemory requires a storage backend.")

        self.storage.save(memory_item)
        self.memories_by_id[memory_item.id] = memory_item
        self._enforce_capacity_limit()
        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """
        检索接口（最小可用版）

        支持 kwargs：
        - memory_type: str
        - since/until: datetime
        - min_importance: float
        - user_id: str（从 metadata.user_id 过滤）
        """
        memory_type: Optional[str] = kwargs.get("memory_type")
        since: Optional[datetime] = kwargs.get("since")
        until: Optional[datetime] = kwargs.get("until")
        min_importance: Optional[float] = kwargs.get("min_importance")
        user_id: Optional[str] = kwargs.get("user_id")

        items = self._backend_query(
            keyword=None,
            memory_type=memory_type,
            since=since,
            until=until,
            min_importance=min_importance,
            limit=max(limit * 5, limit),
        )

        if user_id:
            items = [m for m in items if (m.metadata or {}).get("user_id") == user_id]

        if not query:
            return items[:limit]

        ranked: List[Tuple[float, MemoryItem]] = []
        for m in items:
            score = self._score_query_match(query, m)
            ranked.append((score, m))

        ranked.sort(key=lambda x: (x[0], x[1].importance, x[1].timestamp), reverse=True)
        return [m for _, m in ranked[:limit]]

    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        if self.storage is None:
            raise RuntimeError("EpisodicMemory requires a storage backend.")

        fields: Dict[str, Any] = {}
        if content is not None:
            fields["content"] = content
        if importance is not None:
            fields["importance"] = importance
        if metadata is not None:
            fields["metadata"] = metadata

        if not fields:
            return False

        ok = self.storage.update(memory_id, fields)
        if ok:
            cached = self.memories_by_id.get(memory_id) or self.storage.load(memory_id)
            if cached:
                updated = cached.model_copy(update=fields)
                self.memories_by_id[memory_id] = updated
        return ok

    def remove(self, memory_id: str) -> bool:
        if self.storage is None:
            raise RuntimeError("EpisodicMemory requires a storage backend.")
        ok = self.storage.delete(memory_id)
        self.memories_by_id.pop(memory_id, None)
        return ok

    def has_memory(self, memory_id: str) -> bool:
        if memory_id in self.memories_by_id:
            return True
        if self.storage is None:
            return False
        return self.storage.exists(memory_id)

    def clear(self):
        if self.storage is None:
            self.memories_by_id.clear()
            return
        self.storage.clear()
        self.memories_by_id.clear()

    def get_stats(self) -> Dict[str, Any]:
        count = len(self.memories_by_id)
        avg_importance = (
            sum(m.importance for m in self.memories_by_id.values()) / count if count else 0.0
        )
        backend = self.storage.__class__.__name__ if self.storage else None
        extra: Dict[str, Any] = {}
        if isinstance(self.storage, SQLiteStorage):
            extra["db_path"] = self.storage.db_path
        if isinstance(self.storage, LocalFileStorage):
            extra["storage_path"] = self.storage.storage_path

        return {
            "count": count,
            "avg_importance": avg_importance,
            "max_capacity": self.config.max_capacity,
            "importance_threshold": self.config.importance_threshold,
            "memory_type": self.memory_type,
            "backend": backend,
            **extra,
        }

    # ------------------------------------------------------------------
    # 内部：缓存与后端查询
    # ------------------------------------------------------------------

    def _load_cache(self) -> None:
        if self.storage is None:
            self.memories_by_id = {}
            return
        try:
            items = self.storage.load_all()
        except Exception:
            items = []
        self.memories_by_id = {m.id: m for m in items if m and m.id}
        self._enforce_capacity_limit()

    def _enforce_capacity_limit(self) -> None:
        max_capacity = int(getattr(self.config, "max_capacity", 0) or 0)
        if max_capacity <= 0:
            return
        if len(self.memories_by_id) <= max_capacity:
            return

        # 淘汰策略：优先淘汰 重要性低 + 时间更早 的记忆
        items = list(self.memories_by_id.values())
        items.sort(key=lambda m: (m.importance, m.timestamp))
        to_remove = items[: max(0, len(items) - max_capacity)]
        for m in to_remove:
            try:
                if self.storage:
                    self.storage.delete(m.id)
            finally:
                self.memories_by_id.pop(m.id, None)

    def _backend_query(
        self,
        keyword: Optional[str],
        memory_type: Optional[str],
        since: Optional[datetime],
        until: Optional[datetime],
        min_importance: Optional[float],
        limit: int,
    ) -> List[MemoryItem]:
        if self.storage is None:
            return []

        # SQLiteStorage 支持 query()，尽量用后端过滤减少 IO
        if hasattr(self.storage, "query") and callable(getattr(self.storage, "query")):
            try:
                return self.storage.query(
                    keyword=keyword,
                    memory_type=memory_type,
                    since=since,
                    until=until,
                    min_importance=min_importance,
                    limit=limit,
                )
            except Exception:
                # 后端 query 失败时，退化到全量加载
                pass

        items = list(self.memories_by_id.values())
        if keyword:
            keyword_lower = keyword.lower()
            items = [m for m in items if keyword_lower in (m.content or "").lower()]
        if memory_type:
            items = [m for m in items if m.memory_type == memory_type]
        if since:
            items = [m for m in items if m.timestamp >= since]
        if until:
            items = [m for m in items if m.timestamp <= until]
        if min_importance is not None:
            items = [m for m in items if (m.importance or 0.0) >= min_importance]

        items.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
        return items[:limit]

    @staticmethod
    def _score_query_match(query: str, memory: MemoryItem) -> float:
        """
        一个轻量打分：token 重叠 + 子串匹配，保证无需额外依赖即可工作。
        """
        q = (query or "").strip().lower()
        c = (memory.content or "").strip().lower()
        if not q or not c:
            return 0.0
        if q in c:
            return 2.0

        q_tokens = [t for t in q.split() if t]
        c_tokens = set(t for t in c.split() if t)
        overlap = sum(1 for t in q_tokens if t in c_tokens)
        if overlap:
            return overlap / max(1, len(q_tokens))

        q_chars = set(q.replace(" ", ""))
        c_chars = set(c.replace(" ", ""))
        char_overlap = q_chars.intersection(c_chars)
        if not char_overlap:
            return 0.0
        return len(char_overlap) / max(1, len(q_chars.union(c_chars))) * 0.6



