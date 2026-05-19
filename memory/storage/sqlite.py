import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any

from memory.base import MemoryItem
from memory.storage.base import StorageBackend

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    id        TEXT PRIMARY KEY,
    content   TEXT NOT NULL,
    memory_type TEXT,
    timestamp TEXT NOT NULL,
    importance REAL DEFAULT 0.5,
    metadata  TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_timestamp  ON memories(timestamp);
CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance);
CREATE INDEX IF NOT EXISTS idx_type       ON memories(memory_type);
"""


class SQLiteStorage(StorageBackend):
    """SQLite 持久化存储，单文件，无额外依赖"""

    def __init__(self, db_path: str = "./memory_data/memories.db"):
        self.db_path = db_path
        # 确保父目录存在
        import os
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_CREATE_TABLE)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # 序列化 / 反序列化
    # ------------------------------------------------------------------

    @staticmethod
    def _to_row(item: MemoryItem) -> tuple:
        return (
            item.id,
            item.content,
            item.memory_type,
            item.timestamp.isoformat(),
            item.importance,
            json.dumps(item.metadata, ensure_ascii=False),
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            content=row["content"],
            memory_type=row["memory_type"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            importance=row["importance"],
            metadata=json.loads(row["metadata"]),
        )

    # ------------------------------------------------------------------
    # StorageBackend 接口
    # ------------------------------------------------------------------

    def save(self, item: MemoryItem) -> str:
        sql = """
            INSERT INTO memories (id, content, memory_type, timestamp, importance, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                content     = excluded.content,
                memory_type = excluded.memory_type,
                timestamp   = excluded.timestamp,
                importance  = excluded.importance,
                metadata    = excluded.metadata
        """
        with self._connect() as conn:
            conn.execute(sql, self._to_row(item))
        return item.id

    def load(self, memory_id: str) -> Optional[MemoryItem]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._from_row(row) if row else None

    def load_all(self) -> List[MemoryItem]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY timestamp DESC"
            ).fetchall()
        return [self._from_row(r) for r in rows]

    def delete(self, memory_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
        return cursor.rowcount > 0

    def update(self, memory_id: str, fields: Dict[str, Any]) -> bool:
        if not fields:
            return False

        allowed = {"content", "memory_type", "timestamp", "importance", "metadata"}
        set_clauses = []
        values = []

        for key, value in fields.items():
            if key not in allowed:
                continue
            if key == "timestamp" and isinstance(value, datetime):
                value = value.isoformat()
            if key == "metadata" and isinstance(value, dict):
                value = json.dumps(value, ensure_ascii=False)
            set_clauses.append(f"{key} = ?")
            values.append(value)

        if not set_clauses:
            return False

        values.append(memory_id)
        sql = f"UPDATE memories SET {', '.join(set_clauses)} WHERE id = ?"
        with self._connect() as conn:
            cursor = conn.execute(sql, values)
        return cursor.rowcount > 0

    def exists(self, memory_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return row is not None

    def clear(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM memories")

    # ------------------------------------------------------------------
    # 扩展查询（供 EpisodicMemory 使用）
    # ------------------------------------------------------------------

    def query(
        self,
        keyword: str = None,
        memory_type: str = None,
        since: datetime = None,
        until: datetime = None,
        min_importance: float = None,
        limit: int = 50,
    ) -> List[MemoryItem]:
        """按条件过滤记忆，返回按 importance DESC、timestamp DESC 排序的结果"""
        conditions = []
        values = []

        if keyword:
            conditions.append("content LIKE ?")
            values.append(f"%{keyword}%")
        if memory_type:
            conditions.append("memory_type = ?")
            values.append(memory_type)
        if since:
            conditions.append("timestamp >= ?")
            values.append(since.isoformat())
        if until:
            conditions.append("timestamp <= ?")
            values.append(until.isoformat())
        if min_importance is not None:
            conditions.append("importance >= ?")
            values.append(min_importance)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT * FROM memories
            {where}
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
        """
        values.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, values).fetchall()
        return [self._from_row(r) for r in rows]
