import json
import os
from typing import List, Optional, Dict, Any

from memory.base import MemoryItem
from memory.storage.base import StorageBackend


class LocalFileStorage(StorageBackend):
    """本地 JSON 文件存储，每条记忆存为独立的 .json 文件"""

    def __init__(self, storage_path: str = "./memory_data"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

    def _file_path(self, memory_id: str) -> str:
        return os.path.join(self.storage_path, f"{memory_id}.json")

    def save(self, item: MemoryItem) -> str:
        path = self._file_path(item.id)
        with open(path, "w", encoding="utf-8") as f:
            f.write(item.model_dump_json(indent=2))
        return item.id

    def load(self, memory_id: str) -> Optional[MemoryItem]:
        path = self._file_path(memory_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return MemoryItem.model_validate_json(f.read())

    def load_all(self) -> List[MemoryItem]:
        items = []
        for filename in os.listdir(self.storage_path):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(self.storage_path, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    items.append(MemoryItem.model_validate_json(f.read()))
            except Exception:
                pass  # 跳过损坏的文件
        return items

    def delete(self, memory_id: str) -> bool:
        path = self._file_path(memory_id)
        if not os.path.exists(path):
            return False
        os.remove(path)
        return True

    def update(self, memory_id: str, fields: Dict[str, Any]) -> bool:
        item = self.load(memory_id)
        if item is None:
            return False
        updated = item.model_copy(update=fields)
        self.save(updated)
        return True

    def exists(self, memory_id: str) -> bool:
        return os.path.exists(self._file_path(memory_id))

    def clear(self):
        for filename in os.listdir(self.storage_path):
            if filename.endswith(".json"):
                os.remove(os.path.join(self.storage_path, filename))
