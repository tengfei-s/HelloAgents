import uuid
from datetime import datetime
from mimetypes import init
from typing import Optional

from memory.base import MemoryConfig, MemoryItem
from memory.types.working import WorkingMemory
from memory.types.episodic import EpisodicMemory


class MemoryManager:
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        user_id: str = "default_user",
        enable_working: bool = True,
        enable_episodic: bool = True,
        enable_semantic: bool = False,
        enable_perceptual: bool = False,
    ):
        self.memories = {}
        self.config = config or MemoryConfig()
        self.user_id = user_id

        if enable_working:
            self.memories["working"] = WorkingMemory(self.config)
        if enable_episodic:
            self.memories["episodic"] = EpisodicMemory(self.config)

    def add_memory(
        self,
        content,
        memory_type: str = "working",
        importance: float = 0.5,
        metadata=None,
    ) -> str:
        memory = self.memories.get(memory_type)
        if memory is None:
            raise ValueError(f"Memory type is not enabled or supported: {memory_type}")

        metadata = metadata or {}
        metadata.setdefault("user_id", self.user_id)

        memory_id = str(uuid.uuid4())
        memory_item = MemoryItem(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            timestamp=datetime.now(),
            importance=importance,
            metadata=metadata,
        )
        memory.add(memory_item)
        return memory_id

    def retrieve_memories(self, query, types=None, limit=10, **filters) -> list[MemoryItem]:
        memory_list = []

        if types is None:
            target_types = list(self.memories.keys())
        elif isinstance(types, str):
            target_types = [types]
        else:
            target_types = list(types)

        for memory_type in target_types:
            memory = self.memories.get(memory_type)
            if memory is None:
                continue
            retrieved = memory.retrieve(
                query,
                limit=limit,
                user_id=self.user_id,
                **filters,
            )
            memory_list.extend(retrieved or [])

        memory_list.sort(key=lambda item: (item.importance, item.timestamp), reverse=True)
        return memory_list[:limit]

    def update_memory(self, memory_id, memory_type=None, **fields) -> bool:
        target_types = [memory_type] if memory_type else list(self.memories.keys())

        for target_type in target_types:
            memory = self.memories.get(target_type)
            if memory and memory.has_memory(memory_id):
                return memory.update(memory_id, **fields)
        return False

    def remove_memory(self, memory_id, memory_type=None) -> bool:
        target_types = [memory_type] if memory_type else list(self.memories.keys())

        for target_type in target_types:
            memory = self.memories.get(target_type)
            if memory and memory.has_memory(memory_id):
                return memory.remove(memory_id)
        return False

    def consolidate_memories(self, from_type="working", to_type="episodic") -> int:
        source = self.memories.get(from_type)
        target = self.memories.get(to_type)
        if source is None or target is None:
            return 0

        if not hasattr(source, "memories"):
            return 0

        consolidated_count = 0
        for item in list(source.memories):
            if item.importance >= self.config.importance_threshold:
                copied_item = item.model_copy(update={"memory_type": to_type})
                target.add(copied_item)
                source.remove(item.id)
                consolidated_count += 1
        return consolidated_count

    def get_memory_stats(self) -> dict:
        return {
            "user_id": self.user_id,
            "enabled_types": list(self.memories.keys()),
            "types": {
                name: memory.get_stats()
                for name, memory in self.memories.items()
            },
        }

    def print_all_memories(self):
        for memory_type, memory in self.memories.items():
            print(f"=== {memory_type} memories ===")
            items = self._get_all_items(memory)
            if not items:
                print("(empty)")
                continue

            for item in items:
                print(
                    f"[{item.id}] "
                    f"type={item.memory_type} "
                    f"importance={item.importance} "
                    f"time={item.timestamp}"
                )
                print(f"  content: {item.content}")
                if item.metadata:
                    print(f"  metadata: {item.metadata}")

    def clear(self, types=None):
        if types is None:
            target_types = list(self.memories.keys())
        elif isinstance(types, str):
            target_types = [types]
        else:
            target_types = list(types)

        for memory_type in target_types:
            memory = self.memories.get(memory_type)
            if memory:
                memory.clear()

    @staticmethod
    def _get_all_items(memory) -> list[MemoryItem]:
        if hasattr(memory, "memories"):
            return list(memory.memories)
        if hasattr(memory, "memories_by_id"):
            return list(memory.memories_by_id.values())
        if memory.storage:
            return memory.storage.load_all()
        return []


if __name__ == '__main__':
    memory_config = MemoryConfig()
    memory_manager = MemoryManager(memory_config,user_id="test", enable_working=True,enable_episodic=True)

    memory_manager.add_memory("this is a working memory",memory_type="working")
    memory_manager.add_memory("this is a episodic memory",memory_type="episodic")

    memory_manager.add_memory("用户喜欢使用python写代码",memory_type="working")
    memory_manager.add_memory("用户喜欢使用python写人工智能程序", memory_type="episodic")

    print(memory_manager.retrieve_memories("用户喜欢使用什么编程语言",types=["working","episodic"]))
    memory_manager.print_all_memories()

