import json
from typing import Any, Dict, List, Optional

from memory.manager import MemoryManager
from tools.base import Tool, ToolParameter


class MemoryTool(Tool):
    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        name: str = "memory",
        description: str = "Add, search, update, remove, and inspect agent memory.",
    ) -> None:
        super().__init__(name=name, description=description)
        self.memory_manager = memory_manager or MemoryManager()

    def run(self, parameters: Dict[str, Any]):
        action = str(parameters.get("action", "")).strip().lower()

        if action in ("add", "store", "remember"):
            return self._add(parameters)
        if action in ("search", "retrieve", "recall"):
            return self._search(parameters)
        if action == "update":
            return self._update(parameters)
        if action in ("remove", "delete"):
            return self._remove(parameters)
        if action in ("stats", "stat"):
            return self._stats()
        if action in ("print_all", "list", "all"):
            return self._format_all_memories()

        return (
            "Unknown memory action. Supported actions: "
            "add, search, update, remove, stats, print_all."
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="Action: add/search/update/remove/stats/print_all.",
                required=True,
            ),
            ToolParameter(
                name="content",
                type="string",
                description="Memory content to add or update.",
                required=False,
            ),
            ToolParameter(
                name="query",
                type="string",
                description="Query used to search memory.",
                required=False,
            ),
            ToolParameter(
                name="memory_type",
                type="string",
                description="Memory type: working or episodic.",
                required=False,
                default="working",
            ),
            ToolParameter(
                name="memory_id",
                type="string",
                description="Memory id for update or remove.",
                required=False,
            ),
            ToolParameter(
                name="importance",
                type="number",
                description="Memory importance from 0 to 1.",
                required=False,
                default=0.5,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="Maximum number of search results.",
                required=False,
                default=5,
            ),
        ]

    def _add(self, parameters: Dict[str, Any]) -> str:
        content = parameters.get("content") or parameters.get("input")
        if not content:
            return "Missing required parameter: content."

        memory_type = parameters.get("memory_type", "working")
        importance = self._to_float(parameters.get("importance"), default=0.5)

        try:
            memory_id = self.memory_manager.add_memory(
                content=str(content),
                memory_type=str(memory_type),
                importance=importance,
            )
        except Exception as exc:
            return f"Failed to add memory: {exc}"

        return f"Memory added. id={memory_id}, type={memory_type}, importance={importance}"

    def _search(self, parameters: Dict[str, Any]) -> str:
        query = parameters.get("query") or parameters.get("content") or parameters.get("input")
        if not query:
            return "Missing required parameter: query."

        limit = self._to_int(parameters.get("limit"), default=5)
        memory_type = parameters.get("memory_type")
        types = [str(memory_type)] if memory_type else None

        try:
            results = self.memory_manager.retrieve_memories(
                query=str(query),
                types=types,
                limit=limit,
            )
        except Exception as exc:
            return f"Failed to search memory: {exc}"

        if not results:
            return "No related memories found."

        lines = ["Related memories:"]
        for item in results:
            lines.append(
                f"- [{item.memory_type}] {item.content} "
                f"(id={item.id}, importance={item.importance})"
            )
        return "\n".join(lines)

    def _update(self, parameters: Dict[str, Any]) -> str:
        memory_id = parameters.get("memory_id") or parameters.get("id")
        if not memory_id:
            return "Missing required parameter: memory_id."

        fields: Dict[str, Any] = {}
        if parameters.get("content") is not None:
            fields["content"] = str(parameters["content"])
        if parameters.get("importance") is not None:
            fields["importance"] = self._to_float(parameters.get("importance"), default=0.5)

        if not fields:
            return "No update fields provided."

        memory_type = parameters.get("memory_type")
        ok = self.memory_manager.update_memory(
            str(memory_id),
            memory_type=str(memory_type) if memory_type else None,
            **fields,
        )
        return "Memory updated." if ok else "Memory not found."

    def _remove(self, parameters: Dict[str, Any]) -> str:
        memory_id = parameters.get("memory_id") or parameters.get("id")
        if not memory_id:
            return "Missing required parameter: memory_id."

        memory_type = parameters.get("memory_type")
        ok = self.memory_manager.remove_memory(
            str(memory_id),
            memory_type=str(memory_type) if memory_type else None,
        )
        return "Memory removed." if ok else "Memory not found."

    def _stats(self) -> str:
        return json.dumps(
            self.memory_manager.get_memory_stats(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    def _format_all_memories(self) -> str:
        lines: List[str] = []
        for memory_type, memory in self.memory_manager.memories.items():
            lines.append(f"=== {memory_type} memories ===")
            items = self.memory_manager._get_all_items(memory)
            if not items:
                lines.append("(empty)")
                continue

            for item in items:
                lines.append(
                    f"[{item.id}] type={item.memory_type} "
                    f"importance={item.importance} time={item.timestamp}"
                )
                lines.append(f"  content: {item.content}")
                if item.metadata:
                    lines.append(f"  metadata: {item.metadata}")
        return "\n".join(lines) if lines else "No memory modules enabled."

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        if value is None or value == "":
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_int(value: Any, default: int) -> int:
        if value is None or value == "":
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
