from core.config import Config
from core.llm import HelloAgentsLLM
from agents.simple_agent import SimpleAgent
from tools.buitin import memory_tool
from tools.registry import ToolRegistry
from tools.buitin.memory_tool import MemoryTool
from memory.manager import MemoryManager
from memory.base import MemoryConfig


config = Config()
config.from_env()

llm = HelloAgentsLLM()

memory_config = MemoryConfig()

memory_manager = MemoryManager(memory_config,user_id="test", enable_working=True,enable_episodic=True)

memory_tool = MemoryTool(memory_manager)

tool_registry = ToolRegistry()
tool_registry.register_tool(memory_tool)

agent = SimpleAgent(
        name="TestAgent",
        llm=llm,
        tool_registry=tool_registry,
        enable_tool_calling=True,
        system_prompt=(
        "你是一个有记忆能力的助手。"
        "当用户要求你记住信息时，必须调用 memory 工具。"
        "保存记忆时使用格式："
        "[TOOL_CALL:memory:action=add,content=要记住的内容,memory_type=working,importance=0.8]"
        "查询记忆时使用格式："
        "[TOOL_CALL:memory:action=search,query=查询内容,limit=5]"
        "不要使用 key 或 value 参数。"
    )
)

print(agent.run("请记住：我喜欢使用 Python 写人工智能程序"))
print(agent.run("我喜欢用什么语言写程序？"))

