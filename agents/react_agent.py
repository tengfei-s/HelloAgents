import re
from typing import List

from core.message import Message
from core.agent import Agent
from core.config import Config
from tools.search_tool import search
from tools.registry import ToolRegistry
from core.llm import HelloAgentsLLM

# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""


class ReactAgent(Agent):
    def __init__(self,
                 name: str,
                 llm: HelloAgentsLLM,
                 tool_registry: ToolRegistry,
                 system_prompt:str = None,
                 config:Config = None,
                 max_steps: int = 5
                 ) -> None:
        super().__init__(name,llm,system_prompt,config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps

        self.current_history: List[str] = []

    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent"""
        self.current_history = []
        current_step = 0
        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")
            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=input_text,
                history=history_str
            )

            message = [{"role":"user","content":prompt}]
            response = self.llm.invoke(message)

            thought, action = self._parse_output(response)
            if thought:
                print(f"🤔 思考: {thought}")
            if not action:
                print("⚠️ 警告：未能解析出有效的Action，流程终止。")
                break

            if action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                print(f"🎉 最终答案: {final_answer}")
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or tool_input is None:
                self.current_history.append("Observation: 无效的Action格式，请检查。")
                continue
            print(f"🎬 行动: {tool_name}[{tool_input}]")

            # 调用工具
            observation = self.tool_registry.execute_tool(tool_name, tool_input)
            print(f"👀 观察: {observation}")

            # 更新历史
            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")

        print("⏰ 已达到最大步数，流程终止。")
        final_answer = "抱歉，我无法在限定步数内完成这个任务。"

        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))

        return final_answer


    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。
        """
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def _parse_action_input(self, action_text: str) -> str:
        """解析行动输入"""
        match = re.match(r"\w+\[(.*)\]", action_text)
        return match.group(1) if match else ""



if __name__ == '__main__':
    llm_client = HelloAgentsLLM()
    tool_registry = ToolRegistry()
    tool_registry.register_function("Search","一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。",search)
    reactagent = ReactAgent("searchreact",llm=HelloAgentsLLM(),tool_registry=tool_registry,max_steps=5)
    reactagent.run("英伟达最新的GPU型号是什么")