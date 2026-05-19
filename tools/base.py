from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel

class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None



class Tool(ABC):
    def __init__(self, name: str, description: str,expandable: bool = False):
        self.name: str = name
        self.description: str = description
        self.expandable = expandable
    @abstractmethod
    def run(self,parameters:Dict[str, Any]):
        pass
    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        pass