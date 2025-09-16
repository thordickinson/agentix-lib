from __future__ import annotations
from typing import Protocol, Tuple, List
from .models import Tool, AgentContext
from pydantic import BaseModel

class LLMInput(BaseModel):
    system: str
    tools: list[Tool]


class ContextManager(Protocol):
    def build(self, agent_state: AgentContext) -> LLMInput:
        """
        Devuelve:
          - system_message (str): bloque para añadir al prompt (breadcrumb / instrucciones de vista / memoria)
          - tools (List[Tool]): tools disponibles en el turno actual (derivadas del estado/contexto)
        """
        ...


class SimpleContextManager(ContextManager):
    
    def __init__(self, system: str, tools: list[Tool] = []):
        super().__init__()
        self.system = system
        self.tools = tools
    
    def build(self, agent_state):
        return LLMInput(system=self.system, tools=self.tools)
