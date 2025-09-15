from __future__ import annotations
from typing import Any, Dict, List, Callable, Optional
from dataclasses import dataclass
from agentix.models import Tool, AgentState

@dataclass
class NavIntent:
    kind: str                 # "push_view" | "confirm" | "cancel"
    target: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    return_path: Optional[str] = None

class View:
    screen_key: str
    def instructions(self, agent_state: AgentState, view_state: Dict[str, Any]) -> str: return ""
    def memory_instructions(self, agent_state: AgentState, view_state: Dict[str, Any]) -> str: return ""
    def build_tools(self, agent_state: AgentState, view_state: Dict[str, Any]) -> List[Tool]: return []
    @staticmethod
    def call_view(target_screen_key: str, params: Dict[str, Any] | None = None, return_path: str | None = None) -> Dict[str, Any]:
        return {"nav": "push_view", "target": target_screen_key, "params": params or {}, "return_path": return_path}

class ViewRouter:
    def __init__(self):
        self._factories: Dict[str, Callable[[], View]] = {}
        self._index_key: Optional[str] = None
    def register(self, screen_key: str, factory: Callable[[], View]) -> None:
        self._factories[screen_key] = factory
    def set_index(self, screen_key: str) -> None:
        if screen_key not in self._factories:
            raise KeyError(f"View '{screen_key}' no registrada")
        self._index_key = screen_key
    def get(self, screen_key: str) -> View:
        if screen_key not in self._factories:
            raise KeyError(f"View '{screen_key}' no registrada")
        return self._factories[screen_key]()
    def index_key(self) -> Optional[str]:
        return self._index_key
