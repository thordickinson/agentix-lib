from __future__ import annotations
from typing import Protocol, Tuple, List, Dict, Any
from .models import AgentState, Tool

class ContextManager(Protocol):
    def build(self, agent_state: AgentState, user_id: str, session_id: str) -> Tuple[str, List[Tool]]:
        """
        Devuelve:
          - system_message (str): bloque para añadir al prompt (breadcrumb / instrucciones de vista / memoria)
          - tools (List[Tool]): tools disponibles en el turno actual (derivadas del estado/contexto)
        """
        ...

    async def handle_nav(self, agent_state: AgentState, user_id: str, session_id: str, tool_output: Dict[str, Any]) -> None:
        """
        Procesa la salida de una tool (intenciones de navegación: push_view, confirm, cancel, etc.),
        actualiza agent_state y resuelve retornos (__last_call, return_path, etc.).
        """
        ...
