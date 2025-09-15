from __future__ import annotations
from typing import Any, Dict, List
from pydantic import BaseModel

from agentix.models import Tool, AgentState
from agentix.stack.view import View

class IndexView(View):
    screen_key = "index"

    def instructions(self, agent_state: AgentState, view_state: Dict[str, Any]) -> str:
        return (
            "Bienvenido al panel inmobiliario.\n"
            "- Abre la lista: open_property_list\n"
            "- Crea una nueva: open_property_create\n"
        )

    def build_tools(self, agent_state: AgentState, view_state: Dict[str, Any]) -> List[Tool]:
        class NoInput(BaseModel): pass
        tools: List[Tool] = []

        async def open_list(_i: NoInput, vstate: Dict[str, Any], astate: AgentState, uid: str, sid: str):
            return View.call_view("property_list")

        async def open_create(_i: NoInput, vstate: Dict[str, Any], astate: AgentState, uid: str, sid: str):
            return View.call_view("property_create")

        tools.append(Tool(name="open_property_list", desc="Abrir lista de propiedades", input_model=NoInput, fn=open_list))
        tools.append(Tool(name="open_property_create", desc="Crear nueva propiedad", input_model=NoInput, fn=open_create))
        return tools
