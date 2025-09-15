from __future__ import annotations
from typing import Any, Dict, List

from agentix import tool_from_fn
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
        tools: List[Tool] = []

        async def open_list():
            """
            Abre la lista de propiedades del usuario actual
            """
            return View.call_view("property_list")

        async def open_create():
            return View.call_view("property_create")

        tools.append(tool_from_fn(open_list))
        tools.append(tool_from_fn(open_create))
        return tools
