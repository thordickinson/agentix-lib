from __future__ import annotations
from typing import Any, Dict, List
from pydantic import BaseModel

from agentix.models import Tool, AgentState
from agentix.stack.view import View
from ..repo import PropertyRepo

class PropertyDeleteView(View):
    screen_key = "property_delete"

    def instructions(self, agent_state: AgentState, view_state: Dict[str, Any]) -> str:
        return f"Eliminar propiedad {view_state.get('property_id')}. Usa __confirm o __cancel."

    def build_tools(self, agent_state: AgentState, view_state: Dict[str, Any]) -> List[Tool]:
        class NoInput(BaseModel): pass
        tools: List[Tool] = []

        async def _confirm(_i: NoInput, v: Dict[str, Any], a: AgentState, uid: str, sid: str):
            repo = PropertyRepo()
            pid = v.get("property_id")
            ok = await repo.delete(pid)
            res = {"deleted": ok, "property_id": pid}
            v["__pending_result"] = res
            return {"nav": "confirm", "result": res}

        async def _cancel(_i: NoInput, v: Dict[str, Any], a: AgentState, uid: str, sid: str):
            return {"nav": "cancel"}

        tools += [
            Tool(name="__confirm", desc="Confirmar eliminación", input_model=NoInput, fn=_confirm),
            Tool(name="__cancel", desc="Cancelar", input_model=NoInput, fn=_cancel),
        ]
        return tools
