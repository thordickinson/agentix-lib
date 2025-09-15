from __future__ import annotations
from typing import Any, Dict, List
from pydantic import BaseModel

from agentix import tool_from_fn
from agentix.models import Tool, AgentState
from agentix.stack.view import View
from ..repo import PropertyRepo


class PropertyEditView(View):
    screen_key = "property_edit"

    def instructions(self, agent_state: AgentState, view_state: Dict[str, Any]) -> str:
        pid = view_state.get("property_id")
        return (
            f"Editando propiedad {pid}.\n"
            "- set_field {field,value}\n"
            "- open_client_selector {hint}\n"
            "- __confirm para guardar, __cancel para descartar\n"
        )

    def build_tools(self, agent_state: AgentState, view_state: Dict[str, Any]) -> List[Tool]:
        tools: List[Tool] = []

        async def set_field():
            print("set_field")
            return {"nav": "cancel"}

        async def open_client_selector():
            return {"nav": "cancel"}

        class NoInput(BaseModel): pass

        async def _confirm(_i: NoInput, v: Dict[str, Any], a: AgentState, uid: str, sid: str):
            repo = PropertyRepo()
            pid = v.get("property_id")
            changes = v.get("changes", {})
            res = await repo.update(pid, changes)
            v["__pending_result"] = res
            return {"nav": "confirm", "result": res}

        async def _cancel(_i: NoInput, v: Dict[str, Any], a: AgentState, uid: str, sid: str):
            return {"nav": "cancel"}

        tools += [
            tool_from_fn(set_field),
            tool_from_fn(open_client_selector),
            tool_from_fn(_confirm),
            tool_from_fn(_cancel),
        ]
        return tools
