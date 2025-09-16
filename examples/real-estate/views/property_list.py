from __future__ import annotations
from typing import Any, Dict, List
from pydantic import BaseModel

from agentix import tool_from_fn
from agentix.models import Tool, AgentContext
from agentix.stack.view import View
from ..repo import PropertyRepo

class ListInput(BaseModel):
    limit: int = 10

class OpenEditorInput(BaseModel):
    property_id: str

class OpenDeleteInput(BaseModel):
    property_id: str

class PropertyListView(View):
    screen_key = "property_list"

    def instructions(self, agent_state: AgentContext, view_state: Dict[str, Any]) -> str:
        return (
            "Lista de propiedades.\n"
            "- list_properties {limit}\n"
            "- open_property_editor {property_id}\n"
            "- open_property_delete {property_id}\n"
            "- __cancel para volver\n"
        )

    def build_tools(self, agent_state: AgentContext, view_state: Dict[str, Any]) -> List[Tool]:
        tools: List[Tool] = []

        async def list_properties(inputs: ListInput, vstate: Dict[str, Any], astate: AgentContext, uid: str, sid: str):
            repo = PropertyRepo()
            items = await repo.list(limit=inputs.limit)
            vstate["items"] = items
            return {"properties": items}

        async def open_editor(inputs: OpenEditorInput, vstate: Dict[str, Any], astate: AgentContext, uid: str, sid: str):
            return View.call_view("property_edit", {"property_id": inputs.property_id, "changes": {}, "relations": {}}, return_path=None)

        async def open_delete(inputs: OpenDeleteInput, vstate: Dict[str, Any], astate: AgentContext, uid: str, sid: str):
            return View.call_view("property_delete", {"property_id": inputs.property_id}, return_path=None)

        class NoInput(BaseModel): pass
        async def _cancel(_i: NoInput, v: Dict[str, Any], a: AgentContext, uid: str, sid: str):
            return {"nav": "cancel"}

        tools += [
            tool_from_fn(list_properties), tool_from_fn(open_editor),
            tool_from_fn(open_delete), tool_from_fn(_cancel),
        ]
        return tools
