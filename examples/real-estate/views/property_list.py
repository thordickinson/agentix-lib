from __future__ import annotations
from typing import Any, Dict, List
from pydantic import BaseModel

from agentix.models import Tool, AgentState
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

    def instructions(self, agent_state: AgentState, view_state: Dict[str, Any]) -> str:
        return (
            "Lista de propiedades.\n"
            "- list_properties {limit}\n"
            "- open_property_editor {property_id}\n"
            "- open_property_delete {property_id}\n"
            "- __cancel para volver\n"
        )

    def build_tools(self, agent_state: AgentState, view_state: Dict[str, Any]) -> List[Tool]:
        tools: List[Tool] = []

        async def list_properties(inputs: ListInput, vstate: Dict[str, Any], astate: AgentState, uid: str, sid: str):
            repo = PropertyRepo()
            items = await repo.list(limit=inputs.limit)
            vstate["items"] = items
            return {"properties": items}

        async def open_editor(inputs: OpenEditorInput, vstate: Dict[str, Any], astate: AgentState, uid: str, sid: str):
            return View.call_view("property_edit", {"property_id": inputs.property_id, "changes": {}, "relations": {}}, return_path=None)

        async def open_delete(inputs: OpenDeleteInput, vstate: Dict[str, Any], astate: AgentState, uid: str, sid: str):
            return View.call_view("property_delete", {"property_id": inputs.property_id}, return_path=None)

        class NoInput(BaseModel): pass
        async def _cancel(_i: NoInput, v: Dict[str, Any], a: AgentState, uid: str, sid: str):
            return {"nav": "cancel"}

        tools += [
            Tool(name="list_properties", desc="Listar propiedades", input_model=ListInput, fn=list_properties),
            Tool(name="open_property_editor", desc="Editar propiedad", input_model=OpenEditorInput, fn=open_editor),
            Tool(name="open_property_delete", desc="Eliminar propiedad", input_model=OpenDeleteInput, fn=open_delete),
            Tool(name="__cancel", desc="Volver", input_model=NoInput, fn=_cancel),
        ]
        return tools
