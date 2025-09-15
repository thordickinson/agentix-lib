from __future__ import annotations
from typing import Any, Dict, List
from pydantic import BaseModel

from agentix.models import Tool, AgentState
from agentix.stack.view import View
from ..repo import PropertyRepo

class SetFieldInput(BaseModel):
    field: str
    value: Any

class OpenClientSelectorInput(BaseModel):
    hint: str | None = None

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

        async def set_field(inputs: SetFieldInput, v: Dict[str, Any], a: AgentState, uid: str, sid: str):
            v.setdefault("changes", {})
            v["changes"][inputs.field] = inputs.value
            return {"ok": True, "changed": {inputs.field: inputs.value}}

        async def open_client_selector(inputs: OpenClientSelectorInput, v: Dict[str, Any], a: AgentState, uid: str, sid: str):
            return View.call_view("client_select", {"hint": inputs.hint}, return_path="relations.client")

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
            Tool(name="set_field", desc="Cambiar un campo", input_model=SetFieldInput, fn=set_field),
            Tool(name="open_client_selector", desc="Asociar cliente (subvista)", input_model=OpenClientSelectorInput, fn=open_client_selector),
            Tool(name="__confirm", desc="Guardar cambios", input_model=NoInput, fn=_confirm),
            Tool(name="__cancel", desc="Descartar", input_model=NoInput, fn=_cancel),
        ]
        return tools
