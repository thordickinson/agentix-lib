from __future__ import annotations
from typing import Any, Dict, List
from pydantic import BaseModel

from agentix.models import Tool, AgentState
from agentix.stack.view import View
from ..repo import PropertyRepo

class SetFieldInput(BaseModel):
    field: str
    value: Any

class PropertyCreateView(View):
    screen_key = "property_create"

    def instructions(self, agent_state: AgentState, view_state: Dict[str, Any]) -> str:
        return (
            "Crear propiedad.\n"
            "- set_property_field {field, value}\n"
            "- __confirm para guardar, __cancel para descartar\n"
        )

    def build_tools(self, agent_state: AgentState, view_state: Dict[str, Any]) -> List[Tool]:
        tools: List[Tool] = []

        async def set_field(inputs: SetFieldInput, v: Dict[str, Any], a: AgentState, uid: str, sid: str):
            v.setdefault("fields", {})
            v["fields"][inputs.field] = inputs.value
            return {"ok": True, "changed": {inputs.field: inputs.value}}

        class NoInput(BaseModel): pass

        async def _confirm(_i: NoInput, v: Dict[str, Any], a: AgentState, uid: str, sid: str):
            repo = PropertyRepo()
            data = v.get("fields", {})
            res = await repo.create(data)
            v["__pending_result"] = res
            return {"nav": "confirm", "result": res}

        async def _cancel(_i: NoInput, v: Dict[str, Any], a: AgentState, uid: str, sid: str):
            return {"nav": "cancel"}

        tools += [
            Tool(name="set_property_field", desc="Asignar campo", input_model=SetFieldInput, fn=set_field),
            Tool(name="__confirm", desc="Guardar", input_model=NoInput, fn=_confirm),
            Tool(name="__cancel", desc="Cancelar", input_model=NoInput, fn=_cancel),
        ]
        return tools
