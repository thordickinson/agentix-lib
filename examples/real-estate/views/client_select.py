from __future__ import annotations
from typing import Any, Dict, List
from pydantic import BaseModel

from agentix.models import Tool, AgentContext
from agentix.stack.view import View

class SearchClientsInput(BaseModel):
    query: str
    top_k: int = 5

class SelectClientInput(BaseModel):
    client_id: str
    display_name: str | None = None

class ClientSelectView(View):
    screen_key = "client_select"

    def instructions(self, agent_state: AgentContext, view_state: Dict[str, Any]) -> str:
        return (
            "Selector de clientes.\n"
            "- search_clients {query,top_k}\n"
            "- select_client {client_id,display_name}\n"
            "- __confirm para devolver el cliente, __cancel para salir\n"
        )

    def build_tools(self, agent_state: AgentContext, view_state: Dict[str, Any]) -> List[Tool]:
        tools: List[Tool] = []

        async def search_clients(inputs: SearchClientsInput, v: Dict[str, Any], a: AgentContext, uid: str, sid: str):
            q = (inputs.query or "").strip().lower()
            fake = [{"client_id": f"c_{i}", "name": f"{q.title()} {i}"} for i in range(1, inputs.top_k + 1)]
            v["candidates"] = fake
            return {"results": fake}

        async def select_client(inputs: SelectClientInput, v: Dict[str, Any], a: AgentContext, uid: str, sid: str):
            client = {"client_id": inputs.client_id, "name": inputs.display_name}
            v["selected"] = client
            v["__pending_result"] = client
            return {"selected": client}

        class NoInput(BaseModel): pass

        async def _confirm(_i: NoInput, v: Dict[str, Any], a: AgentContext, uid: str, sid: str):
            return {"nav": "confirm", "result": v.get("__pending_result")}

        async def _cancel(_i: NoInput, v: Dict[str, Any], a: AgentContext, uid: str, sid: str):
            return {"nav": "cancel"}

        tools += [
            Tool(name="search_clients", desc="Buscar clientes", input_model=SearchClientsInput, fn=search_clients),
            Tool(name="select_client", desc="Seleccionar cliente", input_model=SelectClientInput, fn=select_client),
            Tool(name="__confirm", desc="Confirmar selección", input_model=NoInput, fn=_confirm),
            Tool(name="__cancel", desc="Cancelar", input_model=NoInput, fn=_cancel),
        ]
        return tools
