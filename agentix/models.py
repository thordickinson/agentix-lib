from __future__ import annotations
from typing import Any, Dict, Literal, Union, Type
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ValidationError

# ===== Mensajes y estado =====
Role = Literal["system", "user", "assistant"]

class Message(BaseModel):
    role: Role
    content: str
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    meta: Dict[str, Any] = Field(default_factory=dict)

    def to_wire(self) -> Dict[str, str]:
        """Mensaje en formato listo para el LLM."""
        return {"role": self.role, "content": self.content}

class AgentState(BaseModel):
    memory: Dict[str, Any] = Field(default_factory=dict)
    scratchpad: list[str] = Field(default_factory=list)

# ===== Herramientas =====
class Tool(BaseModel):
    name: str
    desc: str
    input_model: Type[BaseModel]
    fn: Any  # async function(inputs, view_state, agent_state, user_id, session_id)

# ===== Salida del modelo =====
class ToolCall(BaseModel):
    name: str
    args: Dict[str, Any]

class ToolCallMsg(BaseModel):
    type: Literal["tool_call"]
    call: ToolCall

class Final(BaseModel):
    type: Literal["final"]
    answer: str

ModelStep = Union[ToolCallMsg, Final]

def parse_model_step(raw: str) -> ModelStep:
    import json
    data = json.loads(raw) if isinstance(raw, str) else raw
    for cls in (ToolCallMsg, Final):
        try:
            return cls.model_validate(data)
        except ValidationError:
            continue
    raise ValueError("Salida LLM inválida: se esperaba 'tool_call' o 'final'")
