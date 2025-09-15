from __future__ import annotations
from typing import Any, Dict, Literal, Union, Optional, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ValidationError
from litellm import ModelResponse

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

class Param(BaseModel):
    name: str
    type: str
    desc: str
    optional: bool = False
    default_value: Optional[Any] = None
    enum_values: Optional[List[str]] = None


class Tool(BaseModel):
    name: str
    desc: str
    params: List[Param]
    fn: Any

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

