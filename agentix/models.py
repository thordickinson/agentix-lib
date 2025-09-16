from __future__ import annotations
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class AgentContext:
    ...

class Message(BaseModel):
    role: str
    content: Optional[str]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    usage_data: Dict[str, Any] = {}
    meta: Dict[str, Any] = Field(default_factory=dict)

    def to_wire(self) -> Dict[str, str]:
        """Mensaje en formato listo para el LLM."""
        return {"role": self.role, "content": self.content}
    
class SystemMessage(Message):
    role: str = Field(default="system", frozen=True)

class ToolResultMessage(BaseModel):
    call_id: str
    result: str

    def from_tool_call():
        ...

class SessionSummary(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content: str

class Session(BaseModel):
    session_id: Optional[str] = None
    user_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    messages: list[Message] = []
    summaries: list[SessionSummary] = []
    state: Dict[str, Any] = {}

class UserMemory(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content: str

class UserInfo(BaseModel):
    id: str
    memories: list[UserMemory] = []

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

