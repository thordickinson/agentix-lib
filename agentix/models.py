from __future__ import annotations
from typing import Any, Dict, Optional, List, Union
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class AgentContext(BaseModel):
    session_id: str
    user_id: str

class Message(BaseModel):
    role: str
    content: Optional[str]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    usage_data: Dict[str, Any] = {}
    meta: Dict[str, Any] = Field(default_factory=dict)

    def to_wire(self) -> Dict[str, str]:
        """Mensaje en formato listo para el LLM."""
        wired = { "role": self.role }
        if self.content:
            wired["content"] = self.content
        return wired
    
class UserMessage(Message):
    role: str = Field(default="user", frozen=True)
    
class SystemMessage(Message):
    role: str = Field(default="system", frozen=True)

class ToolCall(BaseModel):
    tool_call_id: str
    function_name: str
    arguments: str

    def to_wire(self) -> Dict[str, str]:
        return {
            "id": self.tool_call_id,
            "type": "function",
            "function": {
                "name": self.function_name,
                "arguments": self.arguments
            }
        }

class AssistantMessage(Message):
    role: str = Field(default="assistant", frozen=True)
    finish_reason: str
    tool_calls: list[ToolCall] = []

    def to_wire(self):
        wired = super().to_wire()
        if len(self.tool_calls) > 0:
            wired["tool_calls"] = [tc.to_wire() for tc in self.tool_calls]
        return wired


class ToolResultMessage(Message):
    role: str = Field(default="tool", frozen=True)
    tool_call_id: str
    name: str

    def to_wire(self):
        wired = super().to_wire()
        wired["tool_call_id"] = self.tool_call_id
        wired["name"] = self.name
        return wired

class SessionSummary(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content: str



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

MessageType = Union[UserMessage, SystemMessage, AssistantMessage, ToolResultMessage]

class Session(BaseModel):
    session_id: Optional[str] = None
    user_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    messages: list[MessageType] = []
    summaries: list[SessionSummary] = []
    state: Dict[str, Any] = {}
