from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from .models import AgentContext

@dataclass
class StackFrame:
    screen_key: str
    params: Dict[str, Any]
    view_state: Dict[str, Any]
    return_path: Optional[str] = None

class StackManager:
    def __init__(self, frames: Optional[List[StackFrame]] = None):
        self.frames: List[StackFrame] = frames or []

    @classmethod
    def from_state(cls, state: AgentContext) -> "StackManager":
        frames_data = state.memory.get("ui_stack", [])
        frames = [StackFrame(**fd) for fd in frames_data]
        return cls(frames)

    def to_state(self, state: AgentContext):
        state.memory["ui_stack"] = [f.__dict__ for f in self.frames]

    def current(self) -> Optional[StackFrame]:
        return self.frames[-1] if self.frames else None

    def push(self, frame: StackFrame):
        self.frames.append(frame)

    def pop(self, result: Dict[str, Any] | None = None) -> Optional[StackFrame]:
        return self.frames.pop() if self.frames else None

    def return_to_caller(self, child: StackFrame, canceled: bool = False):
        if not self.frames:
            return
        caller = self.frames[-1]
        if child.return_path and not canceled:
            # Guardamos el resultado en el caller
            caller.view_state[child.return_path] = child.view_state.get("__pending_result")

    def breadcrumb(self) -> str:
        return " > ".join(f.screen_key for f in self.frames)
