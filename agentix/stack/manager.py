from __future__ import annotations
from typing import Any, Dict, List, Tuple

from agentix.models import AgentContext, Tool
from agentix.context import ContextManager
from .frames import StackFrame
from .view import ViewRouter

class StackContextManager(ContextManager):
    """
    Impl de ContextManager usando un stack de vistas guardado en agent_state.memory['ui_stack'].
    """
    STATE_KEY = "ui_stack"

    def __init__(self, router: ViewRouter, state_key: str | None = None):
        self.router = router
        self.state_key = state_key or self.STATE_KEY

    # --------- helpers de stack ----------
    def _get_frames(self, agent_state: AgentContext) -> List[StackFrame]:
        raw = agent_state.memory.get(self.state_key, [])
        frames: List[StackFrame] = []
        if isinstance(raw, list):
            for f in raw:
                frames.append(StackFrame(
                    screen_key=f["screen_key"],
                    params=f.get("params", {}),
                    view_state=f.get("view_state", {}),
                    return_path=f.get("return_path"),
                ))
        return frames

    def _save_frames(self, agent_state: AgentContext, frames: List[StackFrame]) -> None:
        agent_state.memory[self.state_key] = [
            {
                "screen_key": fr.screen_key,
                "params": fr.params,
                "view_state": fr.view_state,
                "return_path": fr.return_path,
            }
            for fr in frames
        ]

    @staticmethod
    def _breadcrumb(frames: List[StackFrame]) -> str:
        return " / ".join(f.screen_key for f in frames[-3:])

    # --------- ContextManager API ----------
    def build(self, agent_state: AgentContext, user_id: str, session_id: str) -> Tuple[str, List[Tool]]:
        frames = self._get_frames(agent_state)

        # Si no hay frame, empujar index si está configurado
        if not frames:
            idx = self.router.index_key()
            if idx:
                frames.append(StackFrame(screen_key=idx, params={}, view_state={}))
                self._save_frames(agent_state, frames)

        system_message = ""
        tools: List[Tool] = []
        if frames:
            frame = frames[-1]
            view = self.router.get(frame.screen_key)
            instr = view.instructions(agent_state, frame.view_state) or ""
            memi = view.memory_instructions(agent_state, frame.view_state) or ""
            breadcrumb = self._breadcrumb(frames)

            parts = []
            if breadcrumb:
                parts.append(f"[RUTA] {breadcrumb}")
            if instr:
                parts.append("[VENTANA]\n" + instr)
            if memi:
                parts.append("[MEMORIA]\n" + memi)
            system_message = "\n\n".join(parts)

            tools = view.build_tools(agent_state, frame.view_state)
        
        return system_message, tools

    async def handle_nav(self, agent_state: AgentContext, user_id: str, session_id: str, out: Dict[str, Any]) -> None:
        nav = out.get("nav")
        if not nav:
            return

        frames = self._get_frames(agent_state)

        if nav == "push_view":
            target = out.get("target")
            params = out.get("params") or {}
            return_path = out.get("return_path")
            if not target:
                return
            frames.append(StackFrame(
                screen_key=target,
                params=params,
                view_state=params.copy(),
                return_path=return_path,
            ))
            self._save_frames(agent_state, frames)
            return

        if nav in ("confirm", "cancel"):
            if not frames:
                return
            canceled = (nav == "cancel")
            child = frames.pop()
            result = child.view_state.get("__pending_result")
            if frames:
                caller = frames[-1]
                caller.view_state["__last_call"] = {"canceled": canceled, "result": None if canceled else result}
                if child.return_path and not canceled:
                    self._assign_path(caller.view_state, child.return_path, result)
            self._save_frames(agent_state, frames)

    # --------- utils ----------
    @staticmethod
    def _assign_path(obj: Dict[str, Any], path: str, value: Any) -> None:
        parts = [p for p in (path or "").split(".") if p]
        if not parts:
            return
        cur = obj
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value
