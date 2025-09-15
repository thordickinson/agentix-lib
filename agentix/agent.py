from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Callable

from .config import RETAIN_TAKE, MAIN_MODEL
from .models import Message, AgentState, ToolCallMsg, Final, parse_model_step, Tool
from .summarizer import maybe_summarize_and_rotate
from .repo_protocol import Repo
from .llm import LLMClient
from .context import ContextManager

def _snapshot_state(state: AgentState) -> Dict[str, Any]:
    return state.model_dump(mode="python", round_trip=True)

def _dict_diff(old: Dict[str, Any], new: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    patch: Dict[str, Any] = {}
    for k in new.keys():
        p = f"{prefix}.{k}" if prefix else k
        ov = old.get(k); nv = new[k]
        if isinstance(nv, dict) and isinstance(ov, dict):
            patch.update(_dict_diff(ov, nv, p))
        elif ov != nv:
            patch[p] = nv
    return patch

def build_context_messages(session_doc: Dict[str, Any], summaries_limit: int = 3) -> List[Message]:
    msgs: List[Message] = []
    summaries = session_doc.get("summaries", [])[-summaries_limit:]
    if summaries:
        import json as _json
        stext = "\n---\n".join([
            _json.dumps({"range": s.get("range"), "summary": s.get("summary")}, ensure_ascii=False)
            for s in summaries
        ])
        msgs.append(Message(role="system", content=f"Resumenes previos (no repitas literalmente):\n{stext}"))
    tail = session_doc.get("messages", [])[-RETAIN_TAKE:]
    for m in tail:
        msgs.append(Message(role=m["role"], content=m["content"]))
    return msgs

class Agent:
    """
    El Agent delega TODO el contexto/UI al ContextManager (inyectado).
    """
    def __init__(
        self,
        repo: Repo,
        llm: LLMClient,
        developer_instructions_fn: Callable[[AgentState], str],
        context_manager: ContextManager,
        max_steps: int = 6,
    ):
        self.repo = repo
        self.llm = llm
        self.developer_instructions_fn = developer_instructions_fn
        self.cm = context_manager
        self.max_steps = max_steps

    @staticmethod
    def _schemas(tools: List[Tool]) -> List[Dict[str, Any]]:
        return [{"name": t.name, "description": t.desc, "schema": t.input_model.model_json_schema()} for t in tools]

    async def run(self, user_id: str, session_id: str, agent_input: str, agent_state: Optional[AgentState] = None) -> str:
        agent_state = agent_state or AgentState()
        sdoc = await self.repo.get_or_create_session(session_id, user_id)

        history = build_context_messages(sdoc)
        user_msg = Message(role="user", content=agent_input)
        sdoc = await self.repo.append_message(session_id, user_id, user_msg)
        history += [user_msg]

        developer_instructions = self.developer_instructions_fn(agent_state)

        for _ in range(self.max_steps):
            # Contexto + tools vienen del ContextManager
            system_message, tools = self.cm.build(agent_state, user_id, session_id)
            tool_specs = self._schemas(tools)
            combined_instructions = developer_instructions + ("\n\n" + system_message if system_message else "")

            raw = await self.llm.generate(history, tool_specs, combined_instructions, MAIN_MODEL)
            try:
                action = parse_model_step(raw)
            except Exception:
                reprimand = Message(role="assistant", content='{"type":"final","answer":"Formato inválido"}')
                history.append(reprimand)
                await self.repo.append_message(session_id, user_id, reprimand)
                continue

            if isinstance(action, ToolCallMsg):
                tools_by_name = {t.name: t for t in tools}
                t = tools_by_name.get(action.call.name)
                _before = _snapshot_state(agent_state)

                if not t:
                    obs = {"error": f"Tool '{action.call.name}' no disponible"}
                else:
                    try:
                        parsed = t.input_model.model_validate(action.call.args)
                        result = t.fn(parsed, {}, agent_state, user_id, session_id)  # view_state lo maneja el CM
                        if hasattr(result, "__await__"):
                            result = await result
                        obs = {"ok": True, "tool": t.name, "output": result}
                    except Exception as ex:
                        from pydantic import ValidationError
                        if isinstance(ex, ValidationError):
                            import json as _json
                            obs = {"error": "Args inválidos", "details": _json.loads(ex.json())}
                        else:
                            obs = {"error": f"Fallo tool '{action.call.name}'", "details": str(ex)}

                # navegación/side-effects la maneja el CM
                if obs.get("ok") and isinstance(obs.get("output"), dict):
                    await self.cm.handle_nav(agent_state, user_id, session_id, obs["output"])

                _after = _snapshot_state(agent_state)
                patch = _dict_diff(_before, _after)
                if patch:
                    await self.repo.update_state(session_id, patch)

                import json as _json
                obs_msg = Message(role="user", content=_json.dumps({"observation": obs}, ensure_ascii=False))
                await self.repo.append_message(session_id, user_id, obs_msg)
                history.append(obs_msg)

                await maybe_summarize_and_rotate(self.repo, session_id, user_id)
                developer_instructions = self.developer_instructions_fn(agent_state)
                continue

            if isinstance(action, Final):
                final_msg = Message(role="assistant", content=action.answer)
                await self.repo.append_message(session_id, user_id, final_msg)
                await maybe_summarize_and_rotate(self.repo, session_id, user_id)
                return action.answer

        fallback = "No se obtuvo respuesta final dentro del límite de pasos."
        await self.repo.append_message(session_id, user_id, Message(role="assistant", content=fallback))
        await maybe_summarize_and_rotate(self.repo, session_id, user_id)
        return fallback
