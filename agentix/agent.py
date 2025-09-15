from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

from .config import RETAIN_TAKE
from .models import Message, AgentState
from .summarizer import maybe_summarize_and_rotate
from .repo_protocol import Repo
from .context import ContextManager
from .tools.litellm_formatter import tool_to_dict
import litellm

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
        context_manager: ContextManager,
        model: Optional[str] = "gpt-3.5-turbo",
        max_steps: int = 6,
    ):
        self.repo = repo
        self.cm = context_manager
        self.max_steps = max_steps
        self.model = model

    async def run(self, user_id: str, session_id: str, agent_input: str, agent_state: Optional[AgentState] = None) -> str:
        agent_state = agent_state or AgentState()
        sdoc = await self.repo.get_or_create_session(session_id, user_id)

        history = build_context_messages(sdoc)
        user_msg = Message(role="user", content=agent_input)
        sdoc = await self.repo.append_message(session_id, user_id, user_msg)
        history += [user_msg]

        for _ in range(self.max_steps):
            # Contexto + tools vienen del ContextManager
            system_message, tools = self.cm.build(agent_state, user_id, session_id)
            system_msg = Message(role="system", content=system_message)
            tool_specs = list(map(tool_to_dict, tools))
            messages = list(map(lambda m: m.to_wire(), ([system_msg] + history)))

            raw = await litellm.acompletion(model=self.model, messages=messages, tools=tool_specs, tool_choice="auto")
            response: litellm.Choices = raw.choices[0]

            if response.finish_reason == "tool_calls":
                tool_calls = response.message.tool_calls or []
                tools_by_name = {t.name: t for t in tools}
                for tool_call in tool_calls:
                    function_call = tool_call.function
                    t = tools_by_name.get(function_call.name)
                    _before = _snapshot_state(agent_state)
                    try:
                        parsed = json.loads(function_call.arguments or "{}")
                        result = t.fn(**parsed)
                        if hasattr(result, "__await__"):
                            result = await result
                        obs = {"ok": True, "tool": t.name, "output": result}
                    except Exception as ex:
                        from pydantic import ValidationError
                        if isinstance(ex, ValidationError):
                            import json as _json
                            obs = {"error": "Args inválidos", "details": _json.loads(ex.json())}
                        else:
                            obs = {"error": f"Fallo tool '{tool_call.call.name}'", "details": str(ex)}

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
                continue

            if response.finish_reason == "stop":
                final_msg = Message(role="assistant", content=response.message.content)
                await self.repo.append_message(session_id, user_id, final_msg)
                await maybe_summarize_and_rotate(self.repo, session_id, user_id)
                return final_msg.content

        fallback = "No se obtuvo respuesta final dentro del límite de pasos."
        await self.repo.append_message(session_id, user_id, Message(role="assistant", content=fallback))
        await maybe_summarize_and_rotate(self.repo, session_id, user_id)
        return fallback
