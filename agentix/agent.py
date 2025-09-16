from __future__ import annotations
import json
from textwrap import dedent
from typing import Any, Dict, List, Optional

from datetime import datetime
from .config import RETAIN_TAKE
from .models import Message, AgentState, ToolResultMessage, Tool, AgentContext
from .summarizer import maybe_summarize_and_rotate
from .repo_protocol import Repo
from .context import ContextManager
from .tools.litellm_formatter import tool_to_dict
import inspect
import litellm

from langfuse import get_client
langfuse = get_client()


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

def _format_llm_input(messages: list[dict], tools: list[dict]) -> str:
    def format_message(message: dict) -> str:
        role = message.get("role")
        header = '###' if role == 'system' else '*'
        footer = "\n====================\n\n" if role == 'system' else ''
        separator = '\n\n' if role == 'system' else ':'
        return f"{header} {role}{separator} {message.get("content", "")[:512]}{footer}"
    def format_tool(tool: dict) -> str:
        funct = tool.get("function", {})
        return f"* {funct.get("name")}: {funct.get("description", "")[:64]}"
    
    return dedent(f"""## Tools
{'\n'.join(list(map(format_tool, tools)))}

## Messages
{'\n'.join(list(map(format_message, messages)))}
    """)


class Agent:
    """
    El Agent delega TODO el contexto/UI al ContextManager (inyectado).
    """
    def __init__(
        self,
        name: str,
        repo: Repo,
        context_manager: ContextManager,
        model: Optional[str] = "gpt-3.5-turbo",
        max_steps: int = 6,
    ):
        self.name = name
        self.repo = repo
        self.cm = context_manager
        self.max_steps = max_steps
        self.model = model


    async def invoke_tool(self, tool: Tool, params: dict, agent_context: AgentContext | None = None):
        """
        Invoca tool.fn con params (del LLM) y opcionalmente inyecta AgentContext.
        """
        sig = inspect.signature(tool.fn)
        kwargs = dict(params)

        # Si la función declara AgentContext, inyectarlo
        for pname, param in sig.parameters.items():
            if param.annotation is AgentContext:
                kwargs[pname] = agent_context
                break

        result = tool.fn(**kwargs)
        if hasattr(result, "__await__"):
            result = await result
        return result

    async def run(self, user_id: str, session_id: str, agent_input: str, agent_state: Optional[AgentState] = None) -> str:
        agent_state = agent_state or AgentState()
        sdoc = await self.repo.get_or_create_session(session_id, user_id)
        
        with langfuse.start_as_current_observation(as_type="agent", name=self.name, input=agent_input) as run_span:
            
            run_span.update(session_id=session_id, user_id=user_id)
            history = build_context_messages(sdoc)
            user_msg = Message(role="user", content=agent_input)
            sdoc = await self.repo.append_message(session_id, user_id, user_msg)
            history += [user_msg]
            run_messages = [user_msg]

            for _ in range(self.max_steps):
                # Contexto + tools vienen del ContextManager
                system_message, tools = self.cm.build(agent_state, user_id, session_id)
                system_msg = Message(role="system", content=system_message)
                tool_specs = list(map(tool_to_dict, tools))
                messages = list(map(lambda m: m.to_wire(), ([system_msg] + history + run_messages)))

                with langfuse.start_as_current_observation(name=self.model, as_type="generation",
                                                           completion_start_time=datetime.now(),
                                                           input=_format_llm_input(messages, tool_specs),
                                                           model=self.model) as generation_span:
                    raw = await litellm.acompletion(model=self.model, messages=messages, tools=tool_specs, tool_choice="auto")
                    response: litellm.Choices = raw.choices[0]
                    usage = raw.model_extra.get("usage", {})
                    usage_details = {
                        "completion_tokens": usage.get("completion_tokens", None),
                        "prompt_tokens": usage.get("prompt_tokens", None),
                        "total_tokens": usage.get("total_tokens", None)
                    }
                    generation_span.update(output=raw, usage_details=usage_details)

                if response.finish_reason == "tool_calls":
                    tool_calls = response.message.tool_calls or []
                    tools_by_name = {t.name: t for t in tools}
                    for tool_call in tool_calls:
                        function_call = tool_call.function
                        t = tools_by_name.get(function_call.name)

                        run_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": function_call.arguments
                        })
                        
                        try:
                            with langfuse.start_as_current_observation(as_type="tool", name=function_call.name, input=function_call.arguments) as tool_span:
                                params = json.loads(function_call.arguments or "{}")
                                result = await self.invoke_tool(t, params, None) # TODO : incluir el contexto
                                json_result = json.dumps(result)
                                tool_span.update(output=json_result)
                                run_messages.append({
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_call.name,
                                    "content": json_result,
                                })
                        except Exception as ex:
                            print(ex)
                            raise Exception("Error while invoking function")


                if response.finish_reason == "stop":
                    final_msg = Message(role="assistant", content=response.message.content)
                    run_span.update(output=response.message.content)
                    await self.repo.append_message(session_id, user_id, final_msg)
                    await maybe_summarize_and_rotate(self.repo, session_id, user_id)
                    return final_msg.content

            fallback = "No se obtuvo respuesta final dentro del límite de pasos."
            run_span.update(output="[max-invocation-limit]")
            await self.repo.append_message(session_id, user_id, Message(role="assistant", content=fallback))
            await maybe_summarize_and_rotate(self.repo, session_id, user_id)
            return fallback

