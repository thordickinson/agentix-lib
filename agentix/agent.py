from __future__ import annotations
import json
from textwrap import dedent
from typing import Any, Dict, Optional

from datetime import datetime

from .utils.serializer import to_json
from .models import Session, ToolResultMessage, Tool, ToolCall, AgentContext, SystemMessage, UserMessage, AssistantMessage, SessionSummary, MessageType
from .agent_repository import AgentRepository
from .context import ContextManager
from .tools.litellm_formatter import tool_to_dict
import logging
import inspect
import litellm

logger = logging.getLogger(__name__)

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


def _format_llm_input(messages: list[dict], tools: list[dict]) -> str:
    def format_message(message: dict) -> str:
        role = message.get("role")
        header = '###' if role == 'system' else '*'
        footer = "\n====================\n\n" if role == 'system' else ''
        separator = '\n\n' if role == 'system' else ':'
        return f"{header} {role}{separator} {(message.get("content", None) or "")[:512]}{footer}"
    def format_tool(tool: dict) -> str:
        funct = tool.get("function", {})
        return f"* {funct.get("name")}: {funct.get("description", "")[:64]}"
    
    return dedent(f"""## Tools
{'\n'.join(list(map(format_tool, tools)))}

## Messages
{'\n'.join(list(map(format_message, messages)))}
    """)


def _parse_assistant_response(response: litellm.ModelResponse) -> AssistantMessage:
    choice: litellm.Choices = response.choices[0]
    finish_reason = choice.finish_reason
    usage = response.model_extra.get("usage", {})
    usage_details = {
                        "completion_tokens": usage.get("completion_tokens", None),
                        "prompt_tokens": usage.get("prompt_tokens", None),
                        "total_tokens": usage.get("total_tokens", None)
                    }
    
    def as_tool_call(t: litellm.ChatCompletionMessageToolCall) -> ToolCall:
        return ToolCall(
            tool_call_id=t.id,
            function_name=t.function.name,
            arguments=t.function.arguments
        )

    tool_calls = list(map(as_tool_call, choice.message.tool_calls or []))

    return AssistantMessage(
        finish_reason=finish_reason,
        usage_data=usage_details,
        content=choice.message.content,
        tool_calls=tool_calls
    )
class Agent:
    """
    El Agent delega TODO el contexto/UI al ContextManager (inyectado).
    """
    def __init__(
        self,
        name: str,
        repository: AgentRepository,
        context_manager: ContextManager,
        model: Optional[str] = "gpt-3.5-turbo",
        max_steps: int = 6,
        max_memory_messages: int = 45,
        retain_take: int = 15
    ):
        self.name = name
        self.max_memory_messages = max_memory_messages
        self.retain_take = retain_take
        self.repo = repository
        self.cm = context_manager
        self.max_steps = max_steps
        self.model = model


    async def _invoke_tool(self, tool: Tool, params: dict, agent_context: AgentContext | None = None):
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

    async def run(self, user_id: str, session_id: str, agent_input: str) -> str:
        agent_context = AgentContext(session_id = session_id, user_id=user_id)
        session_data = await self.repo.get_or_create_session(session_id, user_id)
        
        with langfuse.start_as_current_observation(as_type="agent", name=self.name, input=agent_input) as run_span:
            run_span.update(session_id=session_id, user_id=user_id)
            history = session_data.messages
            user_msg = UserMessage(content=agent_input)
            run_messages = [user_msg]

            for _ in range(self.max_steps):
                llm_input = self.cm.build(agent_context)
                system_msg = SystemMessage(content=llm_input.system)
                tool_specs = list(map(tool_to_dict, llm_input.tools))
                messages = list(map(lambda m: m.to_wire(), ([system_msg] + history + run_messages)))

                with langfuse.start_as_current_observation(name=self.model, as_type="generation",
                                                           completion_start_time=datetime.now(),
                                                           input=_format_llm_input(messages, tool_specs),
                                                           model=self.model) as generation_span:
                    raw = await litellm.acompletion(model=self.model, messages=messages, tools=tool_specs, tool_choice="auto")
                    assistant_message = _parse_assistant_response(raw)
                    run_messages.append(assistant_message)
                    generation_span.update(output=raw, usage_details=assistant_message.usage_data)

                if assistant_message.finish_reason == "tool_calls":
                    tool_calls = assistant_message.tool_calls
                    tools_by_name = {t.name: t for t in llm_input.tools}
                    for tool_call in tool_calls:
                        t = tools_by_name.get(tool_call.function_name)
                        
                        try:
                            with langfuse.start_as_current_observation(as_type="tool", name=tool_call.function_name, input=tool_call.arguments) as tool_span:
                                params = json.loads(tool_call.arguments or "{}")
                                result = await self._invoke_tool(t, params, agent_context)
                                json_result = to_json(result)
                                tool_span.update(output=json_result)
                                run_messages.append(ToolResultMessage(
                                    tool_call_id=tool_call.tool_call_id,
                                    name=tool_call.function_name,
                                    content=json_result
                                ))
                        except Exception as ex:
                            logger.error(ex)
                            run_messages.append(ToolResultMessage(
                                tool_call_id=tool_call.tool_call_id,
                                name=tool_call.function_name,
                                content=json.dumps({"status": "error", "message": str(ex)})
                            ))


                if assistant_message.finish_reason == "stop":
                    run_span.update(output=assistant_message.content)
                    await self._end_run(run_messages, session_data)
                    return assistant_message.content

            fallback = "No se obtuvo respuesta final dentro del límite de pasos."
            run_span.update(output="[max-invocation-limit]")
            return fallback

    async def _end_run(self, run_messages: list[MessageType], session: Session):
        old_messages = session.messages
        all_messages = old_messages + run_messages
        if len(all_messages) > self.max_memory_messages:
            summary, rotated = await self._summarize_messages(all_messages)
            session.messages = rotated
            session.summaries.append(summary)
        else:
            session.messages = all_messages
        await self.repo.save_session(session)
        await self.repo.append_messages(session.session_id, session.user_id, run_messages)
    
    async def _summarize_messages(self, messages: list[MessageType]) -> tuple[str, list[MessageType]]:
        remaining = messages[-self.retain_take:]
        to_summarize = messages[:-self.retain_take]
        summary_prompt = dedent(f"""
        Resume brevemente (1-2 frases) la siguiente conversación entre un usuario y un asistente de IA.
        El resumen debe capturar los puntos clave y el contexto necesario para entender la conversación.
        El resumen debe estar en español.
        El resumen debe ser neutral y objetivo, sin interpretaciones ni juicios.
        El resumen debe ser adecuado para que el asistente de IA pueda continuar la conversación sin perder contexto.
        El resumen debe ser en formato texto, sin ningún comentario adicional y listo para ser usado.
        Conversación:
        {''.join([f"- {m.role}: {m.content}\n" for m in to_summarize])}
        """)
        llm_messages = [SystemMessage(content=summary_prompt)] + remaining
        raw = await litellm.acompletion(
            model=self.model,
            messages=[m.to_wire() for m in llm_messages],
        )
        choice = raw.choices[0]
        content = choice.message.content or None
        summary = SessionSummary(content=content)
        return summary, remaining
