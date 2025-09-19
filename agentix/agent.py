from __future__ import annotations
import json
from textwrap import dedent
from typing import Any, Dict, Optional, Callable, Awaitable, Union

from datetime import datetime

from pydantic import BaseModel

from agentix.utils.collections import flatten
from .prompts.summarization import  SUMMARIZATION_SYSTEM_PROMPT, META_SUMMARIZATION_PROMPT

from .utils.serializer import to_json
from .models import Session, ToolResultMessage, Tool, ToolCall, AgentContext, SystemMessage, UserMessage, AssistantMessage, SessionSummary, MessageType
from .agent_repository import AgentRepository
from .context import ContextManager
from .tools.litellm_formatter import tool_to_dict
import logging
import inspect
import litellm
import uuid

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


def _parse_assistant_response(run_id: str, response: litellm.ModelResponse) -> AssistantMessage:
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
        run_id=run_id,
        finish_reason=finish_reason,
        usage_data=usage_details,
        content=choice.message.content,
        tool_calls=tool_calls
    )

class AgentEvent(BaseModel):
    type: str
    message: Optional[str] = None

EventListener = Union[
    Callable[[int], None],
    Callable[[int], Awaitable[None]],
]

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
        max_interactions_in_memory: int = 15,
        max_summaries_in_context = 5,
        interations_retain: int = 5,
        event_listener: Optional[EventListener] = None
    ):
        self.name = name
        self.max_interactions_in_memory = max_interactions_in_memory
        self.interations_retain = interations_retain
        self.repo = repository
        self.cm = context_manager
        self.max_steps = max_steps
        self.model = model
        self.event_listener = event_listener
        self.max_summaries_in_context = max_summaries_in_context


    async def _send_event(self, type: str, message: Optional[str] = None):
        if self.event_listener is None:
            return
        result = self.event_listener(AgentEvent(type=type))
        if inspect.isawaitable(result):
            await result


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
    
    async def get_session_data(self, user_id: str, session_id: str) -> Session:
        session_data = await self.repo.get_or_create_session(session_id, user_id)
        return session_data
    
    async def _add_session_data(self, system_prompt: str, session: Session) -> str:
        parts = [system_prompt]
        if len(session.summaries) > 0:
            parts.append(dedent(f"""
                A continuación se listan resúmenes de partes anteriores de la conversación que ya no están disponibles en detalle.
                Estos resúmenes representan contexto previo que debes tener en cuenta para continuar de forma coherente.
                Utilízalos como si fueran la memoria de lo que ocurrió antes, tanto para responder al usuario como para decidir llamadas a funciones.
            
                ## Resúmenes previos:
                <previous_summaries>
                    {'\n'.join([f'* {summary.content}' for summary in session.summaries])}
                </previous_summaries>
            """))
        return "\n---\n".join(parts)

    async def run(self, user_id: str, session_id: str, agent_input: str) -> str:
        run_id = str(uuid.uuid4())
        agent_context = AgentContext(session_id = session_id, user_id=user_id, run_id=run_id)
        session_data = await self.get_session_data(session_id=session_id, user_id=user_id)
        
        with langfuse.start_as_current_observation(as_type="agent", name=self.name, input=agent_input) as run_span:
            run_span.update(session_id=session_id, user_id=user_id, metadata={"run_id": run_id})
            history = session_data.messages
            user_msg = UserMessage(run_id=run_id, content=agent_input)
            run_messages = [user_msg]

            for _ in range(self.max_steps):
                llm_input = self.cm.build(agent_context)
                full_system_message = await self._add_session_data(llm_input.system, session=session_data)
                system_msg = SystemMessage(run_id=run_id, content=full_system_message)
                tool_specs = list(map(tool_to_dict, llm_input.tools))
                messages = list(map(lambda m: m.to_wire(), ([system_msg] + history + run_messages)))

                with langfuse.start_as_current_observation(name=self.model, as_type="generation",
                                                           completion_start_time=datetime.now(),
                                                           input=_format_llm_input(messages, tool_specs),
                                                           model=self.model) as generation_span:
                    raw = await litellm.acompletion(model=self.model, messages=messages, tools=tool_specs, tool_choice="auto")
                    assistant_message = _parse_assistant_response(run_id=run_id, response=raw)
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
                                    run_id=run_id,
                                    tool_call_id=tool_call.tool_call_id,
                                    name=tool_call.function_name,
                                    content=json_result
                                ))
                        except Exception as ex:
                            logger.error(ex)
                            run_messages.append(ToolResultMessage(
                                run_id=run_id,
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
        

    def _split_in_runs(self, session_messages: list[MessageType]) -> list[list[MessageType]]:
        groups = []
        current_run = []
        current_run_id = None

        for msg in session_messages:
            if getattr(msg, "run_id", None) != current_run_id:
                if current_run:
                    groups.append(current_run)
                current_run = [msg]
                current_run_id = getattr(msg, "run_id", None)
            else:
                current_run.append(msg)
        if current_run:
            groups.append(current_run)
        return groups

    async def _end_run(self, run_messages: list[MessageType], session: Session):
        old_messages = session.messages
        all_messages = old_messages + run_messages

        runs = self._split_in_runs(all_messages)

        if len(runs) > self.max_interactions_in_memory:
            summary, rotated = await self._summarize_runs(runs, session)
            session.messages = rotated
            session.summaries.append(summary)
            summaries = await self._compress_summaries(session)
            session.summaries = summaries
        else:
            session.messages = all_messages
        await self.repo.save_session(session)
        await self.repo.append_messages(session.session_id, session.user_id, run_messages)
    
    async def _summarize_runs(self, runs: list[list[MessageType]], session: Session) -> tuple[str, list[list[MessageType]]]:
        remaining = runs[-self.interations_retain:]
        to_summarize = flatten(runs[:-self.interations_retain])

        def summarizable(message: MessageType):
            return message.role == "user" or isinstance(message, AssistantMessage) and len(message.tool_calls) == 0
        
        to_summarize = list(filter(summarizable, to_summarize))
        to_summarize = {''.join([f"- {m.role}: {m.content}\n" for m in to_summarize])}
        
        to_keep = list(filter(summarizable, flatten(remaining)))
        to_keep = {''.join([f"- {m.role}: {m.content}\n" for m in to_keep])}

        old_summaries = {''.join([f"{s.content}" for s in session.summaries])}

        message = f"""
        Resume la siguiente conversación:
        ---

        ## Resumenes anteriores

        <old_summaries>
        {old_summaries}
        </old_summaries>

        ---

        ## Mensajes a resumir
        
        <conversation>
        {to_summarize}
        </conversation>

        # Mensajes posteriores (No incluir en el resumen, solo dan contexto)

        <messages_to_keep>
        {to_keep}
        </messages_to_keep>
"""
        content = await self._ask_llm(SUMMARIZATION_SYSTEM_PROMPT, message)
        summary = SessionSummary(content=content)
        remaining = flatten(remaining)
        await self._send_event("summarization_completed", content[:64])
        return summary, remaining

    async def _ask_llm(self, system: str, user: str) -> str:
        llm_messages = [SystemMessage(content=system), UserMessage(content=user)]
        raw = await litellm.acompletion(
            model=self.model,
            messages=[m.to_wire() for m in llm_messages],
        )
        choice = raw.choices[0]
        content = choice.message.content or None
        return content
   
    async def _compress_summaries(self, session: Session) -> list[SessionSummary]:
        if len(session.summaries) < self.max_summaries_in_context:
            return session.summaries
        await self._send_event("meta_summarization")
        summaries = "\n".join([f"* {s}" for s in session.summaries])
        message = dedent(f"""
            Sintetiza los siguientes resumenes:
            <summaries>
            {summaries}
            </summaries>
        """)
        summary = await self._ask_llm(META_SUMMARIZATION_PROMPT, message)
        return [SessionSummary(content=summary)]

