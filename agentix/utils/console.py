from __future__ import annotations
import asyncio
import pprint
from typing import Callable, Optional, Sequence

from agentix import Agent, AgentContext
from agentix.agent_repository import AgentRepository
from agentix.models import MessageType

# Tipo para función que imprime/serializa el "stack" según tu ContextManager
StackDumpFn = Callable[[AgentContext], Sequence[dict]]

def _default_stack_dump(state: AgentContext) -> Sequence[dict]:
    """
    Implementación por defecto: asume que el ContextManager tipo 'stack'
    guarda los frames en state.memory['ui_stack'] como lista de dicts.
    """
    stack = state.memory.get("ui_stack", [])
    if isinstance(stack, list):
        return [dict(x) for x in stack]
    return []

async def console_loop(
    *,
    agent: Agent,
    repo: AgentRepository,
    user_id: str = "user_console",
    session_id: str = "session_console",
    prompt: str = "Tú: ",
    intro: str = "=== Agentix Console ===\nComandos: :exit, :summaries :messages",
    messages_tail: int = 10,
    stack_dump_fn: StackDumpFn = _default_stack_dump,
    input_fn: Callable[[str], str] = input,
) -> None:
    """
    Loop de consola reusable para pruebas manuales.

    - agent: instancia de Agent ya configurada (Repo, LLM, ContextManager, etc.)
    - repo: implementación Repo usada por el agent (para mostrar mensajes)
    - user_id, session_id: identificadores de la sesión
    - state: AgentContext (si None, se crea uno nuevo)
    - prompt: prefijo del input
    - intro: mensaje de bienvenida
    - messages_tail: cuántos mensajes mostrar en ':messages'
    - stack_dump_fn: cómo volcar el stack desde AgentContext (por defecto lee 'ui_stack')
    - input_fn: función de entrada (inyectable para tests)

    Comandos soportados:
      :exit / :quit   -> salir
      :state          -> imprimir AgentContext
      :stack          -> imprimir stack (según stack_dump_fn)
      :messages       -> mostrar últimos N mensajes de la sesión
      :reset          -> limpiar stack en AgentContext (clave 'ui_stack')
    """
    print(intro)

    while True:
        user_input = input_fn(prompt).strip()
        if not user_input:
            continue

        # --- Comandos ---
        low = user_input.lower()
        if low in (":exit", ":quit"):
            print("Saliendo...")
            break

        if low == ":summaries":
            session = await agent.get_session_data(user_id, session_id)
            summaries = f"\n".join([f"* {summary.content}" for summary in session.summaries])
            print(f"\033[92mResumenes: \n===\n{summaries}\n===\n\033[0m")
            continue

        if low == ":messages":
            session = await agent.get_session_data(user_id, session_id)
            messages = session.messages

            def format_message(message: MessageType):
                return f"{message.role}: {message.content or "--"}"
            messages_str = "\n".join([f"* {format_message(m)}" for m in messages])
            print(f"\033[92mÚltimos mensajes \n===\n{messages_str}\n===\n\033[0m")

            continue

        # --- Entrada normal: se envía al agente ---
        out = await agent.run(user_id, session_id, user_input)
        print(f"\033[94mAgente: {out}\033[0m")


def run_console_sync(
    *,
    agent: Agent,
    repo: AgentRepository,
    user_id: str = "user_console",
    session_id: str = "session_console",
    state: Optional[AgentContext] = None,
    **kwargs,
) -> None:
    """
    Conveniencia síncrona: crea un loop de eventos y ejecuta console_loop.
    Útil si llamas desde scripts sencillos.
    """
    async def _runner():
        await console_loop(
            agent=agent,
            repo=repo,
            user_id=user_id,
            session_id=session_id,
            state=state,
            **kwargs,
        )

    try:
        asyncio.run(_runner())
    except RuntimeError:
        # Si ya hay un loop corriendo (por ejemplo en notebooks),
        # cae aquí; en ese caso el usuario debería llamar console_loop() directamente.
        raise RuntimeError(
            "Ya hay un event loop activo. Usa await console_loop(...) en vez de run_console_sync(...)."
        )
