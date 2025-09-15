from __future__ import annotations
import asyncio
import pprint
from typing import Callable, Optional, Sequence

from agentix import Agent, AgentState
from agentix.repo_protocol import Repo

# Tipo para función que imprime/serializa el "stack" según tu ContextManager
StackDumpFn = Callable[[AgentState], Sequence[dict]]

def _default_stack_dump(state: AgentState) -> Sequence[dict]:
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
    repo: Repo,
    user_id: str = "user_console",
    session_id: str = "session_console",
    state: Optional[AgentState] = None,
    prompt: str = "Tú: ",
    intro: str = "=== Agentix Console ===\nComandos: :exit, :state, :stack, :messages, :reset",
    messages_tail: int = 10,
    stack_dump_fn: StackDumpFn = _default_stack_dump,
    input_fn: Callable[[str], str] = input,
) -> None:
    """
    Loop de consola reusable para pruebas manuales.

    - agent: instancia de Agent ya configurada (Repo, LLM, ContextManager, etc.)
    - repo: implementación Repo usada por el agent (para mostrar mensajes)
    - user_id, session_id: identificadores de la sesión
    - state: AgentState (si None, se crea uno nuevo)
    - prompt: prefijo del input
    - intro: mensaje de bienvenida
    - messages_tail: cuántos mensajes mostrar en ':messages'
    - stack_dump_fn: cómo volcar el stack desde AgentState (por defecto lee 'ui_stack')
    - input_fn: función de entrada (inyectable para tests)

    Comandos soportados:
      :exit / :quit   -> salir
      :state          -> imprimir AgentState
      :stack          -> imprimir stack (según stack_dump_fn)
      :messages       -> mostrar últimos N mensajes de la sesión
      :reset          -> limpiar stack en AgentState (clave 'ui_stack')
    """
    st = state or AgentState()
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

        if low == ":state":
            print("\n--- AgentState ---")
            pprint.pprint(st.model_dump(mode="python"))
            continue

        if low == ":stack":
            print("\n--- Stack ---")
            frames = stack_dump_fn(st) or []
            for i, fr in enumerate(frames):
                screen = fr.get("screen_key")
                params = fr.get("params")
                vstate = fr.get("view_state")
                print(f"[{i}] screen={screen} params={params} view_state={vstate}")
            continue

        if low == ":messages":
            sdoc = await repo.get_or_create_session(session_id, user_id)
            print(f"\n--- Últimos mensajes (tail={messages_tail}) ---")
            for m in sdoc.get("messages", [])[-messages_tail:]:
                role = m.get("role")
                content = m.get("content")
                print(f"{role}: {content}")
            continue

        if low == ":reset":
            # Por convención limpiamos la clave estándar del StackContextManager
            st.memory["ui_stack"] = []
            print("Stack reseteado (volverá a index en el próximo turno).")
            continue

        # --- Entrada normal: se envía al agente ---
        out = await agent.run(user_id, session_id, user_input, st)
        print(f"Agente: {out}")


def run_console_sync(
    *,
    agent: Agent,
    repo: Repo,
    user_id: str = "user_console",
    session_id: str = "session_console",
    state: Optional[AgentState] = None,
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
