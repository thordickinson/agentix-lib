from __future__ import annotations
import asyncio
import pprint
from typing import Optional

from agentix import Agent, AgentState, LiteLLMAdapter
from agentix.storage.mongo import MongoRepo
from agentix.context import ContextManager
from agentix.stack import StackContextManager
from .router import build_router

def developer_instructions_fn(state: AgentState) -> str:
    tone = state.memory.get("tone", "claro y directo")
    return f"- Sé {tone}. No inventes datos.\n- Responde en español.\n"

async def interactive_loop():
    # Storage (sesiones/mensajes/estado)
    repo = MongoRepo(uri="mongodb://localhost:27017", db_name="agentix_demo")
    await repo.ensure_indexes()

    # Contexto UI (stack) + router de vistas
    router = build_router()
    cm: ContextManager = StackContextManager(router)

    # LLM
    llm = LiteLLMAdapter(model=None)  # usa MAIN_MODEL del core si None

    agent = Agent(
        repo=repo,
        llm=llm,
        developer_instructions_fn=developer_instructions_fn,
        context_manager=cm,
        max_steps=6,
    )

    user_id = "user_demo"
    session_id = "session_demo"
    state: Optional[AgentState] = AgentState(memory={"tone": "crítico y preciso"})

    print("=== Real Estate Agent ===")
    print("Comandos: :exit, :state, :stack, :messages, :reset")

    while True:
        user_input = input("Tú: ").strip()
        if not user_input:
            continue

        if user_input.lower() in (":exit", ":quit"):
            print("Saliendo...")
            break

        if user_input.lower() == ":state":
            print("\n--- AgentState ---")
            pprint.pprint(state.model_dump(mode="python"))
            continue

        if user_input.lower() == ":stack":
            print("\n--- Stack (agent_state.memory['ui_stack']) ---")
            frames = state.memory.get("ui_stack", [])
            for i, fr in enumerate(frames):
                print(f"[{i}] screen={fr.get('screen_key')} params={fr.get('params')} view_state={fr.get('view_state')}")
            continue

        if user_input.lower() == ":messages":
            sdoc = await repo.get_or_create_session(session_id, user_id)
            print("\n--- Últimos mensajes ---")
            for m in sdoc.get("messages", [])[-10:]:
                print(f"{m['role']}: {m['content']}")
            continue

        if user_input.lower() == ":reset":
            state.memory["ui_stack"] = []
            print("Stack reseteado (volverá a index en el próximo turno).")
            continue

        out = await agent.run(user_id, session_id, user_input, state)
        print(f"Agente: {out}")

if __name__ == "__main__":
    asyncio.run(interactive_loop())
