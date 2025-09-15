from __future__ import annotations
import asyncio
from typing import Optional

from agentix import Agent, AgentState, LiteLLMAdapter
from agentix.storage.mongo import MongoRepo
from agentix.context import ContextManager
from agentix.stack import StackContextManager
from agentix.utils.console import console_loop
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

    await console_loop(
        agent=agent,
        repo=repo,
        user_id=user_id,
        session_id=session_id,
        state=state,
        messages_tail=8,
    )

if __name__ == "__main__":
    asyncio.run(interactive_loop())
