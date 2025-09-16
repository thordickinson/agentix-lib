from __future__ import annotations
import asyncio
from typing import Optional

from agentix import Agent, AgentState
from agentix.storage.mongo import MongoRepo
from agentix.context import ContextManager
from agentix.stack import StackContextManager
from agentix.utils.console import console_loop
from .router import build_router
import os



def developer_instructions_fn(state: AgentState) -> str:
    tone = state.memory.get("tone", "claro y directo")
    return f"- Sé {tone}. No inventes datos.\n- Responde en español.\n"

os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = "development"

async def interactive_loop():
    # Storage (sesiones/mensajes/estado)
    repo = MongoRepo(uri="mongodb://localhost:27017", db_name="agentix_demo")
    await repo.ensure_indexes()

    # Contexto UI (stack) + router de vistas
    router = build_router()
    cm: ContextManager = StackContextManager(router)


    agent = Agent(
        name="real_estate",
        repo=repo,
        context_manager=cm
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
