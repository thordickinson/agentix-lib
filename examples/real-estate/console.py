from __future__ import annotations
import asyncio
from typing import Optional

from agentix import Agent, AgentContext
from agentix.storage.mongo import MongoRepo
from agentix.context import ContextManager, SimpleContextManager
from agentix.tools import tool_from_fn
from agentix.stack import StackContextManager
from agentix.utils.console import console_loop
from .router import build_router
import os


os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = "development"


async def get_weather(city: str) -> int:
    """
    Obtiene el clima en la ciudad dada en grados celcius.
    :param str city: el nombre de la ciudad
    """
    return 22

async def interactive_loop():
    # Storage (sesiones/mensajes/estado)
    repo = MongoRepo(uri="mongodb://localhost:27017", db_name="agentix_demo")
    await repo.ensure_indexes()

    # Contexto UI (stack) + router de vistas
    router = build_router()

    system = "Actúa como Napoleón Bonaparte, responde en español antigüo y recordando tus hazañas"
    cm: ContextManager = SimpleContextManager(system, tools=[tool_from_fn(get_weather)])


    agent = Agent(
        name="real_estate",
        repo=repo,
        context_manager=cm
    )

    user_id = "user_demo"
    session_id = "session_demo"

    await console_loop(
        agent=agent,
        repo=repo,
        user_id=user_id,
        session_id=session_id,
        messages_tail=8,
    )

if __name__ == "__main__":
    asyncio.run(interactive_loop())
