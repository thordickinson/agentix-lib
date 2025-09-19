from __future__ import annotations
import asyncio
from textwrap import dedent
from typing import Optional

from agentix import Agent, AgentEvent
from agentix.storage import MongoAgentRepository
from agentix.context import ContextManager, SimpleContextManager
from agentix.tools import tool_from_fn
from agentix.stack import StackContextManager
from agentix.utils.console import console_loop
from .router import build_router
import random
import os
import logging

logging.basicConfig(
    level=logging.ERROR,  # solo imprime ERROR y CRITICAL
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = "development"

async def get_weather(city: str) -> int:
    """
    Obtiene el clima en la ciudad dada en grados celcius.
    :param str city: el nombre de la ciudad
    """
    temperature = random.randint(19, 30) 
    print(f"\tGetting temperature in {city} -> {temperature}")
    return temperature

async def interactive_loop():
    # Storage (sesiones/mensajes/estado)
    repo = MongoAgentRepository(uri="mongodb://localhost:27017", db_name="agentix_demo")
    await repo.ensure_indexes()

    # Contexto UI (stack) + router de vistas
    router = build_router()

    system = dedent("""Eres Napoleón Bonaparte, responde al usuario en español antigüo y 
        relacionandola con alguna de tus hazañas,se breve, como hablando por whatsapp
    """)
    cm: ContextManager = SimpleContextManager(system, tools=[tool_from_fn(get_weather)])
    os.environ["LANGFUSE_TRACING_ENABLED"] = "false"

    def log_events(event: AgentEvent):
        print(f"{event.type}: {event.message or '<no message>'}")


    agent = Agent(
        name="real_estate",
        repository=repo,
        context_manager=cm,
        max_interactions_in_memory=10,
        event_listener=log_events
    )

    user_id = "user_demo"
    session_id = "session_demo"

    await console_loop(
        agent=agent,
        repo=repo,
        user_id=user_id,
        session_id=session_id,
        messages_tail=8
    )

if __name__ == "__main__":
    asyncio.run(interactive_loop())
