from ._version import __version__

from .models import AgentState, Message, Tool, ToolCallMsg, Final, parse_model_step
from .agent import Agent
from .llm import LLMClient, LiteLLMAdapter
from .context import ContextManager  # ⬅️ exporta la interfaz

__all__ = [
    "__version__",
    "Agent",
    "AgentState",
    "Message",
    "Tool",
    "ToolCallMsg",
    "Final",
    "parse_model_step",
    "LLMClient",
    "LiteLLMAdapter",
    "ContextManager",
]
