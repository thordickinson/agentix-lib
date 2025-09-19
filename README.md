# 🐍 Agentix (working name)

> A **lightweight agent library** for Python, designed to create AI agents with **memory** and **tools** in just a few lines.
> Unlike LangGraph, Agentix focuses more on **context management** than workflow orchestration.

---

## 🚀 Motivation

Most agent libraries today (LangGraph, LangChain, etc.) are powerful but often too heavy for simple use cases.
**Agentix** takes a pragmatic approach:

* Define an agent with **memory** and **tools** in minimal code.
* Treat **conversation context** as a first-class concern.
* Simple integration of external **tools** with Python functions.
* Built on **[LiteLLM](https://github.com/BerriAI/litellm)** for flexible LLM backends.
* Async-first design for modern backends and services.

---

## ⚡ Quick Example

```python
from agentix import Agent, Tool

# 1. Define a tool
async def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."

weather_tool = Tool(
    name="get_weather",
    description="Fetch the weather for a given city",
    func=get_weather
)

# 2. Create an agent with memory and tools
agent = Agent(
    name="DemoAssistant",
    memory=True,
    tools=[weather_tool]
)

# 3. Run an interaction
response = await agent.run("What's the weather in Bogotá?")
print(response)
```

Expected output:

```
"The weather in Bogotá is sunny."
```

---

## 🔑 Features

* ✅ **Built on LiteLLM** → easily switch between OpenAI, Anthropic, local models, etc.
* ✅ **Context-first design** → focus on managing memory, not building graphs.
* ✅ **Minimal API** → define tools and agents with almost no boilerplate.
* ✅ **Async-first** → ready for FastAPI, WebSockets, or any async Python runtime.
* ✅ **Memory support (in progress):**

  * Currently backed by **MongoDB**.
  * Roadmap includes **Postgres, MySQL, SQLite, etc. via SQLAlchemy**.

---

## 📦 Installation

*(Work in progress — not yet published to PyPI)*

```bash
pip install agentix
```

---

## 🛠️ Roadmap

* [ ] SQLAlchemy-based memory backends (Postgres, MySQL, SQLite).
* [ ] Persistent memory with Redis and vector stores.
* [ ] Built-in observability (Langfuse / OpenTelemetry).
* [ ] Pre-built agent templates (chatbots, assistants, data explorers).
* [ ] Examples and starter kits.

