from __future__ import annotations
import json
from typing import List, Dict, Protocol
from .models import Message

class LLMClient(Protocol):
    async def generate(
        self,
        history: List[Message],
        tool_specs: List[Dict[str, Any]],
        developer_instructions: str,
        model: str,
    ) -> str: ...

def _contract_system_block(tool_specs: List[Dict], developer_instructions: str) -> str:
    return (
        "Responde SIEMPRE con un JSON válido (un único objeto) y NADA MÁS.\n"
        "Formatos permitidos (elige exactamente UNO):\n"
        '- Llamada a herramienta:\n'
        '  {"type":"tool_call","call":{"name":"<tool_name>","args":{...}}}\n'
        '- Respuesta final al usuario:\n'
        '  {"type":"final","answer":"<texto>"}\n\n'
        "Reglas:\n"
        "- NO uses otros campos ni otros formatos.\n"
        "- NO devuelvas pensamientos o razonamientos internos.\n"
        "- Si necesitas ejecutar una acción, usa 'tool_call'. Si ya puedes contestar, usa 'final'.\n\n"
        "Herramientas disponibles:\n"
        + json.dumps(tool_specs, ensure_ascii=False)
        + "\n\nInstrucciones del developer:\n"
        + developer_instructions
    )

class LiteLLMAdapter:
    """
    Cliente basado en litellm.
    """
    def __init__(self, model: str | None = None):
        self.model = model

    async def generate(
        self,
        history: List[Message],
        tool_specs: List[Dict[str, Any]],
        developer_instructions: str,
        model: str,
    ) -> str:
        import litellm
        msgs = [m.to_wire() for m in history]
        sysmsg = {"role": "system", "content": _contract_system_block(tool_specs, developer_instructions)}
        wire = [sysmsg] + msgs
        resp = await litellm.acompletion(model=model or self.model, messages=wire)
        return resp["choices"][0]["message"]["content"]
