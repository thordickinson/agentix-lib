from typing import List, Any
import inspect
import re
from enum import Enum
from ..models import AgentContext

from agentix.models import Param, Tool


def tool_from_fn(fn: Any) -> Tool:
    name = fn.__name__
    doc = inspect.getdoc(fn) or ""

    # Parse docstring (igual que antes)
    desc_lines = []
    param_docs = {}
    param_re = re.compile(r":param\s+([\w\[\]]+)\s+(\w+):\s*(.+)")

    for line in doc.splitlines():
        line_stripped = line.strip()
        match = param_re.match(line_stripped)
        if match:
            _, pname, pdesc = match.groups()
            param_docs[pname] = pdesc
        elif not line_stripped.startswith(":"):
            desc_lines.append(line_stripped)

    desc = " ".join(desc_lines).strip()

    # Firma
    sig = inspect.signature(fn)
    params: List[Param] = []

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue

        annotation = param.annotation

        # 🔥 Ignorar si es AgentContext
        if annotation is AgentContext:
            continue

        ptype = "Any"
        enum_values = None
        if annotation is not inspect._empty:
            if isinstance(annotation, type) and issubclass(annotation, Enum):
                ptype = annotation.__name__
                enum_values = [e.value for e in annotation]
            else:
                ptype = getattr(annotation, "__name__", str(annotation))

        optional = param.default is not inspect._empty
        default_value = None if param.default is inspect._empty else param.default
        pdesc = param_docs.get(pname, "")

        params.append(Param(
            name=pname,
            type=ptype,
            desc=pdesc,
            optional=optional,
            default_value=default_value,
            enum_values=enum_values
        ))

    return Tool(name=name, desc=desc, params=params, fn=fn)
