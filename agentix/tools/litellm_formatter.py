from ..models import Tool


def tool_to_dict(tool: Tool) -> dict:
    """
    Convierte un Tool en un dict estilo OpenAI/LiteLLM function-calling schema.
    """
    properties = {}
    required = []

    for p in tool.params:
        # Base del schema
        prop_schema = {"type": _map_type(p.type)}

        if p.desc:
            prop_schema["description"] = p.desc

        if p.enum_values:
            # Usamos valores en lugar de nombres
            prop_schema["enum"] = p.enum_values  

        if p.default_value is not None:
            # Representamos default como string si es Enum
            if hasattr(p.default_value, "value"):
                prop_schema["default"] = p.default_value.value
            else:
                prop_schema["default"] = p.default_value

        properties[p.name] = prop_schema

        if not p.optional:
            required.append(p.name)

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.desc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required if required else [],
            },
        },
    }


def _map_type(ptype: str) -> str:
    """
    Mapear tipos de Python a JSON Schema types.
    """
    mapping = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "Any": "string",  # fallback
    }
    return mapping.get(ptype, "string")
