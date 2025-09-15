from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class StackFrame:
    screen_key: str
    params: Dict[str, Any]
    view_state: Dict[str, Any]
    return_path: Optional[str] = None
