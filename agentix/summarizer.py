from typing import Dict, Any, List
from .models import Message
from .config import MAX_CONTEXT_MESSAGES, RETAIN_TAKE

async def maybe_summarize_and_rotate(repo, session_id: str, user_id: str):
    """
    Si el número de mensajes excede MAX_CONTEXT_MESSAGES, rota y genera un summary.
    """
    sdoc = await repo.get_or_create_session(session_id, user_id)
    msgs = sdoc.get("messages", [])
    if len(msgs) <= MAX_CONTEXT_MESSAGES:
        return

    # Tomamos los últimos RETAIN_TAKE mensajes y resumimos el resto
    head = msgs[:-RETAIN_TAKE]
    tail = msgs[-RETAIN_TAKE:]
    summary = {"range": [0, len(head)], "summary": f"{len(head)} mensajes resumidos"}
    # await repo.replace_recent_messages(session_id, tail, summary)
