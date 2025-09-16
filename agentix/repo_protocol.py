from typing import Protocol, Dict, Any, List
from .models import Message, Session


class Repo(Protocol):
    async def get_or_create_session(self, session_id: str, user_id: str) -> Session: ...
    async def save_session(self, session: Session) -> None: ...
    async def append_messages(self, session_id: str, user_id: str, messages: List[Message]) -> None: ...

   