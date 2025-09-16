from __future__ import annotations
from typing import Any, Dict, List
from datetime import datetime, timezone

from pymongo import AsyncMongoClient, ASCENDING, ReturnDocument

from agentix.models import Message, Session
from agentix.repo_protocol import Repo


class MongoRepo(Repo):
    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        db_name: str = "agentix",
        sessions_col: str = "sessions",
        messages_col: str = "messages",
        users_col: str = "users",
        user_memories_col: str = "user_memories",
        audit_messages: bool = True,
    ):
        self.client = AsyncMongoClient(uri)
        self.db = self.client[db_name]
        self.sessions = self.db[sessions_col]
        self.messages = self.db[messages_col]
        self.users = self.db[users_col]
        self.user_memories = self.db[user_memories_col]
        self.audit_messages = audit_messages

    # ---------- Setup ----------
    async def ensure_indexes(self) -> None:
        await self.sessions.create_index([("session_id", ASCENDING), ("user_id", ASCENDING)], unique=True)
        await self.messages.create_index([("session_id", ASCENDING), ("ts", ASCENDING)])
        await self.users.create_index([("user_id", ASCENDING)], unique=True)
        await self.user_memories.create_index([("user_id", ASCENDING), ("key", ASCENDING)], unique=True)

    # ---------- Helpers ----------
    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _msg_to_doc(msg: Message) -> Dict[str, Any]:
        # Mongo acepta datetime; mantenemos ts tal cual
        return {"role": msg.role, "content": msg.content, "ts": msg.ts, "meta": msg.meta}

    # ---------- Repo API ----------
    async def get_or_create_session(self, session_id: str, user_id: str) -> Session:
        doc = await self.sessions.find_one({"session_id": session_id, "user_id": user_id})
        if doc:
            return Session(**doc)
        new_doc = Session(session_id=session_id, user_id=user_id)
        await self.sessions.insert_one(new_doc.model_dump())
        return new_doc

    def save_session(self, session):
        # TODO implement this
        ...

    def append_messages(self, session_id: str, user_id: str, messages: List[Message]) -> None:
        # TODO implement this
        ...
