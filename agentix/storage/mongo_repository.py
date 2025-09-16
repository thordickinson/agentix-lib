from __future__ import annotations
from typing import List
from datetime import datetime, timezone

from pymongo import AsyncMongoClient, ASCENDING

from agentix.models import Message, Session
from agentix.agent_repository import AgentRepository


class MongoAgentRepository(AgentRepository):
    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        db_name: str = "agentix",
        sessions_col: str = "sessions",
        messages_col: str = "messages",
        users_col: str = "users",
        audit_messages: bool = True,
    ):
        self.client = AsyncMongoClient(uri)
        self.db = self.client[db_name]
        self.sessions = self.db[sessions_col]
        self.messages = self.db[messages_col]
        self.users = self.db[users_col]
        self.audit_messages = audit_messages

    # ---------- Setup ----------
    async def ensure_indexes(self) -> None:
        await self.sessions.create_index([("session_id", ASCENDING), ("user_id", ASCENDING)], unique=True)
        await self.messages.create_index([("session_id", ASCENDING), ("ts", ASCENDING)])
        await self.users.create_index([("user_id", ASCENDING)], unique=True)
        
    # ---------- Repo API ----------
    async def get_or_create_session(self, session_id: str, user_id: str) -> Session:
        doc = await self.sessions.find_one({"session_id": session_id, "user_id": user_id})
        if doc:
            return Session(**doc)
        new_doc = Session(session_id=session_id, user_id=user_id)
        await self.sessions.insert_one(new_doc.model_dump())
        return new_doc

    async def save_session(self, session: Session):
        session.updated_at = datetime.now(timezone.utc)
        await self.sessions.find_one_and_update(
            {"session_id": session.session_id, "user_id": session.user_id},
            {"$set": session.model_dump()}
        )

    async def append_messages(self, session_id: str, user_id: str, messages: List[Message]) -> None:
        await self.messages.insert_many([m.model_dump() | {"session_id": session_id, "user_id": user_id, "ts": datetime.now(timezone.utc)} for m in messages])
