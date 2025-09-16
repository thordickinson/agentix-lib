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

    async def append_message(self, session_id: str, user_id: str, msg: Message) -> Dict[str, Any]:
        msg_doc = self._msg_to_doc(msg)
        now = self._utcnow()

        # ✅ Combina $push y $set en el MISMO dict
        res = await self.sessions.find_one_and_update(
            {"session_id": session_id, "user_id": user_id},
            {"$push": {"messages": msg_doc}, "$set": {"updated_at": now}},
            return_document=ReturnDocument.AFTER,
        )

        if self.audit_messages:
            audit_doc = {
                "session_id": session_id,
                "user_id": user_id,
                "role": msg.role,
                "content": msg.content,
                "meta": msg.meta,
                "ts": msg.ts,
                "created_at": now,
            }
            await self.messages.insert_one(audit_doc)

        return res or {}

    async def replace_recent_messages(
        self,
        session_id: str,
        user_id: str,
        new_tail: List[Dict[str, Any]],
        new_summary: Dict[str, Any],
        unlock: bool = True,
    ) -> None:
        """
        Reemplaza los mensajes recientes por 'new_tail' y agrega 'new_summary' a summaries.
        Se usa tras resumir para rotar el contexto.
        """
        now = self._utcnow()
        update: Dict[str, Any] = {
            "$set": {
                "messages": new_tail,
                "updated_at": now,
            },
            "$push": {"summaries": new_summary},
        }
        if unlock:
            update["$set"]["locks.summarize"] = False

        await self.sessions.update_one(
            {"session_id": session_id, "user_id": user_id},
            update,
        )

    async def try_lock_summarize(self, session_id: str) -> bool:
        """
        Intenta adquirir lock de resumen si no está tomado.
        """
        doc = await self.sessions.find_one_and_update(
            {
                "session_id": session_id,
                "locks.summarize": {"$in": [False, None]},
            },
            {"$set": {"locks.summarize": True}},
            return_document=ReturnDocument.BEFORE,
        )
        return doc is not None

    async def unlock_summarize(self, session_id: str) -> None:
        await self.sessions.update_one(
            {"session_id": session_id},
            {"$set": {"locks.summarize": False, "updated_at": self._utcnow()}},
        )

    async def update_state(self, session_id: str, flat_patch: Dict[str, Any]) -> None:
        """
        Aplica un patch plano sobre 'agent_state'. Ejemplo:
          flat_patch = {
            "memory.ui_stack.0.view_state.changes.name": "Nuevo",
            "scratchpad.0": "paso intermedio"
          }
        Se traduce a $set sobre 'agent_state.<path>'.
        """
        if not flat_patch:
            return

        set_ops = {f"agent_state.{k}": v for k, v in flat_patch.items()}
        await self.sessions.update_one(
            {"session_id": session_id},
            {"$set": {**set_ops, "updated_at": self._utcnow()}},
        )

    async def upsert_user_profile(self, user_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        now = self._utcnow()
        doc = await self.users.find_one_and_update(
            {"user_id": user_id},
            {
                "$set": {"profile": patch, "updated_at": now},
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return doc or {}

    async def upsert_user_memories_bulk(self, user_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        items: lista de { "key": str, "value": Any }
        Upsert individual por (user_id, key).
        """
        if not items:
            return {"upserted": 0}

        now = self._utcnow()
        upserted = 0
        for it in items:
            key = it.get("key")
            if not key:
                continue
            value = it.get("value")
            res = await self.user_memories.update_one(
                {"user_id": user_id, "key": key},
                {
                    "$set": {"value": value, "updated_at": now},
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )
            if res.upserted_id is not None:
                upserted += 1

        return {"upserted": upserted}
