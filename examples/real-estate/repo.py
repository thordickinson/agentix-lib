from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from pymongo import AsyncMongoClient, ASCENDING, ReturnDocument
from bson import ObjectId

class PropertyRepo:
    def __init__(self, uri: str = "mongodb://localhost:27017", db_name: str = "agentix_demo"):
        self.client = AsyncMongoClient(uri)
        self.db = self.client[db_name]
        self.col = self.db["properties"]

    async def ensure_indexes(self) -> None:
        await self.col.create_index([("created_at", ASCENDING)])
        await self.col.create_index([("title", ASCENDING)])

    @staticmethod
    def _now():
        return datetime.now(timezone.utc)

    def _to_public(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        if not doc:
            return {}
        d = dict(doc)
        d["property_id"] = str(d["_id"])
        d.pop("_id", None)
        return d

    async def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        now = self._now()
        doc = {**data, "created_at": now, "updated_at": now}
        res = await self.col.insert_one(doc)
        return {"property_id": str(res.inserted_id)}

    async def get(self, property_id: str) -> Optional[Dict[str, Any]]:
        try:
            doc = await self.col.find_one({"_id": ObjectId(property_id)})
        except Exception:
            return None
        return self._to_public(doc) if doc else None

    async def list(self, limit: int = 20) -> List[Dict[str, Any]]:
        cur = self.col.find({}).sort("created_at", -1).limit(limit)
        return [self._to_public(doc) async for doc in cur]

    async def update(self, property_id: str, changes: Dict[str, Any]) -> Dict[str, Any]:
        doc = await self.col.find_one_and_update(
            {"_id": ObjectId(property_id)},
            {"$set": {**changes, "updated_at": self._now()}},
            return_document=ReturnDocument.AFTER,
        )
        if not doc:
            raise ValueError("Propiedad no encontrada")
        return self._to_public(doc)

    async def delete(self, property_id: str) -> bool:
        res = await self.col.delete_one({"_id": ObjectId(property_id)})
        return res.deleted_count > 0
