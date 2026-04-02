"""ChromaDB adapter implementing IVectorDatabase for long-term memory.

Alternative to Zvec. Use by setting vector_db.provider = "chroma" in config.yaml.
"""

from __future__ import annotations

import json
import logging
import uuid

import chromadb

from talm.core.entities import MemoryRecord
from talm.core.interfaces import IVectorDatabase

logger = logging.getLogger(__name__)

COLLECTION_NAME = "talm_memory"


class ChromaDBAdapter(IVectorDatabase):
    """Vector database backed by ChromaDB (local persistent storage)."""

    def __init__(self, persist_dir: str = "./talm_chroma_db", **kwargs) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    async def retrieve(
        self,
        query_embedding: list[float],
        tree_depth: int,
        top_k: int = 3,
        threshold: float = 0.75,
    ) -> list[tuple[MemoryRecord, float]]:
        """Retrieve top-k records filtered by tree_depth with similarity >= threshold."""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 3,
            where={"tree_depth": tree_depth},
            include=["documents", "metadatas", "distances"],
        )

        records: list[tuple[MemoryRecord, float]] = []
        if not results["documents"] or not results["documents"][0]:
            return records

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            similarity = 1.0 - dist
            if similarity < threshold:
                continue
            record = self._doc_to_record(doc, meta)
            records.append((record, similarity))
            if len(records) >= top_k:
                break

        return records

    async def insert(self, record: MemoryRecord, embedding: list[float]) -> str:
        """Insert a new memory record and return its ID."""
        record_id = str(uuid.uuid4())
        doc = json.dumps({
            "task_description": record.task_description,
            "reasoning_trace": record.reasoning_trace,
            "generated_code": record.generated_code,
        })
        self._collection.add(
            ids=[record_id],
            embeddings=[embedding],
            documents=[doc],
            metadatas=[{"tree_depth": record.tree_depth, **record.metadata}],
        )
        logger.info("Inserted memory record %s at depth %d", record_id, record.tree_depth)
        return record_id

    async def delete(self, record_id: str) -> None:
        """Delete a record by ID."""
        self._collection.delete(ids=[record_id])

    async def find_similar(
        self,
        embedding: list[float],
        tree_depth: int,
        threshold: float = 0.95,
    ) -> list[tuple[str, MemoryRecord, float]]:
        """Find records with similarity above the merge threshold."""
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=10,
            where={"tree_depth": tree_depth},
            include=["documents", "metadatas", "distances"],
        )

        matches: list[tuple[str, MemoryRecord, float]] = []
        if not results["ids"] or not results["ids"][0]:
            return matches

        for rid, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - dist
            if similarity >= threshold:
                record = self._doc_to_record(doc, meta)
                matches.append((rid, record, similarity))

        return matches

    async def clear(self) -> None:
        """Clear all records from the collection."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    async def count(self) -> int:
        """Return the number of records stored."""
        return self._collection.count()

    async def list_all(self) -> list[tuple[str, MemoryRecord]]:
        """List all records from the collection."""
        total = self._collection.count()
        if total == 0:
            return []
        results = self._collection.get(
            include=["documents", "metadatas"],
            limit=total,
        )
        records = []
        for rid, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
            records.append((rid, self._doc_to_record(doc, meta)))
        return records

    @staticmethod
    def _doc_to_record(doc: str, meta: dict) -> MemoryRecord:
        """Deserialize a ChromaDB document back into a MemoryRecord."""
        data = json.loads(doc)
        return MemoryRecord(
            task_description=data["task_description"],
            reasoning_trace=data["reasoning_trace"],
            generated_code=data["generated_code"],
            tree_depth=meta.get("tree_depth", 0),
        )
