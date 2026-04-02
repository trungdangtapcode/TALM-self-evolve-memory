"""Zvec adapter implementing IVectorDatabase for long-term memory.

Zvec is an in-process vector database by Alibaba (based on Proxima engine).
Lightweight, no server needed, low-latency cosine similarity search.

Zvec doc IDs must be simple alphanumeric strings, so we store record
metadata in a side-index JSON file alongside the Zvec database.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid

import zvec

from talm.core.entities import MemoryRecord
from talm.core.interfaces import IVectorDatabase

logger = logging.getLogger(__name__)


class ZvecAdapter(IVectorDatabase):
    """Vector database backed by Zvec (local in-process storage)."""

    def __init__(self, persist_dir: str, embedding_dim: int) -> None:
        self._persist_dir = persist_dir
        self._embedding_dim = embedding_dim
        self._collection: zvec.Collection | None = None
        # Side-index: maps doc_id -> MemoryRecord JSON
        self._index_path = os.path.join(persist_dir + "_index.json")
        self._index: dict[str, dict] = {}
        self._load_index()
        self._init_db()

    def _load_index(self) -> None:
        """Load the side-index from disk."""
        if os.path.exists(self._index_path):
            with open(self._index_path) as f:
                self._index = json.load(f)

    def _save_index(self) -> None:
        """Persist the side-index to disk."""
        os.makedirs(os.path.dirname(self._index_path) or ".", exist_ok=True)
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def _init_db(self) -> None:
        """Initialize or open the Zvec collection."""
        schema = zvec.CollectionSchema(
            name="talm_memory",
            vectors=zvec.VectorSchema(
                "embedding",
                zvec.DataType.VECTOR_FP32,
                self._embedding_dim,
                index_param=zvec.HnswIndexParam(
                    metric_type=zvec.MetricType.COSINE,
                ),
            ),
        )
        if os.path.exists(self._persist_dir):
            try:
                self._collection = zvec.open(path=self._persist_dir)
                logger.info("Opened existing Zvec DB at %s", self._persist_dir)
                return
            except Exception:
                logger.warning("Failed to open existing DB, recreating")
                shutil.rmtree(self._persist_dir, ignore_errors=True)

        self._collection = zvec.create_and_open(
            path=self._persist_dir, schema=schema
        )
        logger.info("Created new Zvec DB at %s", self._persist_dir)

    async def retrieve(
        self,
        query_embedding: list[float],
        tree_depth: int,
        top_k: int = 3,
        threshold: float = 0.75,
    ) -> list[tuple[MemoryRecord, float]]:
        """Retrieve top-k records filtered by tree_depth with similarity >= threshold."""
        results = self._collection.query(
            zvec.VectorQuery("embedding", vector=query_embedding),
            topk=top_k * 5,  # over-fetch then filter
        )

        records: list[tuple[MemoryRecord, float]] = []
        for item in results:
            doc_id = item.id
            meta = self._index.get(doc_id)
            if meta is None:
                continue
            if meta.get("tree_depth") != tree_depth:
                continue
            # Zvec cosine returns distance (0=identical, 1=orthogonal)
            similarity = 1.0 - float(item.score)
            if similarity < threshold:
                continue
            record = self._meta_to_record(meta)
            records.append((record, similarity))
            if len(records) >= top_k:
                break

        return records

    async def insert(self, record: MemoryRecord, embedding: list[float]) -> str:
        """Insert a new memory record. Returns the record ID."""
        # Generate a Zvec-compatible alphanumeric ID
        record_id = uuid.uuid4().hex

        # Store vector in Zvec
        self._collection.insert([
            zvec.Doc(id=record_id, vectors={"embedding": embedding}),
        ])

        # Store metadata in side-index
        self._index[record_id] = {
            "task_description": record.task_description,
            "reasoning_trace": record.reasoning_trace,
            "generated_code": record.generated_code,
            "tree_depth": record.tree_depth,
            "metadata": record.metadata,
        }
        self._save_index()

        logger.info("Inserted memory record %s at depth %d", record_id, record.tree_depth)
        return record_id

    async def delete(self, record_id: str) -> None:
        """Delete a record by its ID from both Zvec and side-index."""
        try:
            self._collection.delete([record_id])
        except Exception as e:
            logger.warning("Zvec delete failed for %s: %s", record_id, e)

        self._index.pop(record_id, None)
        self._save_index()
        logger.info("Deleted memory record %s", record_id)

    async def find_similar(
        self,
        embedding: list[float],
        tree_depth: int,
        threshold: float = 0.95,
    ) -> list[tuple[str, MemoryRecord, float]]:
        """Find records with similarity above the merge threshold."""
        results = self._collection.query(
            zvec.VectorQuery("embedding", vector=embedding),
            topk=20,
        )

        matches: list[tuple[str, MemoryRecord, float]] = []
        for item in results:
            doc_id = item.id
            meta = self._index.get(doc_id)
            if meta is None:
                continue
            if meta.get("tree_depth") != tree_depth:
                continue
            # Zvec cosine returns distance (0=identical, 1=orthogonal)
            similarity = 1.0 - float(item.score)
            if similarity >= threshold:
                record = self._meta_to_record(meta)
                matches.append((doc_id, record, similarity))

        return matches

    async def clear(self) -> None:
        """Clear all records by recreating the database and index."""
        if self._collection is not None:
            self._collection.close()
        shutil.rmtree(self._persist_dir, ignore_errors=True)
        self._index.clear()
        self._save_index()
        self._init_db()
        logger.info("Cleared all memory records")

    async def count(self) -> int:
        """Return the number of records stored."""
        return len(self._index)

    @staticmethod
    def _meta_to_record(meta: dict) -> MemoryRecord:
        """Convert a side-index entry back into a MemoryRecord."""
        return MemoryRecord(
            task_description=meta["task_description"],
            reasoning_trace=meta["reasoning_trace"],
            generated_code=meta["generated_code"],
            tree_depth=meta["tree_depth"],
            metadata=meta.get("metadata", {}),
        )
