"""Memory Manager: handles retrieval, update, and consolidation of long-term memory.

Implements the TALM memory lifecycle:
1. Retrieve - depth-filtered K-NN search before planning
2. Update   - store successful task experiences
3. Consolidate - merge near-duplicate records via LLM synthesis
"""

from __future__ import annotations

import json
import logging

from talm.core.entities import MemoryRecord, TALMConfig
from talm.core.interfaces import IEmbedder, ILLMClient, IVectorDatabase
from talm.prompts.templates import consolidation_prompt

logger = logging.getLogger(__name__)


class MemoryManager:
    """Orchestrates all long-term memory operations."""

    def __init__(
        self,
        vector_db: IVectorDatabase,
        embedder: IEmbedder,
        llm: ILLMClient,
        config: TALMConfig,
    ) -> None:
        self._db = vector_db
        self._embedder = embedder
        self._llm = llm
        self._config = config

    async def retrieve(
        self, task_description: str, tree_depth: int
    ) -> list[tuple[MemoryRecord, float]]:
        """Retrieve relevant past experiences for the current task at this depth."""
        embedding = await self._embedder.embed(task_description)
        records = await self._db.retrieve(
            query_embedding=embedding,
            tree_depth=tree_depth,
            top_k=self._config.top_k,
            threshold=self._config.similarity_threshold,
        )
        logger.info(
            "Retrieved %d memory records for depth=%d", len(records), tree_depth
        )
        return records

    async def update(self, record: MemoryRecord) -> None:
        """Store a new experience and consolidate if near-duplicates exist."""
        text = f"{record.task_description}\n{record.reasoning_trace}"
        embedding = await self._embedder.embed(text)

        # Check for near-duplicate records to consolidate
        similar = await self._db.find_similar(
            embedding=embedding,
            tree_depth=record.tree_depth,
            threshold=self._config.merge_threshold,
        )

        if similar:
            # Consolidate: merge new record with the most similar existing one
            old_id, old_record, score = similar[0]
            logger.info(
                "Consolidating memory: similarity=%.3f, merging records", score
            )
            merged = await self._consolidate(old_record, record)
            merged_embedding = await self._embedder.embed(
                f"{merged.task_description}\n{merged.reasoning_trace}"
            )
            await self._db.delete(old_id)
            await self._db.insert(merged, merged_embedding)
        else:
            await self._db.insert(record, embedding)

    async def _consolidate(
        self, record_a: MemoryRecord, record_b: MemoryRecord
    ) -> MemoryRecord:
        """Use LLM to synthesize two similar records into one generalized record."""
        system, user = consolidation_prompt(record_a, record_b)
        raw = await self._llm.generate(system, user)

        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]
            data = json.loads(cleaned)
            return MemoryRecord(
                task_description=data["task_description"],
                reasoning_trace=data["reasoning_trace"],
                generated_code=data["generated_code"],
                tree_depth=record_a.tree_depth,
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Consolidation parse failed (%s), keeping newer record", exc)
            return record_b

    async def clear(self) -> None:
        """Clear all memory (used before benchmark evaluation runs)."""
        await self._db.clear()
        logger.info("Memory cleared")

    async def count(self) -> int:
        """Return the number of memory records stored."""
        return await self._db.count()
