"""Integration tests for ZvecAdapter against the real zvec library.

These tests exercise the production vector_db code path (not FakeVectorDB)
to catch issues that only surface against the actual zvec backend, such as
the missing Collection.close() method that broke clear() in the past.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from talm.core.entities import MemoryRecord
from talm.infrastructure.vector_db_zvec import ZvecAdapter


def _zero_vec(dim: int = 8) -> list[float]:
    return [0.1] * dim


@pytest.fixture
def tmp_persist_dir() -> str:
    d = tempfile.mkdtemp(prefix="zvec_test_")
    yield f"{d}/db"
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def adapter(tmp_persist_dir: str) -> ZvecAdapter:
    return ZvecAdapter(persist_dir=tmp_persist_dir, embedding_dim=8)


@pytest.mark.asyncio
async def test_insert_and_count(adapter: ZvecAdapter):
    rec = MemoryRecord("task", "trace", "code", tree_depth=0)
    rid = await adapter.insert(rec, _zero_vec())
    assert rid
    assert await adapter.count() == 1


@pytest.mark.asyncio
async def test_clear_removes_records_and_allows_reuse(adapter: ZvecAdapter):
    """Regression: clear() used to call a non-existent _collection.close()
    and crash with AttributeError. After clear(), the adapter must still
    be usable for new inserts."""
    rec = MemoryRecord("first", "trace", "code", tree_depth=0)
    await adapter.insert(rec, _zero_vec())
    assert await adapter.count() == 1

    await adapter.clear()  # must not raise
    assert await adapter.count() == 0

    # Adapter must still work after clear
    rec2 = MemoryRecord("second", "trace", "code", tree_depth=0)
    await adapter.insert(rec2, _zero_vec())
    assert await adapter.count() == 1


@pytest.mark.asyncio
async def test_clear_when_empty_is_safe(adapter: ZvecAdapter):
    """Clearing an empty adapter must not raise."""
    assert await adapter.count() == 0
    await adapter.clear()
    assert await adapter.count() == 0


@pytest.mark.asyncio
async def test_metadata_round_trip(adapter: ZvecAdapter):
    """Metadata stored at insert time must survive a retrieve()."""
    rec = MemoryRecord(
        task_description="indexed task",
        reasoning_trace="",
        generated_code="",
        tree_depth=0,
        metadata={"session_id": "abc123"},
    )
    await adapter.insert(rec, _zero_vec())
    results = await adapter.retrieve(
        query_embedding=_zero_vec(),
        tree_depth=0,
        top_k=5,
        threshold=0.0,
    )
    assert len(results) == 1
    retrieved, _sim = results[0]
    assert retrieved.metadata.get("session_id") == "abc123"
