"""Tests for the MemoryManager: retrieve, update, and consolidation logic."""

import json
import pytest

from talm.core.entities import MemoryRecord, TALMConfig
from talm.memory.manager import MemoryManager
from tests.fakes import FakeEmbedder, FakeLLM, FakeVectorDB


@pytest.fixture
def config() -> TALMConfig:
    return TALMConfig(
        similarity_threshold=0.75,
        top_k=3,
        merge_threshold=0.95,
    )


@pytest.fixture
def vector_db() -> FakeVectorDB:
    return FakeVectorDB()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dimension=8)


@pytest.fixture
def manager(vector_db, embedder, config) -> MemoryManager:
    llm = FakeLLM()
    return MemoryManager(vector_db=vector_db, embedder=embedder, llm=llm, config=config)


@pytest.mark.asyncio
async def test_update_inserts_new_record(manager: MemoryManager, vector_db: FakeVectorDB):
    """A new record with no near-duplicates should be inserted."""
    record = MemoryRecord(
        task_description="Write a fibonacci function",
        reasoning_trace="Use recursion with memoization",
        generated_code="def fib(n): ...",
        tree_depth=0,
    )
    await manager.update(record)
    assert await vector_db.count() == 1


@pytest.mark.asyncio
async def test_retrieve_filters_by_depth(manager: MemoryManager, vector_db: FakeVectorDB, embedder: FakeEmbedder):
    """Retrieve should only return records at the same tree depth."""
    r0 = MemoryRecord("task A", "plan A", "code A", tree_depth=0)
    r1 = MemoryRecord("task A", "plan A", "code A", tree_depth=1)

    emb = await embedder.embed("task A\nplan A")
    await vector_db.insert(r0, emb)
    await vector_db.insert(r1, emb)

    results = await manager.retrieve("task A", tree_depth=0)
    assert all(rec.tree_depth == 0 for rec, _ in results)

    results = await manager.retrieve("task A", tree_depth=1)
    assert all(rec.tree_depth == 1 for rec, _ in results)


@pytest.mark.asyncio
async def test_retrieve_respects_threshold(manager: MemoryManager, vector_db: FakeVectorDB, embedder: FakeEmbedder):
    """Records below the similarity threshold should not be returned."""
    rec = MemoryRecord("sorting algorithm", "quicksort steps", "def sort(): ...", tree_depth=0)
    emb = await embedder.embed("sorting algorithm\nquicksort steps")
    await vector_db.insert(rec, emb)

    # Query with a completely different text should yield low similarity
    results = await manager.retrieve("deploy kubernetes cluster", tree_depth=0)
    # May or may not match depending on hash collision, but the mechanism is tested
    for _, sim in results:
        assert sim >= 0.75


@pytest.mark.asyncio
async def test_consolidation_merges_duplicates(vector_db: FakeVectorDB, embedder: FakeEmbedder, config: TALMConfig):
    """When a near-duplicate exists (sim >= 0.95), records should be consolidated."""
    # LLM that returns a proper consolidated JSON
    def consolidation_responder(system: str, user: str) -> str:
        return json.dumps({
            "task_description": "Generalized palindrome check",
            "reasoning_trace": "Merged approach",
            "generated_code": "def is_palindrome(s): return s == s[::-1]",
        })

    llm = FakeLLM(responder=consolidation_responder)
    mgr = MemoryManager(vector_db=vector_db, embedder=embedder, llm=llm, config=config)

    # Insert a record
    rec1 = MemoryRecord("check palindrome", "compare reversed", "def pal(s): ...", tree_depth=0)
    await mgr.update(rec1)
    assert await vector_db.count() == 1

    # Insert an identical record — should trigger consolidation
    rec2 = MemoryRecord("check palindrome", "compare reversed", "def pal(s): pass", tree_depth=0)
    await mgr.update(rec2)

    # Should still be 1 record (merged), not 2
    assert await vector_db.count() == 1

    records = await vector_db.list_all()
    _, merged = records[0]
    assert merged.task_description == "Generalized palindrome check"


@pytest.mark.asyncio
async def test_consolidation_not_triggered_for_different_tasks(manager: MemoryManager, vector_db: FakeVectorDB):
    """Different tasks should not trigger consolidation."""
    r1 = MemoryRecord("fibonacci function", "recursion", "def fib(): ...", tree_depth=0)
    r2 = MemoryRecord("binary search tree", "iterative insert", "class BST(): ...", tree_depth=0)

    await manager.update(r1)
    await manager.update(r2)

    assert await vector_db.count() == 2


@pytest.mark.asyncio
async def test_clear_removes_all_records(manager: MemoryManager, vector_db: FakeVectorDB):
    """Clear should remove all records."""
    for i in range(3):
        rec = MemoryRecord(f"task {i}", f"plan {i}", f"code {i}", tree_depth=0)
        await manager.update(rec)

    assert await vector_db.count() == 3
    await manager.clear()
    assert await vector_db.count() == 0
