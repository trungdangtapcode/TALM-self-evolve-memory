"""Tests for the TALMWorkflow orchestrator: end-to-end integration with fakes."""

import json
import pytest

from talm.core.entities import AgentResponseStatus, TALMConfig
from talm.workflow import TALMWorkflow
from tests.fakes import FakeEmbedder, FakeLLM, FakeSandbox, FakeVectorDB


def _make_workflow_responder():
    """Responder that handles the full pipeline."""

    def responder(system: str, user: str) -> str:
        sys_lower = system.lower()
        if "architect" in sys_lower:
            return "DIRECT_IMPLEMENTATION\n1. Solve directly"
        if "decomposition" in sys_lower:
            return "NO_DELEGATION"
        if "clarify" in sys_lower:
            return '{"status": "clear"}'
        if "programmer" in sys_lower:
            return '```python\ndef solution():\n    return "hello"\n```'
        if "qa" in sys_lower or "test" in sys_lower:
            return '```python\nassert solution() == "hello"\nprint("ALL TESTS PASSED")\n```'
        if "debug" in sys_lower:
            return '```python\ndef solution():\n    return "hello"\n```'
        if "knowledge" in sys_lower:
            return json.dumps({
                "task_description": "merged",
                "reasoning_trace": "merged",
                "generated_code": "def solution(): return 'hello'",
            })
        return "OK"

    return responder


def _build_workflow(max_depth: int = 1, sandbox_pass: bool = True) -> TALMWorkflow:
    config = TALMConfig(
        max_depth=max_depth,
        initial_branching=3,
        decay_rate=1,
        max_retries=2,
        sandbox_enabled=True,
    )
    llm = FakeLLM(responder=_make_workflow_responder())
    embedder = FakeEmbedder(dimension=8)
    vector_db = FakeVectorDB()
    sandbox = FakeSandbox(always_pass=sandbox_pass)

    return TALMWorkflow(
        config=config, llm=llm, embedder=embedder,
        vector_db=vector_db, sandbox=sandbox,
    )


@pytest.mark.asyncio
async def test_workflow_end_to_end():
    """Full workflow should produce successful code and store memory."""
    wf = _build_workflow()

    result = await wf.run("Write a greeting function")

    assert result.status == AgentResponseStatus.SUCCESS
    assert "hello" in result.code
    assert await wf.get_memory_count() == 1


@pytest.mark.asyncio
async def test_workflow_failure_does_not_store_memory():
    """Failed workflow should not store memory."""
    wf = _build_workflow(sandbox_pass=False)

    result = await wf.run("Write something impossible")

    assert result.status == AgentResponseStatus.FAILURE
    assert await wf.get_memory_count() == 0


@pytest.mark.asyncio
async def test_workflow_clear_memory():
    """Clear memory should reset the store."""
    wf = _build_workflow()

    await wf.run("task 1")
    await wf.run("task 2")
    assert await wf.get_memory_count() >= 1

    await wf.clear_memory()
    assert await wf.get_memory_count() == 0


@pytest.mark.asyncio
async def test_workflow_list_memory():
    """list_memory should return all stored records."""
    wf = _build_workflow()

    await wf.run("Write a hello function")
    records = await wf.list_memory()

    assert len(records) >= 1
    assert "task_description" in records[0]
    assert "generated_code" in records[0]


@pytest.mark.asyncio
async def test_workflow_memory_accumulates():
    """Multiple runs should accumulate memory (unless consolidated)."""
    wf = _build_workflow()

    await wf.run("Write fibonacci")
    count1 = await wf.get_memory_count()

    await wf.run("Write binary search")
    count2 = await wf.get_memory_count()

    # Second task is different, should add a new record
    assert count2 >= count1
