"""Tests for the CodeAgent: 5-phase lifecycle, delegation, and re-reasoning."""

import json
import pytest

from talm.agents.code_agent import CodeAgent
from talm.agents.validation_agent import ValidationAgent
from talm.core.entities import AgentResponseStatus, TALMConfig, Task
from talm.memory.manager import MemoryManager
from tests.fakes import FakeEmbedder, FakeLLM, FakeSandbox, FakeVectorDB


def _make_llm_responder(delegate: bool = False):
    """Create a responder that handles all 5 phases deterministically."""

    def responder(system: str, user: str) -> str:
        sys_lower = system.lower()

        # Planning phase
        if "architect" in sys_lower and "plan" in sys_lower:
            if delegate:
                return "1. Parse input\n2. Process data\n3. Format output"
            return "DIRECT_IMPLEMENTATION\n1. Implement the solution directly"

        # Delegation phase
        if "decomposition" in sys_lower:
            if delegate:
                return json.dumps([
                    {"subtask_id": 1, "description": "Parse the input data"},
                    {"subtask_id": 2, "description": "Process and format output"},
                ])
            return "NO_DELEGATION"

        # Clarification check
        if "clarify" in sys_lower and "ambiguous" in sys_lower:
            return '{"status": "clear"}'

        # Integration review
        if "reviewing the outputs" in sys_lower:
            return '{"verdict": "accept"}'

        # Implementation phase
        if "programmer" in sys_lower:
            return '```python\ndef solve():\n    return 42\n```'

        # Validation — test generation
        if "qa engineer" in sys_lower or "test" in sys_lower:
            return '```python\nassert solve() == 42\nprint("ALL TESTS PASSED")\n```'

        # Debug
        if "debug" in sys_lower or "fix" in sys_lower:
            return '```python\ndef solve():\n    return 42\n```'

        # Consolidation
        if "knowledge engineer" in sys_lower:
            return json.dumps({
                "task_description": "generalized",
                "reasoning_trace": "merged",
                "generated_code": "def solve(): return 42",
            })

        return "OK"

    return responder


def _build_agent(
    task_desc: str = "Write a function that returns 42",
    delegate: bool = False,
    max_depth: int = 3,
    sandbox_pass: bool = True,
) -> tuple[CodeAgent, FakeVectorDB, FakeLLM]:
    config = TALMConfig(
        max_depth=max_depth,
        initial_branching=3,
        decay_rate=1,
        max_retries=2,
        sandbox_enabled=True,
    )
    llm = FakeLLM(responder=_make_llm_responder(delegate=delegate))
    embedder = FakeEmbedder(dimension=8)
    vector_db = FakeVectorDB()
    sandbox = FakeSandbox(always_pass=sandbox_pass)
    memory = MemoryManager(vector_db=vector_db, embedder=embedder, llm=llm, config=config)
    validator = ValidationAgent(llm=llm, sandbox=sandbox, config=config)
    task = Task(description=task_desc, tree_depth=0)
    agent = CodeAgent(task=task, llm=llm, memory=memory, validator=validator, config=config)
    return agent, vector_db, llm


@pytest.mark.asyncio
async def test_leaf_agent_full_lifecycle():
    """A leaf agent (no delegation) should: plan → implement → validate → store memory."""
    agent, vector_db, llm = _build_agent(delegate=False)

    result = await agent.execute()

    assert result.status == AgentResponseStatus.SUCCESS
    assert "42" in result.code
    assert result.tree_depth == 0
    # Memory should have 1 record stored
    assert await vector_db.count() == 1


@pytest.mark.asyncio
async def test_agent_stores_memory_on_success():
    """After successful execution, the agent's experience should be in memory."""
    agent, vector_db, _ = _build_agent()

    await agent.execute()

    records = await vector_db.list_all()
    assert len(records) == 1
    _, record = records[0]
    assert "42" in record.generated_code
    assert record.tree_depth == 0


@pytest.mark.asyncio
async def test_agent_does_not_store_memory_on_failure():
    """Failed execution should NOT store anything in memory."""
    agent, vector_db, _ = _build_agent(sandbox_pass=False)

    result = await agent.execute()

    assert result.status == AgentResponseStatus.FAILURE
    assert await vector_db.count() == 0


@pytest.mark.asyncio
async def test_delegation_spawns_children():
    """When delegation is needed, agent should spawn child agents."""
    agent, vector_db, llm = _build_agent(delegate=True, max_depth=3)

    result = await agent.execute()

    assert result.status == AgentResponseStatus.SUCCESS
    # Root + children should all store memory
    count = await vector_db.count()
    assert count >= 1  # at least root stored


@pytest.mark.asyncio
async def test_max_depth_prevents_delegation():
    """At max_depth, agent should NOT delegate regardless of plan complexity."""
    config = TALMConfig(max_depth=0, initial_branching=3, decay_rate=1, max_retries=2, sandbox_enabled=True)
    llm = FakeLLM(responder=_make_llm_responder(delegate=True))
    embedder = FakeEmbedder(dimension=8)
    vector_db = FakeVectorDB()
    sandbox = FakeSandbox(always_pass=True)
    memory = MemoryManager(vector_db=vector_db, embedder=embedder, llm=llm, config=config)
    validator = ValidationAgent(llm=llm, sandbox=sandbox, config=config)
    task = Task(description="complex task", tree_depth=0)
    agent = CodeAgent(task=task, llm=llm, memory=memory, validator=validator, config=config)

    result = await agent.execute()

    # Should succeed as a leaf even though delegate=True
    assert result.status == AgentResponseStatus.SUCCESS


@pytest.mark.asyncio
async def test_planning_queries_memory():
    """Planning phase should query long-term memory for past experiences."""
    agent, vector_db, llm = _build_agent()
    embedder = FakeEmbedder(dimension=8)

    # Pre-populate memory with a relevant record
    from talm.core.entities import MemoryRecord
    rec = MemoryRecord("similar task", "similar plan", "def similar(): pass", tree_depth=0)
    emb = await embedder.embed("similar task\nsimilar plan")
    await vector_db.insert(rec, emb)

    result = await agent.execute()

    assert result.status == AgentResponseStatus.SUCCESS
    # LLM should have been called with planning prompt (first call)
    assert len(llm.calls) >= 1
