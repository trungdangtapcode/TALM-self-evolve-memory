"""Tests for the REST API server endpoints using Starlette test client."""

import json
import pytest
from starlette.testclient import TestClient

from talm.core.entities import TALMConfig
from talm.workflow import TALMWorkflow
from tests.fakes import FakeEmbedder, FakeLLM, FakeSandbox, FakeVectorDB


def _make_responder():
    def responder(system: str, user: str) -> str:
        sys_lower = system.lower()
        if "architect" in sys_lower:
            return "DIRECT_IMPLEMENTATION\n1. Solve directly"
        if "decomposition" in sys_lower:
            return "NO_DELEGATION"
        if "programmer" in sys_lower:
            return '```python\ndef solve(): return 1\n```'
        if "qa" in sys_lower or "test" in sys_lower:
            return '```python\nassert solve() == 1\n```'
        if "knowledge" in sys_lower:
            return json.dumps({
                "task_description": "merged",
                "reasoning_trace": "merged",
                "generated_code": "def solve(): return 1",
            })
        return "OK"
    return responder


@pytest.fixture
def client() -> TestClient:
    """Build test client with all fakes injected."""
    # Inject fake workflow into the server module
    import talm.server as server_module
    config = TALMConfig(max_depth=0, max_retries=1, sandbox_enabled=True)
    wf = TALMWorkflow(
        config=config,
        llm=FakeLLM(responder=_make_responder()),
        embedder=FakeEmbedder(dimension=8),
        vector_db=FakeVectorDB(),
        sandbox=FakeSandbox(always_pass=True),
    )
    server_module._workflow = wf

    app = server_module._build_rest_app()
    return TestClient(app)


def test_health(client: TestClient):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_get_config(client: TestClient):
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "tree" in data
    assert "llm" in data
    assert "embedding" in data


def test_memory_stats(client: TestClient):
    resp = client.get("/api/memory/stats")
    assert resp.status_code == 200
    assert resp.json()["record_count"] == 0


def test_generate_code(client: TestClient):
    resp = client.post("/api/generate", json={
        "task_description": "Return 1",
        "mode": "simple",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "1" in data["code"]


def test_generate_stores_memory(client: TestClient):
    client.post("/api/generate", json={"task_description": "task", "mode": "simple"})
    resp = client.get("/api/memory/stats")
    assert resp.json()["record_count"] == 1


def test_memory_records_endpoint(client: TestClient):
    client.post("/api/generate", json={"task_description": "task", "mode": "simple"})
    resp = client.get("/api/memory/records")
    assert resp.status_code == 200
    records = resp.json()["records"]
    assert len(records) == 1
    assert "task_description" in records[0]
    assert "generated_code" in records[0]


def test_memory_insert(client: TestClient):
    resp = client.post("/api/memory/records", json={
        "task_description": "manual task",
        "reasoning_trace": "manual plan",
        "generated_code": "def manual(): pass",
        "tree_depth": 0,
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "inserted"

    resp = client.get("/api/memory/stats")
    assert resp.json()["record_count"] == 1


def test_memory_insert_validation(client: TestClient):
    resp = client.post("/api/memory/records", json={"task_description": "incomplete"})
    assert resp.status_code == 400


def test_memory_clear(client: TestClient):
    client.post("/api/generate", json={"task_description": "task", "mode": "simple"})
    client.post("/api/memory/clear", json={})
    resp = client.get("/api/memory/stats")
    assert resp.json()["record_count"] == 0


def test_generate_missing_task(client: TestClient):
    resp = client.post("/api/generate", json={"mode": "simple"})
    assert resp.status_code == 400
