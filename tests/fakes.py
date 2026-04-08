"""Fake implementations of all interfaces for testing.

These replace real Gemini, Zvec, and subprocess with deterministic in-memory
implementations, following the Dependency Inversion Principle.
No network calls, no disk I/O, no external dependencies.
"""

from __future__ import annotations

import json
import math
import uuid
from typing import Callable

from talm.core.entities import MemoryRecord, TestResult
from talm.core.interfaces import IEmbedder, ILLMClient, ISandbox, IVectorDatabase


class FakeLLM(ILLMClient):
    """Deterministic LLM that returns scripted responses.

    Pass a responder function to control output per prompt,
    or use the default which echoes back the user prompt.
    """

    def __init__(self, responder: Callable[[str, str], str] | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self._responder = responder

    async def generate(
        self, system_prompt: str, user_prompt: str,
        temperature: float = 0.0, max_tokens: int = 8192,
    ) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self._responder:
            return self._responder(system_prompt, user_prompt)
        return f"response to: {user_prompt[:80]}"


class FakeEmbedder(IEmbedder):
    """Deterministic embedder that produces consistent vectors from text hash."""

    def __init__(self, dimension: int = 8) -> None:
        self._dim = dimension
        self.call_count = 0

    def get_dimension(self) -> int:
        return self._dim

    async def embed(self, text: str) -> list[float]:
        self.call_count += 1
        return self._hash_embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.call_count += len(texts)
        return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text: str) -> list[float]:
        """Produce a deterministic unit vector from text."""
        h = hash(text)
        raw = [(h >> (i * 8) & 0xFF) / 255.0 for i in range(self._dim)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class FakeVectorDB(IVectorDatabase):
    """In-memory vector database for testing. Pure Python, no dependencies."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[MemoryRecord, list[float]]] = {}

    async def retrieve(
        self, query_embedding: list[float], tree_depth: int,
        top_k: int = 3, threshold: float = 0.75,
    ) -> list[tuple[MemoryRecord, float]]:
        results: list[tuple[MemoryRecord, float]] = []
        for _id, (record, emb) in self._store.items():
            if record.tree_depth != tree_depth:
                continue
            sim = _cosine_similarity(query_embedding, emb)
            if sim >= threshold:
                results.append((record, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def insert(self, record: MemoryRecord, embedding: list[float]) -> str:
        record_id = uuid.uuid4().hex
        self._store[record_id] = (record, embedding)
        return record_id

    async def delete(self, record_id: str) -> None:
        self._store.pop(record_id, None)

    async def find_similar(
        self, embedding: list[float], tree_depth: int, threshold: float = 0.95,
    ) -> list[tuple[str, MemoryRecord, float]]:
        matches: list[tuple[str, MemoryRecord, float]] = []
        for rid, (record, emb) in self._store.items():
            if record.tree_depth != tree_depth:
                continue
            sim = _cosine_similarity(embedding, emb)
            if sim >= threshold:
                matches.append((rid, record, sim))
        return matches

    async def clear(self) -> None:
        self._store.clear()

    async def count(self) -> int:
        return len(self._store)

    async def list_all(self) -> list[tuple[str, MemoryRecord]]:
        return [(rid, rec) for rid, (rec, _) in self._store.items()]


class FakeSandbox(ISandbox):
    """Sandbox that executes code for real (subprocess) or returns scripted results."""

    def __init__(self, always_pass: bool = True) -> None:
        self._always_pass = always_pass
        self.executions: list[str] = []

    async def execute(self, code: str, timeout: int = 30) -> TestResult:
        self.executions.append(code)
        if self._always_pass:
            return TestResult(passed=True, stdout="ALL TESTS PASSED", stderr="")
        return TestResult(passed=False, stdout="", stderr="AssertionError: test failed")
