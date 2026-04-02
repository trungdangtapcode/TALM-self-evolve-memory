"""Abstract interfaces (ports) for external dependencies.

Following the Dependency Inversion Principle, the core agents depend on
these interfaces rather than concrete implementations.  Swap adapters
(Gemini -> OpenAI, sentence-transformers -> Gemini embeddings,
Zvec -> ChromaDB -> FAISS, etc.) without touching agent logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from talm.core.entities import MemoryRecord, TestResult


class ILLMClient(ABC):
    """Interface for any LLM provider (Gemini, OpenAI, local models)."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> str:
        """Generate a text completion given system + user prompts."""


class IEmbedder(ABC):
    """Interface for text embedding models (decoupled from LLM).

    Default implementation: sentence-transformers (all-MiniLM-L6-v2).
    Can be swapped to Gemini embeddings, OpenAI embeddings, etc.
    """

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding vector dimension."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return an embedding vector for the given text."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single call for efficiency."""


class IVectorDatabase(ABC):
    """Interface for vector-based long-term memory storage."""

    @abstractmethod
    async def retrieve(
        self,
        query_embedding: list[float],
        tree_depth: int,
        top_k: int = 3,
        threshold: float = 0.75,
    ) -> list[tuple[MemoryRecord, float]]:
        """Retrieve top-k similar records filtered by tree_depth."""

    @abstractmethod
    async def insert(self, record: MemoryRecord, embedding: list[float]) -> str:
        """Insert a new memory record. Returns the record ID."""

    @abstractmethod
    async def delete(self, record_id: str) -> None:
        """Delete a record by its ID."""

    @abstractmethod
    async def find_similar(
        self,
        embedding: list[float],
        tree_depth: int,
        threshold: float = 0.95,
    ) -> list[tuple[str, MemoryRecord, float]]:
        """Find records above the similarity threshold (for consolidation)."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear all records (used before evaluation runs)."""

    @abstractmethod
    async def count(self) -> int:
        """Return the number of records stored."""

    @abstractmethod
    async def list_all(self) -> list[tuple[str, MemoryRecord]]:
        """List all records with their IDs. For debugging/UI inspection."""


class ISandbox(ABC):
    """Interface for isolated code execution environments."""

    @abstractmethod
    async def execute(self, code: str, timeout: int = 30) -> TestResult:
        """Run code in a sandboxed environment and return the result."""
