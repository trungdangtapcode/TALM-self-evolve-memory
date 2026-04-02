"""Workflow Orchestrator: top-level entry point for running TALM.

Creates the root Code Agent and manages the full tree execution.
"""

from __future__ import annotations

import logging

from talm.agents.code_agent import CodeAgent
from talm.agents.validation_agent import ValidationAgent
from talm.config import (
    create_embedder,
    create_llm,
    create_sandbox,
    create_vector_db,
    load_config,
)
from talm.core.entities import AgentResult, TALMConfig, Task
from talm.core.interfaces import IEmbedder, ILLMClient, ISandbox, IVectorDatabase
from talm.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


class TALMWorkflow:
    """Orchestrates a full TALM code-generation run."""

    def __init__(
        self,
        config: TALMConfig | None = None,
        llm: ILLMClient | None = None,
        embedder: IEmbedder | None = None,
        vector_db: IVectorDatabase | None = None,
        sandbox: ISandbox | None = None,
    ) -> None:
        self.config = config or load_config()
        self.llm = llm or create_llm(self.config)
        self.embedder = embedder or create_embedder(self.config)
        self.vector_db = vector_db or create_vector_db(self.config, self.embedder)
        self.sandbox = sandbox or create_sandbox(self.config)
        self.memory = MemoryManager(
            vector_db=self.vector_db,
            embedder=self.embedder,
            llm=self.llm,
            config=self.config,
        )
        self.validator = ValidationAgent(
            llm=self.llm,
            sandbox=self.sandbox,
            config=self.config,
        )

    async def run(self, task_description: str) -> AgentResult:
        """Execute the TALM pipeline for a given task.

        Creates the root Code Agent at depth 0 and starts the recursive
        tree-structured code generation process.
        """
        logger.info("=== TALM Workflow Started ===")
        logger.info("Task: %.100s...", task_description)
        logger.info(
            "Config: max_depth=%d, branching=%d, decay=%d, retries=%d",
            self.config.max_depth,
            self.config.initial_branching,
            self.config.decay_rate,
            self.config.max_retries,
        )

        root_task = Task(
            description=task_description,
            tree_depth=0,
        )
        root_agent = CodeAgent(
            task=root_task,
            llm=self.llm,
            memory=self.memory,
            validator=self.validator,
            config=self.config,
        )

        result = await root_agent.execute()
        logger.info("=== TALM Workflow Completed: %s ===", result.status.value)
        return result

    async def clear_memory(self) -> None:
        """Clear long-term memory (e.g., before evaluation runs)."""
        await self.memory.clear()

    async def get_memory_count(self) -> int:
        """Return the number of records in long-term memory."""
        return await self.memory.count()

    async def list_memory(self) -> list[dict]:
        """List all memory records for UI inspection."""
        records = await self.memory.list_all()
        return [
            {
                "id": rid,
                "task_description": rec.task_description,
                "reasoning_trace": rec.reasoning_trace,
                "generated_code": rec.generated_code,
                "tree_depth": rec.tree_depth,
            }
            for rid, rec in records
        ]
