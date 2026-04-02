"""Core domain entities for the TALM framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentPhase(str, Enum):
    """The 5-phase lifecycle of a Code Agent."""
    PLANNING = "planning"
    DELEGATION = "delegation"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    RETURN = "return"


class AgentResponseStatus(str, Enum):
    """Possible response statuses from an agent."""
    SUCCESS = "success"
    CLARIFY = "clarify"           # child requests clarification from parent
    RESTRUCTURE = "restructure"   # parent decides to restructure subtree
    FAILURE = "failure"


@dataclass
class Task:
    """A task to be solved by an agent node in the tree."""
    description: str
    parent_task: str | None = None
    tree_depth: int = 0
    context: str = ""             # additional context from parent or memory


@dataclass
class AgentResult:
    """Result returned by a Code Agent after execution."""
    status: AgentResponseStatus
    code: str = ""
    reasoning: str = ""
    clarification_request: str = ""
    error_log: str = ""
    task_description: str = ""
    tree_depth: int = 0


@dataclass
class MemoryRecord:
    """A single record stored in the long-term vector memory."""
    task_description: str
    reasoning_trace: str
    generated_code: str
    tree_depth: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of running code in the sandbox."""
    passed: bool
    stdout: str = ""
    stderr: str = ""
    test_code: str = ""


@dataclass
class TALMConfig:
    """System hyperparameters loaded from config.yaml."""
    # Tree structure
    max_depth: int = 3
    initial_branching: int = 3
    decay_rate: int = 1

    # Validation
    max_retries: int = 3

    # Memory
    similarity_threshold: float = 0.75
    top_k: int = 3
    merge_threshold: float = 0.95

    # LLM
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.0-flash"
    temperature: float = 0.0
    max_output_tokens: int = 8192

    # Embedding (decoupled from LLM)
    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector database
    vector_db_provider: str = "zvec"
    vector_db_persist_dir: str = "./talm_vector_db"

    # Sandbox
    sandbox_timeout: int = 30
    sandbox_enabled: bool = True
