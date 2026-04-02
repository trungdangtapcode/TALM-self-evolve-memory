"""FastMCP server exposing TALM as MCP tools.

Run with:
    source .venv/bin/activate
    python -m talm.server

Test with MCP Inspector:
    npx @modelcontextprotocol/inspector python -m talm.server
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from talm.config import load_config
from talm.core.entities import AgentResponseStatus
from talm.workflow import TALMWorkflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP(
    "TALM",
    instructions=(
        "Tree-Structured Multi-Agent Framework with Long-Term Memory "
        "for Scalable Code Generation (TALM paper reproduction)"
    ),
)

# Lazy-initialized workflow instance (created on first tool call)
_workflow: TALMWorkflow | None = None


def _get_workflow() -> TALMWorkflow:
    """Get or create the singleton TALMWorkflow."""
    global _workflow
    if _workflow is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
        config = load_config(config_path)
        _workflow = TALMWorkflow(config=config)
    return _workflow


# -----------------------------------------------------------------------
# MCP Tools
# -----------------------------------------------------------------------


@mcp.tool()
async def generate_code(task_description: str) -> str:
    """Generate Python code for a given task using the TALM multi-agent tree framework.

    The framework will:
    1. Plan a solution strategy (consulting long-term memory)
    2. Optionally decompose into subtasks handled by child agents
    3. Generate and validate code through an iterative debug loop
    4. Store successful experiences in long-term memory

    Args:
        task_description: A clear description of the coding task to solve.

    Returns:
        JSON string with status, generated code, reasoning, and any errors.
    """
    wf = _get_workflow()
    result = await wf.run(task_description)

    return json.dumps(
        {
            "status": result.status.value,
            "code": result.code,
            "reasoning": result.reasoning,
            "error_log": result.error_log,
            "tree_depth": result.tree_depth,
        },
        indent=2,
    )


@mcp.tool()
async def generate_code_simple(task_description: str) -> str:
    """Generate code with a simplified single-agent approach (no tree decomposition).

    Useful for simple tasks where tree decomposition is overkill.
    Temporarily sets max_depth=1 to force leaf-node behavior.

    Args:
        task_description: A clear description of the coding task to solve.

    Returns:
        JSON string with status and generated code.
    """
    wf = _get_workflow()
    # Override depth to prevent delegation
    original_depth = wf.config.max_depth
    wf.config.max_depth = 0
    try:
        result = await wf.run(task_description)
    finally:
        wf.config.max_depth = original_depth

    return json.dumps(
        {
            "status": result.status.value,
            "code": result.code,
            "reasoning": result.reasoning,
            "error_log": result.error_log,
        },
        indent=2,
    )


@mcp.tool()
async def get_memory_stats() -> str:
    """Get statistics about the TALM long-term memory.

    Returns:
        JSON string with memory record count and configuration.
    """
    wf = _get_workflow()
    count = await wf.get_memory_count()

    return json.dumps(
        {
            "record_count": count,
            "config": {
                "embedding_provider": wf.config.embedding_provider,
                "embedding_model": wf.config.embedding_model,
                "vector_db_provider": wf.config.vector_db_provider,
                "similarity_threshold": wf.config.similarity_threshold,
                "merge_threshold": wf.config.merge_threshold,
                "top_k": wf.config.top_k,
            },
        },
        indent=2,
    )


@mcp.tool()
async def clear_memory() -> str:
    """Clear all long-term memory records.

    Use this before evaluation runs to ensure a clean state.

    Returns:
        Confirmation message.
    """
    wf = _get_workflow()
    await wf.clear_memory()
    return json.dumps({"status": "success", "message": "Long-term memory cleared"})


@mcp.tool()
async def get_config() -> str:
    """Get the current TALM system configuration / hyperparameters.

    Returns:
        JSON string with all configuration values.
    """
    wf = _get_workflow()
    c = wf.config
    return json.dumps(
        {
            "tree": {
                "max_depth": c.max_depth,
                "initial_branching": c.initial_branching,
                "decay_rate": c.decay_rate,
            },
            "validation": {"max_retries": c.max_retries},
            "memory": {
                "similarity_threshold": c.similarity_threshold,
                "top_k": c.top_k,
                "merge_threshold": c.merge_threshold,
            },
            "llm": {
                "provider": c.llm_provider,
                "model": c.llm_model,
                "temperature": c.temperature,
            },
            "embedding": {
                "provider": c.embedding_provider,
                "model": c.embedding_model,
            },
            "vector_db": {
                "provider": c.vector_db_provider,
                "persist_dir": c.vector_db_persist_dir,
            },
            "sandbox": {
                "timeout": c.sandbox_timeout,
                "enabled": c.sandbox_enabled,
            },
        },
        indent=2,
    )


@mcp.tool()
async def update_config(
    max_depth: int | None = None,
    initial_branching: int | None = None,
    decay_rate: int | None = None,
    max_retries: int | None = None,
    sandbox_enabled: bool | None = None,
    temperature: float | None = None,
) -> str:
    """Update TALM hyperparameters at runtime (does not persist to config.yaml).

    Args:
        max_depth: Maximum tree depth (m). Default: 3.
        initial_branching: Initial branching factor (n). Default: 3.
        decay_rate: Branching decay per level (k). Default: 1.
        max_retries: Max validation retries (r). Default: 3.
        sandbox_enabled: Enable/disable code sandbox validation.
        temperature: LLM temperature (0 = deterministic).

    Returns:
        JSON string with the updated configuration.
    """
    wf = _get_workflow()
    c = wf.config
    changes: dict = {}

    if max_depth is not None:
        c.max_depth = max_depth
        changes["max_depth"] = max_depth
    if initial_branching is not None:
        c.initial_branching = initial_branching
        changes["initial_branching"] = initial_branching
    if decay_rate is not None:
        c.decay_rate = decay_rate
        changes["decay_rate"] = decay_rate
    if max_retries is not None:
        c.max_retries = max_retries
        changes["max_retries"] = max_retries
    if sandbox_enabled is not None:
        c.sandbox_enabled = sandbox_enabled
        changes["sandbox_enabled"] = sandbox_enabled
    if temperature is not None:
        c.temperature = temperature
        changes["temperature"] = temperature

    return json.dumps({"status": "updated", "changes": changes}, indent=2)


# -----------------------------------------------------------------------
# MCP Resources
# -----------------------------------------------------------------------


@mcp.resource("talm://config")
async def resource_config() -> str:
    """Expose current TALM config as an MCP resource."""
    return await get_config()


@mcp.resource("talm://memory/stats")
async def resource_memory_stats() -> str:
    """Expose memory stats as an MCP resource."""
    return await get_memory_stats()


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------


def main() -> None:
    """Run the TALM MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
