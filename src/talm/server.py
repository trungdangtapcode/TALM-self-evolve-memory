"""TALM server: REST API for web frontend + MCP for Claude/Inspector.

Run modes:
    # HTTP REST API (for web frontend, default)
    python -m talm.server

    # MCP stdio (for Claude Desktop / MCP Inspector)
    python -m talm.server --mcp

    # MCP Inspector
    npx @modelcontextprotocol/inspector python -m talm.server --mcp
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from talm.config import load_config
from talm.workflow import TALMWorkflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Shared workflow (singleton)
# -----------------------------------------------------------------------

_workflow: TALMWorkflow | None = None


def _get_workflow() -> TALMWorkflow:
    global _workflow
    if _workflow is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
        config = load_config(config_path)
        _workflow = TALMWorkflow(config=config)
    return _workflow


# -----------------------------------------------------------------------
# MCP Server (for Claude Desktop / MCP Inspector)
# -----------------------------------------------------------------------

mcp = FastMCP(
    "TALM",
    instructions=(
        "Tree-Structured Multi-Agent Framework with Long-Term Memory "
        "for Scalable Code Generation"
    ),
)


@mcp.tool()
async def generate_code(task_description: str) -> str:
    """Generate Python code using the TALM multi-agent tree framework."""
    wf = _get_workflow()
    result = await wf.run(task_description)
    return json.dumps({
        "status": result.status.value, "code": result.code,
        "reasoning": result.reasoning, "error_log": result.error_log,
        "tree_depth": result.tree_depth,
    }, indent=2)


@mcp.tool()
async def generate_code_simple(task_description: str) -> str:
    """Generate code with single-agent mode (no tree decomposition)."""
    wf = _get_workflow()
    original_depth = wf.config.max_depth
    wf.config.max_depth = 0
    try:
        result = await wf.run(task_description)
    finally:
        wf.config.max_depth = original_depth
    return json.dumps({
        "status": result.status.value, "code": result.code,
        "reasoning": result.reasoning, "error_log": result.error_log,
    }, indent=2)


@mcp.tool()
async def get_memory_stats() -> str:
    """Get long-term memory statistics."""
    wf = _get_workflow()
    count = await wf.get_memory_count()
    return json.dumps({
        "record_count": count,
        "config": {
            "embedding_provider": wf.config.embedding_provider,
            "embedding_model": wf.config.embedding_model,
            "vector_db_provider": wf.config.vector_db_provider,
            "similarity_threshold": wf.config.similarity_threshold,
            "merge_threshold": wf.config.merge_threshold,
            "top_k": wf.config.top_k,
        },
    }, indent=2)


@mcp.tool()
async def clear_memory() -> str:
    """Clear all long-term memory records."""
    wf = _get_workflow()
    await wf.clear_memory()
    return json.dumps({"status": "success", "message": "Long-term memory cleared"})


@mcp.tool()
async def get_config() -> str:
    """Get current TALM system configuration."""
    wf = _get_workflow()
    c = wf.config
    return json.dumps({
        "tree": {"max_depth": c.max_depth, "initial_branching": c.initial_branching, "decay_rate": c.decay_rate},
        "validation": {"max_retries": c.max_retries},
        "memory": {"similarity_threshold": c.similarity_threshold, "top_k": c.top_k, "merge_threshold": c.merge_threshold},
        "llm": {"provider": c.llm_provider, "model": c.llm_model, "temperature": c.temperature},
        "embedding": {"provider": c.embedding_provider, "model": c.embedding_model},
        "vector_db": {"provider": c.vector_db_provider, "persist_dir": c.vector_db_persist_dir},
        "sandbox": {"timeout": c.sandbox_timeout, "enabled": c.sandbox_enabled},
    }, indent=2)


@mcp.tool()
async def update_config(
    max_depth: int | None = None, initial_branching: int | None = None,
    decay_rate: int | None = None, max_retries: int | None = None,
    sandbox_enabled: bool | None = None, temperature: float | None = None,
) -> str:
    """Update TALM hyperparameters at runtime."""
    wf = _get_workflow()
    c = wf.config
    changes: dict = {}
    if max_depth is not None: c.max_depth = max_depth; changes["max_depth"] = max_depth
    if initial_branching is not None: c.initial_branching = initial_branching; changes["initial_branching"] = initial_branching
    if decay_rate is not None: c.decay_rate = decay_rate; changes["decay_rate"] = decay_rate
    if max_retries is not None: c.max_retries = max_retries; changes["max_retries"] = max_retries
    if sandbox_enabled is not None: c.sandbox_enabled = sandbox_enabled; changes["sandbox_enabled"] = sandbox_enabled
    if temperature is not None: c.temperature = temperature; changes["temperature"] = temperature
    return json.dumps({"status": "updated", "changes": changes}, indent=2)


@mcp.resource("talm://config")
async def resource_config() -> str:
    return await get_config()


@mcp.resource("talm://memory/stats")
async def resource_memory_stats() -> str:
    return await get_memory_stats()


# -----------------------------------------------------------------------
# REST API (for web frontend — simple HTTP JSON, no MCP protocol overhead)
# -----------------------------------------------------------------------

def _build_rest_app():
    """Build a Starlette app with REST endpoints that call the same workflow."""
    from starlette.applications import Starlette
    from starlette.middleware.cors import CORSMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def api_generate(request: Request) -> JSONResponse:
        body = await request.json()
        task = body.get("task_description", "")
        mode = body.get("mode", "full")
        if not task:
            return JSONResponse({"error": "task_description required"}, status_code=400)

        wf = _get_workflow()
        if mode == "simple":
            original = wf.config.max_depth
            wf.config.max_depth = 0
            try:
                result = await wf.run(task)
            finally:
                wf.config.max_depth = original
        else:
            result = await wf.run(task)

        return JSONResponse({
            "status": result.status.value,
            "code": result.code,
            "reasoning": result.reasoning,
            "error_log": result.error_log,
            "tree_depth": result.tree_depth,
        })

    async def api_config(request: Request) -> JSONResponse:
        wf = _get_workflow()
        c = wf.config
        return JSONResponse({
            "tree": {"max_depth": c.max_depth, "initial_branching": c.initial_branching, "decay_rate": c.decay_rate},
            "validation": {"max_retries": c.max_retries},
            "memory": {"similarity_threshold": c.similarity_threshold, "top_k": c.top_k, "merge_threshold": c.merge_threshold},
            "llm": {"provider": c.llm_provider, "model": c.llm_model, "temperature": c.temperature},
            "embedding": {"provider": c.embedding_provider, "model": c.embedding_model},
            "vector_db": {"provider": c.vector_db_provider, "persist_dir": c.vector_db_persist_dir},
            "sandbox": {"timeout": c.sandbox_timeout, "enabled": c.sandbox_enabled},
        })

    async def api_config_update(request: Request) -> JSONResponse:
        body = await request.json()
        wf = _get_workflow()
        c = wf.config
        changes = {}
        for key in ["max_depth", "initial_branching", "decay_rate", "max_retries", "sandbox_timeout"]:
            if key in body:
                setattr(c, key, body[key])
                changes[key] = body[key]
        if "sandbox_enabled" in body:
            c.sandbox_enabled = body["sandbox_enabled"]
            changes["sandbox_enabled"] = body["sandbox_enabled"]
        if "temperature" in body:
            c.temperature = body["temperature"]
            changes["temperature"] = body["temperature"]
        return JSONResponse({"status": "updated", "changes": changes})

    async def api_memory_stats(request: Request) -> JSONResponse:
        wf = _get_workflow()
        count = await wf.get_memory_count()
        return JSONResponse({
            "record_count": count,
            "config": {
                "embedding_provider": wf.config.embedding_provider,
                "embedding_model": wf.config.embedding_model,
                "vector_db_provider": wf.config.vector_db_provider,
                "similarity_threshold": wf.config.similarity_threshold,
                "merge_threshold": wf.config.merge_threshold,
                "top_k": wf.config.top_k,
            },
        })

    async def api_memory_clear(request: Request) -> JSONResponse:
        wf = _get_workflow()
        await wf.clear_memory()
        return JSONResponse({"status": "success", "message": "Memory cleared"})

    async def api_memory_records(request: Request) -> JSONResponse:
        """List all memory records with full content."""
        wf = _get_workflow()
        records = await wf.list_memory()
        return JSONResponse({"records": records})

    async def api_memory_insert(request: Request) -> JSONResponse:
        """Manually insert a memory record.

        This goes through the same update() pipeline as auto-stored records,
        meaning consolidation will trigger if a near-duplicate exists (sim >= 0.95).
        """
        body = await request.json()
        for field in ("task_description", "reasoning_trace", "generated_code", "tree_depth"):
            if field not in body:
                return JSONResponse({"error": f"Missing field: {field}"}, status_code=400)

        wf = _get_workflow()
        from talm.core.entities import MemoryRecord
        record = MemoryRecord(
            task_description=body["task_description"],
            reasoning_trace=body["reasoning_trace"],
            generated_code=body["generated_code"],
            tree_depth=body["tree_depth"],
        )
        # Uses the same update() path — consolidation applies automatically
        await wf.memory.update(record)
        return JSONResponse({"status": "inserted", "message": "Record stored (consolidation rules applied)"})

    async def api_health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "service": "talm"})

    app = Starlette(
        routes=[
            Route("/api/health", api_health, methods=["GET"]),
            Route("/api/generate", api_generate, methods=["POST"]),
            Route("/api/config", api_config, methods=["GET"]),
            Route("/api/config", api_config_update, methods=["PUT"]),
            Route("/api/memory/stats", api_memory_stats, methods=["GET"]),
            Route("/api/memory/records", api_memory_records, methods=["GET"]),
            Route("/api/memory/records", api_memory_insert, methods=["POST"]),
            Route("/api/memory/clear", api_memory_clear, methods=["POST"]),
        ],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main() -> None:
    """Run the TALM server.

    Default: REST API on port 8000 (for web frontend).
    --mcp: MCP stdio transport (for Claude Desktop / MCP Inspector).
    """
    if "--mcp" in sys.argv:
        logger.info("Starting TALM MCP server (stdio transport)")
        mcp.run(transport="stdio")
    else:
        import uvicorn
        app = _build_rest_app()
        logger.info("Starting TALM REST API on http://localhost:8000")
        logger.info("Endpoints: /api/health, /api/generate, /api/config, /api/memory/stats, /api/memory/clear")
        uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
