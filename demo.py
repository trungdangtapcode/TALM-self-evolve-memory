"""MVP Demo: Run TALM locally to generate code for a simple task.

Usage:
    source .venv/bin/activate
    export GOOGLE_API_KEY="your-key"
    python demo.py

This demonstrates the full TALM pipeline:
1. Tree-structured multi-agent code generation
2. Long-term memory retrieval and storage
3. Validation via sandbox execution
"""

import asyncio
import json
import logging
import sys

from talm.config import load_config
from talm.workflow import TALMWorkflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# Example tasks of increasing complexity
DEMO_TASKS = [
    # Simple (leaf-node, no delegation expected)
    (
        "Write a Python function `is_palindrome(s: str) -> bool` that checks "
        "if a string is a palindrome, ignoring case and non-alphanumeric characters."
    ),
    # Medium (may delegate to 1-2 children)
    (
        "Write a Python module with two functions: "
        "`fibonacci(n: int) -> list[int]` that returns the first n Fibonacci numbers, "
        "and `is_prime(n: int) -> bool` that checks if a number is prime. "
        "Include a main block that prints the first 20 Fibonacci numbers "
        "and checks which of them are prime."
    ),
]


async def main() -> None:
    config = load_config()

    # For demo, use smaller depth to save API calls
    config.max_depth = 1
    config.max_retries = 2

    print("=" * 60)
    print("TALM Framework - MVP Demo")
    print("=" * 60)
    print(f"LLM:       {config.llm_provider}/{config.llm_model}")
    print(f"Embedder:  {config.embedding_provider}/{config.embedding_model}")
    print(f"VectorDB:  {config.vector_db_provider}")
    print(f"Tree:      depth={config.max_depth}, branching={config.initial_branching}")
    print(f"Sandbox:   {'enabled' if config.sandbox_enabled else 'disabled'}")
    print("=" * 60)

    wf = TALMWorkflow(config=config)

    # Pick a task (default: simple, or pass index as CLI arg)
    task_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    task_idx = min(task_idx, len(DEMO_TASKS) - 1)
    task = DEMO_TASKS[task_idx]

    print(f"\nTask ({task_idx}): {task}\n")
    print("-" * 60)

    result = await wf.run(task)

    print("\n" + "=" * 60)
    print(f"Status: {result.status.value}")
    print("=" * 60)

    if result.reasoning:
        print("\n--- Reasoning / Plan ---")
        print(result.reasoning[:500])

    if result.code:
        print("\n--- Generated Code ---")
        print(result.code)

    if result.error_log:
        print("\n--- Errors ---")
        print(result.error_log[:300])

    # Show memory stats
    mem_count = await wf.get_memory_count()
    print(f"\nMemory records stored: {mem_count}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
