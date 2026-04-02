"""Prompt templates for each phase of the Code Agent lifecycle.

Each template is a function returning (system_prompt, user_prompt) so that
the agent can pass them directly to the LLM client.
"""

from __future__ import annotations

from talm.core.entities import MemoryRecord


# ---------------------------------------------------------------------------
# Planning Phase
# ---------------------------------------------------------------------------

PLANNING_SYSTEM = """\
You are an expert software architect. Given a programming task and optional
memory from previous similar tasks, produce a clear step-by-step plan.

Rules:
- Output ONLY the plan as a numbered list.
- Each step must be concrete and actionable.
- If the task is simple enough to solve directly, say "DIRECT_IMPLEMENTATION"
  and provide a 1-step plan.
- Do NOT write any code in this phase.
"""


def planning_prompt(
    task_description: str,
    memory_context: str,
    parent_context: str = "",
) -> tuple[str, str]:
    """Build the planning prompt pair."""
    user = f"## Task\n{task_description}\n"
    if parent_context:
        user += f"\n## Context from Parent Agent\n{parent_context}\n"
    if memory_context:
        user += f"\n## Relevant Past Experience\n{memory_context}\n"
    return PLANNING_SYSTEM, user


# ---------------------------------------------------------------------------
# Delegation Phase
# ---------------------------------------------------------------------------

DELEGATION_SYSTEM = """\
You are a task decomposition expert. Given a plan for a complex programming
task, decide whether to split it into subtasks for child agents.

Rules:
- If the plan has only 1 step or is marked DIRECT_IMPLEMENTATION, respond
  with exactly: NO_DELEGATION
- Otherwise, decompose into subtasks. For each subtask output a JSON array:
  [
    {"subtask_id": 1, "description": "detailed description of subtask 1"},
    {"subtask_id": 2, "description": "detailed description of subtask 2"},
    ...
  ]
- Maximum number of subtasks: {max_children}
- Each subtask must be self-contained with clear inputs/outputs.
- Include integration notes so the parent can combine results.
"""


def delegation_prompt(
    plan: str,
    task_description: str,
    max_children: int,
) -> tuple[str, str]:
    """Build the delegation prompt pair."""
    system = DELEGATION_SYSTEM.format(max_children=max_children)
    user = f"## Original Task\n{task_description}\n\n## Plan\n{plan}\n"
    return system, user


# ---------------------------------------------------------------------------
# Implementation Phase
# ---------------------------------------------------------------------------

IMPLEMENTATION_SYSTEM = """\
You are an expert Python programmer. Write clean, correct, well-documented
Python code to solve the given task.

Rules:
- Output ONLY valid Python code enclosed in a single ```python ... ``` block.
- Include necessary imports at the top.
- If child agent results are provided, integrate them into a cohesive solution.
- The code must be self-contained and runnable.
- Do NOT include test code; the validation agent handles testing.
"""


def implementation_prompt(
    task_description: str,
    plan: str,
    child_results: list[str] | None = None,
) -> tuple[str, str]:
    """Build the implementation prompt pair."""
    user = f"## Task\n{task_description}\n\n## Plan\n{plan}\n"
    if child_results:
        user += "\n## Code from Child Agents\n"
        for i, code in enumerate(child_results, 1):
            user += f"\n### Child Agent {i}\n```python\n{code}\n```\n"
    return IMPLEMENTATION_SYSTEM, user


# ---------------------------------------------------------------------------
# Validation Phase (test generation)
# ---------------------------------------------------------------------------

VALIDATION_SYSTEM = """\
You are a QA engineer. Given Python code and its task description, write
comprehensive test cases.

Rules:
- Output ONLY valid Python code enclosed in a single ```python ... ``` block.
- Import the necessary modules and paste/import the function(s) being tested.
- Include at least 3 test cases covering: normal input, edge cases, and error cases.
- Use assert statements. Print "ALL TESTS PASSED" at the end if all pass.
- The test code must be fully self-contained and runnable by itself.
"""


def validation_prompt(
    task_description: str,
    code: str,
) -> tuple[str, str]:
    """Build the validation/test-generation prompt pair."""
    user = (
        f"## Task Description\n{task_description}\n\n"
        f"## Code to Test\n```python\n{code}\n```\n"
    )
    return VALIDATION_SYSTEM, user


# ---------------------------------------------------------------------------
# Debug / Fix Phase
# ---------------------------------------------------------------------------

DEBUG_SYSTEM = """\
You are a debugging expert. The code below failed its tests. Fix the code
based on the error output.

Rules:
- Output ONLY the corrected Python code in a single ```python ... ``` block.
- Do NOT change the test cases; only fix the implementation.
- Preserve the original function signatures.
"""


def debug_prompt(
    task_description: str,
    code: str,
    test_code: str,
    error_output: str,
) -> tuple[str, str]:
    """Build the debug prompt pair."""
    user = (
        f"## Task\n{task_description}\n\n"
        f"## Current Code\n```python\n{code}\n```\n\n"
        f"## Test Code\n```python\n{test_code}\n```\n\n"
        f"## Error Output\n```\n{error_output}\n```\n"
    )
    return DEBUG_SYSTEM, user


# ---------------------------------------------------------------------------
# Clarification (Child -> Parent)
# ---------------------------------------------------------------------------

CLARIFICATION_SYSTEM = """\
You are reviewing a subtask description assigned to you. If the description
is ambiguous, underspecified, or missing critical details, respond with a
JSON object:
{"status": "clarify", "question": "your specific question here"}

If the description is clear and complete, respond with:
{"status": "clear"}
"""


def clarification_check_prompt(subtask_description: str) -> tuple[str, str]:
    """Build the clarification-check prompt pair."""
    return CLARIFICATION_SYSTEM, f"## Subtask\n{subtask_description}\n"


# ---------------------------------------------------------------------------
# Parent Reflection (responds to clarification request)
# ---------------------------------------------------------------------------

REFLECTION_SYSTEM = """\
A child agent has requested clarification on a subtask you delegated.
Reflect on your original reasoning and provide a more detailed, unambiguous
task description.

Output ONLY the improved task description as plain text.
"""


def reflection_prompt(
    original_task: str,
    subtask: str,
    question: str,
) -> tuple[str, str]:
    """Build the parent-reflection prompt pair."""
    user = (
        f"## Your Original Task\n{original_task}\n\n"
        f"## Subtask You Delegated\n{subtask}\n\n"
        f"## Child's Question\n{question}\n"
    )
    return REFLECTION_SYSTEM, user


# ---------------------------------------------------------------------------
# Memory Consolidation
# ---------------------------------------------------------------------------

CONSOLIDATION_SYSTEM = """\
You are a knowledge engineer. Two memory records describe solutions to very
similar problems. Merge them into a single, generalized record that captures
the common pattern.

Output a JSON object with these fields:
{
  "task_description": "generalized task description",
  "reasoning_trace": "merged reasoning steps",
  "generated_code": "best-practice code combining both solutions"
}
"""


def consolidation_prompt(
    record_a: MemoryRecord,
    record_b: MemoryRecord,
) -> tuple[str, str]:
    """Build the memory consolidation prompt pair."""
    user = (
        f"## Record A\n"
        f"Task: {record_a.task_description}\n"
        f"Reasoning: {record_a.reasoning_trace}\n"
        f"Code:\n```python\n{record_a.generated_code}\n```\n\n"
        f"## Record B\n"
        f"Task: {record_b.task_description}\n"
        f"Reasoning: {record_b.reasoning_trace}\n"
        f"Code:\n```python\n{record_b.generated_code}\n```\n"
    )
    return CONSOLIDATION_SYSTEM, user


# ---------------------------------------------------------------------------
# Helper: format memory records for inclusion in prompts
# ---------------------------------------------------------------------------

def format_memory_context(records: list[tuple[MemoryRecord, float]]) -> str:
    """Format retrieved memory records into a readable context string."""
    if not records:
        return ""
    parts: list[str] = []
    for i, (rec, score) in enumerate(records, 1):
        parts.append(
            f"### Memory {i} (similarity: {score:.2f})\n"
            f"**Task:** {rec.task_description}\n"
            f"**Reasoning:** {rec.reasoning_trace}\n"
            f"**Code:**\n```python\n{rec.generated_code}\n```\n"
        )
    return "\n".join(parts)
