"""Code Agent: the core recursive agent in the TALM tree.

Each Code Agent is a node in the tree and executes the 5-phase lifecycle:
1. Planning       - Query memory + LLM to create a step-by-step plan
2. Delegation     - Optionally decompose into subtasks for child agents
3. Implementation - Write code (integrating child results if any)
4. Validation     - Test code via ValidationAgent with debug loop
5. Return         - Store experience in memory, return result to parent
"""

from __future__ import annotations

import json
import logging
import re

from talm.agents.validation_agent import ValidationAgent
from talm.core.entities import (
    AgentResponseStatus,
    AgentResult,
    MemoryRecord,
    TALMConfig,
    Task,
)
from talm.core.interfaces import ILLMClient
from talm.memory.manager import MemoryManager
from talm.prompts.templates import (
    clarification_check_prompt,
    delegation_prompt,
    format_memory_context,
    implementation_prompt,
    planning_prompt,
    reflection_prompt,
)

logger = logging.getLogger(__name__)


def _extract_python_code(text: str) -> str:
    """Extract the first ```python ... ``` block from LLM output."""
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _parse_json_array(text: str) -> list[dict] | None:
    """Try to extract a JSON array from LLM output."""
    # Try to find JSON array in the text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


class CodeAgent:
    """A single node in the TALM tree. Recursively spawns children if needed."""

    def __init__(
        self,
        task: Task,
        llm: ILLMClient,
        memory: MemoryManager,
        validator: ValidationAgent,
        config: TALMConfig,
    ) -> None:
        self._task = task
        self._llm = llm
        self._memory = memory
        self._validator = validator
        self._config = config
        self._plan: str = ""
        self._code: str = ""

    async def execute(self) -> AgentResult:
        """Run the full 5-phase lifecycle and return the result."""
        logger.info(
            "CodeAgent executing at depth=%d: %.80s...",
            self._task.tree_depth,
            self._task.description,
        )

        # Phase 1: Planning
        self._plan = await self._phase_planning()
        logger.info("Plan created (%d chars)", len(self._plan))

        # Phase 2: Delegation
        child_results = await self._phase_delegation()

        # Phase 3: Implementation
        self._code = await self._phase_implementation(child_results)
        logger.info("Code generated (%d chars)", len(self._code))

        # Phase 4: Validation
        if self._config.sandbox_enabled:
            passed, self._code, error_log = await self._validator.validate(
                self._task.description, self._code
            )
            if not passed:
                logger.warning("Validation failed after all retries")
                return AgentResult(
                    status=AgentResponseStatus.FAILURE,
                    code=self._code,
                    reasoning=self._plan,
                    error_log=error_log,
                    task_description=self._task.description,
                    tree_depth=self._task.tree_depth,
                )

        # Phase 5: Return — store experience in memory
        record = MemoryRecord(
            task_description=self._task.description,
            reasoning_trace=self._plan,
            generated_code=self._code,
            tree_depth=self._task.tree_depth,
        )
        await self._memory.update(record)

        return AgentResult(
            status=AgentResponseStatus.SUCCESS,
            code=self._code,
            reasoning=self._plan,
            task_description=self._task.description,
            tree_depth=self._task.tree_depth,
        )

    # ------------------------------------------------------------------
    # Phase 1: Planning
    # ------------------------------------------------------------------

    async def _phase_planning(self) -> str:
        """Query long-term memory and generate a step-by-step plan."""
        # Retrieve relevant past experiences at the same tree depth
        memories = await self._memory.retrieve(
            self._task.description, self._task.tree_depth
        )
        memory_context = format_memory_context(memories)

        sys_prompt, usr_prompt = planning_prompt(
            self._task.description, memory_context, self._task.context
        )
        plan = await self._llm.generate(
            sys_prompt,
            usr_prompt,
            temperature=self._config.temperature,
            max_tokens=self._config.max_output_tokens,
        )
        return plan

    # ------------------------------------------------------------------
    # Phase 2: Delegation
    # ------------------------------------------------------------------

    async def _phase_delegation(self) -> list[str]:
        """Decide whether to delegate subtasks to child agents."""
        depth = self._task.tree_depth

        # Leaf nodes cannot delegate
        if depth >= self._config.max_depth:
            return []

        # Calculate max children at this depth: n - k * depth
        max_children = self._config.initial_branching - (
            self._config.decay_rate * depth
        )
        if max_children <= 0:
            return []

        # Ask LLM whether decomposition is needed
        sys_prompt, usr_prompt = delegation_prompt(
            self._plan, self._task.description, max_children
        )
        raw = await self._llm.generate(
            sys_prompt,
            usr_prompt,
            temperature=self._config.temperature,
        )

        if "NO_DELEGATION" in raw:
            logger.info("No delegation needed at depth=%d", depth)
            return []

        # Parse subtask list
        subtasks = _parse_json_array(raw)
        if not subtasks:
            logger.warning("Could not parse delegation output, skipping delegation")
            return []

        # Limit to max_children
        subtasks = subtasks[:max_children]
        logger.info("Delegating %d subtasks at depth=%d", len(subtasks), depth)

        # Spawn child agents and collect results
        child_codes: list[str] = []
        for st in subtasks:
            desc = st.get("description", str(st))
            child_result = await self._execute_child(desc, depth + 1)

            if child_result.status == AgentResponseStatus.CLARIFY:
                # Localized Re-Reasoning: child requests clarification
                desc = await self._handle_clarification(desc, child_result)
                child_result = await self._execute_child(desc, depth + 1)

            if child_result.status == AgentResponseStatus.SUCCESS:
                child_codes.append(child_result.code)
            else:
                logger.warning("Child agent failed for subtask: %.60s...", desc)

        # Structure-Correction: if all children failed, restructure
        if subtasks and not child_codes:
            logger.warning("All children failed — triggering structure correction")
            return await self._structure_correction()

        return child_codes

    async def _execute_child(self, description: str, depth: int) -> AgentResult:
        """Create and execute a child CodeAgent."""
        # Clarification check: let child assess if task is clear
        sys_p, usr_p = clarification_check_prompt(description)
        check = await self._llm.generate(sys_p, usr_p)

        try:
            check_data = json.loads(check.strip())
            if check_data.get("status") == "clarify":
                return AgentResult(
                    status=AgentResponseStatus.CLARIFY,
                    clarification_request=check_data.get("question", ""),
                    task_description=description,
                    tree_depth=depth,
                )
        except (json.JSONDecodeError, AttributeError):
            pass  # Assume task is clear if parsing fails

        child_task = Task(
            description=description,
            parent_task=self._task.description,
            tree_depth=depth,
            context=f"Parent plan:\n{self._plan}",
        )
        child_agent = CodeAgent(
            task=child_task,
            llm=self._llm,
            memory=self._memory,
            validator=self._validator,
            config=self._config,
        )
        return await child_agent.execute()

    async def _handle_clarification(
        self, original_desc: str, result: AgentResult
    ) -> str:
        """Parent reflects and provides a clearer task description."""
        sys_p, usr_p = reflection_prompt(
            self._task.description,
            original_desc,
            result.clarification_request,
        )
        clarified = await self._llm.generate(sys_p, usr_p)
        logger.info("Clarified subtask after child request")
        return clarified.strip()

    async def _structure_correction(self) -> list[str]:
        """Re-plan and re-delegate when all children fail (top-down correction)."""
        logger.info("Structure correction: re-planning delegation")
        sys_prompt, usr_prompt = delegation_prompt(
            self._plan,
            self._task.description + "\n\n[NOTE: Previous decomposition failed. "
            "Please restructure the subtasks differently.]",
            self._config.initial_branching
            - self._config.decay_rate * self._task.tree_depth,
        )
        raw = await self._llm.generate(sys_prompt, usr_prompt)
        subtasks = _parse_json_array(raw)
        if not subtasks:
            return []

        child_codes: list[str] = []
        for st in subtasks:
            desc = st.get("description", str(st))
            result = await self._execute_child(desc, self._task.tree_depth + 1)
            if result.status == AgentResponseStatus.SUCCESS:
                child_codes.append(result.code)

        return child_codes

    # ------------------------------------------------------------------
    # Phase 3: Implementation
    # ------------------------------------------------------------------

    async def _phase_implementation(self, child_results: list[str]) -> str:
        """Generate code from plan + child outputs."""
        sys_prompt, usr_prompt = implementation_prompt(
            self._task.description,
            self._plan,
            child_results or None,
        )
        raw = await self._llm.generate(
            sys_prompt,
            usr_prompt,
            temperature=self._config.temperature,
            max_tokens=self._config.max_output_tokens,
        )
        return _extract_python_code(raw)
