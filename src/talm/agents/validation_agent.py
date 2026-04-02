"""Validation Agent: generates tests, runs code in sandbox, and drives the debug loop.

Implements the TALM validation cycle:
1. Generate test cases via LLM
2. Combine implementation + tests and execute in sandbox
3. If tests fail, feed error logs back to LLM for debugging (up to r retries)
"""

from __future__ import annotations

import logging
import re

from talm.core.entities import TALMConfig, TestResult
from talm.core.interfaces import ILLMClient, ISandbox
from talm.prompts.templates import debug_prompt, validation_prompt

logger = logging.getLogger(__name__)


def _extract_python_code(text: str) -> str:
    """Extract the first ```python ... ``` block from LLM output."""
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


class ValidationAgent:
    """Generates tests and validates code through an iterative debug loop."""

    def __init__(
        self,
        llm: ILLMClient,
        sandbox: ISandbox,
        config: TALMConfig,
    ) -> None:
        self._llm = llm
        self._sandbox = sandbox
        self._config = config

    async def validate(
        self, task_description: str, code: str
    ) -> tuple[bool, str, str]:
        """
        Validate code by generating tests and running them.

        Returns:
            (passed, final_code, error_log)
        """
        # Step 1: Generate test cases
        sys_prompt, usr_prompt = validation_prompt(task_description, code)
        raw_tests = await self._llm.generate(sys_prompt, usr_prompt)
        test_code = _extract_python_code(raw_tests)

        # Step 2: Iterative validation loop (up to r retries)
        current_code = code
        for attempt in range(1, self._config.max_retries + 1):
            # Combine implementation + test code for execution
            combined = current_code + "\n\n" + test_code
            result: TestResult = await self._sandbox.execute(
                combined, timeout=self._config.sandbox_timeout
            )

            if result.passed:
                logger.info("Validation passed on attempt %d", attempt)
                return True, current_code, ""

            logger.warning(
                "Validation failed (attempt %d/%d): %s",
                attempt,
                self._config.max_retries,
                result.stderr[:200],
            )

            # Step 3: Debug — ask LLM to fix the code
            if attempt < self._config.max_retries:
                error_output = result.stderr or result.stdout
                sys_p, usr_p = debug_prompt(
                    task_description, current_code, test_code, error_output
                )
                raw_fix = await self._llm.generate(sys_p, usr_p)
                current_code = _extract_python_code(raw_fix)

        # All retries exhausted
        return False, current_code, result.stderr or result.stdout
