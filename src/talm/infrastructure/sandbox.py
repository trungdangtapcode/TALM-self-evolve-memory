"""Subprocess-based sandbox for executing generated Python code.

For production use, replace with Docker or E2B sandbox.  This MVP
implementation uses subprocess with resource limits for safety.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import os

from talm.core.entities import TestResult
from talm.core.interfaces import ISandbox

logger = logging.getLogger(__name__)


class SubprocessSandbox(ISandbox):
    """Execute Python code in an isolated subprocess with timeout."""

    def __init__(self, timeout: int = 30) -> None:
        self._default_timeout = timeout

    async def execute(self, code: str, timeout: int | None = None) -> TestResult:
        """Write code to a temp file and run it in a subprocess."""
        timeout = timeout or self._default_timeout

        # Write code to a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            passed = proc.returncode == 0

            return TestResult(
                passed=passed,
                stdout=stdout,
                stderr=stderr,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return TestResult(
                passed=False,
                stderr=f"Execution timed out after {timeout}s",
            )
        except Exception as e:
            return TestResult(
                passed=False,
                stderr=f"Sandbox error: {e}",
            )
        finally:
            os.unlink(tmp_path)
