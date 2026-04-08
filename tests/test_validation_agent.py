"""Tests for the ValidationAgent: test generation, sandbox execution, and debug loop."""

import pytest

from talm.agents.validation_agent import ValidationAgent, _extract_python_code
from talm.core.entities import TALMConfig, TestResult
from tests.fakes import FakeLLM, FakeSandbox


@pytest.fixture
def config() -> TALMConfig:
    return TALMConfig(max_retries=3, sandbox_timeout=10)


def test_extract_python_code_from_fenced_block():
    text = 'Some text\n```python\ndef hello():\n    return "hi"\n```\nMore text'
    assert _extract_python_code(text) == 'def hello():\n    return "hi"'


def test_extract_python_code_plain():
    text = 'def hello():\n    return "hi"'
    assert _extract_python_code(text) == text


@pytest.mark.asyncio
async def test_validation_passes_on_first_attempt(config: TALMConfig):
    """Code that passes tests should succeed immediately."""
    llm = FakeLLM(responder=lambda s, u: '```python\nassert True\nprint("ALL TESTS PASSED")\n```')
    sandbox = FakeSandbox(always_pass=True)
    agent = ValidationAgent(llm=llm, sandbox=sandbox, config=config)

    passed, code, error_log = await agent.validate("task", "def foo(): return 1")

    assert passed is True
    assert error_log == ""
    assert len(sandbox.executions) == 1


@pytest.mark.asyncio
async def test_validation_retries_on_failure(config: TALMConfig):
    """Failed tests should trigger debug loop up to max_retries."""
    call_count = 0

    def responder(system: str, user: str) -> str:
        nonlocal call_count
        call_count += 1
        if "debugging" in system.lower() or "fix" in system.lower():
            return '```python\ndef foo(): return 42\n```'
        return '```python\nassert foo() == 42\n```'

    llm = FakeLLM(responder=responder)

    # Sandbox fails first 2 times, passes on 3rd
    exec_count = 0

    class RetryingSandbox(FakeSandbox):
        async def execute(self, code: str, timeout: int = 30) -> TestResult:
            nonlocal exec_count
            exec_count += 1
            self.executions.append(code)
            if exec_count < 3:
                return TestResult(passed=False, stderr="AssertionError")
            return TestResult(passed=True, stdout="OK")

    sandbox = RetryingSandbox()
    agent = ValidationAgent(llm=llm, sandbox=sandbox, config=config)

    passed, code, error_log = await agent.validate("task", "def foo(): return 0")

    assert passed is True
    assert exec_count == 3


@pytest.mark.asyncio
async def test_validation_exhausts_retries(config: TALMConfig):
    """If all retries fail, should return failure with error log."""
    llm = FakeLLM(responder=lambda s, u: '```python\nassert False\n```')
    sandbox = FakeSandbox(always_pass=False)
    agent = ValidationAgent(llm=llm, sandbox=sandbox, config=config)

    passed, code, error_log = await agent.validate("task", "def foo(): return 0")

    assert passed is False
    assert error_log != ""
    assert len(sandbox.executions) == config.max_retries
