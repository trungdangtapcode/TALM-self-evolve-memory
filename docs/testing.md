# Testing

## Philosophy

Tests follow **Clean Architecture testing principles**:

- **No real external dependencies** — all tests use fake implementations of `ILLMClient`, `IEmbedder`, `IVectorDatabase`, and `ISandbox`
- **No network calls** — no Gemini API, no HuggingFace downloads, no disk I/O
- **Deterministic** — fake embedder produces consistent vectors from text hash
- **Fast** — full suite runs in < 1 second

## Test Structure

```
tests/
├── fakes.py                    # Fake implementations of all interfaces
├── test_memory_manager.py      # Unit: retrieve, store, consolidation
├── test_validation_agent.py    # Unit: test generation, debug loop, retries
├── test_code_agent.py          # Integration: 5-phase lifecycle, delegation
├── test_workflow.py            # Integration: end-to-end workflow
└── test_server.py              # API: REST endpoint tests
```

### Fakes (`tests/fakes.py`)

| Fake | Replaces | Behavior |
|------|----------|----------|
| `FakeLLM` | Gemini API | Returns scripted responses based on prompt content |
| `FakeEmbedder` | sentence-transformers | Deterministic hash-based unit vectors |
| `FakeVectorDB` | Zvec / ChromaDB | In-memory dict with cosine similarity |
| `FakeSandbox` | subprocess | Configurable pass/fail |

### Test Layers

| Layer | File | What's Tested |
|-------|------|---------------|
| **Domain** | `test_memory_manager.py` | Retrieve depth-filter, threshold, consolidation merge/skip, clear |
| **Domain** | `test_validation_agent.py` | Code extraction, first-attempt pass, retry loop, exhausted retries |
| **Application** | `test_code_agent.py` | Full lifecycle, memory store on success/failure, delegation, depth limits |
| **Application** | `test_workflow.py` | End-to-end run, failure handling, memory accumulation, clear, list |
| **Presentation** | `test_server.py` | All REST endpoints via Starlette TestClient |

## Running Tests

```bash
source .venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run a specific layer
python -m pytest tests/test_memory_manager.py -v
python -m pytest tests/test_code_agent.py -v

# Run with coverage (install pytest-cov first)
python -m pytest tests/ --cov=talm --cov-report=term-missing
```

## Writing New Tests

1. Import fakes from `tests/fakes.py`
2. Inject them via constructor (dependency injection, not monkey-patching)
3. Use `@pytest.mark.asyncio` for async tests (auto mode enabled in `pyproject.toml`)

Example:

```python
from tests.fakes import FakeEmbedder, FakeLLM, FakeVectorDB

@pytest.mark.asyncio
async def test_my_feature():
    llm = FakeLLM(responder=lambda s, u: "response")
    embedder = FakeEmbedder(dimension=8)
    vector_db = FakeVectorDB()
    # ... build your component with fakes
```
