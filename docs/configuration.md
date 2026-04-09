# Configuration Guide

All settings are in `config.yaml` at the project root. Changes take effect on server restart.

## Full Config Reference

```yaml
# Tree-Structured Collaboration (Paper Section 3.1)
tree:
  max_depth: 3          # m: maximum tree depth. Nodes at depth m are forced leaves.
  initial_branching: 3  # n: root can spawn up to n children.
  decay_rate: 1         # k: branching decreases by k per level (n → n-k → n-2k).

# Validation Agent (Paper Section 3.1)
validation:
  max_retries: 3        # r: max debug attempts when tests fail.

# Long-Term Memory (Paper Section 3.3)
memory:
  similarity_threshold: 0.75   # Minimum cosine similarity for retrieval.
  top_k: 3                     # Max records returned per retrieval query.
  merge_threshold: 0.95        # Similarity threshold for consolidation.
                                # Records above this are merged by LLM.

# LLM Provider
llm:
  provider: "gemini"                  # Currently supported: "gemini"
  model: "gemini-2.5-flash"          # Any Gemini model ID
  temperature: 0                      # 0 = deterministic (paper default)
  max_output_tokens: 8192

# Embedding Model (decoupled from LLM)
embedding:
  provider: "sentence_transformers"   # "sentence_transformers" or "gemini"
  model: "all-MiniLM-L6-v2"          # Any sentence-transformers model name

# Vector Database
vector_db:
  provider: "zvec"                    # "zvec" or "chroma"
  persist_dir: "./talm_vector_db"     # Local storage path

# Sandbox (code execution)
sandbox:
  timeout_seconds: 30                 # Max execution time per code run
  enabled: true                       # Set false to skip validation
```

## Adapter Options

### LLM

| Provider | Config | Notes |
|----------|--------|-------|
| Gemini | `provider: "gemini"`, `model: "gemini-2.5-flash"` | Requires `GOOGLE_API_KEY` env var |

To add a new provider: implement `ILLMClient` in `src/talm/infrastructure/`, add a case in `config.py:create_llm()`.

### Embedding

| Provider | Config | Dimension | Notes |
|----------|--------|-----------|-------|
| sentence-transformers | `provider: "sentence_transformers"`, `model: "all-MiniLM-L6-v2"` | 384 | Local, no API key needed |
| Gemini | `provider: "gemini"` | 768 | Uses `text-embedding-004`, needs API key |

### Vector Database

| Provider | Config | Notes |
|----------|--------|-------|
| Zvec | `provider: "zvec"` | Alibaba's in-process vector DB, cosine similarity, side-index JSON |
| ChromaDB | `provider: "chroma"` | Persistent local storage, built-in metadata filtering |

### Sandbox

Currently only subprocess-based. For production, implement `ISandbox` with Docker or E2B.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Gemini API key |
| `GOOGLE_CLOUD_PROJECT` | No | For Vertex AI |
| `GOOGLE_CLOUD_LOCATION` | No | For Vertex AI |

## Runtime Config Updates

Hyperparameters can be changed at runtime via the API without restarting:

```bash
curl -X PUT http://localhost:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"max_depth": 2, "sandbox_enabled": false}'
```

These changes are **not persisted** to `config.yaml` — they reset on server restart.
