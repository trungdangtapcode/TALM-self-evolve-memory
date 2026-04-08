# API Reference

## REST API (Web Frontend)

Start with: `python -m talm.server` (default port 8000)

### Health Check

```
GET /api/health
```

Response:
```json
{"status": "ok", "service": "talm"}
```

### Generate Code

```
POST /api/generate
Content-Type: application/json
```

Body:
```json
{
  "task_description": "Write a Python function is_palindrome(s) -> bool",
  "mode": "full"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task_description` | string | yes | The coding task to solve |
| `mode` | string | no | `"full"` (tree decomposition) or `"simple"` (single agent). Default: `"full"` |

Response:
```json
{
  "status": "success",
  "code": "def is_palindrome(s): ...",
  "reasoning": "1. Filter string\n2. Compare with reverse",
  "error_log": "",
  "tree_depth": 0
}
```

| Field | Description |
|-------|-------------|
| `status` | `"success"` or `"failure"` |
| `code` | Generated Python code |
| `reasoning` | Agent's plan / reasoning trace |
| `error_log` | Validation errors (empty on success) |
| `tree_depth` | Depth of the root agent that produced this result |

### Configuration

```
GET /api/config
```

Returns all current hyperparameters (tree, validation, memory, LLM, embedding, vector_db, sandbox).

```
PUT /api/config
Content-Type: application/json
```

Body (all fields optional):
```json
{
  "max_depth": 2,
  "temperature": 0.1,
  "sandbox_enabled": false
}
```

### Memory

#### Stats
```
GET /api/memory/stats
```

Response:
```json
{
  "record_count": 5,
  "config": {
    "embedding_provider": "sentence_transformers",
    "embedding_model": "all-MiniLM-L6-v2",
    "vector_db_provider": "zvec",
    "similarity_threshold": 0.75,
    "merge_threshold": 0.95,
    "top_k": 3
  }
}
```

#### Browse Records
```
GET /api/memory/records
```

Response:
```json
{
  "records": [
    {
      "id": "abc123...",
      "task_description": "Write fibonacci",
      "reasoning_trace": "1. Base cases\n2. Recursion",
      "generated_code": "def fib(n): ...",
      "tree_depth": 0
    }
  ]
}
```

#### Insert Record (Manual)
```
POST /api/memory/records
Content-Type: application/json
```

Body:
```json
{
  "task_description": "Sort a list using quicksort",
  "reasoning_trace": "1. Pick pivot\n2. Partition\n3. Recurse",
  "generated_code": "def quicksort(arr): ...",
  "tree_depth": 0
}
```

All 4 fields are required. Consolidation rules apply automatically — if a near-duplicate exists (similarity >= 0.95), records will be merged by the LLM.

#### Clear All
```
POST /api/memory/clear
```

Removes all records. Use before evaluation/benchmark runs.

---

## MCP Server (Claude Desktop / Inspector)

Start with: `python -m talm.server --mcp`

Or via MCP Inspector: `npx @modelcontextprotocol/inspector python -m talm.server --mcp`

### Tools

| Tool | Parameters | Description |
|------|------------|-------------|
| `generate_code` | `task_description: str` | Full tree-structured generation |
| `generate_code_simple` | `task_description: str` | Single-agent mode |
| `get_config` | — | View hyperparameters |
| `update_config` | `max_depth?, temperature?, ...` | Modify at runtime |
| `get_memory_stats` | — | Memory record count + config |
| `clear_memory` | — | Reset long-term memory |

### Resources

| URI | Description |
|-----|-------------|
| `talm://config` | Current configuration |
| `talm://memory/stats` | Memory statistics |
