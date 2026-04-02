# TALM — Tree-Structured Multi-Agent Framework with Long-Term Memory

A reproduction of the [TALM paper](https://arxiv.org/abs/2510.23010): a multi-agent code generation framework that uses **tree-structured task decomposition**, **long-term vector memory**, and **localized re-reasoning** to produce validated Python code.

Exposed as an **MCP server** (via FastMCP) for integration with Claude, Cursor, or any MCP-compatible client.

## Architecture

### System Overview

```mermaid
graph TB
    Client["MCP Client<br/>(Claude / Cursor / Inspector)"]
    Server["FastMCP Server<br/>talm.server"]
    WF["TALMWorkflow<br/>Orchestrator"]

    Client <-->|MCP Protocol| Server
    Server --> WF

    subgraph Core["TALM Core"]
        WF --> RootAgent["Root CodeAgent<br/>depth=0"]
        RootAgent --> MM["MemoryManager"]
        RootAgent --> VA["ValidationAgent"]
    end

    subgraph Infra["Pluggable Adapters"]
        LLM["ILLMClient<br/>(Gemini)"]
        EMB["IEmbedder<br/>(all-MiniLM-L6-v2)"]
        VDB["IVectorDatabase<br/>(Zvec)"]
        SB["ISandbox<br/>(Subprocess)"]
    end

    MM --> EMB
    MM --> VDB
    MM --> LLM
    VA --> LLM
    VA --> SB
    RootAgent --> LLM
```

### Tree-Structured Agent Decomposition

```mermaid
graph TD
    R["Root Agent<br/>depth=0<br/>max children = n = 3"]
    C1["Child Agent 1<br/>depth=1<br/>max children = n-k = 2"]
    C2["Child Agent 2<br/>depth=1"]
    C3["Child Agent 3<br/>depth=1"]
    L1["Leaf Agent<br/>depth=2<br/>no delegation"]
    L2["Leaf Agent<br/>depth=2"]

    R --> C1
    R --> C2
    R --> C3
    C1 --> L1
    C1 --> L2

    style R fill:#4a90d9,color:#fff
    style C1 fill:#7ab648,color:#fff
    style C2 fill:#7ab648,color:#fff
    style C3 fill:#7ab648,color:#fff
    style L1 fill:#f5a623,color:#fff
    style L2 fill:#f5a623,color:#fff
```

### Code Agent 5-Phase Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Planning
    Planning --> Delegation : plan ready
    Delegation --> Implementation : subtasks done / no delegation
    Implementation --> Validation : code generated
    Validation --> Implementation : tests failed (retry ≤ r)
    Validation --> Return : tests passed
    Validation --> Return : retries exhausted (failure)
    Return --> [*]

    state Planning {
        [*] --> QueryMemory
        QueryMemory --> LLMPlan
        LLMPlan --> [*]
    }

    state Delegation {
        [*] --> CheckDepth
        CheckDepth --> SpawnChildren : depth < m
        CheckDepth --> Skip : depth ≥ m
        SpawnChildren --> CollectResults
        CollectResults --> StructureCorrection : all children failed
        StructureCorrection --> CollectResults
        CollectResults --> [*]
        Skip --> [*]
    }

    state Return {
        [*] --> UpdateMemory
        UpdateMemory --> Consolidate
        Consolidate --> ReturnToParent
        ReturnToParent --> [*]
    }
```

### Long-Term Memory Lifecycle

```mermaid
flowchart LR
    subgraph Retrieve["1. Retrieve"]
        A1[Embed task] --> A2[K-NN search<br/>threshold ≥ 0.75]
        A2 --> A3[Filter by<br/>tree_depth]
        A3 --> A4[Top-K = 3<br/>records]
    end

    subgraph Update["2. Update"]
        B1[Task succeeded] --> B2[Embed record]
        B2 --> B3{Similar record<br/>exists? ≥ 0.95}
        B3 -->|No| B4[Insert new]
        B3 -->|Yes| B5[LLM merges<br/>old + new]
        B5 --> B6[Delete old<br/>Insert merged]
    end

    Retrieve --> Update
```

### Localized Re-Reasoning

```mermaid
sequenceDiagram
    participant P as Parent Agent
    participant C as Child Agent

    Note over P,C: Bottom-up: Clarification
    P->>C: Assign subtask
    C-->>P: {"status": "clarify", "question": "..."}
    P->>P: Reflect & improve description
    P->>C: Re-assign clarified subtask
    C-->>P: Code result

    Note over P,C: Top-down: Structure Correction
    P->>C: Assign subtasks (batch)
    C-->>P: All children failed
    P->>P: Discard results, re-plan decomposition
    P->>C: New subtask structure
    C-->>P: Code result
```

### Pluggable Adapters

| Component      | Default                          | Alternatives         |
| -------------- | -------------------------------- | -------------------- |
| **LLM**        | Google Gemini (`gemini-2.0-flash`) | Extend `ILLMClient`  |
| **Embedding**  | `all-MiniLM-L6-v2` (sentence-transformers, 384-dim) | Gemini embeddings    |
| **Vector DB**  | Zvec (Alibaba, cosine)           | ChromaDB             |
| **Sandbox**    | Subprocess (Python)              | Docker / E2B         |

All adapters are swappable via `config.yaml` — no code changes needed.

## Quick Start

```bash
# 1. Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. Set your Gemini API key
export GOOGLE_API_KEY="your-key-here"

# 3. Run the MCP server with Inspector
npx @modelcontextprotocol/inspector python -m talm.server

# Or run the demo directly
python demo.py      # simple task
python demo.py 1    # complex task (multi-module)
```

## MCP Tools

| Tool                   | Description                                     |
| ---------------------- | ----------------------------------------------- |
| `generate_code`        | Full tree-structured multi-agent code generation |
| `generate_code_simple` | Single-agent mode (no tree decomposition)        |
| `get_config`           | View current hyperparameters                     |
| `update_config`        | Modify hyperparameters at runtime                |
| `get_memory_stats`     | Check long-term memory record count              |
| `clear_memory`         | Reset long-term memory (for evaluation runs)     |

## Configuration

All hyperparameters are in [`config.yaml`](config.yaml):

```yaml
tree:
  max_depth: 3          # m
  initial_branching: 3  # n
  decay_rate: 1         # k

validation:
  max_retries: 3        # r

memory:
  similarity_threshold: 0.75
  top_k: 3
  merge_threshold: 0.95

embedding:
  provider: "sentence_transformers"   # or "gemini"
  model: "all-MiniLM-L6-v2"

vector_db:
  provider: "zvec"                    # or "chroma"
```

## Project Structure

```
src/talm/
├── core/
│   ├── entities.py             # Task, AgentResult, MemoryRecord, TALMConfig
│   └── interfaces.py           # ILLMClient, IEmbedder, IVectorDatabase, ISandbox
├── agents/
│   ├── code_agent.py           # 5-phase recursive Code Agent
│   └── validation_agent.py     # Test generation + debug loop
├── memory/
│   └── manager.py              # Retrieve, Update, Consolidate
├── infrastructure/
│   ├── llm_gemini.py           # Gemini LLM adapter
│   ├── embedder_st.py          # sentence-transformers embedder
│   ├── embedder_gemini.py      # Gemini embedder (alternative)
│   ├── vector_db_zvec.py       # Zvec adapter (primary)
│   ├── vector_db.py            # ChromaDB adapter (alternative)
│   └── sandbox.py              # Subprocess sandbox
├── prompts/
│   └── templates.py            # Prompt templates for all phases
├── config.py                   # YAML loader + adapter factories
├── workflow.py                 # TALMWorkflow orchestrator
└── server.py                   # FastMCP server entry point
```

## Key TALM Mechanisms

- **Tree-structured decomposition**: tasks split recursively with branching factor `n - k * depth`, capped at depth `m`
- **Depth-filtered memory**: retrieval only returns records from the same tree depth (matching granularity)
- **Memory consolidation**: near-duplicate records (similarity > 0.95) are merged via LLM synthesis to prevent bloat
- **Localized re-reasoning**:
  - *Bottom-up*: child agents request clarification when tasks are ambiguous
  - *Top-down*: parent agents discard and restructure failed subtree decompositions
- **Validation loop**: LLM-generated tests run in sandbox with up to `r` debug retries

## Environment Variables

| Variable                 | Required | Description           |
| ------------------------ | -------- | --------------------- |
| `GOOGLE_API_KEY`         | Yes      | Gemini API key        |
| `GOOGLE_CLOUD_PROJECT`   | No       | For Vertex AI         |
| `GOOGLE_CLOUD_LOCATION`  | No       | For Vertex AI         |

## License

Research reproduction — refer to the original TALM paper for academic use.
