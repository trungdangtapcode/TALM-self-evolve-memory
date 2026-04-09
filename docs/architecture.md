# Architecture

TALM follows **Clean Architecture** with the Dependency Inversion Principle — core logic depends on interfaces, not concrete implementations.

## Layer Diagram

```mermaid
graph TB
    subgraph Presentation["Presentation Layer"]
        REST["REST API<br/>(Starlette)"]
        MCP["MCP Server<br/>(FastMCP, stdio)"]
        WEB["Web Frontend<br/>(Vite + React)"]
    end

    subgraph Application["Application Layer"]
        WF["TALMWorkflow<br/>workflow.py"]
        CA["CodeAgent<br/>agents/code_agent.py"]
        VA["ValidationAgent<br/>agents/validation_agent.py"]
        MM["MemoryManager<br/>memory/manager.py"]
    end

    subgraph Domain["Domain Layer"]
        ENT["Entities<br/>Task, AgentResult,<br/>MemoryRecord, TALMConfig"]
        INT["Interfaces<br/>ILLMClient, IEmbedder,<br/>IVectorDatabase, ISandbox"]
    end

    subgraph Infrastructure["Infrastructure Layer"]
        GEM["GeminiLLMClient<br/>llm_gemini.py"]
        ST["SentenceTransformerEmbedder<br/>embedder_st.py"]
        ZVEC["ZvecAdapter<br/>vector_db_zvec.py"]
        SB["SubprocessSandbox<br/>sandbox.py"]
    end

    WEB -->|HTTP| REST
    REST --> WF
    MCP --> WF
    WF --> CA
    WF --> MM
    CA --> VA
    CA --> MM
    CA --> INT
    VA --> INT
    MM --> INT
    INT -.->|implements| GEM
    INT -.->|implements| ST
    INT -.->|implements| ZVEC
    INT -.->|implements| SB
```

## Dependency Flow

```
Presentation → Application → Domain ← Infrastructure
```

- **Domain** has zero external dependencies (pure Python dataclasses + ABCs)
- **Application** depends only on Domain interfaces
- **Infrastructure** implements Domain interfaces with real libraries
- **Presentation** wires everything together via `config.py` factories

## Code Agent 5-Phase Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Planning
    Planning --> Delegation : plan ready
    Delegation --> Implementation : subtasks done / no delegation
    Implementation --> Validation : code generated
    Validation --> Implementation : tests failed (retry &le; r)
    Validation --> Return : tests passed
    Validation --> Return : retries exhausted (failure)
    Return --> [*]

    state Planning {
        [*] --> RetrieveMemory : embed task, K-NN search
        RetrieveMemory --> LLMPlan : inject memory context
        LLMPlan --> [*]
    }

    state Delegation {
        [*] --> CheckDepth
        CheckDepth --> Decompose : depth &lt; m
        CheckDepth --> Skip : depth &ge; m (leaf)
        Decompose --> SpawnChildren
        SpawnChildren --> ReviewIntegration
        ReviewIntegration --> StructureCorrection : flawed decomposition
        ReviewIntegration --> [*] : accept
        StructureCorrection --> SpawnChildren
        Skip --> [*]
    }

    state Return {
        [*] --> StoreMemory : auto-store experience
        StoreMemory --> CheckConsolidation
        CheckConsolidation --> LLMMerge : sim &ge; 0.95
        CheckConsolidation --> Insert : no duplicate
        LLMMerge --> [*]
        Insert --> [*]
    }
```

## Tree-Structured Decomposition

```mermaid
graph TD
    R["Root Agent<br/>depth=0, max children = n"]
    C1["Child 1<br/>depth=1, max = n-k"]
    C2["Child 2<br/>depth=1"]
    C3["Child 3<br/>depth=1"]
    L1["Leaf<br/>depth=2, max = n-2k"]
    L2["Leaf<br/>depth=2"]

    R --> C1
    R --> C2
    R --> C3
    C1 --> L1
    C1 --> L2

    style R fill:#4a6cf7,color:#fff
    style C1 fill:#166534,color:#4ade80
    style C2 fill:#166534,color:#4ade80
    style C3 fill:#166534,color:#4ade80
    style L1 fill:#7c2d12,color:#fb923c
    style L2 fill:#7c2d12,color:#fb923c
```

The tree is **not predefined** — the LLM dynamically decides at each node whether to decompose (return subtask JSON) or solve directly (return `NO_DELEGATION`). The `m, n, k` hyperparameters constrain the shape:

- `m=3`: max depth (nodes at depth m are forced leaves)
- `n=3`: root branching factor
- `k=1`: branching decays by k per level (3 → 2 → 1)

## Localized Re-Reasoning

```mermaid
sequenceDiagram
    participant P as Parent Agent
    participant C as Child Agent

    Note over P,C: Bottom-up: Clarification
    P->>C: Assign subtask
    C-->>P: {"status": "clarify", "question": "..."}
    P->>P: Reflect & refine description
    P->>C: Re-assign clarified subtask
    C-->>P: Code result

    Note over P,C: Top-down: Structure Correction
    P->>C: Assign subtasks
    C-->>P: Return outputs
    P->>P: Review integration feasibility
    P->>P: Verdict = "restructure" (flawed decomposition)
    P->>P: Discard all child outputs, re-plan
    P->>C: New subtask structure
    C-->>P: Code result
```

## Long-Term Memory Pipeline

```mermaid
flowchart LR
    subgraph Retrieve["Retrieve (Phase 1)"]
        A1[Embed task] --> A2[K-NN search]
        A2 --> A3[Filter by tree_depth]
        A3 --> A4[Top-K=3, sim &ge; 0.75]
    end

    subgraph Store["Store (Phase 5)"]
        B1[Validation passed] --> B2[Embed record]
        B2 --> B3{Duplicate exists?<br/>sim &ge; 0.95}
        B3 -->|No| B4[Insert new record]
        B3 -->|Yes| B5[LLM merges old+new]
        B5 --> B6[Delete old, insert merged]
    end

    Retrieve ~~~ Store
```

## Pluggable Adapters

All adapters are selected via `config.yaml` and instantiated by factory functions in `config.py`:

| Interface | Default Adapter | Alternative |
|-----------|----------------|-------------|
| `ILLMClient` | `GeminiLLMClient` (gemini-2.5-flash) | Extend for OpenAI, local models |
| `IEmbedder` | `SentenceTransformerEmbedder` (all-MiniLM-L6-v2, 384-dim) | `GeminiEmbedder` (768-dim) |
| `IVectorDatabase` | `ZvecAdapter` (Alibaba Zvec, cosine) | `ChromaDBAdapter` |
| `ISandbox` | `SubprocessSandbox` | Docker, E2B |

To swap: change the `provider` field in `config.yaml`, no code changes needed.
