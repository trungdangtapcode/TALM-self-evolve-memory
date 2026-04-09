# Memory Benchmarking Guide (LongMemEval-style)

A reproducible recipe for benchmarking a long-term memory system using the
**LongMemEval** methodology (Wu et al., ICLR 2025; arXiv:2410.10813).

This guide is written so a coding agent can apply it to **any** memory system
that exposes an embedder + a vector store with cosine similarity. The TALM
implementation in this repo is the worked example, but the harness is
intentionally portable.

---

## 1. What you're measuring

LongMemEval decomposes a memory system into three stages and lets you ablate
each one independently:

```
session/turn texts ──[indexing]──> vectors ──[retrieval]──> top-k ──[reading]──> answer
```

This guide covers the **indexing + retrieval** stages. The reading stage
(LLM-based QA) is optional and needs an API key + a judge model — see §7.

The primary retrieval metric is **session-recall@k**: of the ground-truth
"evidence sessions" that contain the answer, how many appear in the top-k
returned by your retriever. This is the same metric the LongMemEval paper
reports for its retriever baselines (BM25, Contriever, Stella V5, GTE-Qwen2).

LongMemEval evaluates five abilities. Map them to your system as follows:

| LongMemEval ability       | What it tests                                       | Question types in the dataset                |
|---------------------------|-----------------------------------------------------|----------------------------------------------|
| Information Extraction    | Recall a fact stated in one past session            | `single-session-user`, `single-session-assistant`, `single-session-preference` |
| Multi-Session Reasoning   | Synthesize info scattered across many sessions      | `multi-session`                              |
| Knowledge Updates         | Use the *latest* version of a changing fact         | `knowledge-update`                           |
| Temporal Reasoning        | Time-relative queries ("last month", "after X")     | `temporal-reasoning`                         |
| Abstention                | Refuse to answer when info was never given          | any `question_id` ending in `_abs`           |

If your memory system has no notion of timestamps, *report the temporal
result honestly* — don't fake it. The whole point of LongMemEval is to
expose gaps.

---

## 2. Two complementary benchmarks

This repo ships **two** harnesses. Both live in `showcase/` (gitignored).
Use both — they answer different questions.

| Harness                              | Data                          | Needs API key | Runtime  | Answers                                       |
|--------------------------------------|-------------------------------|---------------|----------|-----------------------------------------------|
| `showcase/longmemeval_bench.py`      | 12 hand-crafted seeds         | No            | ~30s     | "Is the retrieval pipeline plumbed correctly? Where are the obvious gaps (recency, abstention threshold)?" |
| `showcase/longmemeval_real.py`       | Real LongMemEval_S (500 ×~47) | No (retrieval-only) | ~13min full / ~40s for 5% slice | "How does my retriever rank against published baselines?" |

The synthetic harness is for **smoke-testing and ablation** — it's fast,
deterministic, and the dataset is small enough to read in one screen so
failures are easy to debug. The real harness gives you the **publishable
headline number** against a recognized baseline.

---

## 3. Setup

### 3.1 Dependencies

You need a Python environment with:
- `sentence-transformers` (or whatever embedder your memory uses)
- A vector store (TALM uses `zvec`; ChromaDB / FAISS / Milvus all work)
- The memory system being benchmarked

For TALM specifically:

```bash
.venv/bin/pip install -e .
.venv/bin/python -c "import sentence_transformers, zvec; print('ok')"
```

### 3.2 Download the LongMemEval dataset

The cleaned variants are on HuggingFace, no authentication required:

```bash
mkdir -p data/
cd data/
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json     # 15 MB,  smoke test
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json  # 265 MB, the real benchmark
# wget longmemeval_m_cleaned.json   # ~3 GB, only if you need the hardest variant
cd ..
```

**Add `data/` to `.gitignore`** — you don't want 280 MB of JSON in version control.

The three variants:
- `longmemeval_oracle.json` — only evidence sessions (≈2 sessions/instance). Trivially easy retrieval; use it to verify your adapter loads the format correctly, *not* to claim a recall number.
- `longmemeval_s_cleaned.json` — the realistic variant: ~47 sessions per instance, only ~2 contain the answer. **This is what you report.**
- `longmemeval_m_cleaned.json` — ~500 sessions per instance, exceeds most context windows. Use only when measuring scalability.

### 3.3 Verify dataset shape

Each instance is a dict with these fields (the only ones you'll use):

```python
{
    "question_id":         "gpt4_2655b836",          # ends in "_abs" for abstention questions
    "question_type":       "temporal-reasoning",     # one of 6 types (see §1)
    "question":            "What was the first issue I had with my new car...",
    "answer":              "GPS system not functioning correctly",
    "haystack_session_ids": ["sess_a1", "sess_a2", ...],   # ~47 IDs
    "haystack_sessions":    [[turn, turn, ...], [...], ...], # parallel list of sessions
    "answer_session_ids":   ["answer_4be1b6b4_2", ...],    # ground-truth evidence (subset)
}
```

Each `turn` inside a session is `{"role": "user"|"assistant", "content": "..."}`.
Sometimes turns also have `has_answer: true` for fine-grained turn-level evaluation.

---

## 4. The minimum interface your memory system must expose

Your memory system needs **three** operations to be benchmarkable. That's it:

```python
class MemorySystem:
    async def insert(self, text: str, metadata: dict) -> str: ...
    async def retrieve(self, query: str, top_k: int) -> list[tuple[str, dict, float]]: ...
        # returns: [(text, metadata, similarity), ...]
    async def clear(self) -> None: ...
```

The `metadata` field is what you use to round-trip the LongMemEval `session_id`
so you can match retrieved hits against the ground truth.

For TALM specifically, the mapping is:

| LongMemEval concept | TALM equivalent |
|---|---|
| `text` (one session, flattened) | `MemoryRecord.task_description` |
| `metadata={"session_id": ...}` | `MemoryRecord.metadata` |
| `insert` | `IVectorDatabase.insert(record, embedding)` (after embedder.embed) |
| `retrieve` | `IVectorDatabase.retrieve(query_emb, tree_depth, top_k, threshold)` |
| `clear` | `IVectorDatabase.clear()` |

**Pitfall:** TALM's retrieval is filtered by `tree_depth`. Pin every benchmark
record and query to `tree_depth=0` or your retrievals will silently return
empty.

---

## 5. The harness algorithm

Per instance, do exactly this:

```python
async def evaluate_instance(instance, memory, top_k):
    await memory.clear()                                    # fresh state

    sessions = instance["haystack_sessions"]
    session_ids = instance["haystack_session_ids"]
    evidence = set(instance["answer_session_ids"] or [])
    is_abstention = instance["question_id"].endswith("_abs")

    # Index: one record per session, text = flattened turns
    for sid, session in zip(session_ids, sessions):
        text = "\n".join(f"{t['role']}: {t['content']}" for t in session)
        await memory.insert(text, metadata={"session_id": sid})

    # Retrieve
    hits = await memory.retrieve(instance["question"], top_k=top_k)
    retrieved_ids = [meta["session_id"] for _text, meta, _score in hits]

    # Score (skip abstention; see §6 for how to handle those separately)
    if is_abstention:
        return {"correct_refusal": len(retrieved_ids) == 0}
    found = set(retrieved_ids) & evidence
    return {
        "recall": len(found) / len(evidence),
        "hit@1":  int(retrieved_ids[0] in evidence) if retrieved_ids else 0,
    }
```

**Performance notes:**
- **Batch-embed sessions** in one call (`embedder.embed_batch`). Embedding 47
  short documents one at a time is the dominant cost on CPU.
- **Don't reuse a stateful vector store across instances if `clear()` is buggy.**
  In TALM today, `ZvecAdapter.clear()` calls a non-existent
  `_collection.close()` and crashes. The workaround is to create a fresh
  adapter (and a fresh persist dir) per instance — slower but reliable.

---

## 6. Metrics and how to report them

### Primary
- **session_recall@k** — averaged over answerable instances. Compare against
  the LongMemEval paper baselines (BM25 ≈70%, Contriever ≈76%, Stella V5 ≈83%,
  GTE-Qwen2 ≈85% on the full S split).
- **session_hit@1** — fraction where the top-1 result is an evidence session.
  Useful for systems that feed only the top-1 to the reader.

### Secondary
- **Per-question-type breakdown.** This is where you find weaknesses. A high
  overall score that's bottlenecked by `multi-session` or `knowledge-update`
  tells you something specific about your system.
- **Latency per instance.** Memory benchmarks are also a free latency
  benchmark — report `avg_latency_s` so reviewers know you're not getting your
  number from a 7B-param embedder at 30s/query.

### Abstention (separate metric)
For `*_abs` instances, ground-truth `answer_session_ids` is empty, so recall
is undefined. Score them with **correct-refusal-rate**: how many returned an
empty result *after* applying your production similarity threshold. A
well-calibrated system should refuse 100% of abstention queries.

If your retriever has no threshold, abstention is N/A — report it as
"system has no abstention mechanism" rather than scoring it.

### Sample size discipline
- Smoke test: 25 instances (5%). Don't report this as a result — it's a
  pipeline check.
- Slice: 100 instances (20%). Acceptable for early iteration.
- Full: all 500. **This is the only number worth claiming externally.**

The full S run takes ~13 minutes on CPU with `all-MiniLM-L6-v2`. There's no
excuse for skipping it once your harness works.

---

## 7. Optional: the reading stage (full QA accuracy)

Retrieval recall is necessary but not sufficient. A retriever with 90% recall
that feeds a weak reader can still produce wrong answers. To measure
end-to-end accuracy:

1. For each instance, retrieve top-k as above.
2. Construct a prompt: `question + retrieved_sessions_text → LLM`.
3. Have the LLM produce a hypothesis answer.
4. Use a judge LLM (LongMemEval uses GPT-4o) to score `hypothesis vs.
   ground-truth answer` as correct/incorrect.
5. Report **QA accuracy**.

This requires API keys and money. The LongMemEval repo has reference
implementations of the judge prompt — copy theirs to stay compatible with
their published numbers.

For TALM, the natural way to wire this is:
- Skip the LongMemEval reader entirely
- Run the full `TALMWorkflow.run()` pipeline on the question, treating
  retrieved sessions as memory context
- Score TALM's final `AgentResult.code` or reasoning against the ground truth

This measures something subtly different (end-to-end agent behavior, not
isolated retrieval+reading), but it's more honest about what TALM actually
does.

---

## 8. Comparing two memory systems fairly

When you swap memory systems for an A/B comparison:

1. **Hold the data fixed.** Same `--seed`, same `--slice`, same dataset
   variant. The harness in this repo accepts `--seed N` to make the slice
   reproducible.
2. **Hold the embedder fixed if possible.** Comparing `MemoryA + MiniLM`
   against `MemoryB + GTE-Qwen2-7B` is comparing embedders, not memories.
   If you must vary the embedder, report both numbers.
3. **Hold `top_k` fixed.** A system that wins at `top_k=10` may lose at
   `top_k=3`. Report multiple `k`s if there's any doubt.
4. **Report the same metrics in the same order.** Don't cherry-pick the
   metric where your system wins.
5. **Show per-category breakdowns.** A 2-point overall lead that comes
   entirely from `single-session-assistant` while losing `multi-session` is
   not actually an improvement for most use cases.

---

## 9. Worked example: TALM on the full LongMemEval_S split

```bash
.venv/bin/python showcase/longmemeval_real.py --slice 500 --top-k 5 --seed 42 --json
```

Result on `longmemeval_s_cleaned.json` (run 2026-04-09, ~11 minutes on CPU):

```
session_recall@5:  85.9%
session_hit@1:     76.0%
Avg latency:       1.44 s/instance       Total: ~11 min
Instances: 470 answerable (30 abstention excluded by default)
Avg sessions/instance: ~47 (only ~2 contain the answer)

By question type:
  single-session-assistant   n= 56   recall=98.2%   hit@1=98.2%
  knowledge-update           n= 72   recall=87.5%   hit@1=77.8%
  multi-session              n=121   recall=85.7%   hit@1=86.0%
  single-session-preference  n= 30   recall=83.3%   hit@1=50.0%
  temporal-reasoning         n=127   recall=82.7%   hit@1=73.2%
  single-session-user        n= 64   recall=81.2%   hit@1=53.1%
```

**System under test:** all-MiniLM-L6-v2 (22M params) + zvec (HNSW, cosine).

**Comparison vs. LongMemEval paper baselines (full S split):**

| System | Embedder size | session_recall@5 |
|---|---:|---:|
| BM25 | — | ~70% |
| Contriever | 110M | ~76% |
| Stella V5 | 1.5B | ~83% |
| GTE-Qwen2-7B | 7B | ~85% |
| **TALM (this run)** | **22M** | **85.9%** |

TALM matches GTE-Qwen2-7B with a 320× smaller embedder.

**Where the failures live.** The weakest category is `single-session-user`
(81.2% recall, 53.1% hit@1). Inspecting the failing instances reveals a
consistent pattern: the question targets one sentence buried in a session
whose dominant topic is something else.

Example failure (`question_id=f4f1d8a4`):
- Question: *"Who gave me a new stand mixer as a birthday gift?"*
- Evidence session: a 12-turn conversation about **caramel pastry recipes**,
  with one user turn that mentions in passing: *"I actually got my new stand
  mixer as a birthday gift from my sister last month, and it's been a
  game-changer for making caramel..."*
- Result: the session's mean embedding is dominated by caramel/pastry/baking
  terms; the brief gift mention is washed out and the session ranks below
  five other dessert-related sessions.

This is exactly the failure mode LongMemEval addresses with **fact-augmented
key expansion** (extract atomic facts from each session and use them as
additional retrieval keys). Adding turn-level indexing or fact extraction
would lift the `single-session-user` numbers significantly without affecting
the categories where session-level indexing already wins.

---

## 10. Known issues and pitfalls

- **TALM filters retrieval by `tree_depth`.** Forgetting to pin records and
  queries to the same depth produces silent zero-recall.
- **Sentence-transformers downloads model weights on first use.** If
  benchmarking in a sandbox without network, pre-warm the model cache.
- **`question_id` ending in `_abs`** has empty `answer_session_ids` — divide-by-zero
  in recall calculation if you don't filter them out.
- **Don't index the question itself.** Some folk implementations accidentally
  insert the question into the haystack and then "retrieve" it, producing
  artificially high scores.
- **Watch the temp-dir cleanup race.** Zvec's RocksDB backend flushes
  asynchronously; `shutil.rmtree` immediately after closing can produce
  scary-looking but harmless `IOError: No such file or directory` lines.

---

## 11. Reference

- **Paper:** Wu et al., *LongMemEval: Benchmarking Chat Assistants on
  Long-Term Interactive Memory*, ICLR 2025. https://arxiv.org/abs/2410.10813
- **Code:** https://github.com/xiaowu0162/LongMemEval
- **Cleaned datasets:** https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
- **Worked harness in this repo:**
  - `showcase/longmemeval_bench.py` — synthetic 4-category benchmark
  - `showcase/longmemeval_real.py` — real LongMemEval_S/Oracle adapter
