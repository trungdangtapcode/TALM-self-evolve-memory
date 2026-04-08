"""Run a slice of the REAL LongMemEval dataset against TALM's memory.

This wires the official LongMemEval JSON (Wu et al., ICLR 2025) into TALM's
production embedder + vector_db so the same retrieval code path used at
runtime is what gets measured.

Pipeline (per instance):
  1. Clear vector_db
  2. Concatenate each session's turns into one text blob (session-granularity
     indexing -- the strongest baseline reported in the paper)
  3. Batch-embed sessions and insert as MemoryRecords (one per session)
  4. Embed the question, retrieve top-k via the production retrieve() path
  5. Score: which retrieved session_ids overlap with answer_session_ids

Metrics:
  * session-recall@k       LongMemEval's primary retrieval metric
  * session-hit@1          top-1 contains an evidence session
  * per question_type breakdown
  * abstention correct-refusal (if --include-abstention)

Usage:
    .venv/bin/python showcase/longmemeval_real.py                      # default: 25 of S
    .venv/bin/python showcase/longmemeval_real.py --variant oracle     # smoke test
    .venv/bin/python showcase/longmemeval_real.py --slice 50 --top-k 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.WARNING)
# Silence the embedding adapter's per-instance INFO logs.
logging.getLogger("talm.infrastructure.embedder_st").setLevel(logging.ERROR)
logging.getLogger("talm.infrastructure.vector_db_zvec").setLevel(logging.ERROR)

from talm.core.entities import MemoryRecord, TALMConfig
from talm.core.interfaces import ILLMClient
from talm.infrastructure.embedder_st import SentenceTransformerEmbedder
from talm.infrastructure.vector_db_zvec import ZvecAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent
TREE_DEPTH = 0


class _UnusedLLM(ILLMClient):
    async def generate(self, *_a, **_kw) -> str:
        raise RuntimeError("LLM should not be called in retrieval-only benchmark")


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------

DATA_PATHS = {
    "s": REPO_ROOT / "data" / "longmemeval_s_cleaned.json",
    "oracle": REPO_ROOT / "data" / "longmemeval_oracle.json",
}


def load_dataset(variant: str) -> list[dict[str, Any]]:
    path = DATA_PATHS[variant]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download with:\n"
            f"  wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
            f"resolve/main/longmemeval_{'s_cleaned' if variant == 's' else 'oracle'}.json"
            f" -O {path}"
        )
    return json.loads(path.read_text())


def session_to_text(session: list[dict[str, Any]]) -> str:
    """Flatten a session (list of turns) into a single embedding-friendly string."""
    parts = []
    for turn in session:
        role = turn.get("role", "?")
        content = turn.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Per-instance evaluation
# ---------------------------------------------------------------------------

@dataclass
class InstanceResult:
    question_id: str
    question_type: str
    is_abstention: bool
    n_sessions: int
    n_evidence: int
    retrieved_session_ids: list[str]      # top-k from retrieval
    evidence_session_ids: set[str]
    elapsed_s: float

    @property
    def found(self) -> set[str]:
        return set(self.retrieved_session_ids) & self.evidence_session_ids

    @property
    def recall(self) -> float:
        if not self.evidence_session_ids:
            return 0.0
        return len(self.found) / len(self.evidence_session_ids)

    @property
    def hit_at_1(self) -> int:
        if not self.retrieved_session_ids:
            return 0
        return int(self.retrieved_session_ids[0] in self.evidence_session_ids)


async def evaluate_instance(
    instance: dict[str, Any],
    embedder: SentenceTransformerEmbedder,
    make_db: "callable",
    top_k: int,
) -> tuple[InstanceResult, ZvecAdapter]:
    t0 = time.perf_counter()

    sessions: list[list[dict]] = instance["haystack_sessions"]
    session_ids: list[str] = instance["haystack_session_ids"]
    evidence_ids = set(instance.get("answer_session_ids") or [])
    qid = instance["question_id"]
    qtype = instance["question_type"]
    is_abs = qid.endswith("_abs")

    # NOTE: ZvecAdapter.clear() has a pre-existing bug (calls a non-existent
    # _collection.close()), so we create a fresh adapter per instance.
    vector_db = make_db()

    # Batch-embed all sessions in one call (the embedder supports it)
    session_texts = [session_to_text(s) for s in sessions]
    session_embeddings = await embedder.embed_batch(session_texts)

    # Insert each session as a MemoryRecord, tagging session_id in metadata
    for sid, text, emb in zip(session_ids, session_texts, session_embeddings):
        record = MemoryRecord(
            task_description=text[:500],   # short head for human inspection
            reasoning_trace="",
            generated_code="",
            tree_depth=TREE_DEPTH,
            metadata={"session_id": sid},
        )
        await vector_db.insert(record, emb)

    # Retrieve for the question
    q_emb = await embedder.embed(instance["question"])
    raw = await vector_db.retrieve(
        query_embedding=q_emb,
        tree_depth=TREE_DEPTH,
        top_k=top_k,
        threshold=0.0,   # we want raw ranking; abstention is scored separately
    )
    retrieved_ids = [rec.metadata.get("session_id", "?") for rec, _ in raw]

    result = InstanceResult(
        question_id=qid,
        question_type=qtype,
        is_abstention=is_abs,
        n_sessions=len(sessions),
        n_evidence=len(evidence_ids),
        retrieved_session_ids=retrieved_ids,
        evidence_session_ids=evidence_ids,
        elapsed_s=time.perf_counter() - t0,
    )
    return result, vector_db


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(results: list[InstanceResult], top_k: int) -> dict[str, Any]:
    answerable = [r for r in results if not r.is_abstention]
    abstention = [r for r in results if r.is_abstention]

    overall = {
        "n_total": len(results),
        "n_answerable": len(answerable),
        "n_abstention": len(abstention),
        "session_recall@k": 0.0,
        "session_hit@1": 0.0,
        "avg_latency_s": 0.0,
    }
    if answerable:
        overall["session_recall@k"] = sum(r.recall for r in answerable) / len(answerable)
        overall["session_hit@1"] = sum(r.hit_at_1 for r in answerable) / len(answerable)
    if results:
        overall["avg_latency_s"] = sum(r.elapsed_s for r in results) / len(results)

    by_type: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[InstanceResult]] = defaultdict(list)
    for r in answerable:
        grouped[r.question_type].append(r)
    for qtype, items in sorted(grouped.items()):
        by_type[qtype] = {
            "n": len(items),
            "session_recall@k": sum(r.recall for r in items) / len(items),
            "session_hit@1": sum(r.hit_at_1 for r in items) / len(items),
        }

    return {
        "top_k": top_k,
        "overall": overall,
        "by_question_type": by_type,
    }


def print_report(results: list[InstanceResult], summary: dict[str, Any]) -> None:
    top_k = summary["top_k"]
    o = summary["overall"]

    print("=" * 78)
    print(f"LongMemEval-Real (S variant) — TALM retrieval microbenchmark")
    print("=" * 78)
    print(f"Instances: {o['n_total']}  (answerable={o['n_answerable']}, "
          f"abstention={o['n_abstention']})    top_k={top_k}")
    print(f"Avg sessions/instance: {sum(r.n_sessions for r in results)/len(results):.1f}    "
          f"Avg latency: {o['avg_latency_s']:.2f}s/instance")
    print()

    # Per-instance detail (compact)
    print(f"{'qid':<24}{'type':<28}{'sess':>5}{'evid':>5}{'rec':>6}{'h@1':>5}")
    print("-" * 78)
    for r in results:
        marker = " (abs)" if r.is_abstention else ""
        rec_str = "-" if r.is_abstention else f"{r.recall:.2f}"
        h1_str = "-" if r.is_abstention else str(r.hit_at_1)
        print(f"{r.question_id[:23]:<24}{(r.question_type+marker)[:27]:<28}"
              f"{r.n_sessions:>5}{r.n_evidence:>5}{rec_str:>6}{h1_str:>5}")
    print()

    print("--- Aggregate ---")
    print(f"  session_recall@{top_k}: {o['session_recall@k']:.1%}")
    print(f"  session_hit@1:        {o['session_hit@1']:.1%}")
    print()
    if summary["by_question_type"]:
        print("--- By question type (answerable only) ---")
        print(f"  {'type':<32}{'n':>4}{'recall@k':>12}{'hit@1':>10}")
        for qtype, m in summary["by_question_type"].items():
            print(f"  {qtype:<32}{int(m['n']):>4}"
                  f"{m['session_recall@k']:>11.1%} {m['session_hit@1']:>9.1%}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["s", "oracle"], default="s")
    parser.add_argument("--slice", type=int, default=25,
                        help="How many instances to evaluate (default 25 = 5%% of 500)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for instance sampling (0 = first N, no shuffle)")
    parser.add_argument("--include-abstention", action="store_true",
                        help="Include _abs instances in the evaluation")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    print(f"Loading LongMemEval_{args.variant}...", flush=True)
    dataset = load_dataset(args.variant)
    print(f"  loaded {len(dataset)} instances", flush=True)

    # Slice selection
    pool = dataset if args.include_abstention else [d for d in dataset if not d["question_id"].endswith("_abs")]
    if args.seed:
        import random
        rng = random.Random(args.seed)
        rng.shuffle(pool)
    instances = pool[: args.slice]
    print(f"  evaluating {len(instances)} instances", flush=True)

    config = TALMConfig()
    tmp_root = tempfile.mkdtemp(prefix="talm_lme_real_")

    print(f"Loading embedder ({config.embedding_model})...", flush=True)
    embedder = SentenceTransformerEmbedder(model_name=config.embedding_model)
    dim = embedder.get_dimension()

    instance_idx = {"i": 0}

    def make_db() -> ZvecAdapter:
        instance_idx["i"] += 1
        persist_dir = f"{tmp_root}/db_{instance_idx['i']}"
        return ZvecAdapter(persist_dir=persist_dir, embedding_dim=dim)

    try:
        results: list[InstanceResult] = []
        t0 = time.perf_counter()
        for i, inst in enumerate(instances, 1):
            r, _db = await evaluate_instance(inst, embedder, make_db, args.top_k)
            results.append(r)
            print(f"  [{i:>3}/{len(instances)}] {r.question_id[:24]:<25} "
                  f"recall={'%.2f' % r.recall if not r.is_abstention else '-':>5}  "
                  f"({r.elapsed_s:.1f}s)", flush=True)

        elapsed = time.perf_counter() - t0
        print(f"\nDone in {elapsed:.1f}s\n", flush=True)

        summary = aggregate(results, args.top_k)
        print_report(results, summary)

        if args.json:
            print()
            print(json.dumps(summary, indent=2))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
