"""LongMemEval-inspired retrieval benchmark for TALM long-term memory.

Adapts the LongMemEval methodology (Wu et al., ICLR 2025; arXiv:2410.10813)
to TALM's code-generation memory. LongMemEval evaluates chat-history QA across
five abilities; we translate four of them to TALM's domain:

  single_hop   <- information_extraction   (paraphrased query of one stored task)
  multi_hop    <- multi_session_reasoning  (query needs >=2 stored tasks)
  update       <- knowledge_updates        (newer record should beat older one)
  abstention   <- abstention               (unrelated query should retrieve nothing)

We skip LongMemEval's `temporal_reasoning` because TALM's memory has no
timestamps -- that gap is reported explicitly rather than papered over.

Pipeline (mirrors LongMemEval's indexing/retrieval/reading decomposition):
  indexing  -> seed MemoryRecords through ZvecAdapter (production code path)
  retrieval -> ZvecAdapter.retrieve with threshold=0 to capture full ranking
  reading   -> not exercised here; this is a retrieval-only microbenchmark

Run:
    .venv/bin/python showcase/longmemeval_bench.py
    .venv/bin/python showcase/longmemeval_bench.py --threshold 0.6
    .venv/bin/python showcase/longmemeval_bench.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

# Silence the chatty INFO logging from TALM internals so the report stays readable.
logging.basicConfig(level=logging.WARNING)

from talm.core.entities import MemoryRecord, TALMConfig
from talm.core.interfaces import ILLMClient
from talm.infrastructure.embedder_st import SentenceTransformerEmbedder
from talm.infrastructure.vector_db_zvec import ZvecAdapter
from talm.memory.manager import MemoryManager


class _UnusedLLM(ILLMClient):
    """Stub LLM: this benchmark never calls update()/consolidate(), so the LLM
    on MemoryManager is never invoked. Raises loudly if that assumption breaks."""

    async def generate(self, system_prompt: str, user_prompt: str,
                       temperature: float = 0.0, max_tokens: int = 8192) -> str:
        raise RuntimeError("LLM should not be called in retrieval-only benchmark")

TREE_DEPTH = 0  # all seeds + queries live at depth 0 (TALM filters by depth)

# ---------------------------------------------------------------------------
# Seed memory: 12 records, each tagged with a stable seed_id in metadata.
# These represent past TALM runs whose solutions were stored after success.
# ---------------------------------------------------------------------------

SEEDS: list[tuple[str, str, str]] = [
    # (seed_id, task_description, generated_code_summary)
    ("S1",
     "Write a Python function is_palindrome(s) that returns True if the string "
     "is a palindrome, ignoring case and non-alphanumeric characters.",
     "def is_palindrome(s):\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]"),
    ("S2",
     "Write a function fibonacci(n) that returns a list with the first n "
     "Fibonacci numbers starting from 0, 1.",
     "def fibonacci(n):\n    a, b, out = 0, 1, []\n    for _ in range(n):\n        out.append(a); a, b = b, a + b\n    return out"),
    ("S3",
     "Write a function is_prime(n) that returns True if n is a prime number.",
     "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True"),
    ("S4",
     "Implement binary_search(arr, target) on a sorted list, returning the "
     "index of target or -1 if not found.",
     "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == target: return mid\n        if arr[mid] < target: lo = mid+1\n        else: hi = mid-1\n    return -1"),
    ("S5",
     "Write merge_sorted(a, b) that merges two pre-sorted lists into one "
     "sorted list in O(n+m) time.",
     "def merge_sorted(a, b):\n    i=j=0; out=[]\n    while i<len(a) and j<len(b):\n        if a[i]<=b[j]: out.append(a[i]); i+=1\n        else: out.append(b[j]); j+=1\n    return out + a[i:] + b[j:]"),
    ("S6",
     "Write word_frequency(text) that returns a dict mapping each lowercase "
     "word in text to its number of occurrences.",
     "from collections import Counter\ndef word_frequency(text):\n    return dict(Counter(text.lower().split()))"),
    # --- Update pair: S7 is the OLD approach, S8 is the NEW approach ---
    ("S7",
     "Read a CSV file into a list of dictionaries (one dict per row, keyed "
     "by header). Use the csv module's reader and build the dicts manually.",
     "import csv\ndef read_csv(path):\n    with open(path) as f:\n        rows = list(csv.reader(f))\n    header = rows[0]\n    return [dict(zip(header, r)) for r in rows[1:]]"),
    ("S8",
     "Read a CSV file into a list of dictionaries. Prefer csv.DictReader "
     "for cleaner code and built-in header handling. Supersedes the manual approach.",
     "import csv\ndef read_csv(path):\n    with open(path, newline='') as f:\n        return list(csv.DictReader(f))"),
    # ---
    ("S9",
     "Save a Python dictionary to disk as a JSON file with pretty-printing.",
     "import json\ndef save_json(obj, path):\n    with open(path, 'w') as f:\n        json.dump(obj, f, indent=2)"),
    ("S10",
     "Make an HTTP GET request to a URL and parse the JSON response body "
     "using the requests library.",
     "import requests\ndef fetch_json(url):\n    r = requests.get(url, timeout=10)\n    r.raise_for_status()\n    return r.json()"),
    ("S11",
     "Implement breadth-first search (BFS) over an adjacency-list graph, "
     "returning the order in which nodes are visited from a start node.",
     "from collections import deque\ndef bfs(graph, start):\n    seen={start}; order=[]; q=deque([start])\n    while q:\n        n=q.popleft(); order.append(n)\n        for nb in graph.get(n, []):\n            if nb not in seen: seen.add(nb); q.append(nb)\n    return order"),
    ("S12",
     "Implement depth-first search (DFS) over an adjacency-list graph using "
     "an explicit stack, returning the visitation order from a start node.",
     "def dfs(graph, start):\n    seen=set(); order=[]; stack=[start]\n    while stack:\n        n=stack.pop()\n        if n in seen: continue\n        seen.add(n); order.append(n)\n        stack.extend(reversed(graph.get(n, [])))\n    return order"),
]

# ---------------------------------------------------------------------------
# Test queries: each tagged with category and the seed_id(s) we expect to retrieve.
# For abstention queries, expected_ids is empty -- the system should return nothing.
# ---------------------------------------------------------------------------

@dataclass
class TestQuery:
    qid: str
    category: str
    text: str
    expected_ids: list[str]  # empty -> abstention
    notes: str = ""


QUERIES: list[TestQuery] = [
    # --- single_hop: paraphrased query of one stored task ----------------
    TestQuery("Q1", "single_hop",
              "Check whether a phrase reads the same forwards and backwards, "
              "ignoring punctuation and case.",
              ["S1"]),
    TestQuery("Q2", "single_hop",
              "Determine whether a given integer is a prime number.",
              ["S3"]),
    TestQuery("Q3", "single_hop",
              "Combine two already-sorted arrays into a single sorted array.",
              ["S5"]),
    TestQuery("Q4", "single_hop",
              "Count the number of occurrences of every word in a body of text.",
              ["S6"]),
    TestQuery("Q5", "single_hop",
              "Persist a Python dictionary to a file in JSON format.",
              ["S9"]),

    # --- multi_hop: query needs >=2 stored tasks -------------------------
    TestQuery("Q6", "multi_hop",
              "Print the first 20 Fibonacci numbers and indicate which of "
              "them are prime.",
              ["S2", "S3"]),
    TestQuery("Q7", "multi_hop",
              "Build a graph traversal utility that supports both "
              "breadth-first and depth-first exploration over an adjacency list.",
              ["S11", "S12"]),
    TestQuery("Q8", "multi_hop",
              "Download a JSON document from a URL and save it to a local file.",
              ["S10", "S9"]),

    # --- update: newer record should rank above older -------------------
    TestQuery("Q9", "update",
              "Parse a CSV file into a list of row dictionaries.",
              ["S8", "S7"],
              notes="S8 (DictReader) supersedes S7 (manual). Newer should rank first."),
    TestQuery("Q10", "update",
              "Load a comma-separated-values file and turn each row into a dict "
              "keyed by column header.",
              ["S8", "S7"],
              notes="Same update test, different paraphrase."),

    # --- abstention: unrelated to anything in memory --------------------
    TestQuery("Q11", "abstention",
              "Train a convolutional neural network on the CIFAR-10 dataset "
              "using PyTorch.",
              []),
    TestQuery("Q12", "abstention",
              "Render a 3D scene with OpenGL fragment shaders.",
              []),
    TestQuery("Q13", "abstention",
              "Write a SQL query that computes a 7-day rolling average over "
              "a sales table.",
              []),
    TestQuery("Q14", "abstention",
              "Implement a Kafka consumer with exactly-once delivery semantics.",
              []),
]


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    qid: str
    category: str
    expected: list[str]
    retrieved: list[tuple[str, float]]   # [(seed_id, similarity), ...] full ranking
    above_threshold: list[tuple[str, float]]  # filtered by prod threshold


async def seed_memory(manager: MemoryManager, embedder: SentenceTransformerEmbedder) -> dict[str, str]:
    """Insert all seed records and return a map: seed_id -> doc_id."""
    seed_id_by_doc: dict[str, str] = {}
    for seed_id, task, code in SEEDS:
        record = MemoryRecord(
            task_description=task,
            reasoning_trace=f"[seed:{seed_id}] solution recorded from prior TALM run.",
            generated_code=code,
            tree_depth=TREE_DEPTH,
            metadata={"seed_id": seed_id},
        )
        # Insert directly through the vector_db so we skip consolidation
        # (consolidation could merge records and break our ground truth).
        text = f"{record.task_description}\n{record.reasoning_trace}"
        embedding = await embedder.embed(text)
        doc_id = await manager._db.insert(record, embedding)  # noqa: SLF001
        seed_id_by_doc[doc_id] = seed_id
    return seed_id_by_doc


async def run_query(
    query: TestQuery,
    embedder: SentenceTransformerEmbedder,
    vector_db: ZvecAdapter,
    prod_threshold: float,
    fetch_k: int = 10,
) -> QueryResult:
    """Run a single retrieval query and tag results with their seed_id."""
    embedding = await embedder.embed(query.text)
    # Threshold=0 to see the full ranking; we apply prod_threshold ourselves below.
    raw = await vector_db.retrieve(
        query_embedding=embedding,
        tree_depth=TREE_DEPTH,
        top_k=fetch_k,
        threshold=0.0,
    )
    full_ranking: list[tuple[str, float]] = []
    for record, sim in raw:
        seed_id = record.metadata.get("seed_id", "?")
        full_ranking.append((seed_id, sim))
    above = [(sid, s) for sid, s in full_ranking if s >= prod_threshold]
    return QueryResult(
        qid=query.qid,
        category=query.category,
        expected=query.expected_ids,
        retrieved=full_ranking,
        above_threshold=above,
    )


# ---------------------------------------------------------------------------
# Per-category metrics
# ---------------------------------------------------------------------------

def metrics_single_hop(results: list[QueryResult], top_k: int) -> dict[str, float]:
    """Hit@1 and Hit@k for queries with exactly one expected record."""
    hit1 = 0
    hitk = 0
    n = len(results)
    for r in results:
        topk_ids = [sid for sid, _ in r.retrieved[:top_k]]
        if topk_ids and topk_ids[0] == r.expected[0]:
            hit1 += 1
        if r.expected[0] in topk_ids:
            hitk += 1
    return {"n": n, f"hit@1": hit1 / n, f"hit@{top_k}": hitk / n}


def metrics_multi_hop(results: list[QueryResult], top_k: int) -> dict[str, float]:
    """Per-query recall and all-found rate for multi-target queries."""
    recalls = []
    all_found = 0
    for r in results:
        topk_ids = set(sid for sid, _ in r.retrieved[:top_k])
        found = topk_ids & set(r.expected)
        recalls.append(len(found) / len(r.expected))
        if len(found) == len(r.expected):
            all_found += 1
    n = len(results)
    avg_recall = sum(recalls) / n if n else 0.0
    return {"n": n, "avg_recall": avg_recall, "all_found_rate": all_found / n}


def metrics_update(results: list[QueryResult], top_k: int) -> dict[str, float]:
    """For each update query, was the newer record (expected[0]) ranked above the older (expected[1])?"""
    correct = 0
    both_present = 0
    for r in results:
        new_id, old_id = r.expected[0], r.expected[1]
        topk = [sid for sid, _ in r.retrieved[:top_k]]
        if new_id in topk and old_id in topk:
            both_present += 1
            if topk.index(new_id) < topk.index(old_id):
                correct += 1
        elif new_id in topk and old_id not in topk:
            # Newer is present alone -- counts as correct preference
            correct += 1
            both_present += 1
    n = len(results)
    return {
        "n": n,
        "newer_preferred_rate": correct / n if n else 0.0,
        "both_present_rate": both_present / n if n else 0.0,
    }


def metrics_abstention(results: list[QueryResult]) -> dict[str, float]:
    """Correct refusal: above_threshold should be empty for abstention queries."""
    n = len(results)
    correct = sum(1 for r in results if not r.above_threshold)
    return {"n": n, "correct_refusal_rate": correct / n if n else 0.0}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_ranking(retrieved: list[tuple[str, float]], k: int = 5) -> str:
    return ", ".join(f"{sid}:{sim:.2f}" for sid, sim in retrieved[:k])


def print_report(
    results: list[QueryResult],
    top_k: int,
    prod_threshold: float,
) -> dict[str, Any]:
    by_cat: dict[str, list[QueryResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    print("=" * 78)
    print("LongMemEval-TALM Retrieval Benchmark")
    print("=" * 78)
    print(f"Seeds inserted: {len(SEEDS)}     Queries: {len(QUERIES)}     "
          f"top_k={top_k}     prod_threshold={prod_threshold}")
    print()

    # Per-query detail table
    print(f"{'QID':<5}{'Category':<13}{'Expected':<12}{'Top-5 ranking (seed_id:sim)':<45}")
    print("-" * 78)
    for r in results:
        exp = ",".join(r.expected) if r.expected else "(none)"
        print(f"{r.qid:<5}{r.category:<13}{exp:<12}{format_ranking(r.retrieved, 5)}")
    print()

    # Aggregate metrics
    summary: dict[str, Any] = {}

    if "single_hop" in by_cat:
        m = metrics_single_hop(by_cat["single_hop"], top_k)
        summary["single_hop"] = m
        print(f"single_hop  (n={m['n']}): "
              f"hit@1={m['hit@1']:.0%}   hit@{top_k}={m[f'hit@{top_k}']:.0%}")

    if "multi_hop" in by_cat:
        m = metrics_multi_hop(by_cat["multi_hop"], top_k)
        summary["multi_hop"] = m
        print(f"multi_hop   (n={m['n']}): "
              f"avg_recall={m['avg_recall']:.0%}   all_found={m['all_found_rate']:.0%}")

    if "update" in by_cat:
        m = metrics_update(by_cat["update"], top_k)
        summary["update"] = m
        print(f"update      (n={m['n']}): "
              f"newer_preferred={m['newer_preferred_rate']:.0%}   "
              f"both_present={m['both_present_rate']:.0%}")

    if "abstention" in by_cat:
        m = metrics_abstention(by_cat["abstention"])
        summary["abstention"] = m
        print(f"abstention  (n={m['n']}): "
              f"correct_refusal={m['correct_refusal_rate']:.0%}")

    print()
    print("Notes:")
    print("  * temporal_reasoning category is omitted: TALM memory has no")
    print("    timestamp field, so the LongMemEval temporal probes do not apply.")
    print("  * 'update' has no recency tiebreaker in TALM today -- ranking")
    print("    falls back to pure cosine similarity. Score reflects whether")
    print("    the newer phrasing alone happens to embed closer to the query.")
    print("=" * 78)
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Production similarity threshold for abstention check (default: 0.75)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Top-k for recall metrics (default: 3)")
    parser.add_argument("--json", action="store_true",
                        help="Also emit machine-readable JSON summary at the end.")
    args = parser.parse_args()

    config = TALMConfig()  # defaults are fine; we use embedder + vector_db directly
    tmp_dir = tempfile.mkdtemp(prefix="talm_longmemeval_")
    persist_dir = f"{tmp_dir}/db"

    try:
        embedder = SentenceTransformerEmbedder(model_name=config.embedding_model)
        vector_db = ZvecAdapter(persist_dir=persist_dir, embedding_dim=embedder.get_dimension())
        memory = MemoryManager(
            vector_db=vector_db,
            embedder=embedder,
            llm=_UnusedLLM(),  # never invoked: we don't call update() in this bench
            config=config,
        )

        await seed_memory(memory, embedder)

        results: list[QueryResult] = []
        for q in QUERIES:
            results.append(await run_query(q, embedder, vector_db, args.threshold))

        summary = print_report(results, top_k=args.top_k, prod_threshold=args.threshold)

        if args.json:
            print()
            print(json.dumps({"summary": summary, "config": {
                "threshold": args.threshold, "top_k": args.top_k,
                "n_seeds": len(SEEDS), "n_queries": len(QUERIES),
            }}, indent=2))

    finally:
        # Close the Zvec collection before removing the dir to avoid
        # rocksdb flush-into-deleted-dir errors during teardown.
        try:
            if vector_db._collection is not None:  # noqa: SLF001
                vector_db._collection.close()  # noqa: SLF001
        except Exception:
            pass
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
