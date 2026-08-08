"""Extended evaluation suite (Part 5 of the RAG evolution spec).

Adds Recall@K, MRR, NDCG@K, latency percentiles, and cache hit rate on top
of the original eval.py's Precision@5 / Hit Rate@5 / keyword-faithfulness —
which stays untouched at src/evaluation/eval.py for backward compatibility.

IMPORTANT — lexical heuristic vs. semantic evaluation:
`faithfulness_lexical` (this file, like the original eval.py) is keyword
overlap against `expected_facts`. It is NOT a semantic faithfulness check —
a correct, well-paraphrased answer can score low, and a wrong answer that
happens to repeat catalog vocabulary can score high. It is included because
it's free (no extra LLM calls) and catches gross hallucination.

`faithfulness_semantic`, `answer_relevance`, `context_relevance`, and
`hallucination_detected` are LLM-as-judge metrics — genuinely semantic, but
they cost one extra Groq call per test case and are therefore OFF by
default (`--llm-judge` to enable). When not measured, this script reports
them as "not_measured" rather than omitting them or defaulting to 0 —
never fabricate a metric that wasn't actually computed.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import httpx
from loguru import logger

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DEFAULT_GROUND_TRUTH = "src/evaluation/ground_truth_100.json"
THRESHOLDS_PATH = "src/evaluation/thresholds.json"
REPORTS_DIR = Path("data/reports")

_API_KEY = os.getenv("BACKEND_API_KEY", "")
REQUEST_HEADERS = {"X-API-Key": _API_KEY} if _API_KEY else {}


# ── Ground truth normalization ────────────────────────────
def load_ground_truth(path: str) -> list[dict]:
    """Accepts either the new structured format (ground_truth_100.json:
    query/level/expected_product_ids/reference_answer/expected_facts) or
    the original legacy format (ground_truth.json:
    query/expected_product_ids/ideal_response_keywords), normalizing both
    to the same shape."""
    with open(path, "r") as f:
        raw_cases = json.load(f)

    normalized = []
    for case in raw_cases:
        normalized.append({
            "query": case["query"],
            "level": case.get("level", 0),
            "level_name": case.get("level_name", "legacy"),
            "expected_product_ids": [str(x) for x in case["expected_product_ids"]],
            "expected_facts": case.get("expected_facts", case.get("ideal_response_keywords", [])),
            "reference_answer": case.get("reference_answer", ""),
        })
    return normalized


# ── Backend query ──────────────────────────────────────────
def query_backend(query: str, limit: int = 5) -> dict:
    response = httpx.post(
        f"{BACKEND_URL}/api/v1/ask",
        json={"q": query, "limit": limit},
        headers=REQUEST_HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


# ── Retrieval metrics ──────────────────────────────────────
def precision_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for rid in top_k if rid in expected) / k


def hit_rate_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """None (undefined) when there's nothing relevant to hit — matches
    recall/MRR/NDCG's treatment, so negative/adversarial cases don't drag
    the average down just for having no expected products. Use
    `refusal_correct` to score those cases instead."""
    if not expected:
        return None
    top_k = set(retrieved[:k])
    return 1.0 if top_k & expected else 0.0


# Substring the LLM is instructed to use verbatim when context doesn't
# support the query (see src/backend/services/llm.py:NO_MATCH_MESSAGE).
# Checked as a substring, not an exact import, to keep the eval suite
# decoupled from backend internals — it evaluates the HTTP API like a
# real client would, not the implementation behind it.
REFUSAL_PHRASE = "cannot find a product matching"


def refusal_correct(answer: str, expected: set[str]) -> float | None:
    """Only meaningful for negative/adversarial cases (expected is empty):
    did the system correctly decline rather than hallucinate a product?"""
    if expected:
        return None
    return 1.0 if REFUSAL_PHRASE.lower() in answer.lower() else 0.0


def recall_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """Of all relevant products for this query, what fraction were
    retrieved in the top K? Undefined (reported as None) for negative/
    adversarial cases where expected is empty by design."""
    if not expected:
        return None
    top_k = set(retrieved[:k])
    return len(top_k & expected) / len(expected)


def mrr(retrieved: list[str], expected: set[str]) -> float:
    if not expected:
        return None
    for rank, rid in enumerate(retrieved, start=1):
        if rid in expected:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """Binary relevance NDCG@K (relevant=1 if in expected_product_ids)."""
    if not expected:
        return None
    top_k = retrieved[:k]
    dcg = sum(
        (1.0 if rid in expected else 0.0) / math.log2(idx + 2)
        for idx, rid in enumerate(top_k)
    )
    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def faithfulness_lexical(answer: str, expected_facts: list[str]) -> float:
    if not expected_facts:
        return None
    answer_lower = answer.lower()
    matched = sum(1 for fact in expected_facts if fact.lower() in answer_lower)
    return matched / len(expected_facts)


# ── LLM-as-judge (optional, --llm-judge) ──────────────────
_JUDGE_PROMPT = """You are grading a RAG system's answer. Respond with ONLY a JSON object, no prose.

QUERY: {query}

RETRIEVED CONTEXT (what the system was allowed to use):
{context}

GENERATED ANSWER:
{answer}

Score each on a 0.0-1.0 scale:
- "faithfulness_semantic": does every claim in the answer follow from the context (1.0) or does it invent/contradict facts (0.0)?
- "answer_relevance": does the answer actually address the query (1.0) or is it off-topic (0.0)?
- "context_relevance": was the retrieved context relevant to the query (1.0) or mostly irrelevant (0.0)?
- "hallucination_detected": true/false — true if the answer states any fact NOT present in the context.

Return exactly: {{"faithfulness_semantic": <float>, "answer_relevance": <float>, "context_relevance": <float>, "hallucination_detected": <bool>}}"""


def llm_judge(query: str, context: str, answer: str, groq_client) -> dict | None:
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": _JUDGE_PROMPT.format(query=query, context=context, answer=answer)}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"LLM judge failed, treating as not_measured: {e}")
        return None


# ── Latency percentiles ────────────────────────────────────
def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(int(len(sorted_vals) * p), len(sorted_vals) - 1)
    return sorted_vals[idx]


# ── Cache hit rate (second pass over the same queries) ─────
def measure_cache_hit_rate(cases: list[dict]) -> float:
    hits = 0
    for case in cases:
        try:
            data = query_backend(case["query"])
            if data.get("source") == "cache":
                hits += 1
        except Exception as e:
            logger.warning(f"Cache-hit-rate re-query failed for '{case['query']}': {e}")
    return hits / len(cases) if cases else 0.0


# ── Main evaluation loop ───────────────────────────────────
def run_evaluation(ground_truth_path: str, use_llm_judge: bool, top_k: int) -> dict:
    cases = load_ground_truth(ground_truth_path)
    groq_client = None
    if use_llm_judge:
        from groq import Groq
        from config.settings import settings
        groq_client = Groq(api_key=settings.GROQ_API_KEY)

    results = []
    for idx, case in enumerate(cases):
        logger.info(f"[{idx + 1}/{len(cases)}] ({case['level_name']}) '{case['query']}'")
        expected = set(case["expected_product_ids"])

        try:
            data = query_backend(case["query"], limit=top_k)
        except Exception as e:
            logger.error(f"Query failed: '{case['query']}' — {e}")
            results.append({**case, "error": str(e)})
            continue

        retrieved_ids = [str(p["id"]) for p in data.get("retrieved_products", [])]
        answer = data.get("answer", "")
        source = data.get("source", "unknown")
        timing = data.get("timing", {})

        row = {
            "query": case["query"],
            "level": case["level"],
            "level_name": case["level_name"],
            "source": source,
            "retrieved_ids": retrieved_ids,
            "expected_ids": sorted(expected),
            "precision_at_k": precision_at_k(retrieved_ids, expected, top_k),
            "hit_rate_at_k": hit_rate_at_k(retrieved_ids, expected, top_k),
            "recall_at_k": recall_at_k(retrieved_ids, expected, top_k),
            "mrr": mrr(retrieved_ids, expected),
            "ndcg_at_k": ndcg_at_k(retrieved_ids, expected, top_k),
            "refusal_correct": refusal_correct(answer, expected),
            "faithfulness_lexical": faithfulness_lexical(answer, case["expected_facts"]),
            "total_ms": timing.get("total_ms", 0.0),
            "search_ms": timing.get("search_ms", 0.0),
            "llm_ms": timing.get("llm_ms", 0.0),
        }

        if use_llm_judge and groq_client is not None:
            context = "\n".join(
                f"- {p.get('title', '')} (${p.get('price', 0.0)}, {p.get('category', '')})"
                for p in data.get("retrieved_products", [])
            )
            judged = llm_judge(case["query"], context, answer, groq_client)
            if judged:
                row["faithfulness_semantic"] = judged.get("faithfulness_semantic")
                row["answer_relevance"] = judged.get("answer_relevance")
                row["context_relevance"] = judged.get("context_relevance")
                row["hallucination_detected"] = judged.get("hallucination_detected")
            else:
                row["faithfulness_semantic"] = "not_measured"
                row["answer_relevance"] = "not_measured"
                row["context_relevance"] = "not_measured"
                row["hallucination_detected"] = "not_measured"
        else:
            row["faithfulness_semantic"] = "not_measured"
            row["answer_relevance"] = "not_measured"
            row["context_relevance"] = "not_measured"
            row["hallucination_detected"] = "not_measured"

        results.append(row)

    cache_hit_rate = measure_cache_hit_rate(cases)

    return {"results": results, "cache_hit_rate": cache_hit_rate}


# ── Aggregation ─────────────────────────────────────────────
def _avg(values: list) -> float | None:
    clean = [v for v in values if isinstance(v, (int, float))]
    return sum(clean) / len(clean) if clean else None


def aggregate(results: list[dict]) -> dict:
    return {
        "count": len(results),
        "precision_at_k": _avg([r.get("precision_at_k") for r in results]),
        "hit_rate_at_k": _avg([r.get("hit_rate_at_k") for r in results]),
        "recall_at_k": _avg([r.get("recall_at_k") for r in results]),
        "mrr": _avg([r.get("mrr") for r in results]),
        "ndcg_at_k": _avg([r.get("ndcg_at_k") for r in results]),
        "refusal_correct": _avg([r.get("refusal_correct") for r in results]),
        "faithfulness_lexical": _avg([r.get("faithfulness_lexical") for r in results]),
        "faithfulness_semantic": _avg([r.get("faithfulness_semantic") for r in results]) or "not_measured",
        "answer_relevance": _avg([r.get("answer_relevance") for r in results]) or "not_measured",
        "context_relevance": _avg([r.get("context_relevance") for r in results]) or "not_measured",
        "latency_p50_ms": percentile([r["total_ms"] for r in results if "total_ms" in r], 0.50),
        "latency_p95_ms": percentile([r["total_ms"] for r in results if "total_ms" in r], 0.95),
        "latency_p99_ms": percentile([r["total_ms"] for r in results if "total_ms" in r], 0.99),
    }


def print_report(overall: dict, by_level: dict, cache_hit_rate: float) -> None:
    print("\n" + "=" * 100)
    print(f"{'OVERALL EVALUATION SUMMARY':^100}")
    print("=" * 100)
    for key, value in overall.items():
        if isinstance(value, float):
            print(f"  {key:<24}: {value:.4f}")
        else:
            print(f"  {key:<24}: {value}")
    print(f"  {'cache_hit_rate':<24}: {cache_hit_rate:.4f}")

    print("\n" + "-" * 100)
    print(f"{'PER-DIFFICULTY-LEVEL BREAKDOWN':^100}")
    print("-" * 100)
    print(f"{'Level':<20}{'N':>5}{'Recall@K':>12}{'MRR':>10}{'NDCG@K':>10}{'HitRate@K':>12}")
    for level_name, metrics in sorted(by_level.items(), key=lambda kv: kv[1]["count"], reverse=True):
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, float) else "n/a"
        print(
            f"{level_name:<20}{metrics['count']:>5}"
            f"{fmt(metrics['recall_at_k']):>12}{fmt(metrics['mrr']):>10}"
            f"{fmt(metrics['ndcg_at_k']):>10}{fmt(metrics['hit_rate_at_k']):>12}"
        )
    print("=" * 100)


def check_thresholds(overall: dict, thresholds_path: str) -> bool:
    with open(thresholds_path, "r") as f:
        thresholds = json.load(f)

    passed = True
    print("\nQuality gate:")
    for metric, spec in thresholds.items():
        if metric.startswith("_"):
            continue
        actual = overall.get(metric)
        required = spec["value"]
        if not isinstance(actual, (int, float)):
            print(f"  [SKIP] {metric}: not measured")
            continue
        ok = actual >= required
        passed = passed and ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {metric}: {actual:.4f} >= {required} ({spec['reason']})")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Extended RAG evaluation suite.")
    parser.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--catalog-version", default=os.getenv("CATALOG_VERSION", "unspecified"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm-judge", action="store_true", help="Enable LLM-as-judge metrics (extra Groq calls).")
    parser.add_argument("--gate", action="store_true", help="Exit non-zero if quality thresholds fail (CI mode).")
    args = parser.parse_args()

    logger.info(f"Running eval_v2 against {args.ground_truth} (catalog_version={args.catalog_version}, llm_judge={args.llm_judge})")

    run = run_evaluation(args.ground_truth, use_llm_judge=args.llm_judge, top_k=args.top_k)
    results = run["results"]

    overall = aggregate(results)
    by_level: dict[str, dict] = {}
    for level_name in {r["level_name"] for r in results}:
        level_results = [r for r in results if r["level_name"] == level_name]
        by_level[level_name] = aggregate(level_results)

    print_report(overall, by_level, run["cache_hit_rate"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"eval_report_{args.catalog_version}.json"
    report_path.write_text(json.dumps({
        "catalog_version": args.catalog_version,
        "ground_truth": args.ground_truth,
        "llm_judge_enabled": args.llm_judge,
        "overall": overall,
        "by_level": by_level,
        "cache_hit_rate": run["cache_hit_rate"],
        "raw_results": results,
    }, indent=2, default=str))
    logger.info(f"Report written to {report_path}")

    if args.gate:
        gate_passed = check_thresholds(overall, THRESHOLDS_PATH)
        if not gate_passed:
            logger.error("Quality gate FAILED.")
            sys.exit(1)
        logger.info("Quality gate PASSED.")


if __name__ == "__main__":
    main()
