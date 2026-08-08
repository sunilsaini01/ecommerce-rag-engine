"""Orchestrates the full /ask request: cache -> hybrid retrieval -> context
-> LLM generation -> cache write. Pulled out of the endpoint handler so it
can be a single `@traceable` root span ("rag_pipeline") with the endpoint,
model, cache-hit, and catalog metadata attached at the call site via
`langsmith_extra` — the shape described in README's LangSmith trace tree.

The endpoint (`src/backend/api/v1/endpoints.py`) stays a thin HTTP adapter:
it builds the response envelope from what this function returns.
"""

import json
import time

from loguru import logger

from src.backend.services.search import ProductSearchService
from src.backend.services.cache import CacheService
from src.backend.services.llm import LLMService, NO_MATCH_MESSAGE
from src.backend.services.context_builder import build_context
from src.backend.observability.tracing import traceable, only_fields


@traceable(
    name="rag_pipeline",
    run_type="chain",
    # search_service/cache_service/llm_service are live objects (one holds
    # the Groq client, which holds the real API key) — never let them near
    # the trace serializer. See tracing.only_fields.
    process_inputs=only_fields("query", "limit"),
)
async def run_rag_pipeline(
    query: str,
    limit: int,
    search_service: ProductSearchService,
    cache_service: CacheService,
    llm_service: LLMService,
) -> dict:
    total_start = time.perf_counter()

    # Step 1: Cache check
    cache_start = time.perf_counter()
    cache_result = await cache_service.lookup(query)
    cache_ms = (time.perf_counter() - cache_start) * 1000

    if cache_result["cache_hit"]:
        logger.info(f"[CACHE HIT] query='{query}'")
        total_ms = (time.perf_counter() - total_start) * 1000
        # Cached value is a JSON envelope {"answer": ..., "retrieved_products": [...]}
        # written below on a cache miss. Older cache entries (pre-dating this
        # format) or anything unexpected fall back to treating the raw value
        # as a plain answer string with no product list — never crash on a
        # cache hit just because the stored shape changed.
        try:
            cached_payload = json.loads(cache_result["answer"])
            cached_answer = cached_payload["answer"]
            cached_products = cached_payload.get("retrieved_products", [])
        except (json.JSONDecodeError, TypeError, KeyError):
            cached_answer = cache_result["answer"]
            cached_products = []
        return {
            "answer": cached_answer,
            "source": "cache",
            "retrieved_products": cached_products,
            "timing": {
                "cache_lookup_ms": round(cache_ms, 2),
                "search_ms": 0.0,
                "llm_ms": 0.0,
                "total_ms": round(total_ms, 2),
            },
        }

    # Step 2: Hybrid search
    logger.info(f"[CACHE MISS] query='{query}' — running hybrid search.")
    search_start = time.perf_counter()
    raw_results = await search_service.search(query=query, limit=limit)
    search_ms = (time.perf_counter() - search_start) * 1000

    if not raw_results:
        total_ms = (time.perf_counter() - total_start) * 1000
        return {
            "answer": NO_MATCH_MESSAGE,
            "source": "no_results",
            "retrieved_products": [],
            "timing": {
                "cache_lookup_ms": round(cache_ms, 2),
                "search_ms": round(search_ms, 2),
                "llm_ms": 0.0,
                "total_ms": round(total_ms, 2),
            },
        }

    # Step 3: Build context blob
    context_result = build_context(raw_results)
    context = context_result["context"]

    # Step 4: LLM generation
    llm_start = time.perf_counter()
    answer, generation_ok = await llm_service.generate_answer(query=query, context=context)
    llm_ms = (time.perf_counter() - llm_start) * 1000

    retrieved_products = [
        {
            "id": r["id"],
            "title": r.get("title", ""),
            "category": r.get("category", ""),
            "price": r.get("price", 0.0),
            "rrf_score": round(r["score"], 6),
        }
        for r in raw_results
    ]

    # Step 5: Cache write — only cache genuine answers, never a failure message.
    # Caches the full envelope (answer + retrieved_products), not just the
    # answer text, so a cache hit can still report which products backed the
    # answer — needed both for the frontend diagnostics sidebar and for
    # evaluation runs (a cache hit must not silently zero out retrieval
    # metrics just because search was skipped).
    if generation_ok:
        cache_payload = json.dumps({"answer": answer, "retrieved_products": retrieved_products})
        await cache_service.set(cache_result["cache_key"], cache_payload)
        logger.info(f"[CACHE SET] key={cache_result['cache_key']}")

    total_ms = (time.perf_counter() - total_start) * 1000

    return {
        "answer": answer,
        "source": "llm" if generation_ok else "llm_error",
        "retrieved_products": retrieved_products,
        "timing": {
            "cache_lookup_ms": round(cache_ms, 2),
            "search_ms": round(search_ms, 2),
            "llm_ms": round(llm_ms, 2),
            "total_ms": round(total_ms, 2),
        },
    }
