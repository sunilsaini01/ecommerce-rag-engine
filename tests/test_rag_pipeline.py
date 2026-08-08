"""Tests for run_rag_pipeline's cache-hit/miss/no-results branches — in
particular, that a cache HIT still returns the original retrieved_products
(a real bug found during this evolution: caching only the answer text made
retrieval metrics silently collapse to zero on any cache-warm evaluation
run — see src/backend/services/rag_pipeline.py)."""

import json
from unittest.mock import AsyncMock

import pytest

from src.backend.services.rag_pipeline import run_rag_pipeline


def _mock_services(cache_hit: bool, cached_payload: dict | None = None, search_results=None, llm_result=("An answer.", True)):
    search_service = AsyncMock()
    search_service.search.return_value = search_results if search_results is not None else [
        {"id": "1", "title": "Widget", "category": "Electronics", "price": 9.99, "score": 0.9}
    ]

    cache_service = AsyncMock()
    cache_service.lookup.return_value = {
        "cache_key": "rag:cache:v:v:v:hash",
        "cache_hit": cache_hit,
        "answer": json.dumps(cached_payload) if cache_hit else None,
    }

    llm_service = AsyncMock()
    llm_service.generate_answer.return_value = llm_result

    return search_service, cache_service, llm_service


@pytest.mark.asyncio
class TestRagPipeline:
    async def test_cache_hit_returns_original_retrieved_products(self):
        cached_payload = {
            "answer": "The Widget costs $9.99.",
            "retrieved_products": [{"id": "1", "title": "Widget", "category": "Electronics", "price": 9.99, "rrf_score": 0.9}],
        }
        search_service, cache_service, llm_service = _mock_services(cache_hit=True, cached_payload=cached_payload)

        result = await run_rag_pipeline("widget", 5, search_service, cache_service, llm_service)

        assert result["source"] == "cache"
        assert result["answer"] == "The Widget costs $9.99."
        assert result["retrieved_products"] == cached_payload["retrieved_products"]
        search_service.search.assert_not_called()
        llm_service.generate_answer.assert_not_called()

    async def test_cache_hit_with_legacy_plain_string_value_degrades_gracefully(self):
        """Before this fix, cache values were plain answer strings. A
        leftover legacy entry must not crash — it should be treated as the
        answer with an empty product list."""
        search_service, cache_service, llm_service = _mock_services(cache_hit=True)
        cache_service.lookup.return_value["answer"] = "a plain legacy cached answer"

        result = await run_rag_pipeline("widget", 5, search_service, cache_service, llm_service)

        assert result["source"] == "cache"
        assert result["answer"] == "a plain legacy cached answer"
        assert result["retrieved_products"] == []

    async def test_cache_miss_runs_full_pipeline_and_writes_envelope(self):
        search_service, cache_service, llm_service = _mock_services(cache_hit=False)

        result = await run_rag_pipeline("widget", 5, search_service, cache_service, llm_service)

        assert result["source"] == "llm"
        assert result["retrieved_products"][0]["id"] == "1"
        search_service.search.assert_called_once()
        llm_service.generate_answer.assert_called_once()
        cache_service.set.assert_called_once()
        written_key, written_value = cache_service.set.call_args[0]
        payload = json.loads(written_value)
        assert payload["answer"] == "An answer."
        assert payload["retrieved_products"][0]["id"] == "1"

    async def test_no_search_results_returns_no_results_without_calling_llm(self):
        search_service, cache_service, llm_service = _mock_services(cache_hit=False, search_results=[])

        result = await run_rag_pipeline("nonexistent product", 5, search_service, cache_service, llm_service)

        assert result["source"] == "no_results"
        llm_service.generate_answer.assert_not_called()
        cache_service.set.assert_not_called()

    async def test_failed_generation_not_cached(self):
        search_service, cache_service, llm_service = _mock_services(
            cache_hit=False, llm_result=("I was unable to process your request at this time. Please try again later.", False)
        )

        result = await run_rag_pipeline("widget", 5, search_service, cache_service, llm_service)

        assert result["source"] == "llm_error"
        cache_service.set.assert_not_called()
