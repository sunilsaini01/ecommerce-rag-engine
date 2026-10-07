import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from config.settings import settings
from src.backend.main import app
from src.backend.services.cache import CacheService
from src.backend.services.errors import DependencyUnavailable
from src.backend.services.llm import LLMService
from src.backend.services.rag_pipeline import run_rag_pipeline
from src.backend.services.search import ProductSearchService


async def stalled(*args, **kwargs):
    await asyncio.sleep(60)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(settings, "REDIS_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(settings, "READINESS_TIMEOUT_SECONDS", 0.01)
    search, cache, llm = AsyncMock(), AsyncMock(), AsyncMock()
    search.search.return_value = [{"id": "1", "title": "Widget", "price": 10, "score": 1}]
    cache.client.incr.return_value = 1
    cache.lookup.return_value = {"cache_key": "test", "cache_hit": False, "answer": None}
    llm.generate_answer.return_value = ("Answer", True)
    monkeypatch.setattr(app.state, "search_service", search, raising=False)
    monkeypatch.setattr(app.state, "cache_service", cache, raising=False)
    monkeypatch.setattr(app.state, "llm_service", llm, raising=False)
    return TestClient(app)


@pytest.mark.parametrize("operation", ["incr", "expire"])
@pytest.mark.parametrize("failure", [RedisConnectionError("private connection info"), stalled])
def test_rate_limit_outage_fails_closed(client, operation, failure):
    getattr(app.state.cache_service.client, operation).side_effect = failure
    response = client.post("/api/v1/ask", json={"q": "widget"})
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"
    assert response.json()["detail"] == "Rate limiting service is temporarily unavailable. Please try again shortly."
    app.state.search_service.search.assert_not_called()
    app.state.llm_service.generate_answer.assert_not_called()


def test_search_still_works_without_redis(client):
    app.state.cache_service.client.incr.side_effect = RedisConnectionError()
    response = client.post("/api/v1/search", json={"q": "widget"})
    assert response.status_code == 200
    app.state.cache_service.client.incr.assert_not_called()


@pytest.mark.parametrize("endpoint", ["ask", "search"])
def test_retrieval_unavailable_returns_503(client, endpoint):
    app.state.search_service.search.side_effect = DependencyUnavailable("Product search service")
    response = client.post(f"/api/v1/{endpoint}", json={"q": "widget"})
    assert response.status_code == 503
    assert "Product search service" in response.json()["detail"]
    app.state.llm_service.generate_answer.assert_not_called()


def test_generation_failure_returns_503_without_caching(client):
    app.state.llm_service.generate_answer.return_value = ("Failure", False)
    response = client.post("/api/v1/ask", json={"q": "widget"})
    assert response.status_code == 503
    assert "Answer generation service" in response.json()["detail"]
    app.state.cache_service.set.assert_not_called()


def test_health_is_independent_of_dependencies(client):
    app.state.cache_service.client.ping.side_effect = RedisConnectionError()
    app.state.search_service.client.get_collection.side_effect = RuntimeError()
    assert client.get("/health").json() == {"status": "ok"}
    app.state.cache_service.client.ping.assert_not_called()
    app.state.search_service.client.get_collection.assert_not_called()


def test_ready_when_configured_and_dependencies_respond(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["checks"]["groq_configuration"] == "configured"
    assert response.headers["Cache-Control"] == "no-store"
    app.state.llm_service.generate_answer.assert_not_called()


@pytest.mark.parametrize("dependency", ["redis", "qdrant_catalog"])
@pytest.mark.parametrize("failure", [RuntimeError("private credentials"), stalled])
def test_ready_fails_on_outage_or_timeout(client, dependency, failure):
    operation = app.state.cache_service.client.ping if dependency == "redis" else app.state.search_service.client.get_collection
    operation.side_effect = failure
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"][dependency] == "unavailable"
    assert "private" not in response.text


def test_ready_reports_missing_groq_configuration(client, monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "")
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["groq_configuration"] == "not_configured"


def test_ready_without_initialized_services(client, monkeypatch):
    for name in ("cache_service", "search_service", "llm_service"):
        monkeypatch.delattr(app.state, name)
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"] == {
        "redis": "unavailable", "qdrant_catalog": "unavailable", "groq_configuration": "not_configured",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RedisConnectionError(), stalled])
async def test_cache_outage_does_not_discard_answer(monkeypatch, failure):
    monkeypatch.setattr(settings, "REDIS_TIMEOUT_SECONDS", 0.01)
    cache = CacheService.__new__(CacheService)
    cache.client = AsyncMock()
    cache.client.get.side_effect = cache.client.set.side_effect = failure
    search, llm = AsyncMock(), AsyncMock()
    search.search.return_value = [{"id": "1", "score": 1, "title": "Widget"}]
    llm.generate_answer.return_value = ("Generated answer", True)
    result = await run_rag_pipeline("widget", 5, search, cache, llm)
    assert result["source"] == "llm"
    assert result["answer"] == "Generated answer"
    cache.client.set.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_deadline_maps_to_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "SEARCH_TIMEOUT_SECONDS", 0.01)
    service = ProductSearchService.__new__(ProductSearchService)
    service._vectorize_query = AsyncMock(side_effect=stalled)
    with pytest.raises(DependencyUnavailable, match="Product search service"):
        await service.search("widget")


@pytest.mark.asyncio
async def test_both_retrieval_paths_failing_maps_to_unavailable():
    service = ProductSearchService.__new__(ProductSearchService)
    service._vectorize_query = AsyncMock(return_value=([1.0], {"indices": [1], "values": [1.0]}))
    service.client = AsyncMock()
    service.client.query_points.side_effect = RuntimeError("Private URL")
    with pytest.raises(DependencyUnavailable) as exc:
        await service.search("widget")
    assert "Private" not in str(exc.value)
    assert service.client.query_points.await_count == 3


@pytest.mark.asyncio
async def test_llm_overall_deadline(monkeypatch):
    monkeypatch.setattr(settings, "LLM_GENERATION_TIMEOUT_SECONDS", 0.01)
    service = LLMService.__new__(LLMService)
    service.client = MagicMock()
    def delayed_completion(prompt):
        time.sleep(0.04)
        return {"content": "Late answer"}
    service._call_groq_sync = delayed_completion
    _, ok = await service.generate_answer("widget", "context")
    assert not ok


@pytest.mark.asyncio
async def test_missing_key_does_not_prevent_startup(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "")
    service = LLMService()
    assert service.client is None
    _, ok = await service.generate_answer("widget", "context")
    assert not ok
    await service.close()
