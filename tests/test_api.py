"""API-layer tests. Real external services (Qdrant/Redis/Groq) are never
started — app.state.{search,cache,llm}_service are replaced with mocks
directly, and the client is used WITHOUT entering the `with` context
manager so the real `lifespan` (which constructs live clients) never runs."""

import json

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from src.backend.main import app
from config.settings import settings


@pytest.fixture
def client():
    app.state.search_service = AsyncMock()
    app.state.cache_service = AsyncMock()
    app.state.llm_service = AsyncMock()

    app.state.search_service.search.return_value = [
        {"id": "1", "title": "Widget", "description": "A test widget.", "price": 9.99, "category": "Electronics", "score": 0.9}
    ]
    app.state.cache_service.lookup.return_value = {"cache_key": "rag:cache:test", "cache_hit": False, "answer": None}
    app.state.cache_service.client.incr.return_value = 1
    app.state.llm_service.generate_answer.return_value = ("The Widget costs $9.99.", True)

    return TestClient(app)


class TestSearchEndpoint:
    def test_search_returns_results(self, client):
        response = client.post("/api/v1/search", json={"q": "widget", "limit": 5})
        assert response.status_code == 200
        body = response.json()
        assert body["results"][0]["id"] == "1"
        assert "latency_ms" in body

    def test_search_rejects_empty_query(self, client):
        response = client.post("/api/v1/search", json={"q": ""})
        assert response.status_code == 422

    def test_search_rejects_limit_over_20(self, client):
        response = client.post("/api/v1/search", json={"q": "widget", "limit": 21})
        assert response.status_code == 422


class TestAskEndpoint:
    def test_ask_cache_miss_calls_full_pipeline(self, client):
        response = client.post("/api/v1/ask", json={"q": "widget", "limit": 5})
        assert response.status_code == 200
        body = response.json()
        assert body["source"] == "llm"
        assert body["retrieved_products"][0]["id"] == "1"
        app.state.cache_service.set.assert_called_once()

    def test_ask_cache_hit_skips_search_and_llm(self, client):
        cached = json.dumps({
            "answer": "Cached answer.",
            "retrieved_products": [{"id": "1", "title": "Widget", "category": "Electronics", "price": 9.99, "rrf_score": 0.9}],
        })
        app.state.cache_service.lookup.return_value = {"cache_key": "k", "cache_hit": True, "answer": cached}

        response = client.post("/api/v1/ask", json={"q": "widget", "limit": 5})

        assert response.status_code == 200
        body = response.json()
        assert body["source"] == "cache"
        assert body["answer"] == "Cached answer."
        app.state.search_service.search.assert_not_called()
        app.state.llm_service.generate_answer.assert_not_called()

    def test_ask_no_results(self, client):
        app.state.search_service.search.return_value = []
        response = client.post("/api/v1/ask", json={"q": "nonexistent product xyz", "limit": 5})
        assert response.status_code == 200
        assert response.json()["source"] == "no_results"

    def test_ask_rejects_empty_query(self, client):
        response = client.post("/api/v1/ask", json={"q": ""})
        assert response.status_code == 422


class TestApiKeyAuth:
    def test_request_allowed_when_api_key_unset(self, client):
        assert settings.API_KEY == ""
        response = client.post("/api/v1/search", json={"q": "widget"})
        assert response.status_code == 200

    def test_request_rejected_without_header_when_api_key_set(self, client, monkeypatch):
        monkeypatch.setattr(settings, "API_KEY", "super-secret")
        response = client.post("/api/v1/search", json={"q": "widget"})
        assert response.status_code == 401

    def test_request_allowed_with_correct_header(self, client, monkeypatch):
        monkeypatch.setattr(settings, "API_KEY", "super-secret")
        response = client.post("/api/v1/search", json={"q": "widget"}, headers={"X-API-Key": "super-secret"})
        assert response.status_code == 200


class TestRateLimit:
    def test_ask_rejected_when_over_limit(self, client, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 1)
        app.state.cache_service.client.incr.return_value = 2  # already over the limit
        response = client.post("/api/v1/ask", json={"q": "widget", "limit": 5})
        assert response.status_code == 429
