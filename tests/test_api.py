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


@pytest.mark.parametrize("endpoint", ["search", "ask"])
class TestRequestValidation:
    @pytest.mark.parametrize("filters", [
        {"min_price": -1}, {"max_price": -1}, {"min_price": 30, "max_price": 20},
        {"min_stock": -1}, {"min_rating": 6}, {"brand": "   "}, {"category": ""},
        {"unsupported": "value"},
    ])
    def test_rejects_invalid_filters(self, client, endpoint, filters):
        response = client.post(f"/api/v1/{endpoint}", json={"q": "widget", "filters": filters})
        assert response.status_code == 422
        app.state.search_service.search.assert_not_called()

    def test_passes_filters_and_returns_attributes(self, client, endpoint):
        product = app.state.search_service.search.return_value[0]
        product.update(brand="Acme", rating=4.5, stock=2, features=["waterproof"],
                       specifications={"battery_hours": 40}, colors=["blue"], sizes=["M"])
        filters = {"max_price": 25, "min_stock": 1, "min_rating": 4, "category": "Electronics", "brand": " Acme "}
        response = client.post(f"/api/v1/{endpoint}", json={"q": "widget", "filters": filters})
        assert response.status_code == 200
        forwarded = app.state.search_service.search.call_args.kwargs["filters"]
        assert forwarded.brand == "Acme" and forwarded.max_price == 25
        result = response.json()["results" if endpoint == "search" else "retrieved_products"][0]
        assert result["features"] == ["waterproof"] and result["specifications"] == {"battery_hours": 40}
        if endpoint == "ask":
            assert app.state.cache_service.lookup.call_args.kwargs["filters"] == forwarded

    @pytest.mark.parametrize("query", [" ", "\t\n\r", "\u2003\u00a0"])
    def test_rejects_whitespace_only_query(self, client, endpoint, query):
        response = client.post(f"/api/v1/{endpoint}", json={"q": query})
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "q"]
        app.state.search_service.search.assert_not_called()
        app.state.cache_service.lookup.assert_not_called()
        app.state.llm_service.generate_answer.assert_not_called()

    @pytest.mark.parametrize("limit", [-1, 0, 21])
    def test_rejects_out_of_range_limit(self, client, endpoint, limit):
        response = client.post(f"/api/v1/{endpoint}", json={"q": "widget", "limit": limit})
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "limit"]
        app.state.search_service.search.assert_not_called()
        app.state.cache_service.lookup.assert_not_called()
        app.state.llm_service.generate_answer.assert_not_called()

    @pytest.mark.parametrize("limit", [1, 20])
    def test_accepts_boundary_limits_and_trims_query(self, client, endpoint, limit):
        response = client.post(f"/api/v1/{endpoint}", json={"q": " \twidget  comparison\n ", "limit": limit})
        assert response.status_code == 200
        assert app.state.search_service.search.call_args.kwargs["query"] == "widget  comparison"
        assert app.state.search_service.search.call_args.kwargs["limit"] == limit
        if endpoint == "ask":
            app.state.cache_service.lookup.assert_awaited_once_with("widget  comparison", limit=limit)
            assert app.state.llm_service.generate_answer.call_args.kwargs["query"] == "widget  comparison"

    @pytest.mark.parametrize("query_length,status", [(200, 200), (201, 422)])
    def test_query_length_is_checked_after_trimming(self, client, endpoint, query_length, status):
        response = client.post(f"/api/v1/{endpoint}", json={"q": "  " + "x" * query_length + "  "})
        assert response.status_code == status

    def test_default_limit_stays_five(self, client, endpoint):
        response = client.post(f"/api/v1/{endpoint}", json={"q": "widget"})
        assert response.status_code == 200
        assert app.state.search_service.search.call_args.kwargs["limit"] == 5
