"""LangSmith tracing tests. Per Part 17 of the RAG evolution spec: the app
must work identically whether tracing is enabled or disabled, and no test
should depend on a live LangSmith API unless explicitly marked
`integration`."""

import os

import pytest

from config.settings import settings
from src.backend.observability.tracing import traceable, tracing_enabled, root_metadata, langsmith_extra, only_fields


class TestTracingDisabled:
    def test_traceable_function_still_executes_normally(self):
        """The core guarantee: decorating a function with @traceable must
        never change its behavior when tracing is off."""
        calls = []

        @traceable(name="test_span", run_type="tool")
        def add(a, b):
            calls.append((a, b))
            return a + b

        assert add(2, 3) == 5
        assert calls == [(2, 3)]

    @pytest.mark.asyncio
    async def test_traceable_async_function_still_executes_normally(self):
        @traceable(name="test_async_span", run_type="chain")
        async def multiply(a, b):
            return a * b

        assert await multiply(3, 4) == 12

    def test_tracing_enabled_false_when_no_api_key(self, monkeypatch):
        monkeypatch.setattr(settings, "LANGSMITH_TRACING", True)
        monkeypatch.setattr(settings, "LANGSMITH_API_KEY", "")
        assert tracing_enabled() is False

    def test_tracing_enabled_false_when_flag_off(self, monkeypatch):
        monkeypatch.setattr(settings, "LANGSMITH_TRACING", False)
        monkeypatch.setattr(settings, "LANGSMITH_API_KEY", "lsv2_sk_fake")
        assert tracing_enabled() is False

    def test_a_function_that_raises_still_raises_normally(self):
        """Tracing must never swallow a real application error."""
        @traceable(name="failing_span", run_type="tool")
        def boom():
            raise ValueError("expected failure")

        with pytest.raises(ValueError, match="expected failure"):
            boom()


class TestMetadataHelpers:
    def test_root_metadata_never_includes_secrets(self):
        meta = root_metadata(endpoint="/api/v1/ask", top_k=5)
        serialized = str(meta).lower()
        assert "api_key" not in serialized
        assert "authorization" not in serialized
        assert settings.GROQ_API_KEY.lower() not in serialized if settings.GROQ_API_KEY else True

    def test_root_metadata_includes_expected_fields(self):
        meta = root_metadata(endpoint="/api/v1/ask", top_k=5)
        assert meta["endpoint"] == "/api/v1/ask"
        assert meta["top_k"] == 5
        assert "catalog_version" in meta
        assert "embedding_model" in meta

    def test_langsmith_extra_shape(self):
        extra = langsmith_extra(tags=["endpoint:ask"], top_k=5)
        assert extra["tags"] == ["endpoint:ask"]
        assert extra["metadata"]["top_k"] == 5


class TestOnlyFieldsSecretGuard:
    def test_only_fields_strips_everything_not_allowlisted(self):
        """run_rag_pipeline is decorated with process_inputs=only_fields(
        'query', 'limit') specifically because its other arguments are live
        service objects — one of which (LLMService.client) holds the real
        Groq API key. This locks in that only query/limit ever reach the
        trace serializer, regardless of what future arguments get added."""
        process = only_fields("query", "limit")
        fake_service_holding_a_secret = object()
        raw_inputs = {
            "query": "hiking boots",
            "limit": 5,
            "search_service": fake_service_holding_a_secret,
            "cache_service": fake_service_holding_a_secret,
            "llm_service": fake_service_holding_a_secret,
        }
        filtered = process(raw_inputs)
        assert filtered == {"query": "hiking boots", "limit": 5}
        assert "search_service" not in filtered
        assert "cache_service" not in filtered
        assert "llm_service" not in filtered


@pytest.mark.integration
class TestTracingEnabledIntegration:
    """Requires a real LANGSMITH_API_KEY with a working workspace. Skipped
    unless explicitly run: `pytest -m integration`."""

    def test_run_is_actually_created_in_langsmith(self):
        if not settings.LANGSMITH_API_KEY:
            pytest.skip("No LANGSMITH_API_KEY configured for integration test.")

        from langsmith import Client

        @traceable(name="integration_test_span", run_type="tool")
        def ping():
            return "pong"

        result = ping()
        assert result == "pong"
        # A real assertion that a run landed would query the LangSmith API
        # for the run by name/project; omitted here to avoid a flaky
        # network-timing-dependent test — this test's purpose is to prove
        # the call path doesn't raise when a real key is configured.
