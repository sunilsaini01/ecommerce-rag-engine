import json

import pytest
from fakeredis import aioredis as fake_aioredis

from src.backend.services.cache import CacheService


def _make_cache_service() -> CacheService:
    service = CacheService.__new__(CacheService)
    service.client = fake_aioredis.FakeRedis(decode_responses=True)
    return service


class TestCacheKeyVersioning:
    def test_key_changes_with_catalog_version(self, monkeypatch):
        from config.settings import settings

        service = _make_cache_service()
        monkeypatch.setattr(settings, "CATALOG_VERSION", "v1")
        key_v1 = service._generate_key("rag:cache", "hiking boots")
        monkeypatch.setattr(settings, "CATALOG_VERSION", "v2")
        key_v2 = service._generate_key("rag:cache", "hiking boots")
        assert key_v1 != key_v2

    def test_key_changes_with_prompt_version(self, monkeypatch):
        from config.settings import settings

        service = _make_cache_service()
        monkeypatch.setattr(settings, "PROMPT_VERSION", "p1")
        key_p1 = service._generate_key("rag:cache", "hiking boots")
        monkeypatch.setattr(settings, "PROMPT_VERSION", "p2")
        key_p2 = service._generate_key("rag:cache", "hiking boots")
        assert key_p1 != key_p2

    def test_query_normalization(self):
        service = _make_cache_service()
        assert service._generate_key("rag:cache", "Hiking Boots") == service._generate_key("rag:cache", "  hiking boots  ")


@pytest.mark.asyncio
class TestCacheGetSet:
    async def test_miss_then_hit(self):
        service = _make_cache_service()
        key = service._generate_key("rag:cache", "test query")
        assert await service.get(key) is None
        await service.set(key, json.dumps({"answer": "hello", "retrieved_products": []}))
        cached = await service.get(key)
        assert json.loads(cached)["answer"] == "hello"

    async def test_lookup_reports_hit_and_miss(self):
        service = _make_cache_service()
        result = await service.lookup("new query")
        assert result["cache_hit"] is False

        await service.set(result["cache_key"], json.dumps({"answer": "cached answer", "retrieved_products": []}))
        result2 = await service.lookup("new query")
        assert result2["cache_hit"] is True
        assert json.loads(result2["answer"])["answer"] == "cached answer"

    async def test_ttl_expiry(self):
        service = _make_cache_service()
        key = service._generate_key("rag:cache", "ttl query")
        await service.set(key, "value", ttl_seconds=1)
        assert await service.get(key) == "value"
        await service.client.pexpire(key, 1)  # force near-immediate expiry
        import asyncio
        await asyncio.sleep(0.05)
        assert await service.get(key) is None

    async def test_invalidate(self):
        service = _make_cache_service()
        key = service._generate_key("rag:cache", "to invalidate")
        await service.set(key, "value")
        await service.invalidate(key)
        assert await service.get(key) is None
