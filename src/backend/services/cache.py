import hashlib
import asyncio
from loguru import logger
import redis.asyncio as aioredis

from config.settings import settings
from src.backend.observability.tracing import traceable


class CacheService:
    def __init__(self) -> None:
        self.client = aioredis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True,
        )
        logger.info(f"CacheService initialized at '{settings.REDIS_HOST}:{settings.REDIS_PORT}'")

    def _generate_key(self, prefix: str, query: str) -> str:
        """Versioned cache key: rag:{catalog_version}:{prompt_version}:
        {model_version}:{query_hash}. Bumping CATALOG_VERSION, PROMPT_VERSION,
        or MODEL_VERSION (config/settings.py) changes every key, so a stale
        answer generated under an old catalog/prompt/model can never be
        served after any of those change — no explicit cache flush needed."""
        normalized = query.lower().strip()
        hash_digest = hashlib.md5(normalized.encode()).hexdigest()
        return (
            f"{prefix}:{settings.CATALOG_VERSION}:{settings.PROMPT_VERSION}:"
            f"{settings.MODEL_VERSION}:{hash_digest}"
        )

    async def get(self, key: str) -> str | None:
        value = await self.client.get(key)
        return value

    async def set(self, key: str, value: str, ttl_seconds: int = 300) -> None:
        await self.client.set(key, value, ex=ttl_seconds)

    async def invalidate(self, key: str) -> None:
        await self.client.delete(key)
        logger.info(f"Cache invalidated for key: {key}")

    @traceable(name="cache_lookup", run_type="tool")
    async def lookup(self, query: str) -> dict:
        """Traced cache-check used by the /ask pipeline. Returns the cache
        key (so a hit can be reused for the write-back check) plus whether
        it was a hit, without ever tracing the query result content twice."""
        key = self._generate_key("rag:cache", query)
        value = await self.get(key)
        return {"cache_key": key, "cache_hit": value is not None, "answer": value}

    async def close(self) -> None:
        await self.client.aclose()
