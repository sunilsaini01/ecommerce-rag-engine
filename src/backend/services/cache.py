import hashlib
import json
import asyncio
from loguru import logger
import redis.asyncio as aioredis
from redis.exceptions import RedisError

from config.settings import settings
from src.backend.models.api_schemas import ProductFilters
from src.backend.observability.tracing import traceable


class CacheService:
    def __init__(self) -> None:
        self.client = aioredis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=settings.REDIS_TIMEOUT_SECONDS,
            socket_timeout=settings.REDIS_TIMEOUT_SECONDS,
        )
        logger.info(f"CacheService initialized at '{settings.REDIS_HOST}:{settings.REDIS_PORT}'")

    def _generate_key(self, prefix: str, query: str, limit: int = 5, filters: ProductFilters | None = None) -> str:
        """Separate answers by catalog, prompt, model, context format, limit,
        normalized query, and canonical filters. The context version prevents
        answers generated with the previous reduced context from being reused.
        Empty filters and omitted filters are equivalent."""
        normalized = query.lower().strip()
        hash_digest = hashlib.md5(normalized.encode()).hexdigest()
        filter_data = filters.model_dump(exclude_none=True) if filters is not None else {}
        filter_digest = hashlib.sha256(json.dumps(filter_data, sort_keys=True).encode()).hexdigest()
        return (
            f"{prefix}:{settings.CATALOG_VERSION}:{settings.PROMPT_VERSION}:"
            f"{settings.MODEL_VERSION}:context:v2:limit:{limit}:filters:{filter_digest}:{hash_digest}"
        )

    async def get(self, key: str) -> str | None:
        try:
            return await asyncio.wait_for(self.client.get(key), settings.REDIS_TIMEOUT_SECONDS)
        except (RedisError, TimeoutError, OSError):
            logger.warning("Redis cache read unavailable; continuing without cached data.")
            return None

    async def set(self, key: str, value: str, ttl_seconds: int = 300) -> None:
        try:
            await asyncio.wait_for(self.client.set(key, value, ex=ttl_seconds), settings.REDIS_TIMEOUT_SECONDS)
        except (RedisError, TimeoutError, OSError):
            logger.warning("Redis cache write unavailable; returning the generated answer.")

    async def invalidate(self, key: str) -> None:
        await asyncio.wait_for(self.client.delete(key), settings.REDIS_TIMEOUT_SECONDS)
        logger.info(f"Cache invalidated for key: {key}")

    @traceable(name="cache_lookup", run_type="tool")
    async def lookup(self, query: str, limit: int = 5, filters: ProductFilters | None = None) -> dict:
        """Traced cache-check used by the /ask pipeline. Returns the cache
        key (so a hit can be reused for the write-back check) plus whether
        it was a hit, without ever tracing the query result content twice."""
        key = self._generate_key("rag:cache", query, limit, filters)
        value = await self.get(key)
        return {"cache_key": key, "cache_hit": value is not None, "answer": value}

    async def close(self) -> None:
        await self.client.aclose()
