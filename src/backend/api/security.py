import time
import asyncio
from redis.exceptions import RedisError
from fastapi import HTTPException, Request

from config.settings import settings
from src.backend.services.errors import DependencyUnavailable


def verify_api_key(request: Request) -> None:
    """No-op when API_KEY is unset (local/dev default). Once set, every
    request must send a matching X-API-Key header."""
    if not settings.API_KEY:
        return
    provided = request.headers.get("X-API-Key")
    if provided != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def enforce_rate_limit(request: Request) -> None:
    """Fixed-window limiter keyed on client IP, backed by the same Redis
    instance used for answer caching. Guards the endpoint that calls Groq,
    since that's the one with an external cost/quota attached."""
    client_ip = request.client.host if request.client else "unknown"
    bucket = int(time.time() // 60)
    key = f"ratelimit:{client_ip}:{bucket}"

    redis_client = request.app.state.cache_service.client
    async def increment_counter():
        count = await redis_client.incr(key)
        if count == 1:
            await redis_client.expire(key, 60)
        return count

    try:
        count = await asyncio.wait_for(increment_counter(), settings.REDIS_TIMEOUT_SECONDS)
    except (RedisError, TimeoutError, OSError) as exc:
        # Fail closed: don't send paid LLM calls when quota enforcement is down.
        raise DependencyUnavailable("Rate limiting service") from exc

    if count > settings.RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({settings.RATE_LIMIT_PER_MINUTE}/min). Try again shortly.",
        )
