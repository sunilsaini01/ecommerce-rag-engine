from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from src.backend.api.middleware import RequestTimingMiddleware
from src.backend.api.v1.router import router as v1_router
from src.backend.services.search import ProductSearchService
from src.backend.services.cache import CacheService
from src.backend.services.llm import LLMService
from src.backend.services.errors import DependencyUnavailable
from config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — connecting to Qdrant and Redis.")
    app.state.search_service = ProductSearchService()
    app.state.cache_service = CacheService()
    app.state.llm_service = LLMService()
    try:
        yield
    finally:
        logger.info("Shutting down.")
        await asyncio.gather(
            app.state.search_service.close(), app.state.cache_service.close(),
            app.state.llm_service.close(), return_exceptions=True,
        )


app = FastAPI(
    title="E-Commerce Search Engine",
    lifespan=lifespan,
)

app.add_middleware(RequestTimingMiddleware)
app.include_router(v1_router, prefix="/api/v1")


@app.exception_handler(DependencyUnavailable)
async def dependency_unavailable(request: Request, exc: DependencyUnavailable):
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc)},
        headers={"Retry-After": "5"},
    )


@app.get("/health")
async def health():
    """Liveness only: does not contact external dependencies."""
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request):
    """Bounded checks of Redis and the catalog; Groq check is configuration only."""
    async def probe(service_name, operation):
        service = getattr(request.app.state, service_name, None)
        if service is None:
            return "unavailable"
        try:
            result = await asyncio.wait_for(operation(service), settings.READINESS_TIMEOUT_SECONDS)
            return "ok" if result else "unavailable"
        except Exception:
            return "unavailable"

    redis_status, qdrant_status = await asyncio.gather(
        probe("cache_service", lambda service: service.client.ping()),
        probe("search_service", lambda service: service.client.get_collection(settings.COLLECTION_NAME)),
    )
    llm = getattr(request.app.state, "llm_service", None)
    groq_status = "configured" if settings.GROQ_API_KEY and llm is not None and llm.client is not None else "not_configured"
    checks = {"redis": redis_status, "qdrant_catalog": qdrant_status, "groq_configuration": groq_status}
    is_ready = redis_status == qdrant_status == "ok" and groq_status == "configured"
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={"status": "ready" if is_ready else "not_ready", "checks": checks},
        headers={"Cache-Control": "no-store"},
    )
