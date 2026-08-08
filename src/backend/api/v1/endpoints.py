import time
from fastapi import APIRouter, Depends, Request
from loguru import logger

from src.backend.models.api_schemas import (
    SearchQueryRequest,
    ProductSearchResult,
    SearchResponseEnvelope,
    RetrievedProduct,
    TimingBreakdown,
    AskResponseEnvelope,
)
from src.backend.services.search import ProductSearchService
from src.backend.services.cache import CacheService
from src.backend.services.llm import LLMService, NO_MATCH_MESSAGE
from src.backend.services.rag_pipeline import run_rag_pipeline
from src.backend.api.security import verify_api_key, enforce_rate_limit
from src.backend.observability.tracing import langsmith_extra

router = APIRouter()


# ── Dependency Injection ──────────────────────────────────
def get_search_service(request: Request) -> ProductSearchService:
    return request.app.state.search_service


def get_cache_service(request: Request) -> CacheService:
    return request.app.state.cache_service


def get_llm_service(request: Request) -> LLMService:
    return request.app.state.llm_service


# ── /search Endpoint ──────────────────────────────────────
@router.post(
    "/search",
    response_model=SearchResponseEnvelope,
    dependencies=[Depends(verify_api_key)],
)
async def search_products(
    request: SearchQueryRequest,
    service: ProductSearchService = Depends(get_search_service),
) -> SearchResponseEnvelope:
    start = time.perf_counter()

    raw_results = await service.search(
        query=request.q,
        limit=request.limit,
        langsmith_extra=langsmith_extra(
            tags=["endpoint:search"],
            endpoint="/api/v1/search",
            top_k=request.limit,
        ),
    )

    latency_ms = (time.perf_counter() - start) * 1000

    results = [
        ProductSearchResult(
            id=r["id"],
            title=r.get("title", ""),
            description=r.get("description", ""),
            price=r.get("price", 0.0),
            category=r.get("category", ""),
            score=r["score"],
        )
        for r in raw_results
    ]

    return SearchResponseEnvelope(
        results=results,
        latency_ms=round(latency_ms, 2),
    )


# ── /ask Endpoint ─────────────────────────────────────────
@router.post(
    "/ask",
    response_model=AskResponseEnvelope,
    dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)],
)
async def ask_products(
    request: SearchQueryRequest,
    http_request: Request,
    search_service: ProductSearchService = Depends(get_search_service),
    cache_service: CacheService = Depends(get_cache_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> AskResponseEnvelope:
    result = await run_rag_pipeline(
        query=request.q,
        limit=request.limit,
        search_service=search_service,
        cache_service=cache_service,
        llm_service=llm_service,
        langsmith_extra=langsmith_extra(
            tags=["endpoint:ask"],
            endpoint="/api/v1/ask",
            top_k=request.limit,
        ),
    )

    return AskResponseEnvelope(
        answer=result["answer"],
        source=result["source"],
        retrieved_products=[RetrievedProduct(**p) for p in result["retrieved_products"]],
        timing=TimingBreakdown(**result["timing"]),
    )