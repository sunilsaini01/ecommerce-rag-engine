import asyncio
from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    SparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)

from config.settings import settings
from src.backend.models.api_schemas import ProductFilters
from src.backend.services.errors import DependencyUnavailable
from src.ingestion.embedder import ProductEmbedder
from src.backend.observability.tracing import traceable

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
RRF_K = 60


def build_product_filter(filters: ProductFilters | None) -> Filter | None:
    if filters is None:
        return None
    conditions = []
    if filters.min_price is not None or filters.max_price is not None:
        conditions.append(FieldCondition(key="price", range=Range(gte=filters.min_price, lte=filters.max_price)))
    for key, value in (("stock", filters.min_stock), ("rating", filters.min_rating)):
        if value is not None:
            conditions.append(FieldCondition(key=key, range=Range(gte=value)))
    for key, value in (("category", filters.category), ("brand", filters.brand)):
        if value is not None:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=conditions) if conditions else None


class ProductSearchService:
    def __init__(self) -> None:
        self.client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            grpc_port=settings.QDRANT_GRPC_PORT,
            prefer_grpc=True,
            timeout=settings.QDRANT_TIMEOUT_SECONDS,
        )
        self.embedder = ProductEmbedder()

    async def close(self) -> None:
        await self.client.close()

    # ── Vector Generation ─────────────────────────────────
    @traceable(name="query_embedding", run_type="chain")
    async def _vectorize_query(
        self, query: str
    ) -> tuple[list[float], dict]:
        dense_result, sparse_result = await asyncio.gather(
            asyncio.to_thread(self.embedder.embed_dense_query, query),
            asyncio.to_thread(self.embedder.embed_sparse_query, query),
        )
        return dense_result["vector"], {
            "indices": sparse_result["indices"],
            "values": sparse_result["values"],
        }

    # ── RRF Python Fallback ───────────────────────────────
    @traceable(name="rrf_fusion", run_type="tool")
    def _rrf_fusion(
        self,
        dense_hits: list[dict],
        sparse_hits: list[dict],
        limit: int,
    ) -> list[dict]:
        scores: dict[str | int, float] = {}
        payloads: dict[str | int, dict] = {}

        for rank, hit in enumerate(dense_hits):
            doc_id = hit["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            payloads[doc_id] = hit["payload"]

        for rank, hit in enumerate(sparse_hits):
            doc_id = hit["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            if doc_id not in payloads:
                payloads[doc_id] = hit["payload"]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [
            {"id": doc_id, "score": score, **payloads[doc_id]}
            for doc_id, score in ranked
        ]

    # ── Native Qdrant RRF (single round trip, default fast path) ──
    @traceable(name="native_rrf_fusion", run_type="retriever")
    async def _native_fusion_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict,
        limit: int,
        query_filter: Filter | None = None,
    ) -> list[dict]:
        prefetch = [
            Prefetch(
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                limit=20,
                filter=query_filter,
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse_vector["indices"],
                    values=sparse_vector["values"],
                ),
                using=SPARSE_VECTOR_NAME,
                limit=20,
                filter=query_filter,
            ),
        ]

        results = await self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )

        return [
            {"id": hit.id, "score": hit.score, **hit.payload}
            for hit in results.points
        ]

    # ── Dense-only query (fallback path building block) ──────
    @traceable(name="dense_retrieval", run_type="retriever")
    async def _dense_query(self, dense_vector: list[float], limit: int, query_filter: Filter | None = None) -> list[dict]:
        response = await self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=dense_vector,
            using=DENSE_VECTOR_NAME,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )
        return [{"id": p.id, "score": p.score, "payload": p.payload} for p in response.points]

    # ── Sparse-only query (fallback path building block) ─────
    @traceable(name="sparse_retrieval", run_type="retriever")
    async def _sparse_query(self, sparse_vector: dict, limit: int, query_filter: Filter | None = None) -> list[dict]:
        response = await self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=SparseVector(
                indices=sparse_vector["indices"],
                values=sparse_vector["values"],
            ),
            using=SPARSE_VECTOR_NAME,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )
        return [{"id": p.id, "score": p.score, "payload": p.payload} for p in response.points]

    # ── Fallback: parallel dense + sparse, fused in Python ────
    # Only reached when the native single-round-trip fusion query fails
    # (e.g. older Qdrant version, config mismatch). As a side effect this
    # path is the one that gives full per-method attribution in traces —
    # the native path can't, since Qdrant's fused response doesn't expose
    # which prefetch branch contributed each hit.
    async def _fallback_fusion_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict,
        limit: int,
        query_filter: Filter | None = None,
    ) -> list[dict]:
        dense_hits, sparse_hits = await asyncio.gather(
            self._dense_query(dense_vector, 20, query_filter),
            self._sparse_query(sparse_vector, 20, query_filter),
        )
        return self._rrf_fusion(dense_hits, sparse_hits, limit)

    # ── Public Entry Point ────────────────────────────────
    @traceable(name="hybrid_retrieval", run_type="retriever")
    async def search(self, query: str, limit: int = 5, filters: ProductFilters | None = None) -> list[dict]:
        try:
            return await asyncio.wait_for(
                self._search(query, limit, filters), settings.SEARCH_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning("Product retrieval is unavailable.")
            raise DependencyUnavailable("Product search service") from exc

    async def _search(self, query: str, limit: int, filters: ProductFilters | None) -> list[dict]:
        dense_vector, sparse_vector = await self._vectorize_query(query)
        query_filter = build_product_filter(filters)

        try:
            results = await self._native_fusion_search(
                dense_vector, sparse_vector, limit, query_filter
            )
            logger.info(f"Native RRF fusion used for query: '{query}'")
        except Exception as e:
            logger.warning(f"Native fusion failed ({e}), falling back to Python RRF.")
            results = await self._fallback_fusion_search(
                dense_vector, sparse_vector, limit, query_filter
            )

        return results
