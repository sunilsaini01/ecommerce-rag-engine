from fastembed import TextEmbedding, SparseTextEmbedding
from typing import Any

from src.backend.observability.tracing import traceable


class ProductEmbedder:
    def __init__(self) -> None:
        self.dense_model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        self.sparse_model = SparseTextEmbedding(
            model_name="Qdrant/bm25"
        )

    def embed_text_chunks(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[dict[str, Any]]]:
        """Batch path used by offline ingestion — intentionally untraced,
        since ingestion is a bulk offline job, not a live RAG request."""

        dense_vectors = list(self.dense_model.embed(texts))

        sparse_results = list(self.sparse_model.embed(texts))
        sparse_vectors = [
            {"indices": s.indices.tolist(), "values": s.values.tolist()}
            for s in sparse_results
        ]

        return dense_vectors, sparse_vectors

    # ── Single-query path used by the online search pipeline ──
    # Split into two functions (rather than reusing embed_text_chunks) so
    # each half of query embedding gets its own LangSmith span — answers
    # "how long did dense vs. sparse embedding take" independently.
    @traceable(name="dense_embedding", run_type="tool")
    def embed_dense_query(self, text: str) -> dict[str, Any]:
        vector = list(self.dense_model.embed([text]))[0].tolist()
        return {"dim": len(vector), "vector": vector}

    @traceable(name="sparse_embedding", run_type="tool")
    def embed_sparse_query(self, text: str) -> dict[str, Any]:
        result = list(self.sparse_model.embed([text]))[0]
        return {
            "indices": result.indices.tolist(),
            "values": result.values.tolist(),
            "nnz": len(result.indices),
        }