"""Retrieval-layer unit tests. These test pure logic (RRF fusion) without a
live Qdrant instance — ProductSearchService.__init__ makes a real
AsyncQdrantClient/ProductEmbedder, which is too heavy for a unit test, so
we exercise `_rrf_fusion` directly via a bare instance whose __init__ is
bypassed."""

from src.backend.services.search import ProductSearchService, RRF_K


def _make_service() -> ProductSearchService:
    return ProductSearchService.__new__(ProductSearchService)


def _hit(doc_id, payload=None) -> dict:
    return {"id": doc_id, "score": 1.0, "payload": payload or {"title": f"Product {doc_id}"}}


class TestRRFFusion:
    def test_single_method_ranking_preserved(self):
        service = _make_service()
        dense_hits = [_hit("a"), _hit("b"), _hit("c")]
        result = service._rrf_fusion(dense_hits, [], limit=5)
        ids = [r["id"] for r in result]
        assert ids == ["a", "b", "c"]

    def test_agreement_across_both_methods_boosts_rank(self):
        """A doc found by BOTH dense and sparse retrieval should outrank a
        doc found by only one, even if the single-method doc ranked #1
        there — this is the entire point of RRF fusion."""
        service = _make_service()
        dense_hits = [_hit("only_dense"), _hit("both")]
        sparse_hits = [_hit("both"), _hit("only_sparse")]
        result = service._rrf_fusion(dense_hits, sparse_hits, limit=5)
        assert result[0]["id"] == "both"

    def test_empty_inputs_returns_empty(self):
        service = _make_service()
        assert service._rrf_fusion([], [], limit=5) == []

    def test_limit_is_respected(self):
        service = _make_service()
        dense_hits = [_hit(str(i)) for i in range(10)]
        result = service._rrf_fusion(dense_hits, [], limit=3)
        assert len(result) == 3

    def test_duplicate_across_methods_not_double_counted_in_output(self):
        """A doc appearing in both hit lists must appear exactly once in
        the fused output, not twice."""
        service = _make_service()
        dense_hits = [_hit("x")]
        sparse_hits = [_hit("x")]
        result = service._rrf_fusion(dense_hits, sparse_hits, limit=5)
        assert len(result) == 1
        assert result[0]["id"] == "x"

    def test_score_matches_rrf_formula(self):
        service = _make_service()
        dense_hits = [_hit("a")]  # rank 0
        result = service._rrf_fusion(dense_hits, [], limit=5)
        expected_score = 1.0 / (RRF_K + 1)
        assert abs(result[0]["score"] - expected_score) < 1e-9
