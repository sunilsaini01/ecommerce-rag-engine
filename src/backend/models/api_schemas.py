from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ProductFilters(BaseModel):
    model_config = {"extra": "forbid", "str_strip_whitespace": True}

    min_price: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    max_price: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    min_stock: int | None = Field(default=None, ge=0)
    min_rating: float | None = Field(default=None, ge=0, le=5, allow_inf_nan=False)
    category: str | None = Field(default=None, min_length=1, max_length=100)
    brand: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_price_range(self):
        if self.min_price is not None and self.max_price is not None and self.min_price > self.max_price:
            raise ValueError("min_price must be less than or equal to max_price")
        return self


class ProductAttributes(BaseModel):
    brand: str | None = None
    rating: float | None = None
    review_count: int | None = None
    stock: int | None = None
    availability: str | None = None
    discount: float | None = None
    tags: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    specifications: dict[str, Any] = Field(default_factory=dict)
    colors: list[str] = Field(default_factory=list)
    sizes: list[str] = Field(default_factory=list)


class SearchQueryRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=200)
    limit: int = Field(default=5, ge=1, le=20)
    filters: ProductFilters | None = None

    @field_validator("q", mode="before")
    @classmethod
    def strip_query(cls, value):
        # Trim before length validation so whitespace-only input is rejected.
        return value.strip() if isinstance(value, str) else value


class ProductSearchResult(ProductAttributes):
    id: str | int
    title: str
    description: str
    price: float
    category: str
    score: float


class SearchResponseEnvelope(BaseModel):
    results: list[ProductSearchResult]
    latency_ms: float


# ── Diagnostic fields for /ask endpoint ──────────────────

class RetrievedProduct(ProductAttributes):
    id: str | int
    title: str
    category: str
    price: float
    rrf_score: float


class TimingBreakdown(BaseModel):
    cache_lookup_ms: float
    search_ms: float
    llm_ms: float
    total_ms: float


class AskResponseEnvelope(BaseModel):
    answer: str
    source: str
    retrieved_products: list[RetrievedProduct]
    timing: TimingBreakdown
