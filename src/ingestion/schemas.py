from pydantic import BaseModel, Field, field_validator
from typing import Any, Optional

# Target category taxonomy for catalog expansion (see catalog_generator.py).
# Records with a category outside this set are NOT rejected — the original
# 39-product baseline predates this taxonomy and stays valid — but they are
# flagged as "unknown_categories" in the ingestion quality report so drift
# is visible instead of silent.
CANONICAL_CATEGORIES: set[str] = {
    "Electronics",
    "Outdoor Gear",
    "Footwear",
    "Sports & Fitness",
    "Clothing",
    "Home Appliances",
    "Home Office",
    "Nutrition",
    "Travel",
    "Kitchen",
    "Personal Care",
    "Gaming",
}

_ALLOWED_AVAILABILITY = {"in_stock", "out_of_stock", "preorder", "discontinued"}

# Optional fields added on top of the original baseline schema
# (id, title, description, price, category, metadata). None of these are
# required so the 39-product baseline dataset — which has none of them —
# keeps validating unchanged.
OPTIONAL_ATTRIBUTE_FIELDS: tuple[str, ...] = (
    "brand",
    "rating",
    "review_count",
    "stock",
    "tags",
    "features",
    "colors",
    "sizes",
    "specifications",
    "discount",
    "availability",
)


class ProductRecord(BaseModel):
    id: str
    title: str
    description: str = ""
    price: float = Field(ge=0.0)
    category: str = "Unknown"
    metadata: dict[str, Any] = {}

    # ── Optional structured attributes (Part 8) ────────────
    brand: Optional[str] = None
    rating: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    review_count: Optional[int] = Field(default=None, ge=0)
    stock: Optional[int] = Field(default=None, ge=0)
    tags: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    colors: list[str] = Field(default_factory=list)
    sizes: list[str] = Field(default_factory=list)
    specifications: dict[str, Any] = Field(default_factory=dict)
    discount: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    availability: Optional[str] = None

    @field_validator("id", mode="before")
    @classmethod
    def coerce_id_to_string(cls, v: Any) -> str:
        return str(v)

    @field_validator("title", mode="before")
    @classmethod
    def strip_and_validate_title(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("title must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("title cannot be empty")
        return stripped

    @field_validator("category", mode="before")
    @classmethod
    def validate_category(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("category must be a non-empty string")
        return v.strip()

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(cls, v: Any) -> dict:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("metadata must be an object")
        return v

    @field_validator("specifications", mode="before")
    @classmethod
    def validate_specifications(cls, v: Any) -> dict:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("specifications must be an object")
        return v

    @field_validator("tags", "features", "colors", "sizes", mode="before")
    @classmethod
    def validate_string_list(cls, v: Any) -> list:
        if v is None:
            return []
        if not isinstance(v, list) or not all(isinstance(item, str) for item in v):
            raise ValueError("must be a list of strings")
        return v

    @field_validator("availability", mode="before")
    @classmethod
    def validate_availability(cls, v: Any) -> Any:
        if v is None:
            return v
        normalized = str(v).strip().lower()
        if normalized not in _ALLOWED_AVAILABILITY:
            raise ValueError(f"availability must be one of {sorted(_ALLOWED_AVAILABILITY)}")
        return normalized

    def get_text_payload(self) -> str:
        parts = [
            f"Title: {self.title}",
            f"Category: {self.category}",
            f"Description: {self.description}",
        ]
        if self.brand:
            parts.append(f"Brand: {self.brand}")
        if self.features:
            parts.append("Features: " + ", ".join(self.features))
        if self.tags:
            parts.append("Tags: " + ", ".join(self.tags))
        return " | ".join(parts)

    def is_known_category(self) -> bool:
        return self.category in CANONICAL_CATEGORIES

    def to_payload(self) -> dict[str, Any]:
        """Full Qdrant point payload, including optional attributes. Fields
        left at their default (None / [] / {}) are still included — Qdrant
        filtering (a possible future step, not implemented yet) needs them
        present, even if empty, to distinguish 'no data' from 'not indexed'."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "price": self.price,
            "category": self.category,
            "metadata": self.metadata,
            "brand": self.brand,
            "rating": self.rating,
            "review_count": self.review_count,
            "stock": self.stock,
            "tags": self.tags,
            "features": self.features,
            "colors": self.colors,
            "sizes": self.sizes,
            "specifications": self.specifications,
            "discount": self.discount,
            "availability": self.availability,
        }


class IngestionReport(BaseModel):
    """Structured ingestion quality report (Part 8). Written to
    data/reports/ and printed to stdout by src/ingestion/ingest.py."""

    source_file: str
    total_records: int
    valid_records: int
    invalid_records: int
    duplicate_ids: list[str]
    categories: dict[str, int]
    unknown_categories: list[str]
    missing_optional_fields: dict[str, int]
    errors: list[dict[str, Any]]

    def is_ingestible(self) -> bool:
        """Hard-fail conditions for CI/ops: nothing valid to ingest, or
        duplicate IDs present (would silently overwrite points in Qdrant).
        A nonzero count of *invalid* records is tolerated and logged —
        matches the existing behavior of skipping bad rows rather than
        aborting the whole batch."""
        return self.valid_records > 0 and not self.duplicate_ids


def validate_catalog(raw_records: list[dict], source_file: str = "") -> tuple[list[ProductRecord], IngestionReport]:
    """Single-pass validation across the WHOLE catalog (not per-batch), so
    duplicate-ID detection works across batch boundaries. Per-record
    Pydantic validation errors are collected, not raised — one malformed
    row does not abort the rest of the catalog."""
    from pydantic import ValidationError

    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    valid_records: list[ProductRecord] = []
    errors: list[dict[str, Any]] = []
    categories: dict[str, int] = {}
    unknown_categories: set[str] = set()
    missing_counts = {field: 0 for field in OPTIONAL_ATTRIBUTE_FIELDS}

    for idx, raw in enumerate(raw_records):
        try:
            record = ProductRecord(**raw)
        except ValidationError as e:
            errors.append({"index": idx, "id": raw.get("id"), "errors": e.errors()})
            continue

        if record.id in seen_ids:
            duplicate_ids.append(record.id)
            errors.append({
                "index": idx,
                "id": record.id,
                "errors": [{"msg": "duplicate product id", "type": "duplicate_id"}],
            })
            continue
        seen_ids.add(record.id)

        categories[record.category] = categories.get(record.category, 0) + 1
        if not record.is_known_category():
            unknown_categories.add(record.category)

        for field in OPTIONAL_ATTRIBUTE_FIELDS:
            value = getattr(record, field)
            if value in (None, [], {}):
                missing_counts[field] += 1

        valid_records.append(record)

    report = IngestionReport(
        source_file=source_file,
        total_records=len(raw_records),
        valid_records=len(valid_records),
        invalid_records=len(raw_records) - len(valid_records),
        duplicate_ids=duplicate_ids,
        categories=categories,
        unknown_categories=sorted(unknown_categories),
        missing_optional_fields=missing_counts,
        errors=errors,
    )
    return valid_records, report
