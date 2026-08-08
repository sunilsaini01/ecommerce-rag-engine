import argparse
import json
import uuid
from pathlib import Path
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
)

from config.settings import settings
from src.ingestion.schemas import ProductRecord, IngestionReport, validate_catalog
from src.ingestion.embedder import ProductEmbedder

# ── Constants ────────────────────────────────────────────
DEFAULT_DATA_PATH = Path("data/raw/sample_products.json")
REPORTS_DIR = Path("data/reports")
BATCH_SIZE = 64
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


# ── Qdrant Setup ─────────────────────────────────────────
def get_client() -> QdrantClient:
    return QdrantClient(
        host=settings.QDRANT_HOST,
        grpc_port=settings.QDRANT_GRPC_PORT,
        prefer_grpc=True,
    )


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if settings.COLLECTION_NAME in existing:
        if not recreate:
            logger.info(f"Collection '{settings.COLLECTION_NAME}' already exists.")
            return
        logger.warning(f"Recreating collection '{settings.COLLECTION_NAME}' (--recreate).")
        client.delete_collection(settings.COLLECTION_NAME)

    client.create_collection(
        collection_name=settings.COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=384,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )
    logger.info(f"Collection '{settings.COLLECTION_NAME}' created.")


# ── Point ID Resolution ────────────────────────────────────
# Qdrant point IDs must be an unsigned int or a UUID — arbitrary strings
# (e.g. a real SKU like "TSHIRT-BLK-M") are not accepted. Numeric catalog
# IDs map straight through; anything else gets a deterministic UUID5 so the
# same product ID always resolves to the same point. The original catalog
# ID is kept in the payload (see build_points) and used as the canonical
# "id" everywhere downstream, so callers never see the UUID.
def resolve_point_id(product_id: str) -> int | str:
    if product_id.isdigit():
        return int(product_id)
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, product_id))


# ── Batch Processing ──────────────────────────────────────
def build_points(
    records: list[ProductRecord],
    dense_vectors: list,
    sparse_vectors: list[dict],
) -> list[PointStruct]:
    points = []
    for record, dense, sparse in zip(records, dense_vectors, sparse_vectors):
        point = PointStruct(
            id=resolve_point_id(record.id),
            vector={
                DENSE_VECTOR_NAME: dense.tolist(),
                SPARSE_VECTOR_NAME: SparseVector(
                    indices=sparse["indices"],
                    values=sparse["values"],
                ),
            },
            payload=record.to_payload(),
        )
        points.append(point)
    return points


# ── Quality Report ─────────────────────────────────────────
def print_report(report: IngestionReport) -> None:
    print("\n" + "=" * 70)
    print(f"{'INGESTION QUALITY REPORT':^70}")
    print("=" * 70)
    print(f"  Source file          : {report.source_file}")
    print(f"  Total records        : {report.total_records}")
    print(f"  Valid records        : {report.valid_records}")
    print(f"  Invalid records      : {report.invalid_records}")
    print(f"  Duplicate IDs        : {len(report.duplicate_ids)} {report.duplicate_ids[:10]}")
    print(f"  Unknown categories   : {report.unknown_categories or 'none'}")
    print("  Categories:")
    for category, count in sorted(report.categories.items(), key=lambda kv: -kv[1]):
        print(f"    - {category:<20} {count}")
    print("  Missing optional fields (count of records without a value):")
    for field, count in report.missing_optional_fields.items():
        print(f"    - {field:<20} {count}/{report.total_records}")
    if report.errors:
        print(f"  First {min(5, len(report.errors))} validation errors:")
        for err in report.errors[:5]:
            print(f"    - index={err['index']} id={err['id']} -> {err['errors']}")
    print("=" * 70)


def write_report(report: IngestionReport, catalog_version: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"ingestion_report_{catalog_version}.json"
    out_path.write_text(json.dumps(report.model_dump(), indent=2))
    logger.info(f"Ingestion report written to {out_path}")
    return out_path


# ── Entry Point ───────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and ingest a product catalog into Qdrant.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to the catalog JSON file.")
    parser.add_argument("--catalog-version", type=str, default=settings.CATALOG_VERSION, help="Label used for the report filename and logs.")
    parser.add_argument("--validate-only", action="store_true", help="Run validation and print/write the report without touching Qdrant.")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the Qdrant collection before ingesting (use when switching catalogs).")
    args = parser.parse_args()

    logger.info(f"Starting ingestion pipeline for {args.data_path} (catalog_version={args.catalog_version}).")

    with open(args.data_path, "r") as f:
        raw_records: list[dict] = json.load(f)
    logger.info(f"Loaded {len(raw_records)} raw records.")

    valid_records, report = validate_catalog(raw_records, source_file=str(args.data_path))
    print_report(report)
    write_report(report, args.catalog_version)

    if not report.is_ingestible():
        logger.error("Catalog failed ingestibility checks (no valid records, or duplicate IDs present). Aborting.")
        raise SystemExit(1)

    if args.validate_only:
        logger.info("Validate-only mode — skipping Qdrant ingestion.")
        return

    embedder = ProductEmbedder()
    client = get_client()
    ensure_collection(client, recreate=args.recreate)

    for batch_start in range(0, len(valid_records), BATCH_SIZE):
        batch = valid_records[batch_start: batch_start + BATCH_SIZE]
        logger.info(f"Processing batch {batch_start} → {batch_start + len(batch)}")

        texts = [r.get_text_payload() for r in batch]
        dense_vectors, sparse_vectors = embedder.embed_text_chunks(texts)

        points = build_points(batch, dense_vectors, sparse_vectors)

        client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=points,
        )
        logger.info(f"Uploaded {len(points)} points to Qdrant.")

    logger.info("Ingestion pipeline complete.")


if __name__ == "__main__":
    main()
