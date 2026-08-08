import pytest
from pydantic import ValidationError

from src.ingestion.schemas import ProductRecord, validate_catalog
from src.ingestion.ingest import resolve_point_id


class TestProductRecord:
    def test_valid_product(self, valid_product):
        record = ProductRecord(**valid_product)
        assert record.id == "999"
        assert record.title == "Test Widget"
        assert record.price == 19.99

    def test_valid_product_with_optional_fields(self, valid_product_with_optional_fields):
        record = ProductRecord(**valid_product_with_optional_fields)
        assert record.brand == "Testcorp"
        assert record.rating == 4.5
        assert record.is_known_category()

    def test_baseline_product_without_optional_fields_still_valid(self, valid_product):
        """The original 39-product baseline has none of the Part 8 optional
        fields — this must keep validating exactly as before."""
        record = ProductRecord(**valid_product)
        assert record.brand is None
        assert record.tags == []
        assert record.specifications == {}

    def test_missing_title_rejected(self, valid_product):
        del valid_product["title"]
        with pytest.raises(ValidationError):
            ProductRecord(**valid_product)

    def test_empty_title_rejected(self, valid_product):
        valid_product["title"] = "   "
        with pytest.raises(ValidationError):
            ProductRecord(**valid_product)

    def test_negative_price_rejected(self, valid_product):
        valid_product["price"] = -5.0
        with pytest.raises(ValidationError):
            ProductRecord(**valid_product)

    def test_malformed_metadata_rejected(self, valid_product):
        valid_product["metadata"] = "not-a-dict"
        with pytest.raises(ValidationError):
            ProductRecord(**valid_product)

    def test_invalid_rating_rejected(self, valid_product):
        valid_product["rating"] = 6.0  # out of 0-5 range
        with pytest.raises(ValidationError):
            ProductRecord(**valid_product)

    def test_negative_stock_rejected(self, valid_product):
        valid_product["stock"] = -1
        with pytest.raises(ValidationError):
            ProductRecord(**valid_product)

    def test_invalid_availability_rejected(self, valid_product):
        valid_product["availability"] = "maybe"
        with pytest.raises(ValidationError):
            ProductRecord(**valid_product)

    def test_id_coerced_to_string(self, valid_product):
        valid_product["id"] = 999
        record = ProductRecord(**valid_product)
        assert record.id == "999"

    def test_unknown_category_flagged_but_not_rejected(self, valid_product):
        """Categories outside CANONICAL_CATEGORIES are allowed (backward
        compatibility) but surfaced via is_known_category()."""
        valid_product["category"] = "Some New Category"
        record = ProductRecord(**valid_product)
        assert not record.is_known_category()


class TestValidateCatalog:
    def test_all_valid(self, valid_product):
        records, report = validate_catalog([valid_product])
        assert report.total_records == 1
        assert report.valid_records == 1
        assert report.invalid_records == 0
        assert report.is_ingestible()

    def test_duplicate_id_detected(self, valid_product):
        raw = [valid_product, dict(valid_product)]
        records, report = validate_catalog(raw)
        assert len(records) == 1
        assert report.duplicate_ids == ["999"]
        assert not report.is_ingestible()

    def test_invalid_record_skipped_not_fatal(self, valid_product):
        bad = dict(valid_product)
        bad["id"] = "1000"
        bad["price"] = -1.0
        raw = [valid_product, bad]
        records, report = validate_catalog(raw)
        assert report.total_records == 2
        assert report.valid_records == 1
        assert report.invalid_records == 1
        assert report.is_ingestible()  # one bad row shouldn't block ingestion

    def test_empty_catalog_not_ingestible(self):
        records, report = validate_catalog([])
        assert not report.is_ingestible()

    def test_category_counts_and_missing_fields(self, valid_product):
        records, report = validate_catalog([valid_product])
        assert report.categories == {"Electronics": 1}
        assert report.missing_optional_fields["brand"] == 1


class TestResolvePointId:
    def test_numeric_id_maps_to_int(self):
        assert resolve_point_id("101") == 101

    def test_non_numeric_id_maps_to_deterministic_uuid(self):
        first = resolve_point_id("SKU-ABC")
        second = resolve_point_id("SKU-ABC")
        assert first == second  # deterministic
        assert resolve_point_id("SKU-XYZ") != first
