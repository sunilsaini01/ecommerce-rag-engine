import os

import pytest


@pytest.fixture(autouse=True)
def _disable_langsmith_tracing_for_unit_tests(request):
    """Unit tests must never depend on network access, but a developer's
    local .env (env_file for pydantic-settings) may have real LangSmith
    tracing enabled for manual testing. Force it off for every test not
    explicitly marked `integration`, so `python -m pytest` behaves
    identically on a clean CI checkout (no .env at all) and on a
    developer machine with tracing configured."""
    if "integration" in request.keywords:
        yield
        return

    from config.settings import settings

    original = settings.LANGSMITH_TRACING
    original_env = os.environ.get("LANGSMITH_TRACING")
    settings.LANGSMITH_TRACING = False
    os.environ["LANGSMITH_TRACING"] = "false"
    yield
    settings.LANGSMITH_TRACING = original
    if original_env is not None:
        os.environ["LANGSMITH_TRACING"] = original_env


@pytest.fixture
def valid_product() -> dict:
    return {
        "id": "999",
        "title": "Test Widget",
        "description": "A widget for testing.",
        "price": 19.99,
        "category": "Electronics",
    }


@pytest.fixture
def valid_product_with_optional_fields() -> dict:
    return {
        "id": "998",
        "title": "Test Gadget Pro",
        "description": "A gadget with full optional attributes.",
        "price": 49.99,
        "category": "Gaming",
        "brand": "Testcorp",
        "rating": 4.5,
        "review_count": 120,
        "stock": 10,
        "tags": ["test", "gadget"],
        "features": ["feature A", "feature B"],
        "colors": ["black", "white"],
        "sizes": ["M", "L"],
        "specifications": {"weight_g": 200},
        "discount": 10.0,
        "availability": "in_stock",
    }
