import os

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = Field(default=6333)
    QDRANT_GRPC_PORT: int = Field(default=6334)
    COLLECTION_NAME: str = "ecommerce_products"
    GROQ_API_KEY: str = ""
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = Field(default=6379)

    # Optional — leave blank to disable auth (local/dev default).
    API_KEY: str = ""
    # Applies per client IP, only to the costly /ask endpoint.
    RATE_LIMIT_PER_MINUTE: int = Field(default=30)

    # ── Observability / environment ───────────────────────
    ENVIRONMENT: str = "local"

    # ── Cache versioning (see src/backend/services/cache.py) ──
    # Bumping any of these invalidates all previously cached answers,
    # so a catalog reload or prompt/model change can't serve a stale
    # answer generated under the old configuration.
    CATALOG_VERSION: str = "baseline-39"
    PROMPT_VERSION: str = "v1"
    MODEL_VERSION: str = "llama-3.1-8b-instant"

    # ── LangSmith observability — disabled unless explicitly configured ──
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "ecommerce-rag-engine"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }


settings = Settings()

# The `langsmith` SDK reads its configuration from process environment
# variables directly (not from this Settings object), so when tracing is
# enabled we mirror the validated values into os.environ. This keeps a
# single source of truth (.env -> Settings) while still satisfying the
# SDK's own env-var contract. When tracing is off or no API key is
# configured, we explicitly force LANGSMITH_TRACING=false so the SDK never
# attempts a network call, even if a stray env var was set elsewhere.
if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
else:
    os.environ["LANGSMITH_TRACING"] = "false"
