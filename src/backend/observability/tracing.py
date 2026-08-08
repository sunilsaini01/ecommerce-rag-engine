"""LangSmith observability layer.

Design goals (see README "LangSmith Observability" section):

* The existing FastAPI/Qdrant/Redis/Groq pipeline is NOT migrated to
  LangChain. `@traceable` from the standalone `langsmith` SDK works on
  plain Python functions, so tracing is layered on top of the existing
  services without touching their control flow.
* Tracing is opt-in. `settings.LANGSMITH_TRACING` (see config/settings.py)
  must be true AND a LANGSMITH_API_KEY must be configured, otherwise every
  `traceable`-wrapped function call is a plain, un-instrumented function
  call — no client is constructed, no network call is attempted.
* Nothing that touches this module ever raises. A tracing failure (bad
  network, SDK error, disabled package) must never break a real user
  request — it only gets logged at DEBUG.
* No secrets ever flow into a traced function's arguments. Request headers
  (X-API-Key, Authorization), Redis/Qdrant credentials, and the Groq API
  key are never passed as parameters to any `@traceable`-decorated
  function in this codebase — tracing only ever sees queries, retrieved
  product ids/scores, prompts, and generated answers.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from loguru import logger

from config.settings import settings

F = TypeVar("F", bound=Callable[..., Any])

try:
    from langsmith import traceable as _langsmith_traceable

    LANGSMITH_INSTALLED = True
except ImportError:  # pragma: no cover - optional dependency missing
    LANGSMITH_INSTALLED = False
    logger.warning(
        "langsmith package is not installed — tracing will be a no-op "
        "regardless of LANGSMITH_TRACING."
    )

    def _langsmith_traceable(*_args: Any, **_kwargs: Any) -> Callable[[F], F]:
        def _decorator(func: F) -> F:
            return func

        return _decorator


def tracing_enabled() -> bool:
    """True only when the SDK is installed, tracing is explicitly enabled,
    and an API key is configured. Mirrors the guard in config/settings.py
    that sets os.environ['LANGSMITH_TRACING']."""
    return LANGSMITH_INSTALLED and settings.LANGSMITH_TRACING and bool(settings.LANGSMITH_API_KEY)


def traceable(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Drop-in re-export of `langsmith.traceable`.

    Safe to apply unconditionally to any function: when tracing is
    disabled the SDK's own `traceable` decorator (or our no-op fallback,
    if the package is missing) is a pass-through with no measurable
    overhead and no network activity.
    """
    return _langsmith_traceable(*args, **kwargs)


def root_metadata(**overrides: Any) -> dict[str, Any]:
    """Base metadata attached to the root span of a request. Extra,
    request-specific fields (endpoint, top_k, ...) are merged in by the
    caller via `overrides`. Never includes secrets — only static,
    non-sensitive configuration describing how the request was served."""
    meta: dict[str, Any] = {
        "environment": settings.ENVIRONMENT,
        "catalog_version": settings.CATALOG_VERSION,
        "prompt_version": settings.PROMPT_VERSION,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "sparse_model": "Qdrant/bm25",
        "llm_model": settings.MODEL_VERSION,
    }
    meta.update({k: v for k, v in overrides.items() if v is not None})
    return meta


def only_fields(*allowed_keys: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Builds a `process_inputs` callback (the `@traceable` hook for
    filtering what gets recorded as a run's inputs) that keeps only an
    explicit allowlist of argument names.

    Defense in depth: functions like `run_rag_pipeline` take live service
    objects (ProductSearchService, CacheService, LLMService) as arguments
    for dependency injection — and LLMService.client is a `groq.Groq`
    instance whose `api_key` attribute is real. The installed langsmith
    SDK's default serializer happens to fall back to `str(obj)` for
    non-JSON-serializable objects today, which does not leak it — but
    relying on that being true forever is exactly the kind of assumption
    that causes credential leaks after a library upgrade. Explicitly
    allowlisting which arguments are ever considered for serialization
    removes the risk entirely, regardless of serializer internals."""
    def _process(inputs: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in inputs.items() if k in allowed_keys}

    return _process


def langsmith_extra(tags: list[str] | None = None, **metadata: Any) -> dict[str, Any]:
    """Builds the `langsmith_extra` kwarg accepted by any `@traceable`
    function, e.g. `run_rag_pipeline(..., langsmith_extra=langsmith_extra(
    tags=["endpoint:ask"], top_k=5))`. Cheap to build even when tracing is
    disabled since it's just a plain dict; the SDK ignores it entirely
    when tracing is off."""
    return {"tags": tags or [], "metadata": root_metadata(**metadata)}
