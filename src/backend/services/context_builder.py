import json

from src.backend.models.api_schemas import ProductAttributes
from src.backend.observability.tracing import traceable


@traceable(name="context_builder", run_type="tool")
def build_context(products: list[dict]) -> dict:
    """Assembles the verified-context blob handed to the LLM. Traced on its
    own so a LangSmith trace can answer 'what context was sent to the LLM'
    without having to cross-reference the retrieval span."""
    lines = []
    for product in products:
        fields = {key: product[key] for key in (
            "id", "title", "category", "price", "description", *ProductAttributes.model_fields,
        ) if key in product and product[key] is not None and product[key] != [] and product[key] != {}}
        lines.append(json.dumps(fields, ensure_ascii=False))
    context = "\n".join(lines)
    return {"context": context, "product_count": len(products)}
