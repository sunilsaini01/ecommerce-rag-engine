from src.backend.observability.tracing import traceable


@traceable(name="context_builder", run_type="tool")
def build_context(products: list[dict]) -> dict:
    """Assembles the verified-context blob handed to the LLM. Traced on its
    own so a LangSmith trace can answer 'what context was sent to the LLM'
    without having to cross-reference the retrieval span."""
    lines = [
        f"- Title: {p.get('title', '')} | "
        f"Category: {p.get('category', '')} | "
        f"Price: ${p.get('price', 0.0)} | "
        f"Description: {p.get('description', '')}"
        for p in products
    ]
    context = "\n".join(lines)
    return {"context": context, "product_count": len(products)}
