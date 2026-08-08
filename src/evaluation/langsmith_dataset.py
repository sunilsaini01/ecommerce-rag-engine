"""Creates/updates a LangSmith evaluation dataset (Part 6) from the
structured ground truth files, so evaluation runs are repeatable and
inspectable from the LangSmith UI, not just the terminal table.

This is a standalone CLI tool — it is NOT imported by the running
application, so it's fine for it to require LANGSMITH_API_KEY to be set
(unlike the app itself, which must work with tracing fully disabled).

Categories map onto Part 6 of the RAG evolution spec:
    simple_semantic, attribute, price, multi_constraint, ambiguous,
    negative, adversarial
via the ground truth's `level_name` field.

Usage:
    python -m src.evaluation.langsmith_dataset --ground-truth src/evaluation/ground_truth_100.json
"""

import argparse
import json
import sys

from loguru import logger

from config.settings import settings

# ground_truth "level_name" -> Part 6 dataset category
_CATEGORY_MAP = {
    "exact_lexical": "simple_semantic",
    "semantic_paraphrase": "simple_semantic",
    "attribute_based": "attribute",
    "price_constraint": "price",
    "multi_constraint": "multi_constraint",
    "ambiguous": "ambiguous",
    "negative": "negative",
    "adversarial": "adversarial",
    "legacy": "simple_semantic",
}


def build_examples(ground_truth_path: str) -> list[dict]:
    with open(ground_truth_path, "r") as f:
        cases = json.load(f)

    examples = []
    for case in cases:
        level_name = case.get("level_name", "legacy")
        examples.append({
            "inputs": {"query": case["query"]},
            "outputs": {
                "expected_product_ids": case["expected_product_ids"],
                "reference_answer": case.get("reference_answer", ""),
                "expected_facts": case.get("expected_facts", case.get("ideal_response_keywords", [])),
            },
            "metadata": {
                "category": _CATEGORY_MAP.get(level_name, level_name),
                "level": case.get("level", 0),
                "level_name": level_name,
                "adversarial_type": case.get("adversarial_type"),
            },
        })
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Create/update a LangSmith evaluation dataset.")
    parser.add_argument("--ground-truth", default="src/evaluation/ground_truth_100.json")
    parser.add_argument("--dataset-name", default="ecommerce-rag-eval")
    args = parser.parse_args()

    if not settings.LANGSMITH_API_KEY:
        logger.error(
            "LANGSMITH_API_KEY is not set. This tool pushes an evaluation "
            "dataset to smith.langchain.com and needs real credentials — "
            "set LANGSMITH_API_KEY in .env and try again."
        )
        sys.exit(1)

    from langsmith import Client

    client = Client(api_key=settings.LANGSMITH_API_KEY, api_url=settings.LANGSMITH_ENDPOINT)
    examples = build_examples(args.ground_truth)

    if client.has_dataset(dataset_name=args.dataset_name):
        dataset = client.read_dataset(dataset_name=args.dataset_name)
        logger.info(f"Dataset '{args.dataset_name}' already exists ({dataset.id}). Clearing old examples before re-upload.")
        existing = list(client.list_examples(dataset_id=dataset.id))
        if existing:
            client.delete_examples(example_ids=[ex.id for ex in existing])
    else:
        dataset = client.create_dataset(
            dataset_name=args.dataset_name,
            description=(
                "E-commerce RAG evaluation queries covering simple semantic, "
                "attribute, price, multi-constraint, ambiguous, negative, and "
                "adversarial (prompt injection / off-topic) categories. "
                "Generated from src/evaluation/ground_truth_100.json — see "
                "src/evaluation/langsmith_dataset.py."
            ),
        )
        logger.info(f"Created dataset '{args.dataset_name}' ({dataset.id}).")

    client.create_examples(dataset_id=dataset.id, examples=examples)

    by_category: dict[str, int] = {}
    for ex in examples:
        cat = ex["metadata"]["category"]
        by_category[cat] = by_category.get(cat, 0) + 1

    logger.info(f"Uploaded {len(examples)} examples to '{args.dataset_name}'.")
    for cat, count in sorted(by_category.items()):
        print(f"  {cat:<20}: {count}")


if __name__ == "__main__":
    main()
