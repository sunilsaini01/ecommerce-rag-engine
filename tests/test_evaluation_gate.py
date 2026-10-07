"""Regression coverage for evaluation failures that used to pass the CI gate."""

import json
import sys

import pytest

from src.evaluation import eval_v2


@pytest.fixture
def thresholds_path(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({
        "_comment": "Test thresholds",
        "recall_at_k": {"value": 0.7, "reason": "Retrieval floor"},
        "refusal_correct": {"value": 0.75, "reason": "Refusal floor"},
    }))
    return str(path)


def passing_results():
    return [
        {"level_name": "semantic", "recall_at_k": 1.0, "source": "llm"},
        {"level_name": "negative", "refusal_correct": 1.0, "source": "cache"},
    ]


def test_complete_successful_run_passes(thresholds_path):
    overall = eval_v2.aggregate(passing_results())
    assert overall["failed_count"] == 0
    assert eval_v2.check_thresholds(overall, thresholds_path)


@pytest.mark.parametrize("results", [
    [],
    [{"error": "Backend unavailable"}],
    passing_results() + [{"error": "Timeout"}],
    passing_results() + [{"source": "llm_error"}],
])
def test_empty_or_failed_run_fails(results, thresholds_path):
    assert not eval_v2.check_thresholds(eval_v2.aggregate(results), thresholds_path)


@pytest.mark.parametrize("value", [None, "not_measured", float("nan"), float("inf"), True])
def test_invalid_required_metric_fails(value, thresholds_path):
    overall = eval_v2.aggregate(passing_results())
    overall["recall_at_k"] = value
    assert not eval_v2.check_thresholds(overall, thresholds_path)


def test_absent_required_metric_fails(thresholds_path):
    overall = eval_v2.aggregate(passing_results())
    del overall["refusal_correct"]
    assert not eval_v2.check_thresholds(overall, thresholds_path)


def test_below_threshold_fails(thresholds_path):
    overall = eval_v2.aggregate(passing_results())
    overall["recall_at_k"] = 0.69
    assert not eval_v2.check_thresholds(overall, thresholds_path)


@pytest.mark.parametrize("results", [
    [],
    [{"level_name": "semantic", "error": "Backend unavailable"}],
    passing_results() + [{"level_name": "semantic", "error": "Timeout"}],
    passing_results() + [{"level_name": "semantic", "source": "llm_error"}],
    [{"level_name": "semantic", "recall_at_k": 1.0}],
])
def test_cli_gate_exits_nonzero_and_preserves_report(monkeypatch, tmp_path, thresholds_path, results):
    monkeypatch.setattr(sys, "argv", ["eval_v2", "--gate", "--catalog-version", "test"])
    monkeypatch.setattr(eval_v2, "THRESHOLDS_PATH", thresholds_path)
    monkeypatch.setattr(eval_v2, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(eval_v2, "run_evaluation", lambda *args, **kwargs: {
        "results": results, "cache_hit_rate": 0.0,
    })

    with pytest.raises(SystemExit) as exc:
        eval_v2.main()

    assert exc.value.code == 1
    report = json.loads((tmp_path / "eval_report_test.json").read_text())
    assert report["raw_results"] == results


def test_cli_gate_success(monkeypatch, tmp_path, thresholds_path):
    monkeypatch.setattr(sys, "argv", ["eval_v2", "--gate", "--catalog-version", "test"])
    monkeypatch.setattr(eval_v2, "THRESHOLDS_PATH", thresholds_path)
    monkeypatch.setattr(eval_v2, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(eval_v2, "run_evaluation", lambda *args, **kwargs: {
        "results": passing_results(), "cache_hit_rate": 1.0,
    })
    eval_v2.main()


@pytest.fixture
def category_thresholds_path(thresholds_path):
    from pathlib import Path

    path = Path(thresholds_path)
    thresholds = json.loads(path.read_text())
    thresholds["by_level"] = {
        "semantic": {
            "min_count": 1,
            "metrics": {"recall_at_k": {"value": 0.7, "reason": "Category retrieval floor"}},
        },
        "negative": {
            "min_count": 1,
            "metrics": {"refusal_correct": {"value": 0.75, "reason": "Category refusal floor"}},
        },
    }
    path.write_text(json.dumps(thresholds))
    return thresholds_path


def category_results(results):
    return {
        name: eval_v2.aggregate([row for row in results if row["level_name"] == name])
        for name in {row["level_name"] for row in results}
    }


def test_all_required_categories_pass(category_thresholds_path):
    results = passing_results()
    assert eval_v2.check_thresholds(
        eval_v2.aggregate(results), category_thresholds_path, category_results(results),
    )


@pytest.mark.parametrize("weak_metric,weak_category", [
    ("recall_at_k", "semantic"),
    ("refusal_correct", "negative"),
])
def test_weak_category_fails_despite_passing_overall(category_thresholds_path, weak_metric, weak_category):
    results = passing_results()
    results += [{"level_name": "other", weak_metric: 1.0} for _ in range(10)]
    for row in results:
        if row["level_name"] == weak_category:
            row[weak_metric] = 0.0
    overall = eval_v2.aggregate(results)
    assert overall[weak_metric] > 0.75
    assert not eval_v2.check_thresholds(overall, category_thresholds_path, category_results(results))


@pytest.mark.parametrize("by_level", [None, {}, {"semantic": {"count": 1, "recall_at_k": 1.0}}])
def test_missing_category_fails(category_thresholds_path, by_level):
    assert not eval_v2.check_thresholds(
        eval_v2.aggregate(passing_results()), category_thresholds_path, by_level,
    )


@pytest.mark.parametrize("change", [
    {"count": 0},
    {"failed_count": 1},
    {"recall_at_k": None},
    {"recall_at_k": float("nan")},
])
def test_incomplete_or_invalid_category_fails(category_thresholds_path, change):
    results = passing_results()
    by_level = category_results(results)
    by_level["semantic"].update(change)
    assert not eval_v2.check_thresholds(eval_v2.aggregate(results), category_thresholds_path, by_level)


def test_cli_rejects_weak_category_and_preserves_report(monkeypatch, tmp_path, category_thresholds_path):
    results = passing_results()
    results[1]["refusal_correct"] = 0.0
    results += [{"level_name": "other", "refusal_correct": 1.0} for _ in range(10)]
    monkeypatch.setattr(sys, "argv", ["eval_v2", "--gate", "--catalog-version", "test"])
    monkeypatch.setattr(eval_v2, "THRESHOLDS_PATH", category_thresholds_path)
    monkeypatch.setattr(eval_v2, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(eval_v2, "run_evaluation", lambda *args, **kwargs: {
        "results": results, "cache_hit_rate": 0.0,
    })
    with pytest.raises(SystemExit) as exc:
        eval_v2.main()
    assert exc.value.code == 1
    report = json.loads((tmp_path / "eval_report_test.json").read_text())
    assert report["overall"]["refusal_correct"] > 0.75
    assert report["by_level"]["negative"]["refusal_correct"] == 0.0


def test_default_thresholds_cover_full_benchmark():
    from collections import Counter
    from pathlib import Path

    thresholds = json.loads(Path(eval_v2.THRESHOLDS_PATH).read_text(encoding="utf-8"))
    counts = Counter(case["level_name"] for case in eval_v2.load_ground_truth(eval_v2.DEFAULT_GROUND_TRUTH))
    assert set(thresholds["by_level"]) == set(counts)
    for name, spec in thresholds["by_level"].items():
        assert spec["min_count"] == counts[name]
    for name in ("negative", "adversarial"):
        assert set(thresholds["by_level"][name]["metrics"]) == {"refusal_correct"}
    assert "recall_at_k" not in thresholds["by_level"]["ambiguous"]["metrics"]
