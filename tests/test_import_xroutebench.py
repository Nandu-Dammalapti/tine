import importlib.util
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "tine_import_xroutebench", Path(__file__).parents[1] / "scripts" / "import_xroutebench.py")
importer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(importer)


def test_answers_are_taken_from_the_end_of_a_response():
    assert importer.choice_answer("I think (a) at first, but the answer is ( C )") == "C"
    assert importer.choice_answer("C. no parenthesis") is None
    assert importer.number_answer("20 feet is 240 inches, so it takes **1,200.50 days**.") == "1200.5"
    assert importer.number_answer("no digits") is None
    assert importer.boxed_answer(r"so \boxed{1} then \boxed{\frac{3}{4} x}") == r"\frac{3}{4}x"
    assert importer.boxed_answer(r"\boxed{unclosed") is None


def rows():
    base = {"ground_truth": "B", "choices": "", "task_id": None, "token_num": "0", "embedding_id": "0", "split": "train"}
    table = [
        ("mmlu", "q1", "em_mc", "small", "it is (B)", "100", "10", "0.5", "1.0"),
        ("mmlu", "q1", "em_mc", "large", "(A)", "100", "20", "0.9", "0.0"),
        ("mmlu", "q1", "em_mc", "large", "(A)", "100", "20", "0.9", "0.0"),  # duplicate row
        ("gsm8k", "q2", "GSM8K", "small", "so 42 apples", "50", "5", "0.1", "1.0"),
        ("gsm8k", "q2", "GSM8K", "large", "the answer is 41", "50", "5", "0.2", "0.0"),
        ("squad", "q3", "f1_score", "small", "Paris", "50", "5", "0.1", "0.4"),
    ]
    return pd.DataFrame([{**base, "task_name": task, "query": query, "metric": metric, "model_name": model,
                          "response": response, "input_tokens": tokens_in, "output_tokens": tokens_out,
                          "response_time": seconds, "performance": score}
                         for task, query, metric, model, response, tokens_in, tokens_out, seconds, score in table])


PRICES = pd.DataFrame({"model_name": ["small", "large"], "input_price_per_1m": [1.0, 10.0],
                       "output_price_per_1m": [2.0, 20.0]})


def test_convert_prices_rows_from_tokens_and_extracts_answers():
    outcomes, answers, report = importer.convert(rows(), PRICES, tasks=["mmlu", "gsm8k"], task_type="exact")
    assert report["queries"] == 2 and report["duplicate_rows_dropped"] == 1 and len(outcomes) == 4
    assert report["multiple_choice_extraction_matches_score"] == 1.0
    small = outcomes[(outcomes.model == "small") & outcomes.task_id.str.contains("mmlu")].iloc[0]
    assert small.cost_usd == pytest.approx((100 * 1.0 + 10 * 2.0) / 1e6) and bool(small.success)
    assert small.latency_ms == pytest.approx(500.0) and small.task_type == "exact"
    assert sorted(row["answer"] for row in answers) == ["41", "42", "A", "B"]
    assert outcomes.task_id.str.startswith("xroutebench:").all()


def test_convert_refuses_fractional_scores_and_unpriced_models():
    with pytest.raises(ValueError, match="fractional"):
        importer.convert(rows(), PRICES, tasks=["squad"], task_type="x")
    with pytest.raises(ValueError, match="No price"):
        importer.convert(rows(), PRICES[PRICES.model_name == "small"], tasks=["mmlu"], task_type="x")


@pytest.mark.parametrize("column,value", [
    ("input_tokens", 100.9), ("output_tokens", 10.9), ("input_tokens", True),
    ("input_tokens", -1), ("output_tokens", float("inf")),
])
def test_invalid_token_counts_cannot_produce_recorded_costs(column, value):
    frame = rows()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        importer.convert(frame, PRICES, tasks=["mmlu"], task_type="x")


@pytest.mark.parametrize("column,value", [
    ("performance", "1.0"), ("response", "a different answer (B)"),
    ("input_tokens", "101"), ("ground_truth", "A"), ("metric", "GSM8K"),
    ("response_time", "1.1"),
])
def test_conflicting_duplicate_evidence_is_not_selected_by_row_order(column, value):
    frame = rows()
    frame.loc[2, column] = value
    with pytest.raises(ValueError, match="Conflicting.*task.*model"):
        importer.convert(frame, PRICES, tasks=["mmlu"], task_type="x")


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, True])
def test_invalid_model_prices_are_rejected(value):
    prices = PRICES.astype(object)
    prices.loc[0, "input_price_per_1m"] = value
    with pytest.raises(ValueError, match="input_price_per_1m"):
        importer.convert(rows(), prices, tasks=["mmlu"], task_type="x")


def test_duplicate_price_rows_are_rejected_explicitly():
    prices = pd.concat([PRICES, PRICES.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique model"):
        importer.convert(rows(), prices, tasks=["mmlu"], task_type="x")


def test_identical_evidence_across_source_splits_is_deduplicated():
    frame = rows()
    frame.loc[2, "split"] = "test"
    outcomes, _, report = importer.convert(frame, PRICES, tasks=["mmlu"], task_type="x")
    assert len(outcomes) == 2 and report["duplicate_rows_dropped"] == 1


def test_unknown_latency_is_preserved_and_representable_large_costs_stay_finite():
    frame = rows()
    frame.loc[0, "response_time"] = None
    prices = PRICES.copy()
    prices.loc[0, ["input_price_per_1m", "output_price_per_1m"]] = 1e308
    outcomes, _, _ = importer.convert(frame, prices, tasks=["mmlu"], task_type="x")
    small = outcomes[outcomes.model == "small"].iloc[0]
    assert pd.isna(small.latency_ms)
    assert small.cost_usd == pytest.approx(1.1e304)


@pytest.mark.parametrize("output,source", [
    ("output", "answers"), ("output", "routing"),
    ("answers", "candidates"), ("answers", "routing"),
])
def test_cli_refuses_paths_that_can_overwrite_input_or_other_output(tmp_path, output, source):
    paths = {name: tmp_path / name for name in ["routing", "candidates", "output", "answers"]}
    paths[output] = paths[source]
    paths[source].write_text("preserve source")
    with pytest.raises(SystemExit) as error:
        importer.main(["--routing", str(paths["routing"]), "--candidates", str(paths["candidates"]),
                       "--tasks", "mmlu", "--output", str(paths["output"]), "--answers", str(paths["answers"])])
    assert error.value.code == 2
    assert paths[source].read_text() == "preserve source"
