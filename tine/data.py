"""Validated outcome tables and an explicit adapter for LLMRouterBench results.

Canonical rows identify one (tenant, task_id, model) outcome. A pending outcome
has ``success=None``; evaluation must exclude it explicitly. Costs are recorded
per request in USD, never inferred from an aggregate or invented from a score.
Token counts, latency, and timestamps may be unknown. No loader executes model
outputs or benchmark tests.

The upstream format was checked against ``ynulihao/LLMRouterBench`` README.md
and baselines/data_loader.py: result JSON contains ``records`` and normally
``dataset_name``, ``split``, and ``model_name``. Each record contains ``index``,
``origin_query``/``prompt``, ``score``, token counts, and ``cost``. Directory
layout is ``<dataset>/<split>/<model>/<timestamp>.json``. Import costs describe
the supplied historical recording, not current provider prices.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import date, datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from tine.artifacts import atomic_write_text

CANONICAL_COLUMNS = [
    "tenant", "task_id", "prompt", "task_type", "model", "success", "cost_usd",
    "input_tokens", "output_tokens", "cached_tokens", "latency_ms", "timestamp",
]
_REQUIRED = CANONICAL_COLUMNS[:7]
_NUMERIC_OPTIONAL = ["input_tokens", "output_tokens", "cached_tokens", "latency_ms"]
_TEXT_COLUMNS = ["tenant", "task_id", "prompt", "task_type", "model"]


def _unique_csv_header(source):
    """Reject duplicate field names before pandas can silently rename them."""
    for header in csv.reader(source):
        if not header or (len(header) == 1 and not header[0].strip()):
            continue
        if len(header) != len(set(header)):
            raise ValueError("CSV must contain unique columns")
        break
    source.seek(0)


def _unique_json_object(pairs):
    """Keep contradictory fields from disappearing during JSON decoding."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key!r}")
        result[key] = value
    return result


def _success(value):
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false", "0", "1"}:
        return value.lower() in {"true", "1"}
    raise ValueError("success must be true, false, 0, 1, or null; scores need explicit conversion")


def validate_outcomes(frame: pd.DataFrame, *, require_known: bool = False) -> pd.DataFrame:
    """Return a normalized copy or raise on ambiguous/invalid outcome data.

    Core columns through cost_usd are required. Missing optional measurements
    become null, never zero. Extra provenance columns are preserved. Set
    require_known=True at evaluation boundaries to reject pending labels.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("outcomes must be a pandas DataFrame")
    if not frame.columns.is_unique:
        raise ValueError("Outcomes must have unique column names")
    missing = sorted(set(_REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing outcome columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Outcome table is empty")
    result = frame.copy()
    for column in ["tenant", "task_id", "prompt", "task_type", "model"]:
        if result[column].isna().any():
            raise ValueError(f"{column} must not be null")
        if column == "task_id":
            result[column] = result[column].astype(str)
        if not result[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{column} must contain nonempty strings")
    result["success"] = result["success"].map(_success).astype(object)
    if require_known and result["success"].isna().any():
        raise ValueError("Pending success labels cannot be evaluated")
    for column in ["cost_usd", *_NUMERIC_OPTIONAL]:
        if column in result and result[column].map(
            lambda value: isinstance(value, (bool, np.bool_, complex, np.complexfloating))
        ).any():
            raise ValueError(f"{column} must contain real numeric measurements; booleans and complex values are invalid")
    costs = pd.to_numeric(result["cost_usd"], errors="raise")
    if costs.isna().any() or not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("cost_usd must contain finite nonnegative recorded per-request costs")
    result["cost_usd"] = costs.astype(float)
    for column in _NUMERIC_OPTIONAL:
        if column not in result:
            result[column] = np.nan
        values = pd.to_numeric(result[column], errors="raise")
        known = values.dropna()
        if not np.isfinite(known).all() or (known < 0).any():
            raise ValueError(f"{column} must be null or finite and nonnegative")
        if column != "latency_ms" and (known != np.floor(known)).any():
            raise ValueError(f"{column} must contain whole token counts")
        result[column] = values
    comparable = result["cached_tokens"].notna() & result["input_tokens"].notna()
    if (result.loc[comparable, "cached_tokens"] > result.loc[comparable, "input_tokens"]).any():
        raise ValueError("cached_tokens cannot exceed input_tokens")
    if "timestamp" not in result:
        result["timestamp"] = None
    known_times = result["timestamp"].dropna()
    if not known_times.empty:
        try:
            pd.to_datetime(known_times, utc=True, format="mixed", errors="raise")
        except (ValueError, TypeError) as error:
            raise ValueError("timestamp must be null or a parseable date/time") from error
    keys = ["tenant", "task_id", "model"]
    if result.duplicated(keys).any():
        raise ValueError("Duplicate (tenant, task_id, model) outcomes; choose one recorded run explicitly")
    grouped = result.groupby(["tenant", "task_id"], sort=False)
    for column in ["prompt", "task_type"]:
        if (grouped[column].nunique(dropna=False) > 1).any():
            raise ValueError(f"Conflicting {column} for the same tenant/task_id")
    return result.reset_index(drop=True)


def read_outcomes(path: str | Path, *, require_known: bool = False) -> pd.DataFrame:
    """Read a canonical CSV, JSONL, or JSON array, then validate it."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        source = StringIO(path.read_text(encoding="utf-8-sig"))
        _unique_csv_header(source)
        frame = pd.read_csv(source, dtype=dict.fromkeys(_TEXT_COLUMNS, str))
        # Preserve literal text while retaining the usual missing-value and
        # numeric parsing for measurements and existing provenance columns.
        source.seek(0)
        text = pd.read_csv(source, dtype=str, keep_default_na=False,
                           usecols=lambda column: column in [*_TEXT_COLUMNS, "source_task_type"])
        if "source_task_type" in text:
            # Empty optional provenance is unknown; literal names such as NA
            # are source labels, just like they are valid task types.
            text["source_task_type"] = text.source_task_type.mask(text.source_task_type.eq(""), None)
        frame[text.columns] = text
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as source:
            records = [json.loads(line, object_pairs_hook=_unique_json_object) for line in source if line.strip()]
        frame = pd.DataFrame(records)
    elif path.suffix.lower() == ".json":
        records = json.loads(path.read_text(), object_pairs_hook=_unique_json_object)
        if not isinstance(records, list):
            raise ValueError("Canonical JSON must be an array; use import_llmrouterbench for result bundles")
        frame = pd.DataFrame(records)
    else:
        raise ValueError("Expected .csv, .jsonl, .ndjson, or .json outcome file")
    return validate_outcomes(frame, require_known=require_known)


def _json_value(value):
    """Normalize nullable table values without rounding finite measurements."""
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def outcome_json_lines(frame: pd.DataFrame):
    """Encode validated outcome records without rounding finite measurements."""
    for row in frame.to_dict(orient="records"):
        yield json.dumps(_json_value(row), ensure_ascii=False, allow_nan=False) + "\n"


def write_outcomes(frame: pd.DataFrame, path: str | Path) -> None:
    """Write validated canonical CSV or JSONL without changing unknown labels."""
    frame = validate_outcomes(frame)
    path = Path(path)
    if path.suffix.lower() == ".csv":
        contents = frame.to_csv(index=False)
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        contents = "".join(outcome_json_lines(frame))
    else:
        raise ValueError("Output must be .csv, .jsonl, or .ndjson")
    atomic_write_text(path, contents)


def combine_outcomes(frames, *, task_type: str | None = None, exclude_models=(),
                     keep_unpriced_models: bool = False):
    """Pool several outcome tables (e.g. coding datasets) into one calibration group.

    Calibration runs per task type, so small datasets (HumanEval has 164 tasks)
    can never certify a threshold alone. Setting ``task_type`` relabels every row
    to one pooled group; the original label is kept in ``source_task_type``. The
    resulting guarantee covers tasks drawn like the pooled mix, not each dataset.

    Only models present in every input are kept, because evaluation needs every
    model's outcome on every task. Models whose recorded cost is zero on every row
    are dropped unless ``keep_unpriced_models``: a zero cost means "not recorded"
    (e.g. self-hosted runs), and routing would treat them as free.

    Returns (frame, report).
    """
    frames = [validate_outcomes(frame) for frame in frames]
    if not frames:
        raise ValueError("Give at least one outcome table")
    model_sets = [set(frame.model) for frame in frames]
    common = set.intersection(*model_sets)
    dropped = {model: "missing_from_some_inputs" for model in sorted(set.union(*model_sets) - common)}
    combined = pd.concat(frames, ignore_index=True)
    if "source_task_type" not in combined:
        combined["source_task_type"] = combined["task_type"]
    known_sources = combined.source_task_type.dropna()
    if not known_sources.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("source_task_type must contain nonempty strings or null")
    sources = combined.groupby(["tenant", "task_id"]).source_task_type
    if sources.nunique().gt(1).any():
        raise ValueError("Conflicting source_task_type for the same tenant/task_id")
    # A prior pool may have provenance while newly appended outcomes do not.
    # Recover a task's recorded source where available, otherwise retain its
    # current task type before the caller relabels this new pool.
    combined["source_task_type"] = sources.transform("first").fillna(combined.task_type)
    if task_type is not None:
        if not isinstance(task_type, str) or not task_type.strip():
            raise ValueError("task_type must be a nonempty string")
        combined["task_type"] = task_type
    for model in exclude_models:
        if model in common:
            dropped[model] = "excluded_by_request"
    costs = combined.groupby("model").cost_usd.agg(lambda c: bool((c == 0).all()))
    if not keep_unpriced_models:
        for model in sorted(common):
            if costs.get(model, False):
                dropped[model] = "cost_not_recorded_all_zero"
    kept = sorted(common - set(dropped))
    if not kept:
        sizes = ", ".join(f"input {i + 1}: {len(models)} models" for i, models in enumerate(model_sets))
        raise ValueError(f"No models remain after combining ({len(common)} models are in every "
                         f"input; {sizes}). Leave out inputs that use a different model set.")
    combined = combined.loc[combined.model.isin(kept)].reset_index(drop=True)
    zero = combined.loc[combined.cost_usd == 0]
    report = {
        "rows": len(combined),
        "tasks_by_source": combined.drop_duplicates(["tenant", "task_id"])
        .groupby("source_task_type").size().to_dict(),
        "task_type": task_type,
        "models_kept": kept,
        "models_dropped": dropped,
        "zero_cost_rows_in_kept_models": {
            model: {"rows": len(rows), "success_rate": float(rows.success.astype(bool).mean())
                    if rows.success.notna().all() else None}
            for model, rows in zero.groupby("model")
        },
    }
    return validate_outcomes(combined), report


def _record_timestamp(path: Path):
    match = re.search(r"(\d{8})_(\d{6})\.json$", path.name)
    if match:
        value = pd.to_datetime("".join(match.groups()), format="%Y%m%d%H%M%S", utc=True)
        return value.isoformat()
    return None


def import_llmrouterbench(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    tenant: str = "benchmark",
    dataset: str | None = None,
    model: str | None = None,
    split: str | None = None,
    score_threshold: float | None = None,
    run_selection: str = "error",
) -> pd.DataFrame:
    """Import extracted upstream result JSON, never download or execute data.

    A file's top-level metadata is authoritative; absent metadata may be supplied
    explicitly or inferred from the documented directory layout. ``dataset``,
    ``model``, and ``split`` also filter a directory import. Fractional scores
    require an explicit score_threshold; null scores stay pending. Cost must be
    present per record. Unknown cached tokens and latency remain null.

    Multiple runs for one dataset/split/model fail by default. ``run_selection=
    'latest'`` explicitly chooses the latest YYYYMMDD_HHMMSS filename, refusing
    ambiguous filenames rather than relying on extraction modification times.
    Provenance includes the source file SHA256, score, and conversion policy.
    """
    if score_threshold is not None and (
        not np.isfinite(score_threshold) or not 0 <= score_threshold <= 1
    ):
        raise ValueError("score_threshold must lie between 0 and 1")
    if run_selection not in {"error", "latest"}:
        raise ValueError("run_selection must be 'error' or 'latest'")
    root = Path(input_path)
    paths = [root] if root.is_file() else sorted(root.rglob("*.json"))
    if not paths:
        raise ValueError(f"No result JSON files found under {root}")
    grouped = {}
    for path in paths:
        raw = path.read_bytes()
        data = json.loads(raw, object_pairs_hook=_unique_json_object)
        if not isinstance(data, dict) or not isinstance(data.get("records"), list):
            raise ValueError(f"{path}: expected a result object with records[]")
        if data.get("demo", False):
            continue
        inferred = path.parts[-4:-1] if len(path.parts) >= 4 else (None, None, None)
        dataset_name = data.get("dataset_name") or (dataset if root.is_file() else None) or inferred[0]
        split_name = data.get("split") or (split if root.is_file() else None) or inferred[1]
        model_name = data.get("model_name") or (model if root.is_file() else None) or inferred[2]
        if not all(isinstance(value, str) and value.strip() for value in (dataset_name, split_name, model_name)):
            raise ValueError(f"{path}: dataset, split, and model metadata are required")
        if any(wanted is not None and actual != wanted for wanted, actual in [
            (dataset, dataset_name), (model, model_name), (split, split_name)
        ]):
            continue
        key = (dataset_name, split_name, model_name)
        entry = (path, raw, data)
        if key in grouped:
            if run_selection == "error":
                raise ValueError(f"Multiple recorded runs for {key}; select files or set run_selection='latest'")
            old_time = _record_timestamp(grouped[key][0])
            new_time = _record_timestamp(path)
            if old_time is None or new_time is None or old_time == new_time:
                raise ValueError(f"Ambiguous run timestamps for {key}")
            if new_time < old_time:
                continue
        grouped[key] = entry
    rows = []
    for (dataset_name, split_name, model_name), (path, raw, data) in sorted(grouped.items()):
        checksum = hashlib.sha256(raw).hexdigest()
        for record in data["records"]:
            if not isinstance(record, dict) or "index" not in record:
                raise ValueError(f"{path}: every record needs an explicit index")
            if record["index"] is None or not isinstance(record["index"], (str, int)):
                raise ValueError(f"{path}: record index must be a nonnull string or integer")
            prompt = record.get("origin_query") or record.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}: text-only origin_query/prompt is required")
            if "score" not in record:
                raise ValueError(f"{path}: missing score is not a failure label")
            score = record["score"]
            if score is None:
                success = None
            elif score_threshold is None:
                success = _success(score)
            else:
                if not isinstance(score, (int, float)) or not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"{path}: score must be numeric in [0,1]")
                success = score >= score_threshold
            if "cost" not in record or record["cost"] is None:
                raise ValueError(f"{path}: missing per-record cost; aggregate costs are insufficient")
            rows.append({
                "tenant": tenant,
                "task_id": f"{dataset_name}:{split_name}:{record['index']}",
                "prompt": prompt,
                "task_type": dataset_name,
                "model": model_name,
                "success": success,
                "cost_usd": record["cost"],
                "input_tokens": record.get("prompt_tokens"),
                "output_tokens": record.get("completion_tokens"),
                "cached_tokens": record.get("cached_tokens"),
                "latency_ms": record.get("latency_ms"),
                "timestamp": _record_timestamp(path),
                "source_path": str(path),
                "source_sha256": checksum,
                "source_score": score,
                "success_label_source": "binary_score" if score_threshold is None else f"score>={score_threshold}",
                "cost_source": "llmrouterbench_recorded_usd",
            })
    result = validate_outcomes(pd.DataFrame(rows))
    if output_path is not None:
        write_outcomes(result, output_path)
    return result


def synthetic_fixture(n_tasks: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic outcomes for CPU integration checks, not evidence.

    Easy boilerplate always succeeds on all models. Hard reasoning tasks have
    complementary random outcomes. Prompts include unique IDs to exercise split
    isolation; cost and outcome patterns are intentionally learnable.
    """
    if not isinstance(n_tasks, int) or isinstance(n_tasks, bool) or n_tasks < 1:
        raise ValueError("n_tasks must be a positive integer")
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n_tasks):
        hard = bool(rng.integers(0, 2))
        prompt = (
            f"Solve deep reasoning algorithm invariant concurrency issue case {index}."
            if hard else f"Write simple boilerplate formatting helper documentation case {index}."
        )
        draw = rng.random()
        inputs = int(rng.integers(40, 160))
        for model_name, probability, cost in [
            ("cheap", 0.30 if hard else 1.0, 0.001),
            ("mid", 0.75 if hard else 1.0, 0.004),
            ("frontier", 0.99 if hard else 1.0, 0.020),
        ]:
            rows.append({
                "tenant": "synthetic", "task_id": f"synthetic-{index}",
                "prompt": prompt, "task_type": "coding", "model": model_name,
                "success": bool(draw < probability), "cost_usd": cost * (1 + inputs / 1000),
                "input_tokens": inputs, "output_tokens": inputs // 2, "cached_tokens": 0,
                "latency_ms": cost * 10000, "timestamp": "2026-01-01T00:00:00Z",
                "cost_source": "synthetic", "success_label_source": "synthetic",
            })
    return validate_outcomes(pd.DataFrame(rows), require_known=True)
