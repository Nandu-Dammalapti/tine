"""Read original coding tasks and checks without producing model outcomes.

Supported sources are OpenAI's HumanEval.jsonl.gz and Google Research's
sanitized-mbpp.json. Checks are retained as text for later grading. This
module never runs checks or includes the reference solutions in its output.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


def _text(record: dict, field: str, location: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: {field} must be nonempty text")
    return value


def _mbpp_split(task_id: int) -> str:
    # The sanitized release inherits the original dataset's ID-based splits.
    for last_id, split in ((10, "prompt"), (510, "test"), (600, "validation"), (974, "train")):
        if task_id <= last_id:
            return split
    raise ValueError("MBPP task_id exceeds the official range")


def load_coding_tasks(
    input_path: str | Path, *, dataset: str, tenant: str = "benchmark"
) -> list[dict]:
    """Return task-only records with dataset-scoped IDs and source checks."""
    if dataset not in {"humaneval", "mbpp"}:
        raise ValueError("dataset must be 'humaneval' or 'mbpp'")
    if not isinstance(tenant, str) or not tenant.strip():
        raise ValueError("tenant must be nonempty text")
    path = Path(input_path)
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if dataset == "humaneval":
        contents = gzip.decompress(raw) if path.suffix.lower() == ".gz" else raw
        # JSON record separators are physical newlines. Unicode separators
        # inside prompt strings belong to those strings.
        records = (
            (number, json.loads(line))
            for number, line in enumerate(contents.split(b"\n"), 1)
            if line.strip()
        )
    else:
        rows = json.loads(raw)
        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected the sanitized MBPP JSON array")
        records = enumerate(rows, 1)

    tasks = []
    seen = set()
    for number, record in records:
        location = f"{path}, record {number}"
        if not isinstance(record, dict):
            raise ValueError(f"{location}: expected a task object")
        prompt = _text(record, "prompt", location)
        if dataset == "humaneval":
            source_id = _text(record, "task_id", location)
            split = "test"
            checks = {
                "entry_point": _text(record, "entry_point", location),
                "test": _text(record, "test", location),
            }
        else:
            source_id = record.get("task_id")
            if type(source_id) is not int or not 1 <= source_id <= 974:
                raise ValueError(f"{location}: MBPP task_id must be an integer from 1 to 974")
            split = _mbpp_split(source_id)
            imports = record.get("test_imports")
            tests = record.get("test_list")
            if not isinstance(imports, list) or not all(isinstance(item, str) for item in imports):
                raise ValueError(f"{location}: test_imports must be a list of strings")
            if not isinstance(tests, list) or not tests or not all(
                isinstance(item, str) and item.strip() for item in tests
            ):
                raise ValueError(f"{location}: test_list must contain nonempty test strings")
            checks = {"test_imports": imports, "test_list": tests}
        if source_id in seen:
            raise ValueError(f"{location}: duplicate task_id {source_id!r}")
        seen.add(source_id)
        tasks.append({
            "tenant": tenant,
            "task_id": f"{dataset}:{source_id}",
            "task_type": dataset,
            "prompt": prompt,
            "source_task_id": source_id,
            "source_split": split,
            "source_sha256": checksum,
            "checks": checks,
        })
    if not tasks:
        raise ValueError(f"{path}: no coding tasks found")
    return tasks


def write_coding_tasks(tasks: list[dict], output_path: str | Path) -> None:
    """Write prepared task-only rows as UTF-8 JSON Lines."""
    path = Path(output_path)
    if path.suffix.lower() not in {".jsonl", ".ndjson"}:
        raise ValueError("Coding task output must be .jsonl or .ndjson")
    contents = "".join(json.dumps(task, ensure_ascii=False, allow_nan=False) + "\n" for task in tasks)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
