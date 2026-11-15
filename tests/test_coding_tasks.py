import gzip
import hashlib
import json

import pytest

from tine.coding_tasks import load_coding_tasks, write_coding_tasks


@pytest.mark.parametrize("compress", [False, True])
def test_load_humaneval_preserves_prompt_and_checks(tmp_path, compress):
    prompt = 'def add_one(n):\n    """Add one.\u2028Preserve the prompt."""\n'
    test = "def check(candidate):\n    assert candidate(1) == 2\n"
    record = {
        "task_id": "HumanEval/0",
        "prompt": prompt,
        "entry_point": "add_one",
        "test": test,
        "canonical_solution": "    return n + 1\n",
    }
    raw = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    path = tmp_path / ("HumanEval.jsonl.gz" if compress else "HumanEval.jsonl")
    path.write_bytes(gzip.compress(raw) if compress else raw)

    tasks = load_coding_tasks(path, dataset="humaneval", tenant="offline")

    assert tasks == [{
        "tenant": "offline",
        "task_id": "humaneval:HumanEval/0",
        "task_type": "humaneval",
        "prompt": prompt,
        "source_task_id": "HumanEval/0",
        "source_split": "test",
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checks": {"entry_point": "add_one", "test": test},
    }]
    output = tmp_path / "processed" / "tasks.jsonl"
    write_coding_tasks(tasks, output)
    assert json.loads(output.read_text(encoding="utf-8")) == tasks[0]


def test_load_sanitized_mbpp_preserves_official_splits_and_checks(tmp_path):
    path = tmp_path / "sanitized-mbpp.json"
    tests = ["assert square_root(4) == 2"]
    imports = ["from math import sqrt"]
    ids = [10, 11, 510, 511, 600, 601]
    rows = [{
        "task_id": task_id,
        "prompt": "Return the square root.",
        "code": "def square_root(n): return sqrt(n)",
        "test_list": tests,
        "test_imports": imports,
    } for task_id in ids]
    path.write_text(json.dumps(rows), encoding="utf-8")

    tasks = load_coding_tasks(path, dataset="mbpp")

    assert [row["task_id"] for row in tasks] == [f"mbpp:{task_id}" for task_id in ids]
    assert [row["source_task_id"] for row in tasks] == ids
    assert [row["source_split"] for row in tasks] == [
        "prompt", "test", "test", "validation", "validation", "train",
    ]
    assert all(row["checks"] == {"test_list": tests, "test_imports": imports} for row in tasks)
    assert all({"code", "model", "success", "cost_usd"}.isdisjoint(row) for row in tasks)


def test_reject_duplicate_tasks_and_missing_checks(tmp_path):
    path = tmp_path / "sanitized-mbpp.json"
    row = {"task_id": 11, "prompt": "Return one.", "test_imports": [], "test_list": ["assert one() == 1"]}
    path.write_text(json.dumps([row, row]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate task_id"):
        load_coding_tasks(path, dataset="mbpp")

    path.write_text(json.dumps([dict(row, test_list=[])]), encoding="utf-8")
    with pytest.raises(ValueError, match="test_list"):
        load_coding_tasks(path, dataset="mbpp")


@pytest.mark.parametrize("format", ["humaneval", "humaneval-gzip", "mbpp"])
@pytest.mark.parametrize("field", ["task_id", "checks"])
def test_conflicting_json_fields_cannot_replace_task_identity_or_checks(tmp_path, format, field):
    if format.startswith("humaneval"):
        record = {"task_id": "HumanEval/0", "prompt": "def one():", "entry_point": "one",
                  "test": "assert candidate() == 1"}
        duplicate = {"task_id": "HumanEval/1"} if field == "task_id" else {"test": "assert candidate() == 2"}
    else:
        record = {"task_id": 11, "prompt": "Return one.", "test_imports": [],
                  "test_list": ["assert one() == 1"]}
        duplicate = {"task_id": 12} if field == "task_id" else {"test_list": ["assert one() == 2"]}
    raw = json.dumps(record)[:-1] + "," + json.dumps(duplicate)[1:]
    raw = ("[" + raw + "]" if format == "mbpp" else raw + "\n").encode()
    path = tmp_path / ("tasks.jsonl.gz" if format == "humaneval-gzip" else "tasks.json")
    path.write_bytes(gzip.compress(raw) if format == "humaneval-gzip" else raw)
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        load_coding_tasks(path, dataset="mbpp" if format == "mbpp" else "humaneval")


def test_failed_task_encoding_preserves_existing_output(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text("previous tasks\n")
    with pytest.raises(UnicodeEncodeError):
        write_coding_tasks([{"prompt": "invalid surrogate: \ud800"}], path)
    assert path.read_text() == "previous tasks\n"


def test_failed_task_publication_preserves_output_and_cleans_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "tasks.ndjson"
    path.write_text("previous tasks\n")

    def fail_replace(*args):
        raise OSError("publication failed")

    monkeypatch.setattr("tine.artifacts.os.replace", fail_replace)
    with pytest.raises(OSError, match="publication failed"):
        write_coding_tasks([{"prompt": "new task"}], path)
    assert path.read_text() == "previous tasks\n"
    assert list(tmp_path.iterdir()) == [path]


def test_task_writer_rejects_symlinks_without_overwriting_the_target(tmp_path):
    target = tmp_path / "original.jsonl"
    target.write_text("preserve original\n")
    alias = tmp_path / "tasks.jsonl"
    alias.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic links"):
        write_coding_tasks([{"prompt": "new task"}], alias)
    assert target.read_text() == "preserve original\n"
    assert alias.is_symlink()
