#!/usr/bin/env python3
"""Static checks of recorded SWE-bench answers: does the patch apply, and do the
Python files it touches still parse?

Nothing is executed. Each answer's patch is applied to a scratch copy of the files
it names, taken from a local bare clone at the task's base commit, and the patched
Python files are parsed with ``ast``. No test is run, no package is imported, and no
model is called, so this is safe on an ordinary machine. A patch naming a path
outside the repository is refused, not applied.

These are checks available at request time for any code change: they need only the
repository. "Applies" follows the SWE-bench harness: ``git apply``, then ``patch
--batch --fuzz=5``. The result depends on the ``patch`` program: the harness runs GNU
patch, which accepts some hunks Apple's ``patch`` rejects (for example a hunk with
no trailing context that is not at the end of the file). Run on Linux to match it. Output is a CSV for ``tine cascade --checks`` (task_id, model, passed)
with the reason for each rejection.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import shutil
import subprocess
import tempfile
import warnings
from concurrent import futures
from pathlib import Path, PurePosixPath

HEADER = re.compile(r"^(?:---|\+\+\+) (\S+)", re.MULTILINE)
RENAME = re.compile(r"^(?:rename (?:from|to)|copy (?:from|to)|Index:) (.+)$", re.MULTILINE)
GIT_HEADER = re.compile(r"^diff --git (\S+) (\S+)", re.MULTILINE)
REASONS = ("ok", "no_patch", "unsafe_path", "no_files", "does_not_apply", "syntax_error")


def patch_paths(patch: str) -> list[str] | None:
    """Repository paths a unified diff names, without the a/ and b/ prefixes.
    ``None`` if any path is absolute or leaves the repository."""
    if re.search(r"^(?:new|old|deleted)? ?(?:file )?mode 120000", patch, re.MULTILINE):
        return None  # symbolic links could redirect a later write
    names = set()
    named = [name for pair in GIT_HEADER.findall(patch) for name in pair]
    for raw in [*HEADER.findall(patch), *RENAME.findall(patch), *named]:
        raw = raw.strip()
        if raw == "/dev/null":
            continue
        path = PurePosixPath(raw)
        if path.is_absolute() or ".." in path.parts or "\x00" in raw:
            return None
        parts = path.parts[1:] if path.parts and path.parts[0] in ("a", "b") else path.parts
        if not parts or parts[0] == ".git":
            return None
        names.add("/".join(parts))
    return sorted(names)


def base_files(repo: Path, commit: str, paths: list[str], destination: Path) -> dict[str, bytes]:
    """Write the named files as they are at ``commit`` under ``destination``.
    Files absent at that commit (new files) are skipped."""
    found = {}
    for path in paths:
        result = subprocess.run(["git", "--git-dir", str(repo), "show", f"{commit}:{path}"],
                                capture_output=True)
        if result.returncode != 0:
            continue
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(result.stdout)
        found[path] = result.stdout
    return found


def apply_patch(patch: str, directory: Path) -> str | None:
    """Apply as the SWE-bench harness does: ``git apply``, then ``patch`` with fuzz.
    Returns the tool that applied it, or ``None``."""
    patch_file = directory.parent / "answer.patch"
    patch_file.write_text(patch if patch.endswith("\n") else patch + "\n", encoding="utf-8")
    for name, command in (("git", ["git", "apply", "--verbose", str(patch_file)]),
                          ("patch", ["patch", "--batch", "--fuzz=5", "-p1", "-i", str(patch_file)])):
        snapshot = directory.parent / "snapshot"
        shutil.copytree(directory, snapshot)
        try:
            result = subprocess.run(command, cwd=directory, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            result = None
        if result is not None and result.returncode == 0:
            shutil.rmtree(snapshot)
            return name
        shutil.rmtree(directory)  # a failed attempt may leave partial edits
        snapshot.rename(directory)
    return None


def parses(source: bytes) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # old code warns about escapes; not an error
            ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    return True


def check_answer(patch, repo: Path, commit: str) -> tuple[bool, str, str]:
    """(passed, reason, applied_by) for one recorded answer."""
    if not isinstance(patch, str) or not patch.strip():
        return False, "no_patch", ""
    paths = patch_paths(patch)
    if paths is None:
        return False, "unsafe_path", ""
    if not paths:
        return False, "no_files", ""
    with tempfile.TemporaryDirectory(prefix="tine-swe-") as scratch:
        tree = Path(scratch) / "tree"
        tree.mkdir()
        before = base_files(repo, commit, paths, tree)
        applied_by = apply_patch(patch, tree)
        if applied_by is None:
            return False, "does_not_apply", ""
        for path in paths:
            target = tree / path
            if not path.endswith(".py") or not target.is_file() or target.is_symlink():
                continue
            # Old code may not parse on this Python; only blame the patch if the
            # file parsed before it, or is new.
            if (path not in before or parses(before[path])) and not parses(target.read_bytes()):
                return False, "syntax_error", applied_by
    return True, "ok", applied_by


def collect(bench_dir: Path, metadata: dict, models: set[str]) -> list[dict]:
    """One job per recorded answer, from the latest run file of each model."""
    jobs = []
    for model_dir in sorted(path for path in bench_dir.iterdir() if path.is_dir()):
        if models and model_dir.name not in models:
            continue
        runs = sorted(model_dir.glob("*.json"))
        if not runs:
            continue
        data = json.loads(runs[-1].read_text(encoding="utf-8"))
        for record in data["records"]:
            instance = metadata.get(record.get("instance_id"))
            if instance is None:
                raise ValueError(f"{runs[-1]}: unknown instance_id {record.get('instance_id')!r}")
            jobs.append({"task_id": f"swe-bench:{data['split']}:{record['index']}",
                         "instance_id": record["instance_id"], "model": data["model_name"],
                         "patch": record.get("prediction"), "repo": instance["repo"],
                         "commit": instance["base_commit"]})
    return jobs


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bench-dir", type=Path, required=True,
                        help="extracted swe-bench split folder (contains <model>/*.json)")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="JSON lines with instance_id, repo and base_commit for every task")
    parser.add_argument("--repos-dir", type=Path, required=True,
                        help="bare clones named <owner>__<name>.git")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="check only the first N answers (debugging)")
    args = parser.parse_args(argv)
    metadata = {}
    for line in args.metadata.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            metadata[row["instance_id"]] = row
    jobs = collect(args.bench_dir, metadata, set(args.models or []))[: args.limit]

    def run(job):
        repo = args.repos_dir / (job["repo"].replace("/", "__") + ".git")
        if not repo.is_dir():
            raise FileNotFoundError(f"Missing bare clone {repo}")
        return job, check_answer(job["patch"], repo, job["commit"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle, futures.ThreadPoolExecutor(args.workers) as pool:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "model", "instance_id", "passed", "reason", "applied_by"])
        for count, (job, (passed, reason, applied_by)) in enumerate(pool.map(run, jobs), 1):
            writer.writerow([job["task_id"], job["model"], job["instance_id"], passed, reason, applied_by])
            if count % 500 == 0:
                print(f"Checked {count}/{len(jobs)} answers", flush=True)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
