import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "tine_swe_checks", Path(__file__).parents[1] / "scripts" / "swe_patch_checks.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)

BASE = "def f():\n    return 1\n\n\ndef g():\n    return 2\n"
# Hunks carry trailing context: without it git anchors a hunk at the end of the file.
GOOD = "--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -1,3 +1,3 @@\n def f():\n-    return 1\n+    return 10\n \n"
WRONG_CONTEXT = "--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -1,2 +1,2 @@\n def nothing_like_this():\n-    return 99\n+    return 10\n"
BREAKS_SYNTAX = "--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -1,3 +1,3 @@\n def f():\n-    return 1\n+    return (\n \n"
NEW_FILE = "--- /dev/null\n+++ b/pkg/new.py\n@@ -0,0 +1,1 @@\n+VALUE = 3\n"


def git(repo, *arguments, stdin=None):
    return subprocess.run(["git", "--git-dir", str(repo), *arguments], input=stdin,
                          capture_output=True, check=True).stdout.decode().strip()


@pytest.fixture
def repo(tmp_path):
    """A bare repository with one commit holding pkg/m.py, built without a work tree."""
    bare = tmp_path / "owner__name.git"
    subprocess.run(["git", "init", "-q", "--bare", "--template=", str(bare)], check=True)
    blob = git(bare, "hash-object", "-w", "--stdin", stdin=BASE.encode())
    inner = git(bare, "mktree", stdin=f"100644 blob {blob}\tm.py\n".encode())
    tree = git(bare, "mktree", stdin=f"040000 tree {inner}\tpkg\n".encode())
    commit = git(bare, "-c", "user.name=t", "-c", "user.email=t@example.com",
                 "commit-tree", tree, "-m", "base")
    return bare, commit


def test_patch_paths_strip_prefixes_and_refuse_paths_outside_the_repository():
    assert checks.patch_paths(GOOD) == ["pkg/m.py"]
    assert checks.patch_paths(NEW_FILE) == ["pkg/new.py"]
    assert checks.patch_paths("no diff here") == []
    for unsafe in ["--- a/../../etc/passwd\n+++ b/../../etc/passwd\n", "--- /etc/passwd\n+++ /etc/passwd\n",
                   "--- a/.git/config\n+++ b/.git/config\n", "Index: ../x\n--- a/y\n+++ b/y\n",
                   "diff --git a/x b/../y\n--- a/x\n+++ b/x\n",
                   "diff --git a/l b/l\nnew file mode 120000\n--- /dev/null\n+++ b/l\n"]:
        assert checks.patch_paths(unsafe) is None


@pytest.mark.parametrize("patch, expected", [
    (GOOD, (True, "ok")), (NEW_FILE, (True, "ok")), (WRONG_CONTEXT, (False, "does_not_apply")),
    (BREAKS_SYNTAX, (False, "syntax_error")), ("", (False, "no_patch")), (None, (False, "no_patch")),
    ("I could not solve this.", (False, "no_files")),
    ("--- a/../x.py\n+++ b/../x.py\n@@ -0,0 +1 @@\n+x\n", (False, "unsafe_path")),
])
def test_check_answer(repo, patch, expected):
    bare, commit = repo
    passed, reason, applied_by = checks.check_answer(patch, bare, commit)
    assert (passed, reason) == expected
    assert bool(applied_by) == (reason in ("ok", "syntax_error"))


def test_old_code_that_no_longer_parses_is_not_blamed_on_the_patch(tmp_path):
    bare = tmp_path / "old.git"
    subprocess.run(["git", "init", "-q", "--bare", "--template=", str(bare)], check=True)
    blob = git(bare, "hash-object", "-w", "--stdin", stdin=b"print 'python 2'\nVALUE = 1\n")
    tree = git(bare, "mktree", stdin=f"100644 blob {blob}\told.py\n".encode())
    commit = git(bare, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit-tree", tree, "-m", "b")
    patch = "--- a/old.py\n+++ b/old.py\n@@ -1,2 +1,2 @@\n print 'python 2'\n-VALUE = 1\n+VALUE = 2\n"
    assert checks.check_answer(patch, bare, commit)[:2] == (True, "ok")


def test_cli_writes_a_checks_file_the_cascade_accepts(repo, tmp_path):
    from tine.checks import prepare_checks, read_checks
    bare, commit = repo
    bench = tmp_path / "bench"
    for model, patches in {"good": [GOOD, NEW_FILE], "bad": [WRONG_CONTEXT, None]}.items():
        folder = bench / model
        folder.mkdir(parents=True)
        (folder / f"run-{model}.json").write_text(json.dumps({
            "model_name": model, "dataset_name": "swe-bench", "split": "verified",
            "records": [{"index": index + 1, "instance_id": "owner__name-1", "prediction": patch}
                        for index, patch in enumerate(patches)]}))
    metadata = tmp_path / "meta.jsonl"
    metadata.write_text(json.dumps({"instance_id": "owner__name-1", "repo": "owner/name",
                                    "base_commit": commit}) + "\n")
    output = tmp_path / "out" / "checks.csv"
    checks.main(["--bench-dir", str(bench), "--metadata", str(metadata),
                 "--repos-dir", str(bare.parent), "--output", str(output), "--workers", "2"])
    table = pd.read_csv(output)
    assert table.groupby("model").passed.sum().to_dict() == {"bad": 0, "good": 2}
    assert set(table.reason) == {"ok", "does_not_apply", "no_patch"}
    tasks = pd.DataFrame({"tenant": ["t", "t"], "task_id": ["swe-bench:verified:1", "swe-bench:verified:2"]})
    passes, coverage = prepare_checks(read_checks(output), tasks, ["bad", "good"])
    assert passes.tolist() == [[False, True], [False, True]] and coverage["unknown_checks"] == 0
    with pytest.raises(ValueError, match="unknown instance_id"):
        metadata.write_text("")
        checks.main(["--bench-dir", str(bench), "--metadata", str(metadata),
                     "--repos-dir", str(bare.parent), "--output", str(output)])
