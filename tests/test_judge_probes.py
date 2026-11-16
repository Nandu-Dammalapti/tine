import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "tine_judge_probes", Path(__file__).parents[1] / "scripts" / "judge_probes.py")
probes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probes)

from tine.judge import GUARD, build_messages, prompt_version  # noqa: E402

PATCH = ("--- a/django/db/query.py\n+++ b/django/db/query.py\n@@ -10,3 +10,4 @@ class Query:\n"
         "     def run(self):\n-        return 1\n+        x = 2\n+        return x\n")
CASE = {"case_id": "c1", "repository": "django/django", "issue": "Django's QuerySet breaks; see djangoproject.com",
        "patch": PATCH}


def test_anonymize_removes_every_spelling_of_the_project_name():
    case = probes.anonymize(CASE)
    text = case["repository"] + case["issue"] + case["patch"]
    assert "django" not in text.lower() and "Quillstack's" in case["issue"]
    assert case["patch"].startswith("--- a/quillstack/db/query.py") and case["case_id"] == "c1"
    with pytest.raises(ValueError):
        probes.anonymize({**CASE, "repository": "other/project"})


def test_inject_adds_one_indented_comment_and_fixes_the_hunk_header():
    case = probes.inject(CASE, "claim")
    lines = case["patch"].split("\n")
    assert lines[2] == "@@ -10,3 +10,5 @@ class Query:"
    assert lines[3] == "+    " + probes.INJECTIONS["claim"] and lines[4] == "     def run(self):"
    assert case["patch"].count("\n") == PATCH.count("\n") + 1


def test_hunks_and_windows_cover_the_changed_lines_with_a_margin():
    ranges = probes.hunks(PATCH + "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x\n")
    assert ranges == {"django/db/query.py": [(10, 12)]}
    assert probes.windows([(10, 12), (20, 21)], 5, 100) == [(5, 26)]
    assert probes.windows([(1, 2)], 5, 4) == [(1, 4)]


def test_context_reads_the_file_at_the_base_commit(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "django" / "db").mkdir(parents=True)
    (work / "django" / "db" / "query.py").write_text("".join(f"line {n}\n" for n in range(1, 31)))
    run = lambda *a: subprocess.run(["git", "-C", str(work), *a], check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("add", ".")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "x")
    commit = subprocess.run(["git", "-C", str(work), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    context = probes.context_for(PATCH, work / ".git", commit, margin=2)
    assert context.startswith("### django/db/query.py, lines 8-14") and "   10 line 10" in context
    with pytest.raises(ValueError):
        probes.file_at(work / ".git", "--upload-pack=x", "a.py")


def test_context_and_hardened_style_change_only_what_they_say():
    plain = build_messages(CASE)
    hardened = build_messages(CASE, "hardened")
    assert GUARD not in plain[0]["content"] and GUARD in hardened[0]["content"]
    assert hardened[1] == plain[1]
    with_context = build_messages({**CASE, "context": "   10 x"})
    assert "<code_before_change>" in with_context[1]["content"] and "with line numbers" in with_context[0]["content"]
    assert prompt_version(CASE) == "2026-11-11" and prompt_version({**CASE, "context": "x"}, "hardened") == \
        "2026-11-11+hardened+context"


def test_compare_counts_paired_movement(tmp_path):
    key = tmp_path / "key.csv"
    key.write_text("case_id,run_resolved\na,True\nb,False\nc,False\nd,True\n")

    def verdicts(name, rows):
        path = tmp_path / f"{name}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["case_id", "verdict", "probability", "cost_usd"])
            writer.writeheader()
            for case_id, verdict in rows:
                writer.writerow({"case_id": case_id, "verdict": verdict, "probability": 0.5, "cost_usd": 0.01})
        return path
    base = verdicts("base", [("a", "accept"), ("b", "reject"), ("c", "reject"), ("d", "accept")])
    probe = verdicts("probe", [("a", "accept"), ("b", "accept"), ("c", "unparsed"), ("d", "reject")])
    result = probes.compare(key, base, probe)
    assert result["baseline"]["catch_rate"] == 1.0 and result["probe"]["catch_rate"] == 0.5
    assert result["wrong_moved_to_accept"] == 1 and result["right_moved_to_reject"] == 1
    assert result["probe"]["unparsed"] == 1 and result["sign_test_p"] == 1.0
