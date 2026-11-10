import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "tine_swe_tests", Path(__file__).parents[1] / "scripts" / "swe_repo_tests.py")
runner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner_module)
GOLD = runner_module.GOLD

INSTANCES = {
    "a__a-1": {"instance_id": "a__a-1", "repo": "a/a", "version": "1.0", "patch": "GOLD-1"},
    "a__a-2": {"instance_id": "a__a-2", "repo": "a/a", "version": "1.0", "patch": "GOLD-2"},
    "b__b-1": {"instance_id": "b__b-1", "repo": "b/b", "version": "2.0", "patch": "GOLD-3"},
}


class FakeShell:
    """Records scripts; a patch containing NOAPPLY fails to apply, HANG times out."""
    run_user = "tester"

    def __init__(self, work_dir, fail_env_for=()):
        self.work_dir, self.calls, self.fail_env_for = Path(work_dir), [], set(fail_env_for)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def script(self, text, *, as_user=False, isolate=False, timeout=None, log_path=None):
        self.calls.append(text)
        patch = (self.work_dir / "answer.patch").read_text() if (self.work_dir / "answer.patch").exists() else ""
        if text.startswith("ENV:"):
            return (1, "boom") if text[4:] in self.fail_env_for else (0, "")
        if "git apply" in text:
            return (1, "") if "NOAPPLY" in patch else (0, "")
        if text.startswith("EVAL:"):
            Path(log_path).write_text(patch)
            return (None, "") if "HANG" in patch else (0, patch)
        if "rev-parse" in text:
            return 0, "abc123\n"
        return 0, ""


def fake_harness():
    def make_test_spec(instance):
        return SimpleNamespace(instance_id=instance["instance_id"], repo=instance["repo"],
                               env_image_key=instance["repo"] + instance["version"],
                               setup_env_script="ENV:" + instance["repo"],
                               install_repo_script=f"git clone https://github.com/{instance['repo']} /testbed",
                               eval_script="EVAL:" + instance["instance_id"])

    def get_eval_report(test_spec, prediction, log_path, include_tests_status):
        text = Path(log_path).read_text()
        if "NOOUTPUT" in text:
            return {test_spec.instance_id: {"resolved": False}}
        broken = 1 if "REGRESS" in text else 0
        fixed = "GOLD" in text or "FIX" in text
        return {test_spec.instance_id: {"resolved": fixed and not broken, "tests_status": {
            "FAIL_TO_PASS": {"success": ["t"] if fixed else [], "failure": [] if fixed else ["t"]},
            "PASS_TO_PASS": {"success": ["p"] * (3 - broken), "failure": ["p"] * broken}}}}

    return make_test_spec, get_eval_report, ">>>>> Tests Timed Out"


def answers(*pairs):
    return [{"task_id": "swe-bench:verified:1", "model": model, "patch": patch} for model, patch in pairs]


def make_runner(tmp_path, monkeypatch, **shell_options):
    monkeypatch.setattr(runner_module.Path, "is_dir", lambda self: True)
    shell = FakeShell(tmp_path / "work", **shell_options)
    return runner_module.Runner(shell, fake_harness(), mirror_dir=None, log_dir=tmp_path / "logs"), shell


def test_gold_patch_is_validated_before_model_patches(tmp_path, monkeypatch):
    runner, shell = make_runner(tmp_path, monkeypatch)
    output, done = tmp_path / "results.csv", {}
    runner.run_instance(INSTANCES["a__a-1"], answers(
        ("fixes", "FIX it"), ("regresses", "FIX but REGRESS"), ("wrong", "nothing"),
        ("unapplied", "NOAPPLY"), ("slow", "HANG"), ("silent", "NOOUTPUT"), ("copy", "FIX it")), output, done)
    rows = {row["model"]: row for row in csv.DictReader(output.open())}
    assert list(rows) == [GOLD, "fixes", "regresses", "wrong", "unapplied", "slow", "silent", "copy"]
    assert rows[GOLD]["status"] == "ok" and rows[GOLD]["resolved"] == "True"
    assert (rows["fixes"]["resolved"], rows["fixes"]["p2p_failed"]) == ("True", "0")
    assert (rows["regresses"]["resolved"], rows["regresses"]["p2p_failed"]) == ("False", "1")
    assert (rows["wrong"]["resolved"], rows["wrong"]["p2p_failed"], rows["wrong"]["f2p_failed"]) == ("False", "0", "1")
    assert (rows["unapplied"]["status"], rows["unapplied"]["applied"]) == ("apply_failed", "False")
    assert rows["slow"]["status"] == "timeout" and rows["silent"]["status"] == "no_test_output"
    # The identical patch from another model reuses the first result.
    assert rows["copy"]["patch_sha"] == rows["fixes"]["patch_sha"]
    assert sum(call.startswith("EVAL:") for call in shell.calls) == 6


def test_invalid_environment_or_gold_stops_the_task(tmp_path, monkeypatch):
    runner, _ = make_runner(tmp_path, monkeypatch, fail_env_for={"b/b"})
    output, done = tmp_path / "results.csv", {}
    runner.run_instance(INSTANCES["b__b-1"], answers(("m", "FIX")), output, done)
    bad_gold = dict(INSTANCES["a__a-2"], patch="does not fix")
    runner.run_instance(bad_gold, answers(("m", "FIX")), output, done)
    rows = list(csv.DictReader(output.open()))
    assert [(row["instance_id"], row["model"], row["status"]) for row in rows] == [
        ("b__b-1", GOLD, "env_failed"), ("a__a-2", GOLD, "gold_unresolved")]


def test_rerun_resumes_and_builds_each_environment_once(tmp_path, monkeypatch):
    runner, shell = make_runner(tmp_path, monkeypatch)
    output = tmp_path / "results.csv"
    done = {}
    runner.run_instance(INSTANCES["a__a-1"], answers(("m1", "FIX")), output, done)
    runner.run_instance(INSTANCES["a__a-2"], answers(("m1", "FIX")), output, done)
    assert sum(call.startswith("ENV:") for call in shell.calls) == 1  # same repository and version
    evaluations = sum(call.startswith("EVAL:") for call in shell.calls)
    resumed = runner_module.read_done(output)
    runner.run_instance(INSTANCES["a__a-1"], answers(("m1", "FIX")), output, resumed)
    assert sum(call.startswith("EVAL:") for call in shell.calls) == evaluations  # nothing left to do
    runner.run_instance(INSTANCES["a__a-1"], answers(("m1", "FIX"), ("m2", "nothing")), output, resumed)
    rows = list(csv.DictReader(output.open()))
    assert [(row["instance_id"], row["model"]) for row in rows].count(("a__a-1", GOLD)) == 1
    assert rows[-1]["model"] == "m2" and sum(call.startswith("EVAL:") for call in shell.calls) == evaluations + 1
    runner.run_instance(INSTANCES["b__b-1"], [], output, resumed, gold_only=True)
    assert list(csv.DictReader(output.open()))[-1]["instance_id"] == "b__b-1"


def bench_and_static(tmp_path):
    bench = tmp_path / "bench"
    static = tmp_path / "static.csv"
    rows = [["task_id", "model", "instance_id", "passed", "reason", "applied_by"]]
    for model in ["m1", "m2"]:
        (bench / model).mkdir(parents=True)
        records = [{"index": index + 1, "instance_id": iid, "prediction": f"{model} patch {iid}"}
                   for index, iid in enumerate(INSTANCES)]
        (bench / model / "run.json").write_text(json.dumps(
            {"model_name": model, "split": "verified", "records": records}))
        for index, iid in enumerate(INSTANCES):
            passed = not (model == "m2" and iid == "a__a-2")
            rows.append([f"swe-bench:verified:{index + 1}", model, iid, passed,
                         "ok" if passed else "does_not_apply", "git" if passed else ""])
    with static.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    return bench, static


def test_plan_keeps_only_static_passes_for_chosen_repositories_and_models(tmp_path):
    bench, static = bench_and_static(tmp_path)
    jobs = runner_module.plan(bench, static, INSTANCES, repos={"a/a"}, models=set())
    assert list(jobs) == ["a__a-1", "a__a-2"]
    assert [a["model"] for a in jobs["a__a-1"]] == ["m1", "m2"] and [a["model"] for a in jobs["a__a-2"]] == ["m1"]
    limited = runner_module.plan(bench, static, INSTANCES, repos=set(), models={"m2"}, limit_instances=1)
    assert list(limited) == ["a__a-1"] and limited["a__a-1"][0]["task_id"] == "swe-bench:verified:1"


def test_checks_file_marks_regressions_failed_and_unrun_answers_unknown(tmp_path):
    from tine.checks import read_checks
    bench, static = bench_and_static(tmp_path)
    results = tmp_path / "results.csv"
    base = {"patch_sha": "x", "applied": True, "resolved": False, "f2p_passed": 0, "f2p_failed": 1,
            "p2p_passed": 3, "seconds": 1}
    for row in [
        {"instance_id": "a__a-1", "model": GOLD, "status": "ok", **base, "resolved": True, "p2p_failed": 0},
        {"instance_id": "a__a-1", "model": "m1", "status": "ok", **base, "p2p_failed": 0},
        {"instance_id": "a__a-1", "model": "m2", "status": "ok", **base, "p2p_failed": 2},
        {"instance_id": "a__a-2", "model": GOLD, "status": "gold_unresolved", **base, "p2p_failed": 0},
        {"instance_id": "a__a-2", "model": "m1", "status": "ok", **base, "p2p_failed": 0},
    ]:
        runner_module.append_row(results, {"task_id": "t", **row})
    counts = runner_module.write_checks(results, static, tmp_path / "checks.csv")
    table = read_checks(tmp_path / "checks.csv").set_index(["task_id", "model"])
    assert table.loc[("swe-bench:verified:1", "m1"), "passed"] == "True"
    assert table.loc[("swe-bench:verified:1", "m2"), "reason"] == "tests:ok"
    assert table.loc[("swe-bench:verified:1", "m2"), "passed"] == "False"
    assert table.loc[("swe-bench:verified:2", "m1"), "passed"] == ""        # task invalid here
    assert table.loc[("swe-bench:verified:2", "m2"), "reason"] == "static:does_not_apply"
    assert table.loc[("swe-bench:verified:3", "m1"), "reason"] == "not_run"
    assert counts == {"passed": 1, "failed": 2, "unknown": 3}
    # Grades: m1 is right on task 1 only, m2 is always wrong.
    for model in ["m1", "m2"]:
        run = json.loads((bench / model / "run.json").read_text())
        for record in run["records"]:
            record["score"] = 1.0 if (model, record["instance_id"]) == ("m1", "a__a-1") else 0.0
        (bench / model / "run.json").write_text(json.dumps(run))
    summary = runner_module.summarize(results, static, bench)
    assert (summary["tasks_attempted"], summary["tasks_valid"]) == (2, 1)
    assert summary["gold_statuses"] == {"ok": 1, "gold_unresolved": 1}
    assert summary["per_model"]["m1"]["accepted_right"] == 1 and summary["per_model"]["m1"]["wrong"] == 0
    assert summary["per_model"]["m2"]["wrong_caught_static"] == 0
    assert summary["per_model"]["m2"]["wrong_caught_static_or_tests"] == 1
    # m1's run says unresolved but the benchmark graded it right: counted as a disagreement.
    assert summary["runs_compared_with_benchmark_grade"] == 2 and summary["agreement_with_benchmark_grade"] == 0.5


def test_refuses_to_run_where_the_machine_is_not_prepared(tmp_path, monkeypatch):
    bench, static = bench_and_static(tmp_path)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("\n".join(json.dumps(row) for row in INSTANCES.values()))
    monkeypatch.setattr(runner_module, "preflight", lambda user, protect: {"ready": False, "root": False})
    with pytest.raises(SystemExit, match="Refusing"):
        runner_module.main(["--bench-dir", str(bench), "--static-checks", str(static),
                            "--dataset", str(dataset), "--output", str(tmp_path / "out.csv")])
