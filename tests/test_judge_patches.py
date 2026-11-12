import csv
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

spec = importlib.util.spec_from_file_location(
    "tine_judge", Path(__file__).parents[1] / "scripts" / "judge_patches.py")
judge_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge_module)


def test_parse_verdict_takes_the_last_valid_object_and_rejects_bad_replies():
    text = 'Thinking {"draft": 1}. Final: {"verdict": "Reject", "probability": 0.2, "reason": "misses the edge case"}'
    assert judge_module.parse_verdict(text) == {
        "verdict": "reject", "probability": 0.2, "reason": "misses the edge case"}
    for bad in ["no json", '{"verdict": "maybe", "probability": 0.5}', '{"verdict": "accept"}',
                '{"verdict": "accept", "probability": 3}', '{"verdict": "accept", "probability": "high"}']:
        with pytest.raises(ValueError):
            judge_module.parse_verdict(bad)


def fixtures(tmp_path):
    """Two tasks. t-1: models a and b give the same patch that passed every free check;
    c failed the existing tests; d failed the static checks. t-2 is invalid here."""
    bench = tmp_path / "bench"
    patches = {"a": "PATCH-SAME", "b": "PATCH-SAME", "c": "PATCH-REGRESS", "d": "PATCH-BROKEN"}
    for model, patch in patches.items():
        (bench / model).mkdir(parents=True)
        (bench / model / "run.json").write_text(json.dumps({"model_name": model, "split": "verified", "records": [
            {"index": 1, "instance_id": "t-1", "prediction": patch, "score": 1.0 if model in "ab" else 0.0},
            {"index": 2, "instance_id": "t-2", "prediction": patch + "2", "score": 0.0}]}))
    static = tmp_path / "static.csv"
    with static.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task_id", "model", "instance_id", "passed", "reason", "applied_by"])
        for model in patches:
            for iid in ["t-1", "t-2"]:
                writer.writerow(["x", model, iid, model != "d", "ok", "git"])
    tests = tmp_path / "tests.csv"
    with tests.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task_id", "model", "instance_id", "patch_sha", "status", "applied", "resolved", "p2p_failed"])
        writer.writerow(["x", "__gold__", "t-1", "g", "ok", "True", "True", "0"])
        writer.writerow(["x", "__gold__", "t-2", "g", "gold_unresolved", "True", "False", "0"])
        writer.writerow(["x", "a", "t-1", "same", "ok", "True", "True", "0"])
        writer.writerow(["x", "b", "t-1", "same", "ok", "True", "True", "0"])
        writer.writerow(["x", "c", "t-1", "regress", "ok", "True", "False", "2"])
        writer.writerow(["x", "a", "t-2", "other", "ok", "True", "False", "0"])
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("\n".join(json.dumps({"instance_id": iid, "repo": "o/r", "problem_statement": f"Issue {iid}"})
                                 for iid in ["t-1", "t-2"]))
    return bench, static, tests, dataset


def test_prepare_keeps_only_patches_that_passed_the_free_checks_and_hides_grades(tmp_path):
    bench, static, tests, dataset = fixtures(tmp_path)
    counts = judge_module.prepare(bench, static, tests, dataset, tmp_path / "cases.jsonl", tmp_path / "key.csv",
                                  models=set(), sample=None, seed=1)
    assert counts == {"cases": 1, "benchmark_right": 1, "tasks": 1}
    case = judge_module.read_jsonl(tmp_path / "cases.jsonl")[0]
    assert set(case) == {"case_id", "repository", "issue", "patch"}  # nothing about grades or models
    assert case["patch"] == "PATCH-SAME" and case["issue"] == "Issue t-1"
    key = judge_module.read_csv(tmp_path / "key.csv")[0]
    assert key["case_id"] == case["case_id"] and key["models"] == "a;b" and key["benchmark_right"] == "True"
    messages = judge_module.build_messages(case)
    assert "o/r" in messages[0]["content"] and "<patch>\nPATCH-SAME\n</patch>" in messages[1]["content"]


def test_judge_records_verdicts_and_tokens_and_resumes(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text("\n".join(json.dumps({"case_id": f"c{n}", "repository": "o/r", "issue": "i", "patch": text})
                               for n, text in enumerate(["GOOD", "BAD", "GARBLED"])))
    seen = []

    def endpoint(request):
        body = json.loads(request.content)
        seen.append(request.headers.get("authorization"))
        assert request.url.path == "/v1/chat/completions" and body["temperature"] == 0
        patch = body["messages"][1]["content"]
        reply = ("not json" if "GARBLED" in patch else json.dumps(
            {"verdict": "accept" if "GOOD" in patch else "reject", "probability": 0.9 if "GOOD" in patch else 0.1,
             "reason": "r"}))
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}],
                                         "usage": {"prompt_tokens": 120, "completion_tokens": 30}})

    client = httpx.Client(transport=httpx.MockTransport(endpoint))
    output = tmp_path / "verdicts.csv"
    counts = judge_module.judge(cases, output, base_url="http://judge.test/v1", model="local-model",
                                api_key="secret", client=client)
    assert counts == {"judged": 3, "skipped": 0, "unparsed": 1} and seen[0] == "Bearer secret"
    rows = {row["case_id"]: row for row in judge_module.read_csv(output)}
    assert (rows["c0"]["verdict"], rows["c1"]["verdict"], rows["c2"]["verdict"]) == ("accept", "reject", "unparsed")
    assert rows["c0"]["prompt_tokens"] == "120" and rows["c0"]["judge"] == "local-model"
    again = judge_module.judge(cases, output, base_url="http://judge.test/v1", model="local-model", client=client)
    assert again == {"judged": 0, "skipped": 3, "unparsed": 0} and len(seen) == 3


def test_score_reports_catch_rate_false_rejects_and_token_cost(tmp_path):
    key = tmp_path / "key.csv"
    with key.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case_id", "benchmark_right", "run_resolved"])
        writer.writerows([["r1", True, True], ["r2", True, True], ["w1", False, False],
                          ["w2", False, True], ["w3", False, False]])
    verdicts = tmp_path / "v.csv"
    for case_id, verdict, probability in [("r1", "accept", 0.95), ("r2", "reject", 0.6), ("w1", "reject", 0.1),
                                          ("w2", "accept", 0.8), ("w3", "unparsed", "")]:
        judge_module.append_verdict(verdicts, {"case_id": case_id, "judge": "j", "verdict": verdict,
                                               "probability": probability, "prompt_tokens": 100,
                                               "completion_tokens": 20, "tokens_estimated": False, "seconds": 1})
    rows = {row["threshold"]: row for row in judge_module.score(key, [verdicts])}
    own = rows["verdict"]
    assert (own["right"], own["wrong"], own["wrong_rejected"], own["right_rejected"], own["unparsed"]) == (2, 3, 2, 1, 1)
    assert own["catch_rate"] == pytest.approx(2 / 3, abs=1e-4) and own["false_reject_rate"] == 0.5
    assert own["accuracy_when_accepted"] == 0.5 and own["mean_prompt_tokens"] == 100
    assert rows[0.9]["catch_rate"] == 1.0 and rows[0.9]["accepted_share"] == 0.2
    assert rows[0.5]["false_reject_rate"] == 0.0  # r2 is accepted at 0.6 despite its verdict
    other = judge_module.score(key, [verdicts], grade="run_resolved")[0]
    assert (other["right"], other["wrong"]) == (3, 2)


def test_think_style_changes_the_instructions_only():
    case = {"case_id": "c", "repository": "o/r", "issue": "i", "patch": "p"}
    plain, think = judge_module.build_messages(case), judge_module.build_messages(case, "think")
    assert plain[1] == think[1] and "strict code reviewer" in think[0]["content"] and "o/r" in think[0]["content"]
    assert judge_module.parse_verdict('1. ... 4. Decide.\n{"verdict": "reject", "probability": 0.2, "reason": "x"}')["verdict"] == "reject"
    with pytest.raises(ValueError):
        judge_module.build_messages(case, "guess")
