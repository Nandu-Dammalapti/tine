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


class FakeClaude:
    """Stands in for anthropic.Anthropic: accepts odd-numbered cases, refuses one."""

    def __init__(self):
        self.messages, self.calls = self, []

    def create(self, **request):
        from types import SimpleNamespace as NS

        self.calls.append(request)
        text = request["messages"][0]["content"]
        if "REFUSE" in text:
            return NS(content=[], stop_reason="refusal", usage=NS(input_tokens=100, output_tokens=0))
        verdict = "accept" if "GOOD" in text else "reject"
        reply = json.dumps({"verdict": verdict, "probability": 0.9 if verdict == "accept" else 0.1, "reason": "r"})
        return NS(content=[NS(type="thinking", thinking=""), NS(type="text", text=reply)], stop_reason="end_turn",
                  usage=NS(input_tokens=1000, output_tokens=2000))


def write_claude_cases(tmp_path, patches):
    path = tmp_path / "cases.jsonl"
    path.write_text("".join(json.dumps({"case_id": f"c{i}", "repository": "o/r", "issue": "i", "patch": patch}) + "\n"
                                    for i, patch in enumerate(patches)))
    return path


def test_claude_judge_records_cost_and_refusals_and_resumes(tmp_path):
    cases = write_claude_cases(tmp_path, ["GOOD", "BAD", "REFUSE"])
    output, client = tmp_path / "verdicts.csv", FakeClaude()
    report = judge_module.judge_anthropic(cases, output, model="claude-sonnet-5-5", max_cost_usd=5.0,
                                          workers=1, client=client)
    assert report["judged"] == 3 and report["unparsed"] == 1 and not report["stopped_at_cap"]
    assert report["spent_usd"] == pytest.approx(2 * (1000 * 2 + 2000 * 10) / 1e6 + 100 * 2 / 1e6)
    rows = {row["case_id"]: row for row in csv.DictReader(output.open())}
    assert rows["c0"]["verdict"] == "accept" and rows["c1"]["verdict"] == "reject"
    assert rows["c2"]["verdict"] == "unparsed" and rows["c2"]["reason"] == "refusal"
    assert float(rows["c0"]["cost_usd"]) == pytest.approx(0.022)
    request = client.calls[0]
    assert request["output_config"] == {"effort": "medium"} and "temperature" not in request
    assert request["system"].startswith("You are reviewing") and request["model"] == "claude-sonnet-5-5"
    again = judge_module.judge_anthropic(cases, output, model="claude-sonnet-5-5", max_cost_usd=5.0,
                                         workers=1, client=client)
    assert again["judged"] == 0 and again["skipped"] == 3 and len(client.calls) == 3


def test_claude_judge_stops_before_its_spending_cap(tmp_path):
    cases = write_claude_cases(tmp_path, ["GOOD"] * 5)
    output, client = tmp_path / "verdicts.csv", FakeClaude()
    # Worst case per request is 8000 output tokens at $10 per million: $0.08. Each real reply costs $0.022.
    report = judge_module.judge_anthropic(cases, output, model="claude-sonnet-5-5", max_cost_usd=0.13,
                                          workers=1, client=client)
    assert report["stopped_at_cap"] and report["judged"] == 3 and report["not_run"] == 2
    assert report["spent_usd"] <= 0.13


def test_chain_inputs_charge_each_judge_only_where_it_was_consulted(tmp_path):
    key = tmp_path / "key.csv"
    key.write_text("case_id,task_id,instance_id,patch_sha,models,benchmark_right,run_resolved\n"
                   "c1,t1,i1,s,a;b,True,True\nc2,t2,i2,s,a,False,False\nc3,t3,i3,s,a,True,True\n")
    def verdicts(name, rows):
        path = tmp_path / f"{name}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=judge_module.VERDICT_FIELDS)
            writer.writeheader()
            for case_id, verdict, probability, cost in rows:
                writer.writerow({"case_id": case_id, "judge": name, "verdict": verdict,
                                 "probability": probability, "cost_usd": cost})
        return path
    cheap = verdicts("cheap", [("c1", "accept", 0.9, 1), ("c2", "reject", 0.2, 1), ("c3", "accept", 0.6, 1)])
    strong = verdicts("strong", [("c1", "accept", 0.9, 10), ("c2", "accept", 0.9, 10), ("c3", "reject", 0.1, 10)])
    free = tmp_path / "free.csv"
    free.write_text("task_id,model,passed\nt1,a,True\nt1,b,True\nt2,a,True\nt3,a,True\nt4,a,False\nt5,a,\nt6,a,True\n")
    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text("".join(json.dumps({"task_id": task, "model": model, "cost_usd": 100.0}) + "\n"
                                for task, model in [("t1", "a"), ("t1", "b"), ("t2", "a"), ("t3", "a"), ("t4", "a")]))
    report = judge_module.chain_inputs(key, [cheap, strong], free, outcomes, tmp_path / "checks.csv",
                                       tmp_path / "judged.jsonl")
    checks = {(row["task_id"], row["model"]): row["passed"] for row in csv.DictReader((tmp_path / "checks.csv").open())}
    assert checks == {("t1", "a"): "True", ("t1", "b"): "True", ("t2", "a"): "False", ("t3", "a"): "False",
                      ("t4", "a"): "False", ("t5", "a"): "", ("t6", "a"): ""}
    costs = {(row["task_id"], row["model"]): row["cost_usd"]
             for row in map(json.loads, (tmp_path / "judged.jsonl").read_text().splitlines())}
    # t2: the cheap judge rejected, so the strong one was never paid. t4 failed the free checks: no judge.
    assert costs == {("t1", "a"): 111.0, ("t1", "b"): 111.0, ("t2", "a"): 101.0, ("t3", "a"): 111.0, ("t4", "a"): 100.0}
    assert report["judge_calls"] == [4, 3] and report["passed"] == 2
    alone = judge_module.chain_inputs(key, [cheap], free, outcomes, tmp_path / "c2.csv", tmp_path / "j2.jsonl",
                                      threshold=0.7)
    assert alone["passed"] == 2 and alone["failed"] == 3  # c3 at 0.6 is below the threshold
