import asyncio
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest

from tine.data import validate_outcomes

SPEC = importlib.util.spec_from_file_location("live_pilot", Path(__file__).resolve().parents[1] / "scripts" / "live_pilot.py")
live_pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(live_pilot)

QUESTIONS = [{"task_id": f"q{i}", "prompt": f"Question {i}?", "ground_truth": "B"} for i in range(4)]
FREE = [{"task_id": "f0", "prompt": "Write a poem (a) about rain.", "ground_truth": None}]


def reply(model: str, prompt: str) -> str:
    if prompt.startswith("Write"):
        return "Rain falls (a) softly."
    return "Answer: B" if model != "claude-haiku-5-5" or prompt != "Question 3?" else "**Answer:** C"


class FakeClient:
    class messages:
        @staticmethod
        def create(model, messages, **_):
            text = reply(model, messages[-1]["content"])

            class Response:
                def model_dump(self):
                    return {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                            "usage": {"input_tokens": 100, "output_tokens": 200}}
            return Response()


def write_questions(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_answer_pattern_reads_common_reply_shapes():
    found = {text: [m.group(1) for m in re.finditer(live_pilot.ANSWER_PATTERN, text)]
             for text in ("Answer: B", "**Answer:** C", "Answer: (D)", "Answer: $F$", "Answer: none")}
    assert found == {"Answer: B": ["B"], "**Answer:** C": ["C"], "Answer: (D)": ["D"], "Answer: $F$": ["F"],
                     "Answer: none": []}


def test_ask_grades_resumes_and_exports_canonical_outcomes(tmp_path):
    questions = write_questions(tmp_path / "q.jsonl", QUESTIONS)
    asked = tmp_path / "asked.jsonl"
    models = ["claude-haiku-5-5", "claude-opus-5-5"]
    first = live_pilot.ask(questions, asked, models=models, max_cost_usd=1.0, client=FakeClient())
    again = live_pilot.ask(questions, asked, models=models, max_cost_usd=1.0, client=FakeClient())
    assert (first["asked"], again["asked"], again["already_done"]) == (8, 0, 8)
    rows = [json.loads(line) for line in asked.read_text().splitlines()]
    wrong = [(r["task_id"], r["model"]) for r in rows if not r["success"]]
    assert wrong == [("q3", "claude-haiku-5-5")]
    live_pilot.export(asked, tmp_path / "o.jsonl", tmp_path / "a.csv")
    assert len(validate_outcomes(pd.read_json(tmp_path / "o.jsonl", lines=True))) == 8
    assert pd.read_csv(tmp_path / "a.csv").shape == (8, 3)


def test_ask_stops_before_the_cap(tmp_path):
    # An Opus call is held to its worst case (4,096 output tokens, about /bin/zsh.082) until it ends.
    questions = write_questions(tmp_path / "q.jsonl", QUESTIONS)
    result = live_pilot.ask(questions, tmp_path / "asked.jsonl", models=["claude-opus-5-5"], max_cost_usd=0.085,
                            client=FakeClient(), workers=1)
    assert result["stopped_at_cap"] and result["asked"] == 1


def policy() -> dict:
    return {"version": 1, "tenant": live_pilot.TENANT, "task_type": live_pilot.TASK_TYPE,
            "frontier": "claude-opus-5-5", "answer_pattern": live_pilot.ANSWER_PATTERN,
            "stages": [{"models": ["claude-haiku-5-5", "claude-sonnet-5-5"], "check": "agreement"}],
            "guarantee": {"epsilon": 0.04, "delta": 0.05, "certified": True, "bound": 0.02, "certify_tasks": 500,
                          "certify_degradation_rate": 0.008}}


def provider(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    text = reply(body["model"], body["messages"][-1]["content"])
    return httpx.Response(200, json={"id": "m", "type": "message", "role": "assistant", "model": body["model"],
                                     "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                                     "usage": {"input_tokens": 100, "output_tokens": 200}})


def test_shadow_run_through_the_proxy_is_graded(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    questions = QUESTIONS + FREE
    result = asyncio.run(live_pilot.through_proxy(policy(), questions, mode="shadow", max_cost_usd=1.0,
                                                  ledger_path=tmp_path / "l.db", inner=httpx.MockTransport(provider)))
    summary = live_pilot.score_proxy(result, questions, "claude-opus-5-5")["summary"]
    assert result["shadow_report"]["shadowed_requests"] == 5
    assert summary["freetext_accepted"] == 0
    # Haiku and Sonnet disagree on q3, so it escalates and nothing is lost.
    assert (summary["accepted_share_graded"], summary["loss_rate"], summary["policy_accuracy"]) == (0.75, 0.0, 1.0)


def test_route_run_through_the_proxy_records_the_served_model(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    result = asyncio.run(live_pilot.through_proxy(policy(), QUESTIONS, mode="route", max_cost_usd=1.0,
                                                  ledger_path=tmp_path / "l.db", inner=httpx.MockTransport(provider)))
    served = sorted(row["ledger_model"] for row in result["requests"])
    assert served == ["claude-haiku-5-5"] * 3 + ["claude-opus-5-5"]
    assert live_pilot.score_proxy(result, QUESTIONS, "claude-opus-5-5")["summary"]["policy_accuracy"] == 1.0


def test_recording_transport_refuses_calls_past_the_cap():
    async def go():
        recorder = live_pilot.RecordingTransport(httpx.MockTransport(provider), max_cost_usd=0.0)
        async with httpx.AsyncClient(transport=recorder) as client:
            response = await client.post("https://x/v1/messages", json={
                "model": "claude-opus-5-5", "messages": [{"role": "user", "content": "Question 0?"}]})
        return response.status_code, recorder.refused
    assert asyncio.run(go()) == (529, 1)


def test_boxed_answers_are_read_and_graded_like_the_benchmarks():
    text = r"so \boxed{\frac{1}{\sqrt{2}}} and finally \boxed{\left( 3, \frac{\pi}{2} \right)}."
    assert [m.group(1) for m in re.finditer(live_pilot.BOXED_PATTERN, text)][-1] == r"\left( 3, \frac{\pi}{2} \right)"
    grade = live_pilot.grade
    assert grade("math500", r"\dfrac{14}{3}", r"\frac{14}{3}")
    assert grade("math500", r"\left( 3, \frac{\pi}{2} \right)", r"(3,\frac{\pi}{2})")
    assert not grade("math500", "9", "10")
    # FinQA counts a number that rounds to the key at the key's precision, as the benchmark does.
    assert grade("finqa", r"14.5\%", "14%") and grade("finqa", "1.74", "2%") and grade("finqa", "Yes", "yes")
    assert not grade("finqa", "14.6", "14.0%") and not grade("finqa", "0.099", "9.9%")
    assert grade("medqa", "B", "B") and not grade("medqa", None, "B") and grade("medqa", "B", None) is None


def test_route_run_with_a_holdout_reports_both_groups(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    questions = [{"task_id": f"q{i}", "prompt": f"Question {i}?", "ground_truth": "B"} for i in range(40)]
    result = asyncio.run(live_pilot.through_proxy(policy(), questions, mode="route", max_cost_usd=5.0,
                                                  ledger_path=tmp_path / "l.db", inner=httpx.MockTransport(provider),
                                                  audit_rate=0.5))
    summary = live_pilot.score_proxy(result, questions, "claude-opus-5-5")["summary"]
    assert summary["routed_requests"] + summary["holdout_requests"] == 40 and summary["holdout_requests"] > 5
    assert summary["holdout_accuracy"] == 1.0 and summary["routed_mean_cost_usd"] < summary["holdout_mean_cost_usd"]
    assert "holdout_estimate" in result["savings_report"]


# Task ids as the benchmarks write them: a colon is not allowed in a batch custom_id.
BATCH_QUESTIONS = [{**question, "task_id": f"medqa:{i}"} for i, question in enumerate(QUESTIONS)]
MESSAGE = {"stop_reason": "end_turn", "usage": {"input_tokens": 100, "output_tokens": 200}}


class FakeBatches:
    """The Message Batches calls with no network. ``ended`` False leaves every batch in
    progress and None kills the process at the first poll; ``kinds`` gives the result
    type of a prompt that does not succeed; ``create_error`` is raised by create."""

    def __init__(self, ended=True, kinds=None, create_error=None):
        self.created, self.ended, self.kinds, self.create_error = [], ended, kinds or {}, create_error

    def create(self, requests):
        if self.create_error:
            raise self.create_error
        ids = [request["custom_id"] for request in requests]
        assert len(set(ids)) == len(ids) and all(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name) for name in ids)
        self.created.append(requests)
        return SimpleNamespace(id=f"msgbatch_{len(self.created)}", processing_status="in_progress")

    def retrieve(self, batch_id):
        if self.ended is None:
            raise KeyboardInterrupt
        return SimpleNamespace(id=batch_id, processing_status="ended" if self.ended else "in_progress")

    def results(self, batch_id):
        for request in reversed(self.created[int(batch_id.rpartition("_")[2]) - 1]):  # not in request order
            params = request["params"]
            prompt = params["messages"][-1]["content"]
            result = {"type": self.kinds.get(prompt, "succeeded")}
            if result["type"] == "succeeded":
                result["message"] = {**MESSAGE, "content": [{"type": "text", "text": reply(params["model"], prompt)}]}
            elif result["type"] == "errored":
                result["error"] = {"type": "error", "error": {"type": "invalid_request_error", "message": "no"}}
            yield SimpleNamespace(model_dump=lambda line={"custom_id": request["custom_id"], "result": result}: line)


def ask_batch(tmp_path, batches, *, models=("claude-haiku-5-5", "claude-opus-5-5"), max_cost_usd=1.0, **options):
    questions = write_questions(tmp_path / "q.jsonl", BATCH_QUESTIONS)
    client = SimpleNamespace(messages=SimpleNamespace(batches=batches))
    return live_pilot.ask_batch(questions, tmp_path / "asked.jsonl", models=list(models), max_cost_usd=max_cost_usd,
                                client=client, poll_seconds=0, **options)


def asked_rows(tmp_path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "asked.jsonl").read_text().splitlines()]


def test_batch_ask_writes_graded_records_at_the_batch_price(tmp_path):
    batches = FakeBatches()
    first = ask_batch(tmp_path, batches)
    assert (first["submitted"], first["asked"], first["errors"], first["pending_batches"]) == (8, 8, 0, [])
    assert [len(requests) for requests in batches.created] == [4, 4]
    # The request is the one the direct path sends.
    assert batches.created[1][0]["params"] == {
        "model": "claude-opus-5-5", "max_tokens": 4096, "output_config": {"effort": "medium"},
        "messages": [{"role": "user", "content": "Question 0?"}]}
    rows = asked_rows(tmp_path)
    assert [(r["task_id"], r["model"]) for r in rows if not r["success"]] == [("medqa:3", "claude-haiku-5-5")]
    opus = next(r for r in rows if r["model"] == "claude-opus-5-5")
    # Half of the direct price: (100 * $4 + 200 * $20) per million tokens.
    assert (opus["cost_usd"], opus["latency_ms"], opus["batch_id"], opus["answer"]) == (0.0022, None, "msgbatch_2", "B")
    assert (opus["input_tokens"], opus["output_tokens"], opus["stop_reason"], opus["reply"]) == (
        100, 200, "end_turn", "Answer: B")
    # A rerun finds every record in the output and submits nothing.
    again = ask_batch(tmp_path, batches)
    assert (again["submitted"], again["asked"], again["already_done"], len(batches.created)) == (0, 0, 8, 2)
    table = {row["model"]: row for row in live_pilot.summarize(tmp_path / "asked.jsonl")}
    assert table["claude-haiku-5-5"]["accuracy"] == 0.75 and table["claude-opus-5-5"]["latency_p50_s"] is None
    live_pilot.export(tmp_path / "asked.jsonl", tmp_path / "o.jsonl", tmp_path / "a.csv")
    assert len(validate_outcomes(pd.read_json(tmp_path / "o.jsonl", lines=True))) == 8
    assert pd.read_csv(tmp_path / "a.csv").shape == (8, 3)


def test_batch_ask_resumes_from_the_sidecar_without_submitting_again(tmp_path):
    batches = FakeBatches(ended=None)
    with pytest.raises(KeyboardInterrupt):  # killed while waiting, after both batches were submitted
        ask_batch(tmp_path, batches)
    sidecar = json.loads((tmp_path / "asked.jsonl.batches.json").read_text())["batches"]
    assert [(b["id"], b["model"], b["collected"]) for b in sidecar] == [
        ("msgbatch_1", "claude-haiku-5-5", False), ("msgbatch_2", "claude-opus-5-5", False)]
    assert not (tmp_path / "asked.jsonl").exists()
    batches.ended = False
    waited = ask_batch(tmp_path, batches, timeout_minutes=0)
    assert (waited["submitted"], waited["asked"], waited["pending_batches"]) == (0, 0, ["msgbatch_1", "msgbatch_2"])
    with pytest.raises(SystemExit):  # the direct path would pay for the same questions again
        live_pilot.ask(tmp_path / "q.jsonl", tmp_path / "asked.jsonl", models=["claude-opus-5-5"], max_cost_usd=1.0,
                       client=FakeClient())
    batches.ended = True
    resumed = ask_batch(tmp_path, batches)
    assert (resumed["submitted"], resumed["asked"], resumed["pending_batches"]) == (0, 8, [])
    assert len(batches.created) == 2 and len(asked_rows(tmp_path)) == 8


def test_batch_ask_submits_only_what_fits_under_the_cap(tmp_path):
    # An Opus request is held to its worst case at the batch price (4,096 output tokens, about $0.041).
    batches = FakeBatches(ended=False)
    first = ask_batch(tmp_path, batches, models=["claude-opus-5-5"], max_cost_usd=0.085, timeout_minutes=0)
    assert (first["submitted"], first["left_out_at_cap"], first["stopped_at_cap"]) == (2, 2, True)
    # Until the batch is collected its worst case stays counted, so nothing more fits.
    batches.ended = True
    second = ask_batch(tmp_path, batches, models=["claude-opus-5-5"], max_cost_usd=0.085)
    assert (second["submitted"], second["asked"], second["left_out_at_cap"]) == (0, 2, 2)
    # Then the two answers count at what they cost ($0.0044), which leaves room for one more.
    third = ask_batch(tmp_path, batches, models=["claude-opus-5-5"], max_cost_usd=0.085)
    assert (third["submitted"], third["asked"], third["left_out_at_cap"], third["spent_usd"]) == (1, 1, 1, 0.0066)


def test_batch_results_that_did_not_succeed_are_counted_and_asked_again(tmp_path):
    failing = FakeBatches(kinds={"Question 1?": "errored", "Question 2?": "expired"})
    first = ask_batch(tmp_path, failing, models=["claude-opus-5-5"])
    assert (first["asked"], first["errors"], first["expired"], first["canceled"]) == (2, 1, 1, 0)
    assert sorted(r["task_id"] for r in asked_rows(tmp_path)) == ["medqa:0", "medqa:3"]
    working = FakeBatches()
    again = ask_batch(tmp_path, working, models=["claude-opus-5-5"])
    assert (again["submitted"], again["asked"], again["errors"]) == (2, 2, 0)
    assert [r["params"]["messages"][0]["content"] for r in working.created[0]] == ["Question 1?", "Question 2?"]
    assert len(asked_rows(tmp_path)) == 4


def test_batch_custom_ids_map_back_to_task_ids(tmp_path, monkeypatch):
    names = ["medqa:123", "mmlupro:test_3000:7", "a b/c"]
    ids = live_pilot.custom_ids(names)
    assert all(live_pilot.CUSTOM_ID.fullmatch(custom_id) for custom_id in ids) and list(ids.values()) == names
    # Three requests to a batch: the fourth question goes in a second batch whose ids start again.
    monkeypatch.setattr(live_pilot, "BATCH_MAX_REQUESTS", 3)
    result = ask_batch(tmp_path, FakeBatches(), models=["claude-haiku-5-5"])
    sidecar = json.loads((tmp_path / "asked.jsonl.batches.json").read_text())["batches"]
    assert [b["requests"] for b in sidecar] == [{"q0": "medqa:0", "q1": "medqa:1", "q2": "medqa:2"}, {"q0": "medqa:3"}]
    assert result["asked"] == 4 and all(b["collected"] for b in sidecar)
    # Results arrive in another order; each reply still lands on its own question.
    assert {r["task_id"]: r["success"] for r in asked_rows(tmp_path)} == {
        "medqa:0": True, "medqa:1": True, "medqa:2": True, "medqa:3": False}


def test_batch_create_that_may_have_gone_through_is_never_repeated(tmp_path):
    lost = ask_batch(tmp_path, FakeBatches(create_error=ConnectionError()), models=["claude-opus-5-5"])
    assert (lost["submitted"], lost["failed_calls"], lost["unconfirmed_batches"]) == (0, 1, 1)
    working = FakeBatches()
    again = ask_batch(tmp_path, working, models=["claude-opus-5-5"])
    assert (again["submitted"], again["unconfirmed_batches"], working.created) == (0, 1, [])
    # A refusal with a 4xx status created nothing, so the next run may submit.
    refused = type("Refused", (Exception,), {"status_code": 429})()
    other = tmp_path / "other"
    other.mkdir()
    assert ask_batch(other, FakeBatches(create_error=refused))["unconfirmed_batches"] == 0
    assert ask_batch(other, FakeBatches())["asked"] == 8


class Killed(BaseException):
    """Stands in for the process dying: nothing in the proxy or the pilot catches it."""


def counting(calls: list, die_at: int | None = None):
    """The stand-in provider, keeping each prompt it is asked and dying at one call."""
    def handler(request: httpx.Request) -> httpx.Response:
        if die_at is not None and len(calls) + 1 == die_at:
            raise Killed
        calls.append(json.loads(request.content)["messages"][-1]["content"])
        return provider(request)
    return httpx.MockTransport(handler)


def run_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def proxy_run(tmp_path, calls, **options):
    options = {"mode": "shadow", "max_cost_usd": 1.0, "concurrency": 1, **options}
    return asyncio.run(live_pilot.through_proxy(
        options.pop("policy", policy()), options.pop("questions", QUESTIONS),
        ledger_path=options.pop("ledger_path", tmp_path / "l.db"), run_file=tmp_path / "out.json.run.jsonl",
        inner=counting(calls, options.pop("die_at", None)), **options))


def test_proxy_run_killed_part_way_keeps_what_it_paid_for_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    run_file = tmp_path / "out.json.run.jsonl"
    # Each request is three calls (the frontier, then the two shadowed models). The run dies
    # on the eighth: two requests are finished and the third has paid for its frontier call.
    with pytest.raises(Killed):
        proxy_run(tmp_path, [], die_at=8)
    kept = run_lines(run_file)
    assert [line["kind"] for line in kept] == (["header"] + ["call"] * 3 + ["request"] + ["call"] * 3 + ["request"]
                                               + ["call"])
    finished = [line for line in kept if line["kind"] == "request"]
    assert [line["task_id"] for line in finished] == ["q0", "q1"]
    assert all(line["accepted"] == 1 and line["ledger_model"] == "claude-opus-5-5" for line in finished)
    # A line cut off by the kill is dropped, and a run file is never started over by accident.
    with run_file.open("a") as stream:
        stream.write('{"kind": "call", "model": "claude-op')
    with pytest.raises(SystemExit, match="--resume"):
        proxy_run(tmp_path, [])
    # Under a cap that the recorded spend already reaches, a resume sends nothing.
    spent = sum(line["cost_usd"] for line in kept if line["kind"] == "call")
    capped = []
    held = proxy_run(tmp_path, capped, resume=True, max_cost_usd=spent)
    assert capped == [] and len(held["requests"]) == 2 and held["spent_usd"] == round(spent, 4)
    second = []
    result = proxy_run(tmp_path, second, resume=True)
    # q0 and q1 are not asked again; q2, in flight when the run died, is asked in full.
    assert sorted(set(second)) == ["Question 2?", "Question 3?"] and len(second) == 6
    assert sorted(row["task_id"] for row in result["requests"]) == ["q0", "q1", "q2", "q3"]
    assert (result["resumes"], len(result["calls"])) == (2, 13)
    assert result["spent_usd"] == round(sum(call["cost_usd"] for call in result["calls"]), 4) > round(spent, 4)
    # The result is what the file holds, and it grades as an unbroken run would.
    lines = run_lines(run_file)
    assert result["calls"] == [{key: value for key, value in line.items() if key != "kind"}
                               for line in lines if line["kind"] == "call"]
    assert lines[-1]["kind"] == "end" and result["shadow_report"] == lines[-1]["shadow_report"]
    summary = live_pilot.score_proxy(result, QUESTIONS, "claude-opus-5-5")["summary"]
    assert (summary["accepted_share_graded"], summary["loss_rate"], summary["policy_accuracy"]) == (0.75, 0.0, 1.0)


def test_resume_refuses_a_changed_policy_question_set_or_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    with pytest.raises(SystemExit, match="no run to resume"):
        proxy_run(tmp_path, [], resume=True, mode="route")
    proxy_run(tmp_path, [], mode="route")
    before = (tmp_path / "out.json.run.jsonl").read_text()
    other_policy = {**policy(), "stages": [{"models": ["claude-sonnet-5-5", "claude-haiku-5-5"], "check": "agreement"}]}
    changed = {"policy_sha256": {"policy": other_policy}, "questions_sha256": {"questions": QUESTIONS[:3] + FREE},
               "mode": {"mode": "shadow"}, "max_tokens": {"max_tokens": 1024},
               "does not exist": {"ledger_path": tmp_path / "missing.db"}}
    for message, change in changed.items():
        calls = []
        with pytest.raises(SystemExit, match=message):
            proxy_run(tmp_path, calls, resume=True, **{"mode": "route", **change})
        assert calls == []
    # A ledger that is there but is not this run's: its requests are not in it.
    from tine.ledger import Ledger

    Ledger(tmp_path / "other.db").close()
    with pytest.raises(SystemExit, match="does not hold 4 of the requests"):
        proxy_run(tmp_path, [], resume=True, mode="route", ledger_path=tmp_path / "other.db")
    assert (tmp_path / "out.json.run.jsonl").read_text() == before
    with pytest.raises(SystemExit, match="more than once"):
        proxy_run(tmp_path, [], mode="route", questions=QUESTIONS + QUESTIONS[:1])


def test_proxy_command_writes_its_result_from_the_run_file_and_resumes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    calls, real = [], live_pilot.through_proxy
    monkeypatch.setattr(live_pilot, "through_proxy",
                        lambda *args, **options: real(*args, inner=counting(calls), **options))
    (tmp_path / "policy.json").write_text(json.dumps(policy()))
    output = tmp_path / "run" / "out.json"
    command = ["proxy", "--policy", str(tmp_path / "policy.json"), "--questions",
               str(write_questions(tmp_path / "q.jsonl", QUESTIONS)), "--mode", "route", "--max-cost-usd", "1",
               "--ledger", str(tmp_path / "l.db"), "--output", str(output), "--concurrency", "1"]
    live_pilot.main(command)
    whole = json.loads(output.read_text())
    assert len(whole["requests"]) == 4 and whole["resumes"] == 0 and len(calls) == 9
    assert json.loads(capsys.readouterr().out)["resumes"] == 0
    with pytest.raises(SystemExit, match="that run finished"):
        live_pilot.main(command + ["--resume"])
    # Killed after the last request but before the result was written: the run file has it all.
    output.unlink()
    with pytest.raises(SystemExit, match="--resume"):
        live_pilot.main(command)
    live_pilot.main(command + ["--resume"])
    again = json.loads(output.read_text())
    assert len(calls) == 9 and again["requests"] == whole["requests"] and again["calls"] == whole["calls"]
    assert again["resumes"] == 1 and json.loads(capsys.readouterr().out)["resumes"] == 1


def test_sample_help_does_not_offer_the_ticket_sets(capsys):
    with pytest.raises(SystemExit) as shown:
        live_pilot.main(["sample", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert shown.value.code == 0 and "{arcc,finqa,math500,medqa,mmlupro}" in text
    assert "scripts/ticket_pilot_prep.py" in text
    with pytest.raises(SystemExit) as refused:
        live_pilot.main(["sample", "--bench-dir", "b", "--output", "o", "--dataset", "tickets"])
    assert refused.value.code == 2 and "invalid choice: 'tickets'" in capsys.readouterr().err


def test_worst_case_counts_text_that_is_not_plain_english_higher():
    from tine.judge import build_messages, estimate_input_tokens, worst_case_cost

    plain = "the printer on the second floor stopped working after the update last night"
    ticket = "RE: [TICKET #48213] VPN-GW01 DOWN!! Kd. 0049-7731 / Fehler 0x80070005 (ACCESS_DENIED)"
    # Plain lower-case English is a third of a token a character, plus 8 for the message.
    assert estimate_input_tokens(plain) == -(-len(plain) // 3) + 8 and estimate_input_tokens("") == 8
    # Capitals, digits, punctuation and line breaks each add a token.
    assert estimate_input_tokens("Ab1!\n") == 2 + 4 + 8
    assert estimate_input_tokens(ticket) / len(ticket) > 2 * estimate_input_tokens(plain) / len(plain)
    # A character outside ASCII counts as its UTF-8 bytes: 2 for "ü", 3 for each of the others.
    assert estimate_input_tokens("ü") == 1 + 2 + 8 and estimate_input_tokens("日本語") == 1 + 9 + 8
    for text in (plain, ticket, "日本語", "x" * 1000):
        assert estimate_input_tokens(text) >= len(text) / 3
    # The caps are built on it: here with no reply allowed, at Opus's input price.
    assert live_pilot._worst_case({"prompt": ticket}, "claude-opus-5-5", 0) == pytest.approx(
        estimate_input_tokens(ticket) * 4.0 / 1e6)
    case = {"repository": "o/r", "issue": ticket, "patch": "--- a/x.py\n+++ b/x.py\n"}
    messages = build_messages(case, "verdict")
    assert worst_case_cost(case, model="claude-opus-5-5", max_tokens=0) == pytest.approx(
        estimate_input_tokens(messages[0]["content"] + messages[1]["content"]) * 4.0 / 1e6)


def test_requests_the_proxy_did_not_shadow_are_not_counted_as_turned_down(tmp_path, monkeypatch):
    """Haiku refuses its shadow call on the second question with a 429. The proxy leaves
    that run out and holds shadow mode back, so the last two requests are never
    shadowed. One request was shadowed and accepted: the share accepted is 1 of 1, not
    1 of 4, and the loss is over that one request."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    def limited(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["model"] == "claude-haiku-5-5" and body["messages"][-1]["content"] == "Question 1?":
            return httpx.Response(429, json={"type": "error", "error": {"type": "rate_limit_error"}})
        return provider(request)
    result = asyncio.run(live_pilot.through_proxy(policy(), QUESTIONS, mode="shadow", max_cost_usd=1.0, concurrency=1,
                                                  ledger_path=tmp_path / "l.db", inner=httpx.MockTransport(limited)))
    rows = {row["task_id"]: row for row in result["requests"]}
    assert all(row["status"] == 200 for row in rows.values()) and rows["q0"]["accepted"] == 1
    assert rows["q1"]["skip_reason"] == live_pilot.RATE_LIMITED_REASON and rows["q1"]["accepted"] == 0
    assert "accepted" not in rows["q2"] and "accepted" not in rows["q3"]
    assert result["shadow_report"]["rate_limited_runs"] == 1 and result["shadow_report"]["skipped_rate_limited"] == 2
    scored = live_pilot.score_proxy(result, QUESTIONS, "claude-opus-5-5")
    summary = scored["summary"]
    assert (summary["graded"], summary["shadowed_graded"], summary["compared"]) == (4, 1, 1)
    assert (summary["accepted_share_graded"], summary["loss_rate"], summary["gain_rate"]) == (1.0, 0.0, 0.0)
    assert [row["shadowed"] for row in scored["rows"]] == [True, False, False, False]
    # A loss on a request that was shadowed still counts, and one that was not cannot.
    wrong = {**result, "calls": [{**call, "answer": "C"} if call["model"] == "claude-haiku-5-5" else call
                                 for call in result["calls"]]}
    assert live_pilot.score_proxy(wrong, QUESTIONS, "claude-opus-5-5")["summary"]["loss_rate"] == 1.0
    # Routed runs have no shadow rows: every request counts, as before. Only the one
    # request that reached the frontier can be compared with it.
    routed = asyncio.run(live_pilot.through_proxy(policy(), QUESTIONS, mode="route", max_cost_usd=1.0,
                                                  ledger_path=tmp_path / "r.db", inner=httpx.MockTransport(provider)))
    summary = live_pilot.score_proxy(routed, QUESTIONS, "claude-opus-5-5")["summary"]
    assert "shadowed_graded" not in summary and (summary["accepted_share_graded"], summary["compared"]) == (0.75, 1)


def test_time_the_machine_slept_through_is_reported_beside_the_run(tmp_path, monkeypatch):
    """The laptop slept for 106 seconds during one paid run: the wall clock moved on and
    the monotonic clock, which the proxy times requests with, did not."""
    import time

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    asleep = {"seconds": 0.0}
    monkeypatch.setattr(live_pilot, "time", SimpleNamespace(
        time=lambda: time.time() + asleep["seconds"], monotonic=time.monotonic, sleep=time.sleep))

    def sleepy(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["model"] == "claude-opus-5-5" and body["messages"][-1]["content"] == "Question 2?":
            asleep["seconds"] += 106.0
        return provider(request)
    result = asyncio.run(live_pilot.through_proxy(policy(), QUESTIONS, mode="shadow", max_cost_usd=1.0, concurrency=1,
                                                  ledger_path=tmp_path / "l.db", inner=httpx.MockTransport(sleepy)))
    rows = {row["task_id"]: row for row in result["requests"]}
    assert rows["q2"]["client_seconds_including_shadow"] >= 106.0 and rows["q2"]["client_monotonic_seconds"] < 5.0
    assert rows["q2"]["latency_ms"] < 5000  # the ledger's latency does not see the sleep
    summary = live_pilot.score_proxy(result, QUESTIONS, "claude-opus-5-5")["summary"]
    assert summary["suspended_requests"] == 1 and 105.0 <= summary["suspended_seconds"] <= 107.0
    # No sleep, or a result recorded before the monotonic time was kept: nothing is reported.
    awake = asyncio.run(live_pilot.through_proxy(policy(), QUESTIONS, mode="shadow", max_cost_usd=1.0,
                                                 ledger_path=tmp_path / "a.db", inner=httpx.MockTransport(provider)))
    assert "suspended_seconds" not in live_pilot.score_proxy(awake, QUESTIONS, "claude-opus-5-5")["summary"]
    old = {**result, "requests": [{key: value for key, value in row.items() if key != "client_monotonic_seconds"}
                                  for row in result["requests"]]}
    assert "suspended_seconds" not in live_pilot.score_proxy(old, QUESTIONS, "claude-opus-5-5")["summary"]


def bench_with_scores(tmp_path, monkeypatch, scores_by_model):
    """A two-set medqa archive: 40 questions, each model's score per question index."""
    monkeypatch.setitem(live_pilot.DATASETS, "medqa", {**live_pilot.DATASETS["medqa"],
                                                       "sets": {"calibrate": 20, "shadow": 20}})
    for model, scores in scores_by_model.items():
        folder = tmp_path / "bench" / "medqa" / "test" / model
        folder.mkdir(parents=True)
        (folder / "run.json").write_text(json.dumps({"records": [
            {"index": index, "prompt": f"Question {index}?", "ground_truth": "B", "score": scores[index]}
            for index in range(40)]}))
    return tmp_path / "bench"


def test_sample_prints_how_hard_each_set_is_and_warns_about_an_uneven_draw(tmp_path, monkeypatch, capsys):
    import random

    order = list(range(40))
    random.Random(0).shuffle(order)  # the draw sample_dataset makes: the first 20 are the history
    history = set(order[:20])
    # Model a solves everything; model b solves the later set and half of the history.
    half = set(order[:10])
    bench = bench_with_scores(tmp_path, monkeypatch, {
        "model-a": [1.0] * 40, "model-b": [0.0 if index in half else 1.0 for index in range(40)]})
    live_pilot.main(["sample", "--bench-dir", str(bench), "--output", str(tmp_path / "sets"), "--dataset", "medqa"])
    printed = capsys.readouterr()
    counts, scores = (json.loads(part) for part in printed.out.replace("}\n{", "}\x00{").split("\x00"))
    assert counts == {"calibrate": 20, "shadow": 20}
    drawn = {json.loads(line)["task_id"] for line in (tmp_path / "sets" / "medqa-calibrate.jsonl").read_text().splitlines()}
    assert drawn == {f"medqa:{index}" for index in history}
    assert scores["mean_recorded_score"] == {"medqa-calibrate": 0.75, "medqa-shadow": 1.0}
    assert len(scores["uneven"]) == 1 and "medqa-shadow is 25.0 points easier than medqa-calibrate" in scores["uneven"][0]
    assert printed.err.startswith("Warning: medqa-shadow is 25.0 points easier") and "not like for like" in printed.err
    # --compare reads sets that exist, the history first, and samples nothing.
    live_pilot.main(["sample", "--bench-dir", str(bench), "--compare", str(tmp_path / "sets" / "medqa-shadow.jsonl"),
                     str(tmp_path / "sets" / "medqa-calibrate.jsonl")])
    printed = capsys.readouterr()
    assert "medqa-calibrate is 25.0 points harder than medqa-shadow" in printed.err
    with pytest.raises(SystemExit) as refused:
        live_pilot.main(["sample", "--bench-dir", str(bench)])
    assert refused.value.code == 2 and "--output" in capsys.readouterr().err


def test_an_even_draw_and_sets_outside_the_archive_give_no_warning(tmp_path, monkeypatch):
    bench = bench_with_scores(tmp_path, monkeypatch, {"model-a": [1.0, 0.0] * 20, "model-b": [1.0] * 40})
    sets = {"history": [{"task_id": f"medqa:{index}", "ground_truth": "B", "dataset": "medqa"} for index in range(20)],
            "later": [{"task_id": f"medqa:{index}", "ground_truth": "B", "dataset": "medqa"} for index in range(20, 40)],
            "freetext": FREE, "tickets": [{"task_id": "tickets:1", "ground_truth": "A", "dataset": "tickets"}]}
    assert live_pilot.recorded_scores(bench, sets) == {"mean_recorded_score": {"history": 0.75, "later": 0.75},
                                                       "uneven": []}
