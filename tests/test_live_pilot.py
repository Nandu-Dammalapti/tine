import asyncio
import importlib.util
import json
import re
from pathlib import Path

import httpx
import pandas as pd

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
