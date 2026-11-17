import asyncio
import hashlib
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from tine.ledger import Ledger
from tine.proxy import ProxyConfig, create_app
from tine.shadow import extract_answer

SPEC = importlib.util.spec_from_file_location(
    "load_test", Path(__file__).resolve().parents[1] / "scripts" / "load_test.py")
lt = importlib.util.module_from_spec(SPEC)
sys.modules["load_test"] = lt  # dataclasses look their module up while it is being loaded
SPEC.loader.exec_module(lt)

AUTH = {"authorization": f"Bearer {lt.TOKEN}"}


def config(base_url, **changes):
    values = dict(tenant_keys={lt.TOKEN: lt.TENANT}, openai_api_key=lt.FAKE_KEY, anthropic_api_key=lt.FAKE_KEY,
                  openai_frontier_model=lt.FRONTIER, anthropic_frontier_model=lt.FRONTIER, openai_base_url=base_url,
                  anthropic_base_url=base_url, prices=lt.PRICES)
    values.update(changes)
    return ProxyConfig(**values)


def through_proxy(settings, bodies, **changes):
    """Send requests through the real proxy app to the stand-in on a real local socket.
    Returns, per request, the caller's response, its ledger row and the stand-in's own
    bytes for the same request asked directly."""
    async def scenario():
        standin = lt.StandInServer(settings)
        server = await asyncio.start_server(standin.handle, "127.0.0.1", 0)
        base_url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        ledger, out = Ledger(), []
        # trust_env=False: never through an HTTP proxy named in the environment.
        async with httpx.AsyncClient(trust_env=False) as upstream:
            app = create_app(config(base_url, **changes), ledger=ledger, client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://tine") as client:
                for route, body in bodies:
                    response = await client.post(lt.ROUTES[route], headers=AUTH, json=body)
                    row = ledger.get(lt.TENANT, response.headers["x-tine-request-id"])
                    direct, conn = lt.Client(base_url), lt.Conn()
                    reply = await direct.send(conn, direct.message("POST", lt.ROUTES[route], b"",
                                                                   json.dumps(body).encode()), keep=True)
                    direct.close(conn)
                    out.append((response, row, reply.body))
        server.close()
        return out, standin
    return asyncio.run(scenario())


def ask(route, stream=False, usage=True, text="Which option? End with a line 'Answer: X'."):
    body = {"model": lt.FRONTIER, "messages": [{"role": "user", "content": text}]}
    if route == "anthropic":
        body["max_tokens"] = 64
    if stream:
        body["stream"] = True
        if route == "openai" and usage:
            body["stream_options"] = {"include_usage": True}
    return route, body


def expected_cost(row):
    rates = lt.PRICES[f"{row['provider']}:{lt.FRONTIER}"]
    return (row["input_tokens"] * rates["input_per_million"] + row["output_tokens"] * rates["output_per_million"]) / 1e6


def test_buffered_replies_parse_as_the_proxy_expects_and_a_cost_is_recorded():
    settings = lt.StandIn(reply_tokens=30)
    results, standin = through_proxy(settings, [ask("anthropic"), ask("openai")])
    for (response, row, direct), provider in zip(results, ("anthropic", "openai")):
        assert response.status_code == 200 and response.content == direct
        letter = extract_answer(provider, response.json(), lt.ANSWER_PATTERN)
        assert letter in lt.LETTERS
        assert row["output_tokens"] == 30 and row["input_tokens"] > 0
        assert row["cost_usd"] == pytest.approx(expected_cost(row)) and row["cost_usd"] > 0
        assert row["baseline_cost_usd"] == pytest.approx(row["cost_usd"])
    assert standin.by_model == {lt.FRONTIER: 4} and standin.connections >= 1


def test_streams_are_relayed_byte_for_byte_and_the_stream_usage_reader_prices_them():
    settings = lt.StandIn(reply_tokens=25, chunks=6)
    results, _ = through_proxy(settings, [ask("anthropic", stream=True), ask("openai", stream=True),
                                          ask("openai", stream=True, usage=False)])
    (anthropic, anthropic_row, anthropic_direct), (openai, openai_row, openai_direct), (bare, bare_row, _) = results
    for response, row, direct in ((anthropic, anthropic_row, anthropic_direct), (openai, openai_row, openai_direct)):
        assert response.headers["content-type"].startswith("text/event-stream") and response.content == direct
        assert row["stream"] == 1 and row["output_tokens"] == 25
        assert row["cost_usd"] == pytest.approx(expected_cost(row)) and row["cost_usd"] > 0
    events = [line[6:] for line in anthropic.text.split("\n") if line.startswith("data: ")]
    kinds = [json.loads(event)["type"] for event in events]
    assert kinds[:3] == ["message_start", "content_block_start", "ping"]
    assert kinds[-3:] == ["content_block_stop", "message_delta", "message_stop"]
    assert kinds.count("content_block_delta") == 6
    text = "".join(json.loads(event)["delta"]["text"] for event in events
                   if json.loads(event)["type"] == "content_block_delta")
    assert text.rsplit("\n", 1)[1].startswith("Answer: ")
    chunks = [line[6:] for line in openai.text.split("\n") if line.startswith("data: ")]
    assert chunks[-1] == "[DONE]" and json.loads(chunks[-2])["usage"]["completion_tokens"] == 25
    assert all(json.loads(chunk)["usage"] is None for chunk in chunks[:-2])
    # OpenAI sends usage in a stream only when the request asks for it: no cost then.
    assert b'"usage"' not in bare.content and bare_row["cost_usd"] is None and bare_row["output_tokens"] is None


def test_error_replies_come_back_with_the_providers_shape_and_status():
    results, _ = through_proxy(lt.StandIn(error_share=1.0, error_statuses=(529,)), [ask("anthropic"), ask("openai")])
    (anthropic, anthropic_row, _), (openai, _, _) = results
    assert anthropic.status_code == 529 and anthropic.json()["error"]["type"] == "overloaded_error"
    assert openai.status_code == 529 and openai.json()["error"]["type"] == "server_error"
    assert anthropic_row["upstream_status"] == 529 and anthropic_row["cost_usd"] is None
    limited, _ = through_proxy(lt.StandIn(error_share=1.0, error_statuses=(429,)), [ask("anthropic")])
    assert limited[0][0].status_code == 429 and limited[0][0].headers["retry-after"] == "1"


def test_the_agreement_settings_give_the_shares_they_name():
    settings = lt.StandIn(agree_share=0.7, differ_share=0.1)
    digests = [hashlib.sha256(f"prompt {index}".encode()).digest() for index in range(20_000)]
    small_a, small_b = lt.CHEAP
    agree = sum(lt.answer_letter(settings, d, small_a) == lt.answer_letter(settings, d, small_b) for d in digests)
    differ = sum(lt.answer_letter(settings, d, small_a) != lt.answer_letter(settings, d, lt.FRONTIER) for d in digests)
    assert agree / 20_000 == pytest.approx(0.7, abs=0.02) and differ / 20_000 == pytest.approx(0.1, abs=0.02)
    always = lt.StandIn(agree_share=1.0, differ_share=0.0)
    assert all(len({lt.answer_letter(always, d, model) for model in (*lt.CHEAP, lt.FRONTIER)}) == 1 for d in digests[:200])


def test_the_synthetic_policy_is_accepted_by_the_proxy_and_marked_as_a_load_test():
    policy = lt.synthetic_policy()
    assert policy["tenant"] == policy["task_type"] == "load-test" and policy["synthetic"] is True
    assert "Nothing was certified" in policy["note"]
    routes = {lt.TENANT: {"openai": policy, "anthropic": policy}}
    config("http://127.0.0.1:1", route_policies=routes).validate()
    config("http://127.0.0.1:1", shadow_policies=routes, shadow_sample_rate=1.0, shadow_max_cost_usd=1.0).validate()
    agreed, standin = through_proxy(lt.StandIn(agree_share=1.0), [ask("openai"), ask("anthropic")], route_policies=routes)
    assert [response.headers["x-tine-route"] for response, _, _ in agreed] == ["stage-0", "stage-0"]
    assert agreed[0][1]["model"] == lt.CHEAP[0] and agreed[0][1]["routing_cost_usd"] > 0
    assert standin.by_model[lt.CHEAP[0]] == standin.by_model[lt.CHEAP[1]] == 2
    split, _ = through_proxy(lt.StandIn(agree_share=0.0), [ask("openai")], route_policies=routes)
    assert split[0][0].headers["x-tine-route"] == "frontier" and split[0][1]["model"] == lt.FRONTIER


def test_percentiles_are_nearest_rank_on_fixed_numbers():
    hundred = list(range(1, 101))
    assert [lt.percentile(hundred, p) for p in (50, 90, 95, 99, 100)] == [50, 90, 95, 99, 100]
    assert [lt.percentile([40, 10, 30, 20], p) for p in (25, 50, 51, 95)] == [10, 20, 30, 40]
    assert lt.percentile([7], 99) == 7 and lt.percentile([], 50) is None and lt.distribution([]) is None
    assert lt.distribution([5.0, 1.0, 2.0, 4.0, 3.0]) == {
        "n": 5, "min": 1.0, "p50": 3.0, "p90": 5.0, "p95": 5.0, "p99": 5.0, "max": 5.0, "mean": 3.0}
    # 20 values: the 95th percentile is the 19th, not a value between the last two.
    assert lt.distribution(list(range(1, 21)))["p95"] == 19
    assert lt.spread([3.0, None, 1.0, 2.0]) == {"median": 2.0, "min": 1.0, "max": 3.0, "rounds": [3.0, None, 1.0, 2.0]}


def row(index, scheduled, sent, done, *, first=None, status=200, stream=False, measured=True, error=None, via=None):
    return lt.Row(index, "openai", stream, measured, scheduled, sent, first, done, status, error, 100, None, via,
                  None, None)


def test_open_loop_latency_runs_from_the_scheduled_send_time():
    assert lt.schedule(4, 5) == [0.0, 0.25, 0.5, 0.75, 1.0]
    # Scheduled at 1.0 s, left 0.3 s late, first byte at 1.4 s, done at 2.0 s.
    assert lt.timings(1.0, 1.3, 1.4, 2.0) == pytest.approx(
        {"latency_ms": 1000.0, "service_ms": 700.0, "send_delay_ms": 300.0, "ttfb_ms": 400.0})
    assert lt.timings(1.0, 1.0, None, 1.5)["ttfb_ms"] is None
    # Ten requests a second for one second behind a server that takes 0.25 s for each,
    # one at a time: every request waits for the ones before it. Service time stays
    # 250 ms; the latency a caller sees grows by 150 ms with every request.
    load = lt.Load(base_url="http://x", rate=10.0, duration=1.0)
    rows, free_at = [], 0.0
    for index, offset in enumerate(lt.schedule(10.0, 10)):
        sent = max(offset, free_at)
        free_at = sent + 0.25
        rows.append(row(index, offset, sent, free_at))
    summary = lt.summarize(load, rows, measure_from=0.0, stop_at=1.0)
    assert summary["service_ms"]["p50"] == pytest.approx(250.0) and summary["service_ms"]["max"] == pytest.approx(250.0)
    assert summary["latency_ms"]["min"] == pytest.approx(250.0)
    assert summary["latency_ms"]["p50"] == pytest.approx(250.0 + 4 * 150.0)
    assert summary["latency_ms"]["max"] == pytest.approx(250.0 + 9 * 150.0)
    assert summary["send_delay_ms"]["max"] == pytest.approx(9 * 150.0)
    # Only four replies arrived inside the one-second window: that is the throughput.
    assert summary["throughput_rps"] == 4.0 and summary["loop"] == "open" and summary["measured_requests"] == 10


def test_a_summary_leaves_out_the_warm_up_and_keeps_errors_apart():
    load = lt.Load(base_url="http://x", in_flight=2, requests=4, warmup_requests=1)
    rows = [row(0, 0.0, 0.0, 9.0, measured=False), row(1, 1.0, 1.0, 1.1), row(2, 1.0, 1.0, 1.3, stream=True, first=1.05),
            row(3, 1.1, 1.1, 1.15, status=429), row(4, 1.3, 1.3, 2.0, status=0, error="timeout")]
    summary = lt.summarize(load, rows, measure_from=0.0, stop_at=None)
    assert summary["warmup_requests"] == 1 and summary["measured_requests"] == 4
    assert summary["statuses"] == {"200": 2, "429": 1} and summary["client_errors"] == {"timeout": 1}
    assert summary["latency_ms"]["n"] == 2 and summary["latency_ms"]["max"] == pytest.approx(300.0)
    assert summary["streamed_ttfb_ms"]["p50"] == pytest.approx(50.0) and summary["buffered_latency_ms"]["n"] == 1
    assert summary["not_2xx_latency_ms"]["p50"] == pytest.approx(50.0)
    # Three replies between the first measured send (1.0 s) and the last finish (2.0 s).
    assert summary["window_seconds"] == pytest.approx(1.0) and summary["throughput_rps"] == 3.0
    stretches = lt.windows(rows, 0.25)
    assert [stretch["requests"] for stretch in stretches] == [3, 1] and stretches[1]["client_errors"] == 1


def test_requests_are_the_same_on_both_sides_and_new_with_every_nonce():
    load = lt.Load(base_url="http://x", in_flight=1, requests=1, stream_share=0.5, prompt_chars=600, nonce="a")
    first, again = lt.Requests(load, lt.Client("http://x")), lt.Requests(load, lt.Client("http://y:9"))
    bodies = [first.body(index) for index in range(200)]
    assert bodies == [again.body(index) for index in range(200)]
    assert 60 < sum(stream for _, stream, _ in bodies) < 140 and {route for route, _, _ in bodies} == set(lt.ROUTES)
    prompts = {body["messages"][0]["content"] for _, _, body in bodies}
    assert len(prompts) == 200 and all(500 < len(prompt) <= 600 for prompt in prompts)
    other = lt.Requests(lt.Load(base_url="http://x", in_flight=1, requests=1, prompt_chars=600, nonce="b"))
    assert other.body(0)[2]["messages"][0]["content"] not in prompts


def test_servers_this_tool_starts_get_no_real_key_and_the_docker_line_uses_only_fake_settings(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-would-be-real")
    monkeypatch.setenv("TINE_ANTHROPIC_API_KEY", "sk-would-be-real-too")
    monkeypatch.setenv("HTTPS_PROXY", "http://corporate-proxy:3128")
    env = lt.child_environment({"TINE_LEDGER_PATH": "/x"})
    assert "ANTHROPIC_API_KEY" not in env and "TINE_ANTHROPIC_API_KEY" not in env and "HTTPS_PROXY" not in env
    assert env["TINE_LEDGER_PATH"] == "/x"
    scenario = lt.Scenario("s", mode="shadow", sample_rate=0.5)
    line = lt.docker_line(scenario, provider_port=18001, base_url="http://127.0.0.1:18080",
                          policies=Path("/tmp/work/config/load-test-policies.json"))
    assert line.startswith("docker run --rm -d --name tine-load-proxy -p 127.0.0.1:18080:8000 ")
    assert "http://host.docker.internal:18001" in line and "--env-file" not in line and "sk-" not in line
    assert f"TINE_ANTHROPIC_API_KEY={lt.FAKE_KEY}" in line and "-v /tmp/work/config:/config:ro" in line
    assert "TINE_SHADOW_POLICIES=/config/load-test-policies.json" in line and line.endswith(" tine-proxy:dev")
    # What docker-compose.yml sets for the documented deployment, so the same container is measured.
    compose = (Path(__file__).resolve().parents[1] / "docker-compose.yml").read_text()
    for flag, written in (("--read-only", "read_only: true"), ("--tmpfs /tmp:size=64m,mode=1777", "/tmp:size=64m,mode=1777"),
                          ("--user 10001:10001", 'user: "10001:10001"'), ("--cap-drop ALL", "- ALL"),
                          ("--security-opt no-new-privileges:true", "no-new-privileges:true"), ("--init", "init: true")):
        assert flag in line and written in compose
    assert "/data" not in line and "TINE_LEDGER_PATH" not in line
    mounted = lt.docker_line(lt.Scenario("p"), provider_port=18001, base_url="http://127.0.0.1:18080", policies=None,
                             data=Path("/tmp/work/data"))
    assert "-v /tmp/work/data:/data" in mounted and "TINE_LEDGER_PATH=/data/ledger.sqlite3" in mounted


def test_a_tiny_compare_runs_end_to_end_through_a_real_proxy_process(tmp_path):
    scenario = lt.Scenario("tiny", provider={"reply_tokens": 20, "chunks": 4}, rounds=1,
                           load={"in_flight": 2, "requests": 12, "stream_share": 0.5, "prompt_chars": 300,
                                 "outcome_share": 1.0}, unloaded_seconds=0.3)
    result = asyncio.run(lt.compare(scenario, tmp_path))
    assert result["direct"]["statuses"] == result["proxy"]["statuses"] == {"200": 12}
    assert result["proxy"]["client_errors"] == {} and result["added_latency_ms"]["p50"]["median"] is not None
    relayed = result["relayed_bytes"]
    assert relayed["streamed"]["compared"] + relayed["buffered"]["compared"] == 12
    assert relayed["streamed"]["different"] == relayed["buffered"]["different"] == 0
    assert relayed["streamed"]["compared"] > 0 and result["added_ttfb_ms"]["p50"] is not None
    assert result["ledger"]["rows"]["requests"] == 12 and result["ledger"]["rows"]["outcome_events"] == 12
    assert result["rounds"][0]["proxy"]["outcome_reports"]["statuses"] == {"200": 12}
    reports = result["report_endpoints"]
    assert reports["ledger_report"]["status"] == 200 and reports["shadow_report"]["status"] == 404
    process = result["proxy_process"]
    assert process["peak_rss_bytes"] > 1_000_000 and process["cpu_seconds"] > 0 and not process["killed_after_grace"]
    # Twelve directly, twelve through the proxy, and the stand-in asked one request at a
    # time beforehand (one of those is its warm-up).
    alone = result["over_planned_ms"]["direct_one_request_at_a_time"]["requests"] + 1
    assert result["upstream"]["requests"] == 12 and result["standin"]["by_model"] == {lt.FRONTIER: 24 + alone}
    assert result["docker_run"] is None and lt.csv_rows(result)[0]["proxy_statuses"] == "200:12"
    # Two in flight: the stand-in was first asked one request at a time, and both ways of
    # working out the added latency are in the result beside the one chosen.
    assert result["over_planned_ms"]["direct_one_request_at_a_time"]["requests"] >= 1
    assert result["added_latency_basis"]["name"] in ("percentiles", "over_planned")
    assert result["added_paired_ms"]["pairs"] == [12] and result["added_by_percentiles_ms"]["latency_ms"]["p50"]
    assert set(lt.csv_rows(result)[0]) <= set(lt.CSV_COLUMNS)
    assert lt.csv_rows(result)[0]["prompt_chars"] == 300 and lt.csv_rows(result)[0]["reply_tokens"] == 20
    assert keys_about_savings_or_drift(result) == []


def keys_about_savings_or_drift(value, path=""):
    """Keys of a result that name a saving or a drift check, outside the block that says
    such figures are synthetic."""
    found = []
    if isinstance(value, dict):
        for key, inner in value.items():
            if key == "synthetic_not_evidence":
                continue
            if "saving" in str(key) or "drift" in str(key):
                found.append(f"{path}/{key}")
            found += keys_about_savings_or_drift(inner, f"{path}/{key}")
    elif isinstance(value, list):
        for inner in value:
            found += keys_about_savings_or_drift(inner, path)
    return found


def test_paired_differences_hold_where_a_difference_of_medians_does_not():
    # Replies of 1.0 to 2.9 s, the proxy adds 5 ms to each. Two short requests count on
    # the proxy side only: the difference of medians says the proxy is 95 ms faster.
    direct = [row(index, 0.0, 0.0, 1.0 + index / 10) for index in range(20)]
    through = [row(index, 0.0, 0.0, 1.005 + index / 10) for index in range(20)]
    direct[0] = row(0, 0.0, 0.0, 1.0, measured=False)  # in the warm-up on one side
    direct[3] = row(3, 0.0, 0.0, 0.01, status=429)  # refused on one side: not a pair
    pairs = lt.paired(direct, through)
    assert pairs["pairs"] == 18 and pairs["added_ms"]["p50"] == pytest.approx(5.0)
    assert pairs["added_ms"]["min"] == pytest.approx(5.0) and pairs["added_ms"]["max"] == pytest.approx(5.0)
    medians = [lt.percentile([(r.done - r.scheduled) * 1000 for r in rows if r.measured and r.status == 200], 50)
               for rows in (direct, through)]
    assert medians[1] - medians[0] == pytest.approx(-95.0)
    streams = lt.paired([row(0, 0.0, 0.0, 2.0, stream=True, first=0.40)], [row(0, 5.0, 5.0, 7.004, stream=True, first=5.403)])
    assert streams["added_ttfb_ms"]["p50"] == pytest.approx(3.0) and streams["added_ms"]["p50"] == pytest.approx(4.0)
    assert lt.paired(direct, [])["added_ms"] is None


def test_latency_over_the_stand_ins_set_time_and_the_rule_for_a_queueing_direct_side():
    settings = lt.StandIn(latency_ms=100.0, latency_spread_ms=50.0, ttfb_ms=20.0, error_share=0.2)
    load = lt.Load(base_url="http://x", in_flight=1, requests=1, stream_share=0.5, prompt_chars=300, nonce="n")
    requests, standin = lt.Requests(load), lt.StandInServer(settings)
    rows, expected = [], {}
    for index in range(40):
        route, stream, body = requests.body(index)
        plan = standin.plan(route, json.dumps(body).encode())
        if plan["status"] != 200:
            rows.append(row(index, 0.0, 0.0, 0.001, status=plan["status"]))
            continue
        expected[index] = plan["seconds"] * 1000
        rows.append(row(index, 10.0, 10.0, 10.0 + plan["seconds"] + 0.002, stream=stream,
                        first=10.0 + standin.first_byte_seconds(plan) + 0.001 if stream else None))
    planned = lt.planned_times(settings, load, rows)
    assert {index: total for index, (total, _) in planned.items()} == pytest.approx(expected)
    assert 20 < len(planned) < 40 and all(100.0 <= total < 150.0 for total, _ in planned.values())
    over = lt.over_planned(rows, planned)
    assert over["latency_ms"]["n"] == len(planned) and over["latency_ms"]["p50"] == pytest.approx(2.0, abs=1e-6)
    assert over["latency_ms"]["max"] == pytest.approx(2.0, abs=1e-6) and over["ttfb_ms"]["p50"] == pytest.approx(1.0, abs=1e-6)
    # Settings without any spread need no request rebuilt.
    assert lt.planned_times(lt.StandIn(latency_ms=30.0), load, [row(7, 0.0, 0.0, 0.03)]) == {7: (30.0, None)}
    # A relay known to add 5 ms, instant replies, 50 in flight at a stand-in good for
    # about 40,000 a second: directly the median is 1.2 ms of queueing against 0.1 ms
    # alone, and through the relay 5.9 ms. Subtracting gives 4.7; the relay adds 5.8.
    assert lt.direct_is_saturated(1.2, 0.1, 5.9)
    # Today's proxy at 10 in flight: 0.26 ms of queueing against 13.9 ms. Negligible.
    assert not lt.direct_is_saturated(0.26, 0.1, 13.9)
    # Replies with a set latency: timers run late by the same amount on both sides.
    assert not lt.direct_is_saturated(1.5, 1.4, 6.0)
    assert not lt.direct_is_saturated(None, 0.1, 5.0) and not lt.direct_is_saturated(1.2, None, 5.0)
    assert set(lt.BASES) == {"percentiles", "paired", "over_planned", "holdout"}


def test_routed_requests_are_reported_by_route_and_only_the_holdout_is_overhead():
    # Directly every request asks the frontier (300 ms). Through the proxy: the holdout
    # asks the frontier too (302 ms), stage 0 waits for the slower cheaper model
    # (350 ms), an escalated request asks the cheaper ones and then the frontier (650 ms).
    direct = [row(index, 0.0, 0.0, 0.300) for index in range(9)]
    through = ([row(index, 0.0, 0.0, 0.302, via="holdout") for index in range(3)]
               + [row(index, 0.0, 0.0, 0.350, via="stage-0") for index in range(3, 6)]
               + [row(index, 0.0, 0.0, 0.650, via="frontier") for index in range(6, 9)])
    routes = lt.by_route(direct, through)
    assert set(routes) == {"holdout", "stage-0", "frontier"} and routes["holdout"]["requests"] == 3
    assert routes["holdout"]["minus_direct_ms"]["p50"] == pytest.approx(2.0)
    assert routes["stage-0"]["minus_direct_ms"]["p50"] == pytest.approx(50.0)
    assert routes["frontier"]["latency_ms"]["p50"] == pytest.approx(650.0)
    assert "policy's shape" in lt.ROUTE_FIGURES["stage"] and "overhead" in lt.ROUTE_FIGURES["holdout"]


def test_a_tiny_routed_compare_reports_added_latency_from_the_holdout_only_and_sets_synthetic_figures_apart(tmp_path):
    scenario = lt.Scenario("tiny-route", provider={"reply_tokens": 20, "latency_ms": 20.0, "latency_spread_ms": 20.0},
                           rounds=1, mode="route", audit_rate=0.5, unloaded_seconds=0.3,
                           load={"in_flight": 4, "requests": 40, "prompt_chars": 300})
    result = asyncio.run(lt.compare(scenario, tmp_path))
    routing = result["routing"]
    assert result["proxy"]["statuses"] == {"200": 40} and sum(routing["x_tine_route"].values()) == 40
    held = routing["latency_by_route_ms"]["holdout"]
    assert held["requests"] == routing["x_tine_route"]["holdout"] > 0 and "overhead" in held["what"]
    assert result["added_latency_basis"]["name"] == "holdout"
    assert result["added_latency_ms"]["p50"] == held["minus_direct_p50"]
    assert result["added_by_percentiles_ms"] is None and result["added_paired_ms"] is None
    assert any("policy's shape" in note and "holdout" in note for note in result["observations"])
    line = lt.csv_rows(result)[0]
    assert line["added_p50_ms"] is None and line["added_p95_ms"] is None and line["added_basis"] == "holdout"
    assert line["holdout_added_p50_ms"] == held["minus_direct_p50"]["median"] and "holdout:" in line["route_p50_ms"]
    # Counts stay in the result. What the proxy worked out from made-up prices and the
    # stand-in's agreement setting is under one key, with the policy's note beside it.
    assert routing["routed_requests"] + routing["holdout_requests"] == 40
    apart = routing["synthetic_not_evidence"]
    assert "Nothing was certified" in apart["note"] and "saving_fraction" in apart["holdout_estimate"]
    assert keys_about_savings_or_drift(result) == []
    assert keys_about_savings_or_drift({"routing": {"saving_fraction": 0.7}}) == ["/routing/saving_fraction"]


def test_a_shadow_run_paused_by_provider_refusals_says_so(tmp_path):
    scenario = lt.Scenario("tiny-shadow", provider={"reply_tokens": 20, "error_share": 0.3, "error_statuses": (529,)},
                           rounds=1, mode="shadow", sample_rate=1.0, unloaded_seconds=0.0,
                           load={"in_flight": 1, "requests": 40, "prompt_chars": 300})
    result = asyncio.run(lt.compare(scenario, tmp_path))
    shadow = result["shadow"]
    assert shadow["skipped_rate_limited"] > 0 and shadow["sampled_requests"] > 0
    assert any("paused by the stand-in's 429 and 529 replies" in note and "do not measure shadow mode's overhead" in note
               for note in result["observations"])
    assert any("skipped_rate_limited" in note and "skipped_rate_cap" in note and "skipped_stopped" in note
               and "rate_limited_runs" in note for note in result["observations"])
    line = lt.csv_rows(result)[0]
    assert line["shadow_skipped_rate_limited"] == shadow["skipped_rate_limited"]
    assert {"shadow_skipped_rate_cap", "shadow_skipped_stopped", "shadow_rate_limited_runs"} <= set(lt.CSV_COLUMNS)
    assert "Nothing was certified" in shadow["synthetic_not_evidence"]["note"]
    assert "drift_check" in shadow["synthetic_not_evidence"] and "policy_would_accept_share" not in shadow
    assert keys_about_savings_or_drift(result) == []


def test_the_matrix_covers_agent_sized_requests_and_long_replies_and_every_row_states_its_sizes():
    for quick in (False, True):
        chosen = lt.scenarios(quick=quick)
        assert {2000, 50_000, 200_000} <= {s.load.get("prompt_chars", 2000) for s in chosen}
        long_streams = [s for s in chosen if s.provider.get("reply_tokens", 200) >= 2000 and s.load.get("stream_share")]
        assert long_streams and long_streams[0].provider["chunks"] >= 400
        assert any(s.kind == "ledger" and s.load.get("prompt_chars", 2000) >= 50_000 for s in chosen)
        limited = [s for s in chosen if s.mode == "shadow" and s.provider.get("error_share")]
        assert len(limited) == 1 and "not shadow mode's overhead" in limited[0].about
        assert len({s.name for s in chosen}) == len(chosen)
    assert {"prompt_chars", "reply_tokens", "added_basis", "added_is_queueing_at_saturation"} <= set(lt.CSV_COLUMNS)
    saturated = [s for s in lt.scenarios() if s.name.startswith("a-passthrough-instant-") and s.load["in_flight"] > 1]
    assert len(saturated) == 3 and all("not per-request overhead" in s.about for s in saturated)


def test_another_server_on_the_stand_ins_port_is_refused_and_a_taken_port_is_not_called_free(tmp_path):
    async def scenario():
        # Someone else's server: it answers GET /health with 200, as the stand-in does.
        foreign = await asyncio.start_server(lt.StandInServer(lt.StandIn(latency_ms=50.0)).handle, "127.0.0.1", 0)
        port = foreign.sockets[0].getsockname()[1]
        try:
            # Either the answer is seen not to be ours, or our own stand-in is seen to have
            # failed to take the port first: both refuse to measure.
            with pytest.raises(RuntimeError, match="Another server answers on port|before it answered"):
                await lt.start_standin(lt.StandIn(), tmp_path, port=port)
        finally:
            foreign.close()
        ours, url = await lt.start_standin(lt.StandIn(), tmp_path)
        try:
            health = json.loads((await lt.Client(url).get("/health")).body)
        finally:
            ours.stop()
        return health, ours.pid
    health, pid = asyncio.run(scenario())
    assert health["standin"] is True and health["pid"] == pid and len(health["instance"]) == 16
    # A listener on every address must not leave the port looking free on 127.0.0.1.
    with socket.socket() as taken:
        taken.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        taken.bind(("0.0.0.0", 0))
        taken.listen()
        lt._next_port = taken.getsockname()[1]
        assert lt.free_port() != taken.getsockname()[1]


def listening(port):
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


@pytest.mark.parametrize("name", ["SIGTERM", "SIGHUP"])
def test_a_terminated_run_stops_its_servers_and_removes_its_folder(tmp_path, name):
    env = lt.child_environment({"TMPDIR": str(tmp_path)})
    tool = subprocess.Popen(
        [sys.executable, str(Path(lt.__file__).resolve()), "compare", "--in-flight", "2", "--duration", "60",
         "--latency-ms", "50", "--rounds", "1", "--unloaded-seconds", "0"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        ports, deadline = [], time.monotonic() + 60
        while len(ports) < 2 and time.monotonic() < deadline and tool.poll() is None:
            time.sleep(0.2)
            ports = []
            for folder in tmp_path.glob("tine-load-*"):
                ports += [int(log.stem.split("-")[1]) for log in folder.glob("standin-*.log")]
                for line in (folder / "proxy.log").read_text().splitlines() if (folder / "proxy.log").exists() else []:
                    if "running on http://127.0.0.1:" in line:
                        ports.append(int(line.split("http://127.0.0.1:")[1].split()[0]))
        assert len(ports) == 2 and all(listening(port) for port in ports)
        time.sleep(1.0)  # the load is running
        tool.send_signal(getattr(signal, name))
        assert tool.wait(timeout=60) == 128 + getattr(signal, name)
        assert not any(listening(port) for port in ports)
        assert list(tmp_path.glob("tine-load-*")) == []
    finally:
        try:
            os.killpg(tool.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


# --- the stand-in against the providers' documented wire format -----------------------
#
# No raw reply from a real provider is recorded in this repository to compare with (no
# paid call was made for this tool, and replies are not committed). The shapes below
# are written out from the providers' API references as literals, so that a change to
# the stand-in has to be made here too, and the Anthropic ones are also read by the
# anthropic SDK, which fails on a field a real client needs. The openai SDK is not
# installed in the development environment.

PLAN_BODY = {"model": lt.FRONTIER, "max_tokens": 64, "messages": [{"role": "user", "content": "Which option?"}]}


def planned(provider, settings=None, **more):
    standin = lt.StandInServer(settings or lt.StandIn(reply_tokens=12, chunks=3))
    return standin, standin.plan(provider, json.dumps({**PLAN_BODY, **more}).encode())


def test_buffered_replies_and_errors_have_the_documented_keys_and_values():
    standin, plan = planned("anthropic")
    message = standin.payload("anthropic", plan)
    assert set(message) == {"id", "type", "role", "model", "content", "stop_reason", "stop_sequence", "usage"}
    assert (message["type"], message["role"], message["stop_reason"], message["stop_sequence"]) == (
        "message", "assistant", "end_turn", None)
    assert message["id"].startswith("msg_") and message["model"] == lt.FRONTIER
    assert message["content"] == [{"type": "text", "text": plan["text"]}]
    assert set(message["usage"]) == {"input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
                                     "output_tokens"}
    standin, plan = planned("openai")
    completion = standin.payload("openai", plan)
    assert set(completion) == {"id", "object", "created", "model", "choices", "usage"}
    assert completion["object"] == "chat.completion" and completion["id"].startswith("chatcmpl-")
    assert isinstance(completion["created"], int) and len(completion["choices"]) == 1
    assert completion["choices"][0] == {
        "index": 0, "message": {"role": "assistant", "content": plan["text"], "refusal": None}, "logprobs": None,
        "finish_reason": "stop"}
    usage = completion["usage"]
    assert set(usage) == {"prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details"}
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["prompt_tokens_details"] == {"cached_tokens": 0}
    assert standin.error("anthropic", 529) == {
        "type": "error", "error": {"type": "overloaded_error", "message": "stand-in provider: injected 529"}}
    assert standin.error("anthropic", 429)["error"]["type"] == "rate_limit_error"
    assert standin.error("anthropic", 500)["error"]["type"] == "api_error"
    assert standin.error("openai", 429) == {"error": {
        "message": "stand-in provider: injected 429", "type": "rate_limit_exceeded", "param": None,
        "code": "rate_limit_exceeded"}}
    assert set(standin.error("openai", 500)["error"]) == {"message", "type", "param", "code"}


def test_the_anthropic_stream_is_the_documented_event_sequence_with_event_lines():
    standin, plan = planned("anthropic", stream=True)
    opening, deltas, closing = standin.stream_parts("anthropic", plan)
    events = []
    for block in (opening + b"".join(deltas) + closing).decode().split("\n\n")[:-1]:
        # Every event is exactly an "event:" line naming it and a "data:" line.
        name, data = block.split("\n")
        assert name.startswith("event: ") and data.startswith("data: ")
        events.append((name[7:], json.loads(data[6:])))
    assert [name for name, _ in events] == [
        "message_start", "content_block_start", "ping", "content_block_delta", "content_block_delta",
        "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
    assert all(data["type"] == name for name, data in events)
    start = events[0][1]
    assert set(start) == {"type", "message"}
    assert set(start["message"]) == {"id", "type", "role", "model", "content", "stop_reason", "stop_sequence", "usage"}
    assert (start["message"]["type"], start["message"]["role"], start["message"]["content"],
            start["message"]["stop_reason"]) == ("message", "assistant", [], None)
    assert set(start["message"]["usage"]) == {"input_tokens", "cache_creation_input_tokens",
                                              "cache_read_input_tokens", "output_tokens"}
    assert events[1][1] == {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
    assert events[2][1] == {"type": "ping"}
    for _, data in events[3:6]:
        assert set(data) == {"type", "index", "delta"} and data["index"] == 0
        assert set(data["delta"]) == {"type", "text"} and data["delta"]["type"] == "text_delta"
    assert "".join(data["delta"]["text"] for _, data in events[3:6]) == plan["text"]
    assert events[6][1] == {"type": "content_block_stop", "index": 0}
    assert events[7][1] == {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                            "usage": {"output_tokens": 12}}
    assert events[8][1] == {"type": "message_stop"}


def test_the_openai_stream_is_the_documented_chunk_sequence_with_a_usage_chunk_and_done():
    standin, plan = planned("openai", stream=True, stream_options={"include_usage": True})
    opening, deltas, closing = standin.stream_parts("openai", plan)
    blocks = (opening + b"".join(deltas) + closing).decode().split("\n\n")[:-1]
    # No "event:" lines on this route: every block is one "data:" line.
    assert all(block.startswith("data: ") and "\n" not in block for block in blocks) and blocks[-1] == "data: [DONE]"
    chunks = [json.loads(block[6:]) for block in blocks[:-1]]
    assert len(chunks) == 1 + 3 + 1 + 1
    for chunk in chunks:
        assert set(chunk) == {"id", "object", "created", "model", "choices", "usage"}
        assert chunk["object"] == "chat.completion.chunk" and chunk["id"] == chunks[0]["id"]
        assert chunk["id"].startswith("chatcmpl-") and chunk["model"] == lt.FRONTIER
    assert chunks[0]["choices"] == [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
    for chunk in chunks[1:4]:
        assert len(chunk["choices"]) == 1 and set(chunk["choices"][0]) == {"index", "delta", "finish_reason"}
        assert set(chunk["choices"][0]["delta"]) == {"content"} and chunk["choices"][0]["finish_reason"] is None
    assert "".join(chunk["choices"][0]["delta"]["content"] for chunk in chunks[1:4]) == plan["text"]
    assert chunks[4]["choices"] == [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    assert all(chunk["usage"] is None for chunk in chunks[:5])
    # The usage chunk has no choices at all.
    assert chunks[5]["choices"] == [] and set(chunks[5]["usage"]) == {
        "prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details"}
    # Not asked for: no usage field anywhere and no extra chunk.
    standin, plan = planned("openai", stream=True)
    opening, deltas, closing = standin.stream_parts("openai", plan)
    bare = (opening + b"".join(deltas) + closing).decode().split("\n\n")[:-1]
    assert len(bare) == 1 + 3 + 1 + 1 and bare[-1] == "data: [DONE]"
    assert all(set(json.loads(block[6:])) == {"id", "object", "created", "model", "choices"} for block in bare[:-1])


def test_the_anthropic_sdk_reads_the_stand_ins_replies_streams_and_errors():
    anthropic = pytest.importorskip("anthropic")
    try:
        import httpx2 as web  # the HTTP library of recent versions of the SDK
    except ImportError:
        web = httpx

    def client_for(settings):
        standin = lt.StandInServer(settings)

        def answer(request):
            assert request.url.path == lt.ROUTES["anthropic"]
            plan = standin.plan("anthropic", request.content)
            if plan["status"] != 200:
                return web.Response(plan["status"], json=standin.error("anthropic", plan["status"]))
            if plan["stream"]:
                opening, deltas, closing = standin.stream_parts("anthropic", plan)
                return web.Response(200, headers={"content-type": "text/event-stream"},
                                    content=opening + b"".join(deltas) + closing)
            return web.Response(200, json=standin.payload("anthropic", plan))
        # A local transport: nothing leaves the process, and the key is the made-up one.
        return anthropic.Anthropic(api_key=lt.FAKE_KEY, base_url="http://standin.invalid", max_retries=0,
                                   http_client=web.Client(transport=web.MockTransport(answer)))
    asked = {"model": lt.FRONTIER, "max_tokens": 64, "messages": [{"role": "user", "content": "Which option?"}]}
    client = client_for(lt.StandIn(reply_tokens=12, chunks=5))
    message = client.messages.create(**asked)
    assert (message.type, message.role, message.stop_reason, message.model) == (
        "message", "assistant", "end_turn", lt.FRONTIER)
    assert message.content[0].type == "text" and message.content[0].text.rsplit("\n", 1)[1].startswith("Answer: ")
    assert message.usage.output_tokens == 12 and message.usage.input_tokens > 0
    with client.messages.stream(**asked) as stream:
        kinds = [event.type for event in stream]
        final = stream.get_final_message()
    assert kinds.count("content_block_delta") == 5 and kinds[0] == "message_start" and kinds[-1] == "message_stop"
    assert {"content_block_start", "content_block_stop", "message_delta"} <= set(kinds)
    # The same request gets the same text streamed as buffered, and the SDK puts it together.
    assert final.content[0].text == message.content[0].text and final.stop_reason == "end_turn"
    assert final.usage.output_tokens == 12 and final.usage.input_tokens == message.usage.input_tokens
    for status, kind, error in ((429, "rate_limit_error", anthropic.RateLimitError),
                                (529, "overloaded_error", anthropic.APIStatusError),
                                (500, "api_error", anthropic.InternalServerError)):
        refusing = client_for(lt.StandIn(error_share=1.0, error_statuses=(status,)))
        with pytest.raises(error) as raised:
            refusing.messages.create(**asked)
        assert raised.value.status_code == status and raised.value.body["type"] == "error"
        assert raised.value.body["error"]["type"] == kind
