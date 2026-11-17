import asyncio
import hashlib
import importlib.util
import json
import sys
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


def row(index, scheduled, sent, done, *, first=None, status=200, stream=False, measured=True, error=None):
    return lt.Row(index, "openai", stream, measured, scheduled, sent, first, done, status, error, 100, None, None,
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


def test_a_tiny_compare_runs_end_to_end_through_a_real_proxy_process(tmp_path):
    scenario = lt.Scenario("tiny", provider={"reply_tokens": 20, "chunks": 4}, rounds=1,
                           load={"in_flight": 2, "requests": 12, "stream_share": 0.5, "prompt_chars": 300,
                                 "outcome_share": 1.0})
    result = asyncio.run(lt.compare(scenario, tmp_path))
    assert result["direct"]["statuses"] == result["proxy"]["statuses"] == {"200": 12}
    assert result["proxy"]["client_errors"] == {} and result["added_latency_ms"]["p50"]["median"] is not None
    relayed = result["relayed_bytes"]
    assert relayed["streamed"]["compared"] + relayed["buffered"]["compared"] == 12
    assert relayed["streamed"]["different"] == relayed["buffered"]["different"] == 0
    assert relayed["streamed"]["compared"] > 0 and result["added_ttfb_ms"]["p50"] is not None
    assert result["ledger"]["rows"]["requests"] == 12 and result["ledger"]["rows"]["outcome_events"] == 12
    assert result["rounds"][0]["proxy"]["outcome_reports"]["statuses"] == {"200": 12}
    assert result["report_endpoints"]["savings"]["status"] == 200 and result["report_endpoints"]["shadow"]["status"] == 404
    process = result["proxy_process"]
    assert process["peak_rss_bytes"] > 1_000_000 and process["cpu_seconds"] > 0 and not process["killed_after_grace"]
    assert result["upstream"]["requests"] == 12 and result["standin"]["by_model"] == {lt.FRONTIER: 24}
    assert result["docker_run"] is None and lt.csv_rows(result)[0]["proxy_statuses"] == "200:12"
