import html
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tine.ledger import Ledger
from tine.report_page import NOT_KNOWN, count, headline, headline_sentences, money, percent, render
from tine.shadow import SHADOW_SCOPE, ShadowStore

spec = importlib.util.spec_from_file_location(
    "tine_savings_report", Path(__file__).parents[1] / "scripts" / "savings_report.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

HOSTILE = '<script>alert("x")</script><img src=x onerror=1>\'&'
POLICY = {
    "version": 1, "tenant": "acme", "task_type": "quiz", "frontier": "big",
    "stages": [{"models": ["small-a", "small-b"], "check": "agreement"},
               {"models": ["mid"], "check": "tests+judge:sonnet"}],
    "guarantee": {"epsilon": 0.02, "delta": 0.05, "certified": True, "bound": 0.0162, "certify_tasks": 1426,
                  "certify_degradation_rate": 0.0105, "statement": "With confidence 1 - delta, the loss is small."},
    "estimate": {"scope": "held-out test tasks", "test_tasks": 1414, "savings": 0.525},
}


def filled_ledger() -> Ledger:
    """Three routed requests: two confirmed with every cost, one still pending."""
    ledger = Ledger()
    for cost, routing, baseline in ((0.25, 0.05, 1.0), (0.5, 0.2, 1.0)):
        ledger.record(tenant="acme", provider="openai", model="small-a", request_body={"q": cost}, success=True,
                      cost_usd=cost, routing_cost_usd=routing, baseline_cost_usd=baseline)
    ledger.record(tenant="acme", provider="openai", model="big", request_body={"q": "pending"}, cost_usd=1.0)
    return ledger


def routed_ledger() -> Ledger:
    """Live routing through a policy, as the proxy records it with the baseline estimate
    on: three answers from a cheaper model with an estimated baseline, one more whose
    routing cost is not known, one with no baseline at all, one request that escalated
    to the frontier (its baseline measured), one failure and one pending."""
    ledger = Ledger()
    for index in range(3):
        ledger.record(tenant="acme", provider="openai", model="small-a", request_body={"q": index}, success=True,
                      cost_usd=0.1, routing_cost_usd=0.05, baseline_cost_usd=1.0, baseline_estimated=True)
    ledger.record(tenant="acme", provider="openai", model="small-a", request_body={"q": "unpriced"}, success=True,
                  cost_usd=0.1, routing_cost_usd=None, baseline_cost_usd=1.0, baseline_estimated=True)
    ledger.record(tenant="acme", provider="openai", model="small-a", request_body={"q": "no baseline"}, success=True,
                  cost_usd=0.1, routing_cost_usd=0.05)
    ledger.record(tenant="acme", provider="openai", model="big", request_body={"q": "hard"}, success=True,
                  cost_usd=1.0, routing_cost_usd=0.15, baseline_cost_usd=1.0)
    ledger.record(tenant="acme", provider="openai", model="small-a", request_body={"q": "wrong"}, success=False,
                  cost_usd=0.1, routing_cost_usd=0.0, baseline_cost_usd=1.0, baseline_estimated=True)
    ledger.record(tenant="acme", provider="openai", model="small-a", request_body={"q": "later"}, cost_usd=0.1,
                  routing_cost_usd=0.05)
    return ledger


def add_shadow_rows(ledger: Ledger) -> ShadowStore:
    """Four shadowed requests: two accepted and matching, one accepted by a code check
    (not compared), one not accepted. The last has an unknown shadow cost."""
    store = ShadowStore(ledger)
    call = {"model": "small-a", "cost_usd": 0.1, "error": None}
    accepted = {"accepted": True, "stage": 0, "served_by": "small-a", "answer": "B", "calls": [call]}
    runs = [("s1", accepted, "B"), ("s2", accepted, "C"),
            ("s3", {**accepted, "comparable": False}, "anything"),
            ("s4", {"accepted": False, "stage": None, "served_by": None, "answer": None,
                    "calls": [{"model": "small-a", "cost_usd": None, "error": "timeout"}]}, "B")]
    for request_id, outcome, frontier_answer in runs:
        store.record(request_id=request_id, tenant="acme", provider="openai", policy=POLICY, outcome=outcome,
                     frontier_answer=frontier_answer, frontier_cost_usd=1.0)
    return store


def text_of(page: str) -> str:
    """The page as a reader sees it: tags and the style block removed, entities decoded."""
    body = re.sub(r"<style>.*?</style>", "", page, flags=re.S)
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", body)).split())


def poisoned(value):
    """The same structure with every leaf replaced by a hostile string."""
    if isinstance(value, dict):
        return {key: poisoned(item) for key, item in value.items()}
    if isinstance(value, list):
        return [poisoned(item) for item in value]
    return HOSTILE


def test_money_is_readable_for_tiny_and_large_amounts():
    assert money(0.000127) == "$0.000127" and money(0.0127) == "$0.0127"
    assert money(0.69812) == "$0.698" and money(0.25) == "$0.25" and money(0.3) == "$0.30"
    assert money(0.9999) == "$1.00" and money(1) == "$1.00" and money(1234567.891) == "$1,234,567.89"
    assert money(-0.15) == "-$0.15" and money(0) == "$0.00" and money(0.0) == "$0.00"
    assert money(1e-9) == "under $0.000001" and money(-1e-9) == "above -$0.000001"


def test_unknown_values_are_never_shown_as_zero():
    for unknown in (None, float("nan"), float("inf"), True, "12", HOSTILE, [1]):
        assert money(unknown) == NOT_KNOWN and percent(unknown) == NOT_KNOWN and count(unknown) == NOT_KNOWN
    assert count(1.5) == NOT_KNOWN and count(1414) == "1,414" and percent(0.5346) == "53.5%"
    assert percent(-0.00001) == "0.0%"
    # JSON allows an integer of any length. It must not crash the page.
    for huge in (10**400, -(10**400)):
        assert money(huge) == NOT_KNOWN and percent(huge) == NOT_KNOWN and count(huge) == NOT_KNOWN
    assert percent(1e300) == NOT_KNOWN


def test_percentages_are_never_rounded_in_the_policys_favour():
    # A saving or a share is rounded down, a loss bound up.
    assert percent(0.5346324414140835, rounding="down") == "53.4%"
    assert percent(0.016151136041472752, rounding="up", digits=2) == "1.62%"
    assert percent(-0.2005, rounding="down") == "-20.1%"  # a loss is not rounded towards zero
    assert percent(0.02, rounding="up", digits=2) == "2.00%" and percent(0.5, rounding="down") == "50.0%"
    # Float noise does not cost a whole step: 1 - 0.1 style results stay where they belong.
    assert percent(0.8999999999999999, rounding="down") == "90.0%"
    assert percent(0.020000000000000004, rounding="up", digits=2) == "2.00%"
    assert percent(-0.00000001, rounding="down") == "-0.1%" and percent(0.0, rounding="down") == "0.0%"
    # The figures of the recorded replay (results/2026-11-16-shadow-replay), as a report holds them.
    replay = {"shadowed_requests": 1414, "policy_would_accept": 1099, "policy_would_accept_share": 1099 / 1414,
              "accepted_matching_frontier": 1041, "accepted_matching_frontier_share": 1041 / 1099,
              "cost_known_requests": 1414, "unknown_cost_requests": 0, "frontier_cost_usd": 0.69812,
              "projected_policy_cost_usd": 0.3248824, "projected_savings_usd": 0.69812 - 0.3248824,
              "projected_savings_fraction": 1 - 0.3248824 / 0.69812}
    assert headline(None, replay) == (
        "Of 1,414 shadowed requests, the policy would have answered 77.7% with cheaper models; its projected cost "
        "is $0.325 against the frontier model's $0.698: a projected saving of $0.373 (53.4%).")
    text = text_of(render(None, replay, tenant="benchmark", generated_at="now",
                          policy={**POLICY, "guarantee": {**POLICY["guarantee"], "bound": 0.016151136041472752,
                                                          "certify_degradation_rate": 0.010518934081346423},
                                  "estimate": {"test_tasks": 1414, "savings": 0.5251849252277546}}))
    assert "same answer as the frontier model 94.7% (1,041 answers)" in text
    assert "Upper bound on the loss, from the certify tasks 1.62%" in text
    assert "Loss observed on those tasks 1.06%" in text and "held-out test tasks was 52.5%" in text
    assert "Savings and shares are rounded down" not in text and "rounded down" in text


def test_headline_says_so_when_there_is_nothing_to_report():
    empty_ledger, empty_shadow = Ledger().report("acme"), ShadowStore(Ledger()).report("acme")
    for ledger_report, shadow_report in ((None, None), (empty_ledger, empty_shadow), ({}, {})):
        sentence = headline(ledger_report, shadow_report)
        assert sentence.startswith("Nothing has been measured yet") and "$" not in sentence and "%" not in sentence
    page = text_of(render(empty_ledger, empty_shadow, tenant="acme", generated_at="now"))
    assert "No request has been shadowed" in page and "Ledger: requests" not in page and "$" not in page
    # A count that cannot be read is not the same as a count of zero.
    unread = text_of(render(None, {"shadowed_requests": None}, tenant="acme", generated_at="now"))
    assert "The number of shadowed requests is not known" in unread and "No request has been shadowed" not in unread


def test_headline_for_shadow_only_leads_with_what_was_shadowed():
    ledger = Ledger()
    report = add_shadow_rows(ledger).report("acme")
    sentence = headline(ledger.report("acme"), report)
    assert sentence.startswith("Of 4 shadowed requests, the policy would have answered 75.0% with cheaper models")
    assert "on the 3 requests whose costs are all known" in sentence
    assert "projected cost is $0.30 against the frontier model's $3.00" in sentence
    assert sentence.endswith("a projected saving of $2.70 (90.0%).")


def test_headline_does_not_project_a_saving_from_unknown_costs():
    report = {"shadowed_requests": 10, "policy_would_accept": 4, "policy_would_accept_share": 0.4,
              "frontier_cost_usd": None, "projected_policy_cost_usd": None, "projected_savings_usd": None}
    sentence = headline(None, report)
    assert sentence == ("Of 10 shadowed requests, the policy would have answered 40.0% with cheaper models; "
                        "their costs are not known, so no saving is projected.")


def test_headline_for_live_routing_leads_with_the_measured_ledger():
    ledger = filled_ledger()
    sentence = headline(ledger.report("acme"), add_shadow_rows(ledger).report("acme"))
    # The saving is a subtotal, so the sentence says how many recorded requests it covers.
    assert sentence == ("On 2 of 3 recorded requests, those with a confirmed successful outcome and known costs, "
                        "answers and routing together cost $1.00 against a baseline of $2.00: "
                        "a net saving of $1.00 (50.0%).")
    whole = Ledger()
    whole.record(tenant="acme", provider="openai", model="m", request_body={}, success=True, cost_usd=0.25,
                 routing_cost_usd=0.05, baseline_cost_usd=1.0)
    assert headline(whole.report("acme"), None).startswith(
        "On 1 request with a confirmed successful outcome and known costs, answers")


def test_a_saving_on_escalated_requests_only_shows_what_it_leaves_out():
    """With no baseline estimate, only escalated requests are comparable, and they cost
    more than the baseline. The page must show how many successes are left out."""
    ledger = Ledger()
    for index in range(3):  # answered by a cheaper model: the frontier baseline was not observed
        ledger.record(tenant="acme", provider="openai", model="small-a", request_body={"q": index}, success=True,
                      cost_usd=0.1, routing_cost_usd=0.05)
    ledger.record(tenant="acme", provider="openai", model="big", request_body={"q": "hard"}, success=True,
                  cost_usd=1.0, routing_cost_usd=0.15, baseline_cost_usd=1.0)
    report = ledger.report("acme")
    assert headline_sentences(report, None) == [
        "The measured saving covers only 1 of 4 confirmed successful requests, those whose frontier cost was "
        "measured.",
        "On them, answers and routing together cost $1.15 against a baseline of $1.00: $0.15 more than the "
        "baseline, so there is no net saving.",
        "If the proxy routes through a policy, the frontier model's cost is measured only for requests that "
        "escalated to it, unless a baseline is supplied; the requests cheaper models answered are then the ones "
        "left out, and this figure is not the policy's saving.",
        "No estimated or projected figure is available for the requests left out: supply measured baselines "
        "through the outcome endpoint, or run shadow mode."]
    page = render(report, None, tenant="acme", generated_at="now")
    # The title line states the scope, not the loss.
    assert "<h1>The measured saving covers only 1 of 4 confirmed successful requests" in page
    assert "$" not in page[page.index("<h1>"):page.index("</h1>")]
    text = text_of(page)
    assert "Confirmed successful but in neither figure, because a cost or the baseline is not known 3" in text
    assert "The measured saving covers 1 of 4 confirmed successes." in text
    unknown = text_of(render({**report, "verified_successes": None}, None, tenant="acme", generated_at="now"))
    assert "because a cost or the baseline is not known not known" in unknown
    assert "The measured saving covers only" not in unknown  # nothing is claimed about a count not known


def test_live_routing_keeps_estimated_baselines_out_of_the_measured_saving():
    """With live routing, the measured saving covers only the escalated request. The
    headline must say so and point to the estimated and projected figures, each
    labelled as what it is, and the estimated saving must never join the measured one."""
    ledger = routed_ledger()
    report = ledger.report("acme")
    assert (report["comparable_verified_requests"], report["estimated_baseline_requests"]) == (1, 3)
    shadow = add_shadow_rows(ledger).report("acme")
    assert headline_sentences(report, shadow) == [
        "The measured saving covers only 1 of 6 confirmed successful requests, those whose frontier cost was "
        "measured.",
        "On them, answers and routing together cost $1.15 against a baseline of $1.00: $0.15 more than the "
        "baseline, so there is no net saving.",
        "Under live routing the frontier model's cost is measured only for requests that escalated to it, unless "
        "a baseline is supplied, so this figure leaves out the requests cheaper models answered and is not the "
        "policy's saving.",
        "Estimated, not measured: on 3 confirmed successful requests that cheaper models answered, with the "
        "frontier model's cost estimated from each answer's token counts, the result is an estimated saving of "
        "$2.55.",
        "Projected, not measured: on 3 shadowed requests with all costs known, shadow mode puts the policy at a "
        "projected saving of $2.70 (90.0%)."]
    page = render(report, shadow, tenant="acme", generated_at="now")
    title = page[page.index("<h1>"):page.index("</h1>")]
    assert "$" not in title and "covers only 1 of 6" in title
    assert page.index("</h1>") < page.index('class="lead"') < page.index("</header>")
    text = text_of(page)
    for expected in (
        "Requests recorded 8", "Confirmed successful 6", "Confirmed failed 1", "Outcome still pending 1",
        "Requests with a cost that is not known 1 (answer cost missing on 0, routing cost missing on 1)",
        "Requests the measured saving is computed on 1",
        "Baseline: the frontier model's measured cost $1.00",
        "Net saving, measured -$0.15 (-15.0%)",
        "Estimated, not measured Confirmed successful, with the frontier model's cost estimated 3",
        "Net saving on those, estimated $2.55 (estimated, not measured)",
        "Confirmed successful but in neither figure, because a cost or the baseline is not known 2",
        "Known answer spend, all recorded requests $1.70",
        "Known routing spend, all recorded requests $0.40",
        "Known spend, answers and routing together $2.10",
        "Known spend per confirmed success not known",
        "The measured saving covers 1 of 6 confirmed successes.",
        "On the requests the measured saving covers, routing cost more than the baseline.",
        report["estimated_baseline_scope"],
    ):
        assert expected in text, expected
    assert "$2.55" not in text[text.index("Measured against the frontier model"):text.index("Estimated, not measured C")]


def test_estimated_figure_is_named_when_nothing_is_measured():
    ledger = Ledger()
    for index in range(2):
        ledger.record(tenant="acme", provider="openai", model="small-a", request_body={"q": index}, success=True,
                      cost_usd=0.1, routing_cost_usd=0.05, baseline_cost_usd=1.0, baseline_estimated=True)
    report = ledger.report("acme")
    assert headline_sentences(report, None) == [
        "Of 2 recorded requests, none is a confirmed success with known costs and a measured frontier cost, so no "
        "measured saving can be stated yet.",
        "Estimated, not measured: on 2 confirmed successful requests that cheaper models answered, with the "
        "frontier model's cost estimated from each answer's token counts, the result is an estimated saving of "
        "$1.70."]
    text = text_of(render(report, None, tenant="acme", generated_at="now"))
    assert "Net saving, measured not known" in text and "Net saving on those, estimated $1.70 (estimated" in text
    # With a shadow projection the projection leads, and the estimate follows it, labelled.
    shadow = add_shadow_rows(ledger).report("acme")
    sentences = headline_sentences(report, shadow)
    assert sentences[0].startswith("Of 4 shadowed requests") and sentences[1].startswith("Estimated, not measured")
    # An estimate that costs more is called what it is; one that is not known says so.
    loss = {**report, "estimated_baseline_savings_usd": -0.2}
    assert headline(loss, None).endswith(
        "the result is $0.20 more than the estimated baseline, so there is no estimated saving.")
    unknown = {**report, "estimated_baseline_savings_usd": None}
    assert headline(unknown, None).endswith("token counts, the estimated saving is not known.")
    assert "Net saving on those, estimated not known" in text_of(render(unknown, None, tenant="acme",
                                                                        generated_at="now"))


def test_shadow_sampling_state_is_shown_only_when_the_report_has_it():
    report = {**add_shadow_rows(Ledger()).report("acme"), "sampling": "stopped: the spending cap is reached"}
    assert "Shadow sampling when the report was read stopped: the spending cap is reached" in text_of(
        render(None, report, tenant="acme", generated_at="now"))
    assert "Shadow sampling when the report was read not known" in text_of(
        render(None, {**report, "sampling": None}, tenant="acme", generated_at="now"))
    report.pop("sampling")
    assert "Shadow sampling" not in render(None, report, tenant="acme", generated_at="now")
    # Shadow spend with every cost known is shown as it is, without "at least".
    known = {**report, "unknown_cost_shadow_calls": 0}
    assert "on calls with a known cost $0.30 Shadow calls and checks whose cost is not known 0" in text_of(
        render(None, known, tenant="acme", generated_at="now"))


def test_headline_calls_a_loss_a_loss():
    ledger = Ledger()
    ledger.record(tenant="acme", provider="openai", model="m", request_body={}, success=True, cost_usd=0.25,
                  routing_cost_usd=0.9, baseline_cost_usd=1.0)
    sentence = headline(ledger.report("acme"), None)
    assert sentence.endswith("$0.15 more than the baseline, so there is no net saving.")
    assert "On these requests, routing cost more than the baseline" in render(ledger.report("acme"), None, tenant="acme", generated_at="now")
    shadow = {"shadowed_requests": 2, "policy_would_accept": 0, "policy_would_accept_share": 0.0,
              "unknown_cost_requests": 0, "cost_known_requests": 2, "frontier_cost_usd": 1.0,
              "projected_policy_cost_usd": 1.2, "projected_savings_usd": -0.2, "projected_savings_fraction": -0.2}
    assert headline(None, shadow).endswith("$0.20 more than the frontier model, so there is no projected saving.")


def test_a_ledger_with_only_pending_requests_states_no_saving():
    ledger = Ledger()
    ledger.record(tenant="acme", provider="openai", model="big", request_body={}, cost_usd=1.0)
    report = ledger.report("acme")
    assert headline(report, None).startswith("Of 1 recorded request, none has a confirmed successful outcome")
    page = text_of(render(report, None, tenant="acme", generated_at="now"))
    assert "Net saving, measured not known" in page and "Until one does, no measured saving is stated." in page
    assert "Outcome still pending 1" in page and "Net saving, measured $" not in page


def test_round_trip_from_a_ledger_with_shadow_rows_to_the_page():
    ledger = filled_ledger()
    shadow_report = add_shadow_rows(ledger).report("acme")
    when = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
    page = render(ledger.report("acme"), shadow_report, tenant="acme", generated_at=when, policy=POLICY)
    text = text_of(page)
    for expected in (
        "Tenant acme . Generated 2026-11-16T12:00:00+00:00.",
        "Requests shadowed 4",
        "Policy would have answered with cheaper models 3 (75.0%)",
        "Of those, same answer as the frontier model 50.0% (1 of 2 compared)",
        "Accepted by a code check, not compared 1",
        "Requests with all costs known 3",
        "Requests left out because a cost is not known 1",
        "Frontier model cost, on requests with known costs $3.00",
        "Projected policy cost, on the same requests $0.30",
        "Projected saving $2.70 (90.0%)",
        "What shadow mode itself cost, on calls with a known cost at least $0.30 (1 call or check of unknown "
        "cost not included)",
        "Shadow calls and checks whose cost is not known 1",
        "Shadow calls made 4",
        "Shadow calls that failed 1",
        "Requests recorded 3",
        "Confirmed successful 2",
        "Outcome still pending 1",
        "Requests with a cost that is not known 1 (answer cost missing on 0, routing cost missing on 1)",
        "Verified answer cost $0.75",
        "Routing cost (second opinions, escalated attempts, checks) $0.25",
        "Requests the measured saving is computed on 2",
        "Baseline: the frontier model's measured cost $2.00",
        "Net saving, measured $1.00 (50.0%)",
        "Confirmed successful, with the frontier model's cost estimated 0",
        "Net saving on those, estimated no such requests",
        "Confirmed successful but in neither figure, because a cost or the baseline is not known 0",
        "Known answer spend, all recorded requests $1.75",
        "Known routing spend, all recorded requests $0.25",
        "Known spend, answers and routing together $2.00",
        "Known spend per confirmed success not known",
    ):
        assert expected in text, expected
    # Sections come in the promised order, after the headline.
    order = [page.index(mark) for mark in ("<h1>", "Shadow mode: what", "Ledger: requests", "The policy in plain",
                                           "What this does and does not show")]
    assert order == sorted(order)


def test_policy_is_described_in_plain_words():
    text = text_of(render(None, None, tenant="acme", generated_at="now", policy=POLICY))
    assert "Ask small-a and small-b . Accept the answer only if all 2 give the same one." in text
    assert "Ask mid . Accept the answer only if the check named tests+judge:sonnet passes." in text
    assert "If no stage accepted an answer, send the request to big ." in text
    assert "Quality limit 2.0% (epsilon = 0.02)" in text and "Confidence 95.0% (delta = 0.05)" in text
    assert "Upper bound on the loss, from the certify tasks 1.62% " in text
    assert "Loss observed on those tasks 1.05% " in text
    assert "This describes the policy file supplied when the page was made." in text
    # The bound is an upper limit, so it is rounded up: 1.611% must not print as 1.61%.
    tight = {**POLICY, "guarantee": {**POLICY["guarantee"], "bound": 0.016111}}
    assert "from the certify tasks 1.62% " in text_of(render(None, None, tenant="acme", generated_at="now", policy=tight))
    assert "Graded tasks the bound rests on 1,426" in text and "Certified at this limit yes" in text
    empty = text_of(render(None, None, tenant="acme", generated_at="now",
                           policy={"version": 1, "frontier": "big", "stages": [], "guarantee": {"certified": False}}))
    assert "Every request goes to big ." in empty and "Certified at this limit no" in empty
    assert "Quality limit not known" in empty


def test_every_caveat_from_the_scope_texts_is_carried():
    ledger = filled_ledger()
    ledger_report, shadow_report = ledger.report("acme"), add_shadow_rows(ledger).report("acme")
    text = text_of(render(ledger_report, shadow_report, tenant="acme", generated_at="now", policy=POLICY))
    for scope in (SHADOW_SCOPE, ledger_report["scope"], ledger_report["cost_scope"], ledger_report["savings_scope"],
                  ledger_report["estimated_baseline_scope"], POLICY["estimate"]["scope"]):
        assert scope in text
    for caveat in ("Equal answers are not correctness.", "They cover only requests whose costs are all known.",
                   "not a billing amount", "The measured saving uses measured baselines only.",
                   "shown on its own row as estimated, not measured",
                   "Under live routing the measured saving leans to the hardest requests.", "Shadow mode is not free.",
                   "A missing cost or outcome is never counted as zero.", "Only some requests are shadowed.",
                   "The figures describe the shadowed requests, not all traffic.", "rounded down"):
        assert caveat in text
    assert text.index("What this does and does not show") < text.index("Equal answers are not correctness.")
    assert "The baseline may be an estimate" not in text and "does not say how many baselines" not in text


def test_code_stages_are_shown_as_not_compared():
    report = {"shadowed_requests": 3, "policy_would_accept": 2, "policy_would_accept_share": 2 / 3,
              "accepted_compared_with_frontier": 0, "accepted_matching_frontier": 0,
              "accepted_matching_frontier_share": None}
    text = text_of(render(None, report, tenant="acme", generated_at="now"))
    assert "Of those, same answer as the frontier model not compared" in text
    assert "Accepted by a code check, not compared 2" in text


def test_nulls_and_missing_keys_render_as_not_known_never_zero():
    shadow = {"shadowed_requests": 5, "policy_would_accept": 2, "policy_would_accept_share": 0.4}
    ledger = {"requests": 5}
    text = text_of(render(ledger, shadow, tenant="acme", generated_at="now"))
    for line in ("Frontier model cost, on requests with known costs not known", "Projected saving not known",
                 "What shadow mode itself cost, on calls with a known cost not known", "Shadow calls made not known",
                 "Of those, same answer as the frontier model not known", "Net saving, measured not known",
                 "Verified answer cost not known", "Outcome still pending not known",
                 "Requests with a cost that is not known not known",
                 "Shadow calls and checks whose cost is not known not known",
                 "Confirmed successful, with the frontier model's cost estimated not known",
                 "Net saving on those, estimated not known", "Known spend per confirmed success not known",
                 "Known answer spend, all recorded requests not known",
                 "This ledger report does not say how many baselines were estimated."):
        assert line in text, line
    assert "$" not in text and " 0 " not in text and not re.search(r"(?<![\d.])0\.0%", text)
    nulled = dict.fromkeys(Ledger().report("acme"), None) | {"requests": 1}
    assert "$" not in text_of(render(nulled, dict.fromkeys(shadow, None), tenant="acme", generated_at="now"))


def test_shadow_spend_is_labelled_as_covering_known_costs_only():
    """ShadowStore.report sums the calls with a known cost, so it gives 0 when no call
    has one. The page must not present that as shadow mode having cost nothing."""
    store = ShadowStore(Ledger())
    unpriced = {"accepted": True, "stage": 0, "served_by": "small-a", "answer": "B",
                "calls": [{"model": "small-a", "cost_usd": None, "error": None}]}
    store.record(request_id="s1", tenant="acme", provider="openai", policy=POLICY, outcome=unpriced,
                 frontier_answer="B", frontier_cost_usd=1.0)
    report = store.report("acme")
    assert report["shadow_spend_usd"] == 0.0 and report["cost_known_requests"] == 0
    text = text_of(render(None, report, tenant="acme", generated_at="now"))
    assert ("What shadow mode itself cost, on calls with a known cost at least $0.00 (1 call or check of unknown "
            "cost not included)") in text
    assert "Shadow calls and checks whose cost is not known 1" in text
    assert "It sums only calls whose cost is known, so it is too low, and can read as zero" in text
    assert headline(None, report) == ("Of 1 shadowed request, the policy would have answered 100.0% with cheaper "
                                      "models; their costs are not known, so no saving is projected.")


def test_headline_reads_well_when_a_count_is_missing():
    shadow = {"shadowed_requests": 2, "frontier_cost_usd": 1.0, "projected_policy_cost_usd": 0.5,
              "projected_savings_usd": 0.5}
    assert headline(None, shadow) == (
        "Of 2 shadowed requests, the policy would have answered a share that is not known with cheaper models; "
        "its projected cost is $0.50 against the frontier model's $1.00 on the requests whose costs are all "
        "known: a projected saving of $0.50.")


def test_note_is_shown_under_the_headline_and_escaped():
    page = render(None, None, tenant="acme", generated_at="now", note="A replay of benchmark answers. " + HOSTILE)
    assert 'class="note"' in page and "A replay of benchmark answers." in page and HOSTILE not in page
    assert page.index("</h1>") < page.index('class="note"') < page.index("</header>")
    for absent in (None, "", "   ", 5):
        assert 'class="note"' not in render(None, None, tenant="acme", generated_at="now", note=absent)


def test_hostile_strings_in_every_data_field_are_escaped():
    ledger = filled_ledger()
    ledger_report = ledger.report("acme")
    shadow_report = {**add_shadow_rows(ledger).report("acme"), "sampling": "on"}
    pages = [render(poisoned(ledger_report), poisoned(shadow_report), tenant=HOSTILE, generated_at=HOSTILE,
                    policy=poisoned(POLICY), note=HOSTILE)]
    # One field at a time, so a field that is only reached when the rest is well formed is covered too. The
    # routed ledger reaches the estimated-baseline and partial-measurement wording.
    for full in (ledger_report, routed_ledger().report("acme")):
        assert {"estimated_baseline_requests", "estimated_baseline_savings_usd", "estimated_baseline_scope",
                "known_cost_per_verified_success_usd"} <= set(full)
        for name, report in (("ledger", full), ("shadow", shadow_report)):
            for key in report:
                reports = {"ledger": full, "shadow": shadow_report} | {name: {**report, key: HOSTILE}}
                pages.append(render(reports["ledger"], reports["shadow"], tenant="acme", generated_at="now",
                                    policy=POLICY))
    assert "unknown_cost_shadow_calls" in shadow_report
    for key in ("estimated_baseline_scope", "sampling"):
        page = render(routed_ledger().report("acme") | {key: HOSTILE}, shadow_report | {key: HOSTILE},
                      tenant="acme", generated_at="now")
        assert html.escape(HOSTILE) in page
        pages.append(page)
    for path in (("tenant",), ("task_type",), ("frontier",), ("stages", 0, "models", 0), ("stages", 1, "check"),
                 ("guarantee", "epsilon"), ("guarantee", "delta"), ("guarantee", "bound"),
                 ("guarantee", "certify_tasks"), ("guarantee", "statement"), ("guarantee", "certified"),
                 ("estimate", "scope"), ("estimate", "savings"), ("estimate", "test_tasks")):
        policy = json.loads(json.dumps(POLICY))
        target = policy
        for step in path[:-1]:
            target = target[step]
        target[path[-1]] = HOSTILE
        pages.append(render(ledger_report, shadow_report, tenant="acme", generated_at="now", policy=policy))
    for page in pages:
        assert "<script" not in page.lower() and "<img" not in page.lower() and "onerror=1>" not in page
        assert HOSTILE not in page
    assert html.escape(HOSTILE) in pages[0]
    assert pages[0].count(html.escape(HOSTILE)) >= 15  # tenant (twice), time, note, sampling, scopes, policy names


def test_page_is_self_contained():
    ledger = filled_ledger()
    page = render(ledger.report("acme"), add_shadow_rows(ledger).report("acme"), tenant="acme",
                  generated_at="now", policy=POLICY)
    lowered = page.lower()
    for forbidden in ("http", "<script", "<link", "<img", "<iframe", "<object", "<embed", "<form", "@import",
                      "url(", " src=", " href=", "javascript:", " onload", " onclick"):
        assert forbidden not in lowered, forbidden
    assert page.startswith("<!DOCTYPE html>") and page.rstrip().endswith("</html>")
    assert "prefers-color-scheme:dark" in page and 'name="viewport"' in page
    assert page.count("<style>") == 1 and "<title>Savings report: acme</title>" in page


def test_cost_bars_are_scaled_and_dropped_when_a_cost_is_unknown():
    shadow = {"shadowed_requests": 1, "frontier_cost_usd": 2.0, "projected_policy_cost_usd": 0.5}
    page = render(None, shadow, tenant="acme", generated_at="now")
    assert 'style="width:100.0%"' in page and 'style="width:25.0%"' in page
    assert 'class="bars"' not in render(None, {**shadow, "projected_policy_cost_usd": None}, tenant="acme",
                                        generated_at="now")
    assert 'class="bars"' not in render(None, {**shadow, "frontier_cost_usd": HOSTILE}, tenant="acme",
                                        generated_at="now")


def test_cli_reads_a_ledger_file_read_only_and_writes_the_page(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    ledger = Ledger(path)
    ledger.record(tenant="acme", provider="openai", model="small-a", request_body={}, success=True, cost_usd=0.25,
                  routing_cost_usd=0.05, baseline_cost_usd=1.0)
    ledger.record(tenant="other", provider="openai", model="big", request_body={}, success=True, cost_usd=9.0,
                  routing_cost_usd=0.0, baseline_cost_usd=9.0)
    add_shadow_rows(ledger)
    ledger.close()
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(POLICY), encoding="utf-8")
    before = path.read_bytes()
    output = tmp_path / "out" / "report.html"
    cli.main(["--ledger", str(path), "--tenant", "acme", "--policy", str(policy_path), "--output", str(output),
              "--generated-at", "2026-11-16"])
    text = text_of(output.read_text(encoding="utf-8"))
    assert "a net saving of $0.70 (70.0%)" in text and "Requests shadowed 4" in text
    assert "Generated 2026-11-16." in text and "The policy in plain words" in text
    assert "$9.00" not in text  # the other tenant's requests stay out
    assert "Note." not in text  # the policy given is the one the shadow rows were run with
    assert path.read_bytes() == before
    with pytest.raises(Exception):
        cli.open_read_only(path)._db.execute("DELETE FROM requests")


def test_cli_says_so_when_shadow_rows_were_run_with_another_policy(tmp_path, capsys):
    path = tmp_path / "ledger.sqlite3"
    ledger = Ledger(path)
    add_shadow_rows(ledger)
    ledger.close()
    newer = tmp_path / "policy.json"
    newer.write_text(json.dumps({**POLICY, "stages": POLICY["stages"][:1]}), encoding="utf-8")
    text = text_of(cli.build(tenant="acme", ledger=path, policy=newer, note="Made for a test."))
    assert ("Note. Made for a test. 4 of the 4 shadowed requests were run with a different policy file from the "
            "one described on this page.") in text
    assert "warning: 4 of the 4 shadowed requests" in capsys.readouterr().err
    same = tmp_path / "same.json"
    same.write_text(json.dumps(POLICY, indent=2), encoding="utf-8")  # layout of the file does not matter
    assert "Note." not in text_of(cli.build(tenant="acme", ledger=path, policy=same))
    assert "Note." not in text_of(cli.build(tenant="nobody", ledger=path))


def test_cli_leaves_out_the_shadow_report_when_the_ledger_has_no_shadow_table(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    Ledger(path).close()
    ledger_report, shadow_report = cli.reports_from_ledger(path, "acme")
    assert shadow_report is None and ledger_report["requests"] == 0
    assert "Shadow mode" not in text_of(cli.build(tenant="acme", ledger=path)).replace("Shadow mode is not", "")


def test_cli_builds_from_a_saved_report_and_refuses_mismatches(tmp_path, capsys):
    saved = tmp_path / "replay.json"
    saved.write_text(json.dumps({"shadow_report": {"tenant": "acme", "shadowed_requests": 2, "policy_would_accept": 1,
                                                   "policy_would_accept_share": 0.5}}), encoding="utf-8")
    assert "Of 2 shadowed requests" in cli.build(tenant="acme", report_json=saved)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(POLICY), encoding="utf-8")
    output = tmp_path / "report.html"
    for arguments in (["--report-json", str(saved), "--tenant", "someone-else"],
                      ["--report-json", str(policy_path), "--tenant", "acme"],
                      ["--ledger", str(tmp_path / "missing.sqlite3"), "--tenant", "acme"]):
        with pytest.raises(SystemExit) as stopped:
            cli.main([*arguments, "--output", str(output)])
        assert "savings_report:" in str(stopped.value)
    saved.write_text(json.dumps({"shadow_report": {"shadowed_requests": 2}}), encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.main(["--report-json", str(saved), "--tenant", "zed", "--policy", str(policy_path), "--output", str(output)])
    # A report that is not an object, and an output path that cannot be written, end cleanly too.
    saved.write_text(json.dumps({"shadow_report": [1], "ledger_report": "x"}), encoding="utf-8")
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--report-json", str(saved), "--tenant", "acme", "--output", str(output)])
    assert "is not a JSON object" in str(stopped.value)
    saved.write_text(json.dumps({"shadow_report": {"shadowed_requests": 2}}), encoding="utf-8")
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--report-json", str(saved), "--tenant", "acme", "--output", str(tmp_path)])
    assert "savings_report:" in str(stopped.value)
    assert not output.exists()
    # An integer too long for a float is shown as not known, not a traceback.
    saved.write_text('{"shadow_report": {"shadowed_requests": 3, "frontier_cost_usd": 1' + "0" * 400 + "}}",
                     encoding="utf-8")
    cli.main(["--report-json", str(saved), "--tenant", "acme", "--note", "A replay.", "--output", str(output)])
    text = text_of(output.read_text(encoding="utf-8"))
    assert "their costs are not known, so no saving is projected" in text and "Note. A replay." in text
