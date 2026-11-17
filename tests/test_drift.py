"""The drift check of shadow mode (``tine.shadow.drift``).

The check compares the share of shadowed requests whose accepted cheaper answer
differs from the frontier's with what the policy's certification allows for: the
quality limit plus the share that differed without a loss when the policy was frozen.

The counts below are those of recorded shadow runs and of the graded histories their
policies were frozen from (``results/2026-11-17-live-pilot``,
``results/2026-11-17-live-pilot-more``, ``results/2026-11-16-shadow-replay``): counts
only, no prompt and no answer. In every one of these runs the graded loss was at or
under the policy's limit, so none of them may warn.
"""

import math

import numpy as np
import pytest

from tine.ledger import Ledger
from tine.shadow import (
    DRIFT_BLOCK, DRIFT_EVIDENCE_NEEDED, ShadowStore, drift, drift_allowance, drift_evidence, drift_scopes,
    policy_row_digest,
)

# Run, epsilon, the policy's baseline (differing without a loss, graded tasks), then the
# run's differing answers, compared answers and drift requests, and the evidence.
HEALTHY = [
    ("MMLU-Pro first pilot", 0.04, (7, 692), (9, 268, 317), 0.0181),
    ("long-1", 0.04, (7, 692), (9, 273, 300), 0.0199),
    ("long-2", 0.04, (7, 692), (17, 283, 300), 0.0566),
    ("long-3", 0.04, (7, 692), (11, 277, 300), 0.0243),
    ("long-4", 0.04, (7, 692), (14, 281, 300), 0.0350),
    ("long-5", 0.04, (7, 692), (7, 276, 300), 0.0167),
    ("MedQA", 0.04, (7, 556), (1, 191, 200), 0.0170),
    ("ARC", 0.04, (2, 556), (7, 200, 200), 0.0382),
    ("FinQA exact", 0.04, (70, 554), (25, 102, 200), 0.0361),
    ("FinQA by value", 0.04, (83, 554), (29, 118, 200), 0.0369),
    ("MATH-500 exact", 0.04, (10, 227), (5, 85, 100), 0.0492),
    ("MATH-500 by value", 0.04, (0, 227), (2, 97, 100), 0.0460),
    ("Tickets", 0.04, (19, 692), (32, 300, 300), 0.541),
]
# Runs of more than one block of requests. The evidence then depends on the order.
LONG = [
    ("Long pooled", 0.04, (7, 692), (58, 1390, 1500), 0.0145),
    ("All pooled", 0.04, (7, 692), (67, 1658, 1817), 0.0096),
    ("xRouteBench replay", 0.02, (76, 2840), (58, 1099, 1414), 0.0425),
]
MMLU = (7, 692)


def baseline(harmless: int, tasks: int) -> dict:
    return {"tasks": tasks, "differing": harmless, "differing_not_lost": harmless}


def check(differing, compared, requests, epsilon, recorded=None, **options) -> dict:
    return drift(compared, differing, epsilon, requests=requests,
                 baseline=baseline(*recorded) if recorded else None, **options)


def spread(total: int, ones: int) -> list[bool]:
    """True at ``ones`` evenly spaced positions of ``total``."""
    return [(index + 1) * ones // total > index * ones // total for index in range(total)]


def spread_rows(differing: int, compared: int, requests: int) -> list[tuple[bool, bool]]:
    """Per drift request, in order: whether its answer differs and whether it was
    compared, with the differing ones and the ones sent to the frontier spread evenly."""
    differs = spread(requests, differing)
    matching = iter(spread(requests - differing, compared - differing))
    return [(flag, True if flag else next(matching)) for flag in differs]


def check_rows(rows, epsilon, recorded=None) -> dict:
    return drift(sum(compared for _, compared in rows), sum(differs for differs, _ in rows), epsilon,
                 requests=len(rows), baseline=baseline(*recorded) if recorded else None, latest=drift_scopes(rows))


@pytest.mark.parametrize("name,epsilon,recorded,counts,evidence", HEALTHY)
def test_healthy_recorded_runs_are_compatible_with_their_certification(name, epsilon, recorded, counts, evidence):
    report = check(*counts, epsilon, recorded)
    assert report["drift_check"].startswith("compatible with the certification: ") and report["drift_warning"] is None
    assert report["drift_basis"] == "baseline" and report["drift_evidence"] == pytest.approx(evidence, rel=0.01)
    assert report["drift_scope"] == {"requests": counts[2], "compared": counts[1], "differing": counts[0],
                                     "latest": False}


@pytest.mark.parametrize("name,epsilon,recorded,counts,evidence", LONG)
def test_healthy_long_runs_are_compatible_with_their_certification(name, epsilon, recorded, counts, evidence):
    report = check_rows(spread_rows(*counts), epsilon, recorded)
    assert report["drift_check"].startswith("compatible with the certification: ") and report["drift_evidence"] < 1
    assert report["drift_evidence"] == pytest.approx(evidence, rel=0.01)
    assert (report["accepted_differing_from_frontier"], report["drift_requests"]) == (counts[0], counts[2])


def test_the_same_runs_under_a_policy_without_a_baseline_get_a_rough_check_and_no_warning():
    evidence = {}
    for name, epsilon, _, counts, _ in HEALTHY:
        report = check(*counts, epsilon)
        assert report["drift_check"].startswith("rough check, no warning: ") and report["drift_warning"] is None, name
        assert (report["drift_basis"], report["drift_baseline"]) == ("default", None)
        assert report["drift_allowance"] == pytest.approx(epsilon + 0.10)
        evidence[name] = report["drift_evidence"]
    for name, epsilon, _, counts, _ in LONG:
        report = check_rows(spread_rows(*counts), epsilon)
        assert report["drift_check"].startswith("rough check, no warning: "), name
        evidence[name] = report["drift_evidence"]
    assert max(evidence, key=evidence.get) == "FinQA by value"
    assert evidence["FinQA by value"] == pytest.approx(0.133, rel=0.01)


def test_the_mmlu_pro_chain_is_compatible_at_the_two_percent_limit_too():
    """The chain certified at 4% is also the one certified at 2%. There the allowance
    is 3.89%, and the plain comparison with 2% warned on most of these runs."""
    assert drift_allowance(0.02, baseline(*MMLU))["drift_allowance"] == pytest.approx(0.0389, abs=5e-5)
    for name, _, recorded, counts, _ in HEALTHY:
        if recorded == MMLU:
            report = check(*counts, 0.02, recorded)
            assert report["drift_check"].startswith("compatible with the certification: "), name
            assert counts[0] / counts[2] > 0.02 and report["drift_evidence"] < 1  # above the limit, and no warning
            if name == "long-2":
                assert report["drift_evidence"] == pytest.approx(0.576, rel=0.01)
    for name, _, recorded, counts, _ in LONG:
        if recorded == MMLU:
            report = check_rows(spread_rows(*counts), 0.02, recorded)
            assert report["drift_check"].startswith("compatible with the certification: "), name


@pytest.mark.parametrize("epsilon,recorded", [
    (0.04, (7, 692)), (0.04, (7, 556)), (0.04, (2, 556)), (0.04, (10, 227)), (0.04, (0, 227)), (0.04, (19, 692)),
    (0.02, (76, 2840)), (0.04, None), (0.02, None)])
def test_cheaper_models_agreeing_on_wrong_answers_still_warn(epsilon, recorded):
    """The case the check is for: cheaper models agree with each other on wrong
    answers for about 40% of tasks. 13 of the first 30 accepted answers differ."""
    report = check(13, 30, 30, epsilon, recorded)
    assert report["drift_check"] == "warning" and report["drift_evidence"] >= DRIFT_EVIDENCE_NEEDED
    assert report["drift_warning"].startswith(
        "Cannot confirm the certification on this traffic: 13 of 30 accepted cheaper answers differ from the "
        "frontier model's answer, 43.33% of the 30 shadowed requests whose answer was compared with the frontier's "
        "or sent to the frontier. That is clearly more than ")
    if recorded is None:
        assert report["drift_evidence"] == pytest.approx({0.04: 132.9, 0.02: 630.0}[epsilon], rel=0.01)


def test_where_harmless_differences_are_common_only_a_larger_change_shows():
    """A quarter of FinQA's accepted answers already differ from the frontier's, mostly
    by rounding, so the allowance is wide and wrong agreement needs more to show."""
    exact, by_value = (70, 554), (83, 554)
    assert drift_allowance(0.04, baseline(*exact))["drift_allowance"] == pytest.approx(0.19197, abs=1e-5)
    calm = check(13, 30, 30, 0.04, exact)
    assert calm["drift_check"].startswith("compatible") and calm["drift_evidence"] == pytest.approx(8.01, rel=0.01)
    for counts, recorded, evidence in (((14, 30, 30), exact, 24.6), ((15, 30, 30), by_value, 25.2),
                                       ((30, 45, 85), exact, 24.6)):  # 40.5 points more among accepted answers
        report = check(*counts, 0.04, recorded)
        assert report["drift_check"] == "warning" and report["drift_evidence"] == pytest.approx(evidence, rel=0.01)


@pytest.mark.parametrize("differing,requests,warns,evidence", [
    (90, 1000, True, 148.8),  # 9.0%: a loss at twice the limit on top of the usual differences
    (86, 1000, True, 25.8), (85, 1000, False, 17.2), (32, 300, True, 22.1), (31, 300, False, 11.85),
    (70, 1000, False, 0.183)])
def test_a_differing_share_from_the_start_warns_once_it_is_clearly_above_the_allowance(
        differing, requests, warns, evidence):
    report = check(differing, requests, requests, 0.04, MMLU)
    assert (report["drift_check"] == "warning") is warns and (report["drift_warning"] is not None) is warns
    assert report["drift_evidence"] == pytest.approx(evidence, rel=0.01)
    assert report["drift_allowance"] == pytest.approx(0.058916, abs=1e-6)
    assert report["drift_baseline"] == {"tasks": 692, "differing_not_lost": 7,
                                        "upper": pytest.approx(0.018916, abs=1e-6)}


def test_the_sentences_say_what_was_compared_and_what_the_level_is_made_of():
    calm = check(9, 268, 317, 0.04, MMLU)
    assert calm["drift_check"] == (
        "compatible with the certification: 9 of 268 accepted answers differ from the frontier's, 2.84% of the 317 "
        "shadowed requests whose answer was compared with the frontier's or sent to the frontier. That is not "
        "clearly above the 5.89% the certification allows for: the quality limit of 4.00% plus 1.89% for answers "
        "that differed without a loss on the 692 graded tasks the policy was certified and tested on")
    assert check(32, 300, 300, 0.04, MMLU)["drift_warning"] == (
        "Cannot confirm the certification on this traffic: 32 of 300 accepted cheaper answers differ from the "
        "frontier model's answer, 10.67% of the 300 shadowed requests whose answer was compared with the frontier's "
        "or sent to the frontier. That is clearly more than the 5.89% the certification allows for: the quality "
        "limit of 4.00% plus 1.89% for answers that differed without a loss on the 692 graded tasks the policy was "
        "certified and tested on. A different answer is not always a wrong one; grade a sample of these requests "
        "before relying on the policy or its projected saving.")
    assert check(13, 30, 30, 0.04)["drift_warning"] == (
        "Cannot confirm the certification on this traffic: 13 of 30 accepted cheaper answers differ from the "
        "frontier model's answer, 43.33% of the 30 shadowed requests whose answer was compared with the frontier's "
        "or sent to the frontier. That is clearly more than 14.00%: the quality limit of 4.00% plus a default "
        "10.00%, used because this policy file records no baseline of its own. A different answer is not always a "
        "wrong one; grade a sample of these requests before relying on the policy or its projected saving.")
    # Too few compared answers, or no limit: the two earlier sentences, unchanged.
    assert check(13, 29, 29, 0.04, MMLU)["drift_check"] == ("not checked yet: 29 accepted answers compared with the "
                                                            "frontier's, 30 needed")
    assert check(13, 30, 30, None, MMLU)["drift_check"] == "not checked: the policy's quality limit was not given"
    assert check(13, 30, 30, None, MMLU)["drift_allowance"] is None


def test_a_change_late_in_a_long_history_shows_in_the_latest_requests():
    """10,500 healthy requests, then 40% of answers differing. All the requests together
    barely move; the requests since the last full block show it."""
    healthy = [(index % 125 < 3, True) for index in range(10_500)]
    before = check_rows(healthy, 0.04, MMLU)
    assert sum(differs for differs, _ in healthy) == 252
    assert before["drift_check"].startswith("compatible") and before["drift_evidence"] == pytest.approx(0.00125, rel=0.01)
    drifted = [(index % 5 < 2, True) for index in range(150)]
    after = check_rows(healthy + drifted, 0.04, MMLU)
    assert after["drift_check"] == "warning" and after["drift_evidence"] == pytest.approx(305.6, rel=0.01)
    assert after["drift_scope"] == {"requests": 650, "compared": 650, "differing": 72, "latest": True}
    assert "72 of 650 accepted cheaper answers differ from the frontier model's answer, 11.08% of the latest 650 " \
           "shadowed requests whose answer was compared with the frontier's or sent to the frontier." in after["drift_warning"]
    # The fields beside the sentence still count every request.
    assert (after["accepted_differing_from_frontier"], after["drift_requests"]) == (312, 10_650)
    # All the requests together say almost nothing: without the latest ones there is no warning.
    alone = check(312, 10_650, 10_650, 0.04, MMLU)
    assert alone["drift_check"].startswith("compatible") and alone["drift_evidence"] == pytest.approx(0.0006, rel=0.05)
    first = next(count for count in range(1, 151)
                 if check_rows(healthy + drifted[:count], 0.04, MMLU)["drift_check"] == "warning")
    assert first == 136


def test_a_warning_goes_away_when_later_requests_no_longer_support_it():
    rows = [(index < 13, True) for index in range(30)]
    assert check_rows(rows, 0.04, MMLU)["drift_check"] == "warning"
    later = check_rows(rows + [(False, True)] * 100, 0.04, MMLU)
    assert later["drift_check"].startswith("compatible") and later["drift_warning"] is None
    assert later["drift_evidence"] == pytest.approx(1.02, rel=0.01)


def test_reading_the_report_after_every_request_keeps_false_alarms_under_five_percent():
    """With each answer differing at exactly the allowed rate, the chance that a warning
    shows at any read in the first 1,000 requests, worked out exactly."""
    allowance = drift_allowance(0.04, baseline(*MMLU))["drift_allowance"]
    least = {}  # per number of requests, the fewest differing answers that warn
    count = 1
    for requests in range(30, DRIFT_BLOCK + 1):
        while check(count, requests, requests, 0.04, MMLU)["drift_check"] != "warning":
            count += 1
        assert count == 1 or check(count - 1, requests, requests, 0.04, MMLU)["drift_check"] != "warning"
        least[requests] = count
    assert [least[requests] for requests in (30, 100, 300, 1000)] == [8, 15, 32, 86]
    alive, warned = np.zeros(least[DRIFT_BLOCK] + 2), 0.0
    alive[0] = 1.0
    for requests in range(1, DRIFT_BLOCK + 1):
        alive[1:] = alive[1:] * (1 - allowance) + alive[:-1] * allowance
        alive[0] *= 1 - allowance
        if requests in least:
            warned += alive[least[requests]:].sum()
            alive[least[requests]:] = 0.0
    assert warned == pytest.approx(0.0089, rel=0.02) and warned <= 1 / DRIFT_EVIDENCE_NEEDED


def test_evidence_weighs_all_the_requests_and_each_later_start_with_weights_that_sum_to_one():
    scopes = [{"requests": 0, "compared": 0, "differing": 0}] * 4
    assert drift_evidence(scopes, 0.05, 0.04, min_compared=0) == (
        pytest.approx(1 / 2 + 1 / 4 + 1 / 12 + 1 / 24),
        {"requests": 0, "compared": 0, "differing": 0, "latest": False})  # no request: each scope gives 1
    assert drift_evidence(scopes, 0.05, 0.04) == (0.0, None)  # a scope needs 30 compared answers to count
    assert drift_evidence(scopes, 1.0, 0.04, min_compared=0)[0] == 0.0  # nothing can be above an allowance of 1
    rows = [(False, True)] * 2500
    assert drift_scopes(rows) == [{"requests": 1500, "compared": 1500, "differing": 0},
                                  {"requests": 500, "compared": 500, "differing": 0}]
    assert drift_scopes(rows[:DRIFT_BLOCK]) == [] and drift_scopes([]) == []
    # Counts far past the level needed are reported at the cap, not as an overflow.
    assert check(1000, 1000, 1000, 0.04, MMLU)["drift_evidence"] == 1e12
    assert math.isfinite(check(0, 100_000, 100_000, 0.04, MMLU)["drift_evidence"])


def accepted_run(answer="A", **changes):
    return {"accepted": True, "stage": 0, "served_by": "small-a", "answer": answer,
            "calls": [{"model": "small-a", "cost_usd": 0.01, "error": None}], **changes}


def policy(epsilon=0.04, recorded=None, frontier="frontier"):
    return {"version": 1, "tenant": "a", "frontier": frontier,
            "guarantee": {"certified": True, "epsilon": epsilon, "delta": 0.05, "bound": epsilon / 2},
            "stages": [{"models": ["small-a", "small-b"], "check": "agreement"}],
            **({"drift_baseline": baseline(*recorded)} if recorded else {})}


def record(store, used, name, differing, matching, provider="openai", **changes):
    for index in range(differing + matching):
        store.record(request_id=f"{name}{index}", tenant="a", provider=provider, policy=used,
                     outcome=accepted_run(**changes), frontier_answer="B" if index < differing else "A",
                     frontier_cost_usd=1.0)


def test_each_policy_is_checked_on_the_requests_run_with_it_against_its_own_limit_and_baseline():
    store = ShadowStore(Ledger())
    current, older = policy(0.04, MMLU), policy(0.04)
    assert policy_row_digest(current) != policy_row_digest(older)
    record(store, older, "old", differing=30, matching=10)  # an earlier policy file drifted badly
    record(store, current, "new", differing=2, matching=38)
    report = store.report("a", policies={"openai": current})
    assert report["shadowed_requests"] == 80 and report["drift_other_policy_requests"] == 40
    assert (report["drift_policy"], report["drift_policies"]) == ("openai", None)
    assert (report["drift_requests"], report["accepted_differing_from_frontier"], report["drift_epsilon"]) == (40, 2, 0.04)
    assert report["drift_basis"] == "baseline" and report["drift_baseline"]["tasks"] == 692
    assert report["drift_check"].startswith("compatible with the certification: 2 of 40 accepted answers")
    # The same rows pooled under one limit, as before policies were told apart, do warn.
    assert store.report("a", epsilon=0.04)["drift_check"] == "warning"
    # A policy file that ran nothing here has nothing to check yet, and says how much it left out.
    unused = store.report("a", policies={"openai": policy(0.02)})
    assert unused["drift_check"].startswith("not checked yet: 0 accepted answers") and unused["drift_requests"] == 0
    assert unused["drift_other_policy_requests"] == 80


def test_two_providers_with_different_policies_are_checked_separately():
    store = ShadowStore(Ledger())
    openai, anthropic = policy(0.04, MMLU), policy(0.02, frontier="claude")
    record(store, openai, "o", differing=1, matching=59)
    record(store, anthropic, "c", differing=13, matching=17, provider="anthropic")
    report = store.report("a", policies={"openai": openai, "anthropic": anthropic})
    # The policy with a warning is the one shown, though the other has more requests.
    assert report["drift_policy"] == "anthropic" and report["drift_check"] == "warning"
    assert (report["drift_epsilon"], report["drift_basis"], report["drift_requests"]) == (0.02, "default", 30)
    assert report["drift_other_policy_requests"] == 0 and set(report["drift_policies"]) == {"openai", "anthropic"}
    calm = report["drift_policies"]["openai"]
    assert calm["drift_check"].startswith("compatible with the certification: 1 of 60") and calm["drift_epsilon"] == 0.04
    assert report["drift_policies"]["anthropic"]["drift_warning"] == report["drift_warning"]
    # Without a warning, the policy with the most requests is shown.
    quiet = ShadowStore(Ledger())
    record(quiet, openai, "o", differing=1, matching=59)
    record(quiet, anthropic, "c", differing=0, matching=30, provider="anthropic")
    assert quiet.report("a", policies={"anthropic": anthropic, "openai": openai})["drift_policy"] == "openai"
    # A policy with no stages shadows nothing and is not checked; with only such policies nothing is.
    empty = {**openai, "stages": [], "guarantee": {"certified": False, "epsilon": 0.04, "delta": 0.05}}
    alone = quiet.report("a", policies={"openai": empty})
    assert alone["drift_check"].startswith("not checked:") and alone["drift_policy"] is None
    assert alone["drift_other_policy_requests"] == 90


def test_answers_accepted_by_a_named_check_and_rate_limited_runs_stay_out_of_the_drift_check():
    from tine.shadow import RATE_LIMITED_REASON

    store = ShadowStore(Ledger())
    used = policy(0.04, MMLU)
    record(store, used, "kept", differing=1, matching=39)
    record(store, used, "code", differing=30, matching=0, comparable=False)  # two right patches rarely match
    for index in range(30):
        store.record(request_id=f"limited{index}", tenant="a", provider="openai", policy=used,
                     outcome=accepted_run(), frontier_answer="B", frontier_cost_usd=1.0,
                     skip_reason=RATE_LIMITED_REASON)
    report = store.report("a", policies={"openai": used})
    assert report["shadowed_requests"] == 70 and report["rate_limited_runs"] == 30
    assert (report["drift_requests"], report["accepted_differing_from_frontier"]) == (40, 1)
    assert report["drift_check"].startswith("compatible") and report["drift_other_policy_requests"] == 0


def test_the_store_weighs_the_latest_requests_in_the_order_they_were_stored():
    store = ShadowStore(Ledger())
    used = policy(0.04, MMLU)
    for index in range(DRIFT_BLOCK + 150):
        late = index >= DRIFT_BLOCK
        differs = (index - DRIFT_BLOCK) % 5 < 2 if late else index % 125 < 3
        store.record(request_id=f"r{index:05d}", tenant="a", provider="openai", policy=used,
                     outcome=accepted_run(), frontier_answer="B" if differs else "A", frontier_cost_usd=1.0)
    report = store.report("a", policies={"openai": used})
    assert report["drift_scope"] == {"requests": 150, "compared": 150, "differing": 60, "latest": True}
    assert "60 of 150 accepted cheaper answers differ from the frontier model's answer, 40.00% of the latest 150 " \
           "shadowed requests" in report["drift_warning"]
    assert (report["drift_requests"], report["accepted_differing_from_frontier"]) == (1150, 84)
