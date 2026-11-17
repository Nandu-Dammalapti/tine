"""The drift check of shadow mode (``tine.shadow.drift``).

The check compares the share of shadowed requests whose accepted cheaper answer
differs from the frontier's with what the policy's certification allows for: the
quality limit plus the share that differed without a loss when the policy was frozen.
It does the same among the accepted answers alone, and it gives a notice when a share
is above the one on the graded tasks without being above the allowance.

The counts below are those of recorded shadow runs and of the graded histories their
policies were frozen from (``results/2026-11-17-live-pilot``,
``results/2026-11-17-live-pilot-more``, ``results/2026-11-16-shadow-replay``,
``results/2026-11-17-drift-check``): counts only, no prompt and no answer. In every
run of ``HEALTHY`` the graded loss was under the policy's limit, so none may warn.
"""

import math

import numpy as np
import pytest

from tine.ledger import Ledger
from tine.shadow import (
    DRIFT_BLOCK, DRIFT_DEFAULT_ALLOWANCE_ACCEPTED, DRIFT_EVIDENCE_NEEDED, NOT_COMPARED, TOOLS_SKIP_REASON,
    ShadowStore, drift, drift_allowance, drift_evidence, drift_scopes, policy_chain_digest, policy_row_digest,
)

# The drift baselines of the pilot policies, as freezing them again from their recorded
# graded histories writes them: graded tasks, accepted answers compared with the
# frontier's, those that differed, and those that differed without a loss.
FROZEN = {
    "MMLU-Pro": (692, 612, 16, 7), "MedQA": (556, 512, 13, 7), "ARC": (556, 556, 15, 2),
    "FinQA exact": (554, 294, 77, 70), "FinQA by value": (554, 357, 95, 83), "MATH-500 exact": (227, 197, 11, 10),
    "MATH-500 by value": (227, 223, 2, 0), "Tickets": (692, 692, 38, 19), "xRouteBench": (2840, 2190, 105, 76),
}
# Run, epsilon, the policy, then the run's differing answers, compared answers and drift
# requests, and the evidence against the share of requests alone.
HEALTHY = [
    ("MMLU-Pro first pilot", 0.04, "MMLU-Pro", (9, 268, 317), 0.0181),
    ("long-1", 0.04, "MMLU-Pro", (9, 273, 300), 0.0199),
    ("long-2", 0.04, "MMLU-Pro", (17, 283, 300), 0.0566),
    ("long-3", 0.04, "MMLU-Pro", (11, 277, 300), 0.0243),
    ("long-4", 0.04, "MMLU-Pro", (14, 281, 300), 0.0350),
    ("long-5", 0.04, "MMLU-Pro", (7, 276, 300), 0.0167),
    ("MedQA", 0.04, "MedQA", (1, 191, 200), 0.0170),
    ("ARC", 0.04, "ARC", (7, 200, 200), 0.0382),
    ("FinQA exact", 0.04, "FinQA exact", (25, 102, 200), 0.0361),
    ("FinQA by value", 0.04, "FinQA by value", (29, 118, 200), 0.0369),
    ("MATH-500 exact", 0.04, "MATH-500 exact", (5, 85, 100), 0.0492),
    ("MATH-500 by value", 0.04, "MATH-500 by value", (2, 97, 100), 0.0460),
]
# Runs of more than one block of requests. The evidence then depends on the order.
LONG = [
    ("Long pooled", 0.04, "MMLU-Pro", (58, 1390, 1500), 0.0138),
    ("All pooled", 0.04, "MMLU-Pro", (67, 1658, 1817), 0.0122),
    ("xRouteBench replay", 0.02, "xRouteBench", (58, 1099, 1414), 0.0415),
]
# The ticket run is apart: its graded loss was 12 of 300, at the 4% limit and not under it.
TICKETS = (32, 300, 300)


def frozen(name: str) -> dict:
    """A policy's whole drift baseline, as its file holds it."""
    tasks, compared, differing, harmless = FROZEN[name]
    return {"tasks": tasks, "compared": compared, "differing": differing, "differing_not_lost": harmless}


def baseline(name: str) -> dict:
    """The part of a baseline the share of requests is checked against, and no more:
    with this alone the check does not look at the accepted answers apart and gives
    no notice."""
    return {"tasks": FROZEN[name][0], "differing_not_lost": FROZEN[name][3]}


def check(differing, compared, requests, epsilon, recorded=None, **options) -> dict:
    return drift(compared, differing, epsilon, requests=requests, baseline=recorded, **options)


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
                 requests=len(rows), baseline=recorded, latest=drift_scopes(rows))


@pytest.mark.parametrize("name,epsilon,policy_name,counts,evidence", HEALTHY)
def test_healthy_recorded_runs_show_no_sign_of_drift(name, epsilon, policy_name, counts, evidence):
    report = check(*counts, epsilon, baseline(policy_name))
    assert report["drift_check"].startswith("no clear sign of drift: ") and report["drift_warning"] is None
    assert report["drift_basis"] == "baseline" and report["drift_evidence"] == pytest.approx(evidence, rel=0.01)
    assert report["drift_scope"] == {"requests": counts[2], "compared": counts[1], "differing": counts[0],
                                     "latest": False, "among": "requests"}
    # With the whole baseline the accepted answers are checked apart too, and the share
    # is held against the graded tasks' own: still nothing, by a wide margin.
    whole = check(*counts, epsilon, frozen(policy_name))
    assert whole["drift_check"].startswith("no clear sign of drift: ") and whole["drift_notice"] is None
    assert whole["drift_evidence"] < 0.06 and whole["drift_notice_evidence"] < 0.6


@pytest.mark.parametrize("name,epsilon,policy_name,counts,evidence", LONG)
def test_healthy_long_runs_show_no_sign_of_drift(name, epsilon, policy_name, counts, evidence):
    report = check_rows(spread_rows(*counts), epsilon, baseline(policy_name))
    assert report["drift_check"].startswith("no clear sign of drift: ") and report["drift_evidence"] < 1
    assert report["drift_evidence"] == pytest.approx(evidence, rel=0.01)
    assert (report["accepted_differing_from_frontier"], report["drift_requests"]) == (counts[0], counts[2])
    whole = check_rows(spread_rows(*counts), epsilon, frozen(policy_name))
    assert whole["drift_check"].startswith("no clear sign of drift: ") and whole["drift_notice_evidence"] < 0.1


def test_the_same_runs_under_a_policy_without_a_baseline_get_a_rough_check_and_no_warning():
    evidence = {}
    for name, epsilon, _, counts, _ in HEALTHY + [("Tickets", 0.04, "Tickets", TICKETS, None)]:
        report = check(*counts, epsilon)
        assert report["drift_check"].startswith("rough check, no warning: ") and report["drift_warning"] is None, name
        assert (report["drift_basis"], report["drift_baseline"], report["drift_history"]) == ("default", None, None)
        assert report["drift_allowance"] == pytest.approx(epsilon + 0.10)
        assert report["drift_allowance_accepted"] == DRIFT_DEFAULT_ALLOWANCE_ACCEPTED == 1 / 3
        evidence[name] = report["drift_evidence"]
    for name, epsilon, _, counts, _ in LONG:
        report = check_rows(spread_rows(*counts), epsilon)
        assert report["drift_check"].startswith("rough check, no warning: "), name
        evidence[name] = report["drift_evidence"]
    # A quarter of FinQA's accepted answers differ, 14.5% of its requests: above the default
    # 14%, though not clearly at 200 requests. A longer run like it would warn.
    assert max(evidence, key=evidence.get) == "FinQA by value"
    assert evidence["FinQA by value"] == pytest.approx(0.0989, rel=0.01)


def test_the_mmlu_pro_chain_shows_no_sign_of_drift_at_the_two_percent_limit_too():
    """The chain certified at 4% is also the one certified at 2%. There the allowance
    is 3.89%, and the plain comparison with 2% warned on most of these runs."""
    assert drift_allowance(0.02, baseline("MMLU-Pro"))["drift_allowance"] == pytest.approx(0.0389, abs=5e-5)
    for name, _, policy_name, counts, _ in HEALTHY:
        if policy_name == "MMLU-Pro":
            report = check(*counts, 0.02, baseline(policy_name))
            assert report["drift_check"].startswith("no clear sign of drift: "), name
            assert counts[0] / counts[2] > 0.02 and report["drift_evidence"] < 1  # above the limit, and no warning
            if name == "long-2":
                assert report["drift_evidence"] == pytest.approx(0.576, rel=0.01)
            whole = check(*counts, 0.02, frozen(policy_name))
            assert whole["drift_check"].startswith("no clear sign of drift: ") and whole["drift_notice_evidence"] < 1
    for name, _, policy_name, counts, _ in LONG:
        if policy_name == "MMLU-Pro":
            report = check_rows(spread_rows(*counts), 0.02, frozen(policy_name))
            assert report["drift_check"].startswith("no clear sign of drift: "), name


@pytest.mark.parametrize("epsilon,policy_name", [
    (0.04, "MMLU-Pro"), (0.04, "MedQA"), (0.04, "ARC"), (0.04, "MATH-500 exact"), (0.04, "MATH-500 by value"),
    (0.04, "Tickets"), (0.02, "xRouteBench"), (0.04, None), (0.02, None)])
def test_cheaper_models_agreeing_on_wrong_answers_still_warn(epsilon, policy_name):
    """The case the check is for: cheaper models agree with each other on wrong
    answers for about 40% of tasks. 13 of the first 30 accepted answers differ."""
    for recorded in ((baseline(policy_name), frozen(policy_name)) if policy_name else (None,)):
        report = check(13, 30, 30, epsilon, recorded)
        assert report["drift_check"] == "warning" and report["drift_evidence"] >= DRIFT_EVIDENCE_NEEDED
        assert report["drift_warning"].startswith(
            "Cannot confirm the certification on this traffic: 13 of 30 accepted cheaper answers differ from the "
            "frontier model's answer, 43.33% of the 30 shadowed requests the policy was run on. That is clearly more "
            "than ")
    if policy_name is None:
        # Half of the evidence is that of the share of requests (132.9 and 630 alone).
        assert report["drift_evidence"] == pytest.approx({0.04: 66.7, 0.02: 315.2}[epsilon], rel=0.01)


def test_where_harmless_differences_are_common_only_a_larger_change_shows():
    """A quarter of FinQA's accepted answers already differ from the frontier's, mostly
    by rounding, so the allowance is wide and wrong agreement needs more to show."""
    exact, by_value = baseline("FinQA exact"), baseline("FinQA by value")
    assert drift_allowance(0.04, exact)["drift_allowance"] == pytest.approx(0.19197, abs=1e-5)
    calm = check(13, 30, 30, 0.04, exact)
    assert calm["drift_evidence"] == pytest.approx(8.01, rel=0.01) and calm["drift_check"].startswith(
        "no clear sign of drift: ") and calm["drift_check"].endswith(
        "Answers that differed without a loss were common on the graded tasks, so only a large change shows")
    for counts, recorded, evidence in (((14, 30, 30), exact, 24.6), ((15, 30, 30), by_value, 25.2),
                                       ((30, 45, 85), exact, 24.6)):  # 40.5 points more among accepted answers
        report = check(*counts, 0.04, recorded)
        assert report["drift_check"] == "warning" and report["drift_evidence"] == pytest.approx(evidence, rel=0.01)
    # With the whole baseline: 13 of 30 says nothing, 14 is above the graded tasks' own share, 15 warns.
    verdicts = [check(count, 30, 30, 0.04, frozen("FinQA exact"))["drift_check"].split(":")[0] for count in (13, 14, 15)]
    assert verdicts == ["no clear sign of drift", "notice", "warning"]
    among = check(30, 45, 85, 0.04, frozen("FinQA exact"))
    assert among["drift_check"] == "warning" and among["drift_scope"]["among"] == "accepted"


@pytest.mark.parametrize("differing,requests,warns,evidence", [
    (90, 1000, True, 148.8),  # 9.0%: a loss at twice the limit on top of the usual differences
    (86, 1000, True, 25.8), (85, 1000, False, 17.2), (32, 300, True, 22.1), (31, 300, False, 11.85),
    (70, 1000, False, 0.183)])
def test_a_differing_share_from_the_start_warns_once_it_is_clearly_above_the_allowance(
        differing, requests, warns, evidence):
    report = check(differing, requests, requests, 0.04, baseline("MMLU-Pro"))
    assert (report["drift_check"] == "warning") is warns and (report["drift_warning"] is not None) is warns
    assert report["drift_evidence"] == pytest.approx(evidence, rel=0.01)
    assert report["drift_allowance"] == pytest.approx(0.058916, abs=1e-6)
    assert report["drift_baseline"] == {"tasks": 692, "differing_not_lost": 7,
                                        "upper": pytest.approx(0.018916, abs=1e-6)}
    assert (report["drift_allowance_accepted"], report["drift_history"], report["drift_notice_evidence"]) == (None,) * 3


def test_the_levels_come_from_the_counts_the_policy_file_records():
    levels = drift_allowance(0.04, frozen("MMLU-Pro"))
    assert levels["drift_allowance"] == pytest.approx(0.058916, abs=1e-6)
    # Per accepted answer: 4% of 692 tasks over 612 compared answers, plus the bound on 7 of 612.
    assert levels["drift_epsilon_accepted"] == pytest.approx(0.04 * 692 / 612)
    assert levels["drift_allowance_accepted"] == pytest.approx(0.04 * 692 / 612 + 0.021376, abs=1e-6)
    assert levels["drift_accepted_share"] == pytest.approx(612 / 692)
    assert levels["drift_baseline"] == {"tasks": 692, "differing_not_lost": 7, "upper": pytest.approx(0.018916, abs=1e-6),
                                        "compared": 612, "upper_accepted": pytest.approx(0.021376, abs=1e-6)}
    # What differed on the graded tasks, losses included: 16 of 692 and of 612.
    assert levels["drift_history"] == {"tasks": 692, "differing": 16, "upper": pytest.approx(0.03490, abs=1e-5),
                                       "compared": 612, "upper_accepted": pytest.approx(0.03943, abs=1e-5)}
    finqa = drift_allowance(0.04, frozen("FinQA exact"))
    assert (finqa["drift_allowance"], finqa["drift_allowance_accepted"]) == (
        pytest.approx(0.19197, abs=1e-5), pytest.approx(0.35791, abs=1e-5))
    default = drift_allowance(0.04)
    assert (default["drift_basis"], default["drift_allowance"], default["drift_allowance_accepted"]) == (
        "default", pytest.approx(0.14), 1 / 3)


def test_the_sentences_say_what_was_compared_and_what_the_level_is_made_of():
    allowed = ("the 5.89% the certification allows for: the quality limit of 4.00% plus 1.89% for answers that "
               "differed without a loss on the 692 graded tasks the policy was certified and tested on")
    calm = check(9, 268, 317, 0.04, baseline("MMLU-Pro"))
    assert calm["drift_check"] == (
        "no clear sign of drift: 9 of 268 accepted answers differ from the frontier's, 2.84% of the 317 "
        f"shadowed requests the policy was run on. That is not clearly above {allowed}")
    assert check(9, 268, 317, 0.04, frozen("MMLU-Pro"))["drift_check"] == (
        calm["drift_check"] + ". Of the accepted answers alone 3.36% differ, against 6.66% allowed for")
    grade = "A different answer is not always a wrong one; grade a sample of these requests"
    assert check(32, 300, 300, 0.04, baseline("MMLU-Pro"))["drift_warning"] == (
        "Cannot confirm the certification on this traffic: 32 of 300 accepted cheaper answers differ from the "
        "frontier model's answer, 10.67% of the 300 shadowed requests the policy was run on. That is clearly more "
        f"than {allowed}. {grade} before relying on the policy or its projected saving.")
    rough = check(13, 30, 30, 0.04)
    assert rough["drift_warning"] == (
        "Cannot confirm the certification on this traffic: 13 of 30 accepted cheaper answers differ from the "
        "frontier model's answer, 43.33% of the 30 shadowed requests the policy was run on. That is clearly more "
        "than 14.00%: the quality limit of 4.00% plus a default 10.00%, used because this policy file records no "
        f"baseline of its own. {grade} before relying on the policy or its projected saving.")
    assert check(9, 268, 317, 0.04)["drift_check"] == (
        "rough check, no warning: 9 of 268 accepted answers differ from the frontier's, 2.84% of the 317 shadowed "
        "requests the policy was run on. That is not clearly above 14.00%: the quality limit of 4.00% plus a default "
        "10.00%, used because this policy file records no baseline of its own. Of the accepted answers alone 3.36% "
        "differ, against 33.33% allowed for. Only a large change shows at this level; freeze the policy again from "
        "recorded answers to record a baseline")
    # Too few compared answers, or no limit: the two earlier sentences, unchanged.
    assert check(13, 29, 29, 0.04, frozen("MMLU-Pro"))["drift_check"] == (
        "not checked yet: 29 accepted answers compared with the frontier's, 30 needed")
    assert check(13, 30, 30, None, frozen("MMLU-Pro"))["drift_check"] == (
        "not checked: the policy's quality limit was not given")
    assert check(13, 30, 30, None, frozen("MMLU-Pro"))["drift_allowance"] is None


def test_requests_of_another_kind_do_not_hide_accepted_answers_that_differ():
    """Only part of a tenant's traffic is of the kind the policy was certified on. The
    rest is tried and never accepted, so the share of requests stays small whatever
    the accepted answers do. The share among the accepted answers does not."""
    # A fifth of 20,000 requests is of the certified kind, half of it is accepted, and every accepted answer differs.
    report = check(2087, 2087, 20_000, 0.04, frozen("FinQA exact"))
    assert report["drift_share"] < report["drift_allowance"] and report["drift_check"] == "warning"
    assert report["drift_scope"] == {"requests": 20_000, "compared": 2087, "differing": 2087, "latest": False,
                                     "among": "accepted"}
    # The share of requests alone, as the check was before, saw nothing.
    assert check(2087, 2087, 20_000, 0.04, baseline("FinQA exact"))["drift_evidence"] < 0.01
    # The simulated buyer's wrong agreement, 44% of accepted answers, on 15% of the traffic.
    report = check(1154, 2644, 20_000, 0.04, frozen("MMLU-Pro"))
    assert report["drift_share"] == pytest.approx(0.0577) and report["drift_warning"] == (
        "Cannot confirm the certification on this traffic: 1,154 of 2,644 accepted cheaper answers differ from the "
        "frontier model's answer, 43.65% of them, in the 20,000 shadowed requests the policy was run on. That is "
        "clearly more than the 6.66% of accepted answers the certification allows for: the quality limit of 4.00% "
        "of requests, which is 4.52% of accepted answers when 88.44% of requests are accepted as on the graded "
        "tasks, plus 2.14% for answers that differed without a loss among the 612 compared on the graded tasks the "
        "policy was certified and tested on. A different answer is not always a wrong one; grade a sample of these "
        "requests before relying on the policy or its projected saving.")
    # A policy file without a baseline: clearly more than a third of the accepted answers.
    for counts, epsilon in (((2295, 5321, 20_000), 0.02), ((2644, 2644, 20_000), 0.04)):
        report = check(*counts, epsilon)
        assert report["drift_share"] < report["drift_allowance"] and report["drift_check"] == "warning"
        assert "% of them, in the 20,000 shadowed requests the policy was run on. That is clearly more than 33.33% of " \
               "accepted answers, a default used because this policy file records no baseline of its own." \
               in report["drift_warning"]
    # Healthy mixed traffic says nothing either way: 3% of the accepted answers differ.
    healthy = check(80, 2644, 20_000, 0.04, frozen("MMLU-Pro"))
    assert healthy["drift_check"].startswith("no clear sign of drift") and healthy["drift_notice_evidence"] < 0.01


def test_a_share_above_the_graded_tasks_own_and_within_the_allowance_gets_a_notice():
    """MedQA: 13 of 556 graded tasks differed (2.3%), 7 of them without a loss. A loss of
    twice the limit makes about 8% differ: not clearly above the 6.35% allowed for at
    1,000 requests, and clearly above what the graded tasks showed."""
    report = check(80, 920, 1000, 0.04, frozen("MedQA"))
    assert report["drift_check"] == "notice" and report["drift_warning"] is None
    assert report["drift_evidence"] < 1 and report["drift_notice_evidence"] > 1e6
    assert report["drift_notice"] == (
        "Accepted answers differ from the frontier's more often than on the graded tasks: 80 of 920 accepted answers "
        "differ from the frontier's, 8.70% of them, in the 1,000 shadowed requests the policy was run on. On the "
        "graded tasks the policy was certified and tested on, 13 of 512 did, 2.54%, and at most 4.01% with 95% "
        "confidence. The share is not clearly above the 6.90% of accepted answers the certification allows for: the "
        "quality limit of 4.00% of requests, which is 4.34% of accepted answers when 92.09% of requests are accepted "
        "as on the graded tasks, plus 2.55% for answers that differed without a loss among the 512 compared on the "
        "graded tasks the policy was certified and tested on. The check cannot tell a higher loss from more harmless "
        "differences. A different answer is not always a wrong one; grade a sample of these requests to find out "
        "which.")
    # So does one and a half times the limit. Traffic like the graded tasks gets none.
    assert check(60, 920, 1000, 0.04, frozen("MedQA"))["drift_check"] == "notice"
    calm = check(20, 920, 1000, 0.04, frozen("MedQA"))
    assert calm["drift_check"].startswith("no clear sign of drift") and calm["drift_notice_evidence"] < 0.01
    # A baseline that does not say how many answers differed on the graded tasks gives no notice.
    assert check(80, 920, 1000, 0.04, baseline("MedQA"))["drift_check"].startswith("no clear sign of drift")
    # At the 2% limit the MMLU-Pro allowance (3.89%) is barely above the graded tasks' own
    # share (at most 3.49%), so a notice has little room there.
    levels = drift_allowance(0.02, frozen("MMLU-Pro"))
    assert 0.0349 < levels["drift_history"]["upper"] < levels["drift_allowance"] < 0.039


def test_a_loss_of_twice_the_limit_does_not_show_at_1000_requests_where_harmless_differences_are_common():
    """What the check cannot do. FinQA: 12.5% of requests differ on healthy traffic, so
    a loss of 8% makes 18.8% differ, under the 19.2% allowed for. It never warns, and
    the notice needs more than 1,000 requests."""
    report = check(188, 510, 1000, 0.04, frozen("FinQA exact"))
    assert report["drift_check"].startswith("no clear sign of drift") and report["drift_notice_evidence"] < 5
    longer = check(1880, 5100, 10_000, 0.04, frozen("FinQA exact"))
    assert longer["drift_check"] == "notice" and longer["drift_evidence"] < 1
    # Three times the limit warns at 1,000.
    assert check(224, 510, 1000, 0.04, frozen("FinQA exact"))["drift_check"] == "warning"


def test_the_ticket_run_stays_quiet_only_because_it_is_short():
    """IT tickets: the graded loss of the shadow run was 12 of 300, at the 4% limit, and
    answers that differed without a loss were 20 of 300 against 19 of 692 at freezing.
    The run's 10.67% is above its own allowance of 8.00%, not clearly at 300 requests.
    The same share on more requests gets a notice and then a warning. Whether that is
    wanted is open; it is not a healthy run that the check leaves alone."""
    recorded = frozen("Tickets")
    assert drift_allowance(0.04, recorded)["drift_allowance"] == pytest.approx(0.08003, abs=1e-5)
    short = check(*TICKETS, 0.04, recorded)
    assert short["drift_check"].startswith("no clear sign of drift") and short["drift_share"] > short["drift_allowance"]
    assert short["drift_evidence"] == pytest.approx(0.541, rel=0.01)
    assert short["drift_notice_evidence"] == pytest.approx(1.76, rel=0.01)
    assert check(107, 1000, 1000, 0.04, recorded)["drift_check"] == "notice"
    assert check(320, 3000, 3000, 0.04, recorded)["drift_check"] == "warning"
    # Traffic like the ticket history itself (38 of 692, 5.5%) shows nothing.
    assert check(55, 1000, 1000, 0.04, recorded)["drift_notice_evidence"] < 0.1


def test_a_change_late_in_a_long_history_shows_in_the_latest_requests():
    """10,500 healthy requests, then 40% of answers differing. All the requests together
    barely move; the requests since the last full block show it."""
    recorded = baseline("MMLU-Pro")
    healthy = [(index % 125 < 3, True) for index in range(10_500)]
    before = check_rows(healthy, 0.04, recorded)
    assert sum(differs for differs, _ in healthy) == 252
    assert before["drift_check"].startswith("no clear sign") and before["drift_evidence"] == pytest.approx(0.0011, rel=0.02)
    drifted = [(index % 5 < 2, True) for index in range(150)]
    after = check_rows(healthy + drifted, 0.04, recorded)
    assert after["drift_check"] == "warning" and after["drift_evidence"] == 1e12
    assert after["drift_scope"] == {"requests": 150, "compared": 150, "differing": 60, "latest": True,
                                    "among": "requests"}
    assert "60 of 150 accepted cheaper answers differ from the frontier model's answer, 40.00% of the latest 150 " \
           "shadowed requests the policy was run on." in after["drift_warning"]
    # The fields beside the sentence still count every request.
    assert (after["accepted_differing_from_frontier"], after["drift_requests"]) == (312, 10_650)
    # All the requests together say almost nothing: without the latest ones there is no warning.
    alone = check(312, 10_650, 10_650, 0.04, recorded)
    assert alone["drift_check"].startswith("no clear sign") and alone["drift_evidence"] == pytest.approx(0.0006, rel=0.05)
    first = next(count for count in range(1, 151)
                 if check_rows(healthy + drifted[:count], 0.04, recorded)["drift_check"] == "warning")
    assert first == 30  # as soon as the latest requests hold 30 compared answers


def test_drift_that_climbs_in_the_second_half_of_a_thousand_requests_shows():
    """Nothing in the first 500 requests, then cheaper models agree on wrong answers
    more and more often, up to 40% of requests at the last one. FinQA's shape: half
    the requests accepted, a quarter of the accepted answers differing harmlessly.
    Over the whole run the share is not clearly above the allowance; the last block is."""
    rows = []
    for index in range(1000):
        accepted = index % 2 == 0
        differs = accepted and index % 8 == 0
        if index >= 500 and (index * 7919) % 500 < 0.4 * (index - 499):  # a climbing share of requests
            accepted = differs = True
        rows.append((differs, accepted))
    counts = (sum(compared for _, compared in rows), sum(differs for differs, _ in rows))
    whole = drift(*counts, 0.04, requests=1000, baseline=frozen("FinQA exact"))
    assert whole["drift_check"] != "warning" and whole["drift_evidence"] < 1
    report = check_rows(rows, 0.04, frozen("FinQA exact"))
    assert report["drift_check"] == "warning" and report["drift_scope"]["latest"]
    assert report["drift_scope"]["requests"] in (250, 500) and DRIFT_BLOCK == 250


def test_a_warning_goes_away_when_later_requests_no_longer_support_it():
    rows = [(index < 13, True) for index in range(30)]
    assert check_rows(rows, 0.04, baseline("MMLU-Pro"))["drift_check"] == "warning"
    later = check_rows(rows + [(False, True)] * 100, 0.04, baseline("MMLU-Pro"))
    assert later["drift_check"].startswith("no clear sign of drift") and later["drift_warning"] is None
    assert later["drift_evidence"] == pytest.approx(1.02, rel=0.01)


def test_reading_the_report_after_every_request_keeps_false_alarms_under_five_percent():
    """With each answer differing at exactly the allowed rate, the chance that all the
    requests together warn at any read in the first 1,000, worked out exactly."""
    recorded = baseline("MMLU-Pro")
    allowance = drift_allowance(0.04, recorded)["drift_allowance"]
    least = {}  # per number of requests, the fewest differing answers that warn
    count = 1
    for requests in range(30, 1001):
        while check(count, requests, requests, 0.04, recorded)["drift_check"] != "warning":
            count += 1
        assert count == 1 or check(count - 1, requests, requests, 0.04, recorded)["drift_check"] != "warning"
        least[requests] = count
    assert [least[requests] for requests in (30, 100, 300, 1000)] == [8, 15, 32, 86]
    alive, warned = np.zeros(least[1000] + 2), 0.0
    alive[0] = 1.0
    for requests in range(1, 1001):
        alive[1:] = alive[1:] * (1 - allowance) + alive[:-1] * allowance
        alive[0] *= 1 - allowance
        if requests in least:
            warned += alive[least[requests]:].sum()
            alive[least[requests]:] = 0.0
    assert warned == pytest.approx(0.0089, rel=0.02) and warned <= 1 / DRIFT_EVIDENCE_NEEDED


def test_the_whole_check_keeps_false_alarms_under_five_percent_at_the_allowed_rates():
    """The whole rule, with the latest requests weighed apart and the accepted answers
    checked too: 200 streams of 2,000 requests in which 88.44% are accepted and an
    accepted answer differs at exactly the allowed rate, read every 20 requests."""
    recorded = frozen("MMLU-Pro")
    levels = drift_allowance(0.04, recorded)
    rng = np.random.default_rng(11)
    warned = 0
    for _ in range(200):
        accepted = rng.random(2000) < levels["drift_accepted_share"]
        differs = accepted & (rng.random(2000) < levels["drift_allowance_accepted"])
        assert abs(differs.mean() - levels["drift_allowance"]) < 0.03
        rows = list(zip(differs.tolist(), accepted.tolist()))
        warned += any(check_rows(rows[:upto], 0.04, recorded)["drift_check"] == "warning"
                      for upto in range(40, 2001, 20))
    assert warned / 200 <= 1 / DRIFT_EVIDENCE_NEEDED


def test_evidence_weighs_all_the_requests_and_each_later_start_with_weights_that_sum_to_one():
    scopes = [{"requests": 0, "compared": 0, "differing": 0}] * 4
    empty = {"requests": 0, "compared": 0, "differing": 0, "latest": False, "among": "requests"}
    assert drift_evidence(scopes, 0.05, 0.04, min_compared=0) == (
        pytest.approx(1 / 2 + 1 / 4 + 1 / 12 + 1 / 24), empty)  # no request: each scope gives 1
    # With the accepted answers checked too, each of the two sums has half the weight.
    assert drift_evidence(scopes, 0.05, 0.04, min_compared=0, accepted_allowance=0.1, accepted_epsilon=0.05) == (
        pytest.approx(1 / 2 + 1 / 4 + 1 / 12 + 1 / 24), empty)
    assert drift_evidence(scopes, 0.05, 0.04) == (0.0, None)  # a scope needs 30 compared answers to count
    assert drift_evidence(scopes, 1.0, 0.04, min_compared=0)[0] == 0.0  # nothing can be above an allowance of 1
    rows = [(False, True)] * (2 * DRIFT_BLOCK + 100)
    assert drift_scopes(rows) == [{"requests": DRIFT_BLOCK + 100, "compared": DRIFT_BLOCK + 100, "differing": 0},
                                  {"requests": 100, "compared": 100, "differing": 0}]
    assert drift_scopes(rows[:DRIFT_BLOCK]) == [] and drift_scopes([]) == []
    # Counts far past the level needed are reported at the cap, not as an overflow.
    assert check(1000, 1000, 1000, 0.04, frozen("MMLU-Pro"))["drift_evidence"] == 1e12
    assert math.isfinite(check(0, 100_000, 100_000, 0.04, frozen("MMLU-Pro"))["drift_evidence"])


def accepted_run(answer="A", **changes):
    return {"accepted": True, "stage": 0, "served_by": "small-a", "answer": answer,
            "calls": [{"model": "small-a", "cost_usd": 0.01, "error": None}], **changes}


def policy(epsilon=0.04, recorded=None, frontier="frontier", **changes):
    return {"version": 1, "tenant": "a", "frontier": frontier,
            "guarantee": {"certified": True, "epsilon": epsilon, "delta": 0.05, "bound": epsilon / 2},
            "stages": [{"models": ["small-a", "small-b"], "check": "agreement"}],
            **({"drift_baseline": frozen(recorded)} if recorded else {}), **changes}


def record(store, used, name, differing, matching, provider="openai", **changes):
    for index in range(differing + matching):
        store.record(request_id=f"{name}{index}", tenant="a", provider=provider, policy=used,
                     outcome=accepted_run(**changes), frontier_answer="B" if index < differing else "A",
                     frontier_cost_usd=1.0)


def test_each_policy_is_checked_on_the_requests_run_with_it_against_its_own_limit_and_baseline():
    store = ShadowStore(Ledger())
    current, older = policy(0.04, "MMLU-Pro"), policy(0.04, answer_pattern=r"ANSWER: (\w)")  # read answers another way
    assert policy_chain_digest(current) != policy_chain_digest(older)
    record(store, older, "old", differing=30, matching=10)  # an earlier policy drifted badly
    record(store, current, "new", differing=2, matching=38)
    report = store.report("a", policies={"openai": current})
    assert report["shadowed_requests"] == 80 and report["drift_other_policy_requests"] == 40
    assert (report["drift_policy"], report["drift_policies"]) == ("openai", None)
    assert (report["drift_requests"], report["drift_differing"], report["drift_compared"]) == (40, 2, 40)
    assert report["drift_epsilon"] == 0.04
    assert report["drift_basis"] == "baseline" and report["drift_baseline"]["tasks"] == 692
    assert report["drift_check"].startswith("no clear sign of drift: 2 of 40 accepted answers")
    # The figures outside the drift check cover every request, and agree with each other.
    assert (report["accepted_compared_with_frontier"], report["accepted_matching_frontier"],
            report["accepted_differing_from_frontier"]) == (80, 48, 32)
    assert report["accepted_differing_from_frontier_share"] == pytest.approx(0.4)
    # The same rows pooled under one limit, as before policies were told apart, do warn.
    assert store.report("a", epsilon=0.04)["drift_check"] == "warning"
    # A policy that ran nothing here has nothing to check yet, and says how much it left out.
    unused = store.report("a", policies={"openai": policy(0.02, frontier="another")})
    assert unused["drift_check"].startswith("not checked yet: 0 accepted answers") and unused["drift_requests"] == 0
    assert unused["drift_other_policy_requests"] == 80 and unused["accepted_differing_from_frontier"] == 32


def test_a_policy_frozen_again_keeps_the_requests_and_the_warning_of_the_file_before_it():
    """Freezing a policy again, to record a baseline or on a later day, writes a new
    file. It runs the same way, so its drift check goes on from the same requests."""
    store = ShadowStore(Ledger())
    first = policy(0.04, provenance={"created": "2026-11-16T10:00:00+00:00"})
    again = policy(0.04, "MMLU-Pro", provenance={"created": "2026-11-17T10:00:00+00:00"})
    assert policy_row_digest(first) != policy_row_digest(again) and policy_chain_digest(first) == policy_chain_digest(again)
    record(store, first, "old", differing=40, matching=0)  # cheaper models agreeing on wrong answers
    assert store.report("a", policies={"openai": first})["drift_check"] == "warning"
    record(store, again, "new", differing=0, matching=5)
    report = store.report("a", policies={"openai": again})
    assert report["drift_check"] == "warning" and report["drift_other_policy_requests"] == 0
    assert (report["drift_requests"], report["drift_differing"], report["drift_basis"]) == (45, 40, "baseline")
    assert (report["accepted_differing_from_frontier"], report["accepted_matching_frontier"]) == (40, 5)
    # A file with other stages, another frontier, pattern or comparison is another policy.
    for other in (policy(0.04, frontier="other"), policy(0.04, answer_compare="math"), policy(0.04, answer_pattern="x"),
                  {**policy(0.04), "stages": [{"models": ["small-a", "small-c"], "check": "agreement"}]}):
        assert policy_chain_digest(other) != policy_chain_digest(first)
        assert store.report("a", policies={"openai": other})["drift_other_policy_requests"] == 45
    # The limit, the evidence and the date are not part of how a policy runs.
    assert policy_chain_digest(policy(0.02, "MedQA")) == policy_chain_digest(first)
    assert policy_chain_digest({**first, "answer_compare": "exact"}) == policy_chain_digest(first)


def test_rows_stored_before_the_chain_digest_count_for_their_own_file_until_the_proxy_adopts_them():
    store = ShadowStore(Ledger())
    first, again = policy(0.04), policy(0.04, "MMLU-Pro")
    record(store, first, "old", differing=40, matching=0)
    with store._ledger._db:
        store._ledger._db.execute("UPDATE shadow_runs SET policy_chain_sha256=NULL")  # as an older proxy stored them
    assert store.report("a", policies={"openai": first})["drift_requests"] == 40
    missed = store.report("a", policies={"openai": again})
    assert (missed["drift_requests"], missed["drift_other_policy_requests"]) == (0, 40)
    # The proxy marks the rows of the file it holds when it starts; another tenant's stay as they are.
    assert store.adopt("b", first) == 0 and store.adopt("a", again) == 0
    assert store.adopt("a", first) == 40 and store.adopt("a", first) == 0
    kept = store.report("a", policies={"openai": again})
    assert (kept["drift_requests"], kept["drift_other_policy_requests"], kept["drift_check"]) == (40, 0, "warning")


def test_two_providers_with_different_policies_are_checked_separately():
    store = ShadowStore(Ledger())
    openai, anthropic = policy(0.04, "MMLU-Pro"), policy(0.02, frontier="claude")
    record(store, openai, "o", differing=1, matching=59)
    record(store, anthropic, "c", differing=13, matching=17, provider="anthropic")
    report = store.report("a", policies={"openai": openai, "anthropic": anthropic})
    # The policy with a warning is the one shown, though the other has more requests.
    assert report["drift_policy"] == "anthropic" and report["drift_check"] == "warning"
    assert (report["drift_epsilon"], report["drift_basis"], report["drift_requests"]) == (0.02, "default", 30)
    assert report["drift_other_policy_requests"] == 0 and set(report["drift_policies"]) == {"openai", "anthropic"}
    calm = report["drift_policies"]["openai"]
    assert calm["drift_check"].startswith("no clear sign of drift: 1 of 60") and calm["drift_epsilon"] == 0.04
    assert report["drift_policies"]["anthropic"]["drift_warning"] == report["drift_warning"]
    # The count at the top is of both policies' requests; each policy's own is beside its check.
    assert (report["accepted_differing_from_frontier"], report["drift_differing"], calm["drift_differing"]) == (14, 13, 1)
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


def test_a_policy_with_a_notice_is_shown_before_a_quiet_one():
    store = ShadowStore(Ledger())
    openai, anthropic = policy(0.04, "MedQA"), policy(0.04, "MMLU-Pro", frontier="claude")
    record(store, openai, "o", differing=80, matching=840)
    record(store, anthropic, "c", differing=20, matching=1980, provider="anthropic")
    report = store.report("a", policies={"openai": openai, "anthropic": anthropic})
    assert (report["drift_policy"], report["drift_check"]) == ("openai", "notice")
    assert report["drift_notice"].startswith("Accepted answers differ from the frontier's more often than on the "
                                             "graded tasks: 80 of 920 accepted answers")
    assert report["drift_policies"]["anthropic"]["drift_notice"] is None


def test_requests_sent_to_the_frontier_untried_and_rate_limited_runs_stay_out_of_the_drift_check():
    from tine.shadow import RATE_LIMITED_REASON

    store = ShadowStore(Ledger())
    used = policy(0.04, "MMLU-Pro")
    record(store, used, "kept", differing=1, matching=39)
    for index in range(30):
        store.record(request_id=f"limited{index}", tenant="a", provider="openai", policy=used,
                     outcome=accepted_run(), frontier_answer="B", frontier_cost_usd=1.0,
                     skip_reason=RATE_LIMITED_REASON)
    for index in range(500):  # requests that carry tools: not of the kind the policy was certified on
        store.record(request_id=f"tools{index}", tenant="a", provider="openai", policy=used, frontier_answer=None,
                     frontier_cost_usd=1.0, skip_reason=TOOLS_SKIP_REASON,
                     outcome={"accepted": False, "stage": None, "served_by": None, "answer": None, "calls": []})
    report = store.report("a", policies={"openai": used})
    assert (report["shadowed_requests"], report["sent_to_frontier_for_tools"], report["rate_limited_runs"]) == (540, 500, 30)
    assert (report["drift_requests"], report["drift_differing"], report["drift_share"]) == (40, 1, 0.025)
    assert report["drift_check"].startswith("no clear sign of drift") and report["drift_other_policy_requests"] == 0


def test_answers_accepted_by_a_named_check_are_requests_of_the_drift_check_and_are_not_compared():
    """The quality limit is a share of all tasks, so a request whose answer a named
    check accepted counts as a request, at freezing and here. It is not compared: two
    right patches rarely match."""
    store = ShadowStore(Ledger())
    judged = {"models": ["small-c"], "check": "tests+judge"}
    used = {**policy(0.04, "MMLU-Pro"), "stages": [judged, {"models": ["small-a", "small-b"], "check": "agreement"}]}
    record(store, used, "code", differing=60, matching=0, comparable=False)
    record(store, used, "kept", differing=4, matching=36)
    report = store.report("a", policies={"openai": used})
    assert (report["drift_requests"], report["drift_compared"], report["drift_differing"]) == (100, 40, 4)
    assert report["drift_share"] == 0.04 and report["drift_check"].startswith("no clear sign of drift: 4 of 40")
    # A policy whose stages are all named checks has nothing to compare, and says so.
    code = {**policy(0.04), "stages": [judged]}
    alone = ShadowStore(Ledger())
    record(alone, code, "code", differing=50, matching=0, comparable=False)
    report = alone.report("a", policies={"openai": code})
    assert report["drift_check"] == NOT_COMPARED == (
        "not checked: this policy's answers are accepted by a named check and are never compared with the frontier's")
    assert (report["drift_basis"], report["drift_allowance"], report["drift_epsilon"]) == (None, None, 0.04)
    assert (report["drift_requests"], report["drift_other_policy_requests"]) == (50, 0)


def test_the_store_weighs_the_latest_requests_in_the_order_they_were_stored():
    store = ShadowStore(Ledger())
    used = policy(0.04, "MMLU-Pro")
    for index in range(4 * DRIFT_BLOCK + 150):
        late = index >= 4 * DRIFT_BLOCK
        differs = index % 5 < 2 if late else index % 125 < 3
        store.record(request_id=f"r{index:05d}", tenant="a", provider="openai", policy=used,
                     outcome=accepted_run(), frontier_answer="B" if differs else "A", frontier_cost_usd=1.0)
    report = store.report("a", policies={"openai": used})
    assert report["drift_scope"] == {"requests": 150, "compared": 150, "differing": 60, "latest": True,
                                     "among": "requests"}
    assert "60 of 150 accepted cheaper answers differ from the frontier model's answer, 40.00% of the latest 150 " \
           "shadowed requests" in report["drift_warning"]
    assert (report["drift_requests"], report["accepted_differing_from_frontier"]) == (1150, 84)
