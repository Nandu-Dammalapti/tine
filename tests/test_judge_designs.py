import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(f"tine_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


designs = load("judge_designs")


def verdict(kind, p, cost):
    return {"verdict": kind, "probability": p, "cost": cost, "consulted": False}


HAIKU = {"sure-yes": verdict("accept", 0.95, 1.0), "edge-yes": verdict("accept", 0.9, 1.0),
         "middle": verdict("accept", 0.8, 1.0), "edge-no": verdict("reject", 0.1, 1.0),
         "low-middle": verdict("reject", 0.2, 1.0), "no-reply": verdict("unparsed", None, 1.0),
         "only-haiku": verdict("accept", 0.5, 1.0)}
SONNET = {"sure-yes": verdict("reject", 0.05, 10.0), "edge-yes": verdict("reject", 0.05, 10.0),
          "middle": verdict("reject", 0.3, 10.0), "edge-no": verdict("accept", 0.99, 10.0),
          "low-middle": verdict("accept", 0.97, 10.0), "no-reply": verdict("accept", 0.92, 10.0)}


def test_tiered_decides_on_the_first_judge_only_when_sure():
    table = designs.tiered_verdicts(HAIKU, SONNET, hi=0.9, lo=0.1)
    # >= hi and <= lo are the first judge's own decisions, paid at its cost only.
    assert table["sure-yes"]["verdict"] == "accept" and table["sure-yes"]["cost"] == 1.0
    assert table["edge-yes"]["verdict"] == "accept" and not table["edge-yes"]["consulted"]
    assert table["edge-no"]["verdict"] == "reject" and table["edge-no"]["cost"] == 1.0
    # Between the two, and with no probability, the second judge decides and both are paid.
    for case_id, expected in (("middle", "reject"), ("low-middle", "accept"), ("no-reply", "accept")):
        assert table[case_id]["verdict"] == expected
        assert table[case_id]["cost"] == 11.0 and table[case_id]["consulted"]
    # The second judge never saw it: no verdict, which rejects.
    assert table["only-haiku"]["verdict"] == "unparsed" and table["only-haiku"]["cost"] == 1.0
    assert not designs.accepts(table["only-haiku"], None)


def test_tiered_grid_edges_and_validation():
    wide = designs.tiered_verdicts(HAIKU, SONNET, hi=0.95, lo=0.3)
    assert wide["edge-yes"]["consulted"] and wide["edge-yes"]["verdict"] == "reject"
    assert not wide["low-middle"]["consulted"] and wide["low-middle"]["verdict"] == "reject"
    with pytest.raises(ValueError):
        designs.tiered_verdicts(HAIKU, SONNET, hi=0.3, lo=0.3)


def test_judge_quality_counts_consults_and_cost():
    key = {case_id: {"right": case_id in {"sure-yes", "low-middle"}} for case_id in SONNET}
    table = designs.tiered_verdicts(HAIKU, SONNET, hi=0.9, lo=0.1)
    q = designs.judge_quality(table, key)
    # wrong: edge-yes (accepted), middle (rejected), edge-no (rejected), no-reply (accepted)
    assert q["wrong"] == 4 and q["wrong_rejected"] == 2 and q["catch_rate"] == 0.5
    assert q["right"] == 2 and q["right_rejected"] == 0
    assert q["second_judge_share"] == pytest.approx(3 / 6)
    assert q["cost_per_check_usd"] == pytest.approx((1 + 1 + 11 + 1 + 11 + 11) / 6)


def test_screened_needs_both_and_charges_second_only_after_first_accepts():
    table = designs.screened_verdicts(HAIKU, SONNET)
    assert table["edge-no"]["verdict"] == "reject" and table["edge-no"]["cost"] == 1.0
    assert table["middle"]["verdict"] == "reject" and table["middle"]["cost"] == 11.0


JUDGE = {"c1": verdict("accept", 0.8, 0.1), "c2": verdict("accept", 0.95, 0.2),
         "c3": verdict("reject", 0.2, 0.3), "c4": verdict("accept", 0.95, 0.4)}


def test_selector_picks_highest_accepted_probability_and_pays_every_read():
    result = designs.select_patch(["a", "b", "c"], {"a": "c1", "b": "c2", "c": "c3"}, JUDGE, None)
    assert result == {"passed": True, "picked": "b", "judge_calls": 3, "judge_cost_usd": pytest.approx(0.6)}
    # A tie goes to the cheaper member.
    tie = designs.select_patch(["a", "b", "c"], {"b": "c4", "c": "c2"}, JUDGE, None)
    assert tie["picked"] == "b"
    # At tau 0.9 the 0.8 patch is not eligible on its own.
    strict = designs.select_patch(["a", "c"], {"a": "c1", "c": "c3"}, JUDGE, 0.9)
    assert strict == {"passed": False, "picked": None, "judge_calls": 2, "judge_cost_usd": pytest.approx(0.4)}


def test_selector_reads_an_identical_patch_once_and_skips_missing_verdicts():
    result = designs.select_patch(["a", "b", "c"], {"a": "c1", "b": "c1", "c": "unjudged"}, JUDGE, None)
    assert result["judge_calls"] == 1 and result["judge_cost_usd"] == pytest.approx(0.1)
    assert result["picked"] == "a"
    nothing = designs.select_patch(["a", "b"], {}, JUDGE, None)
    assert nothing == {"passed": False, "picked": None, "judge_calls": 0, "judge_cost_usd": 0.0}


def test_sequential_stops_at_first_accept_and_pays_only_called_members():
    own = {"a": 1.0, "b": 2.0, "c": 4.0}
    result = designs.sequential_patch(["a", "b", "c"], {"a": "c3", "b": "c1", "c": "c2"}, JUDGE, None, own)
    assert result["picked"] == "b" and result["model_cost"] == 3.0
    assert result["judge_calls"] == 2 and result["judge_cost_usd"] == pytest.approx(0.4)


def test_oracle_picks_a_right_usable_patch():
    right = {"a": False, "b": True, "c": True}
    assert designs.oracle_patch(["a", "b", "c"], {"a": "x", "c": "z"}, right)["picked"] == "c"
    assert not designs.oracle_patch(["a", "b"], {"a": "x"}, right)["passed"]


# A small world: two cheap models, a frontier, three tasks.
OWN = {"a": 1.0, "b": 2.0, "front": 100.0}
GRADE = {("t1", "a"): False, ("t1", "b"): True, ("t2", "a"): True, ("t2", "b"): True,
         ("t3", "a"): False, ("t3", "b"): False}
CASE_OF = {("t1", "a"): "x1", ("t1", "b"): "y1", ("t2", "a"): "x2", ("t2", "b"): "x2", ("t3", "a"): "x3"}
FREE = {("t1", "a"): True, ("t1", "b"): True, ("t2", "a"): True, ("t2", "b"): True,
        ("t3", "a"): True, ("t3", "b"): False}
VERDICTS = {"x1": verdict("accept", 0.7, 0.1), "y1": verdict("accept", 0.9, 0.1),
            "x2": verdict("accept", 0.9, 0.1), "x3": verdict("accept", 0.8, 0.1)}


def outcomes():
    rows = []
    for task in ("t1", "t2", "t3"):
        for model, cost in OWN.items():
            rows.append({"tenant": "x", "task_id": task, "prompt": task, "task_type": "code", "model": model,
                         "success": GRADE.get((task, model), True), "cost_usd": cost})
    return pd.DataFrame(rows)


def test_build_stages_charges_members_plus_judge_and_grades_the_chosen_patch():
    frame, checks, members = designs.build_stages(outcomes(), FREE, CASE_OF, VERDICTS, "front", sizes=(2,))
    group = frame[frame.model == "a+b"].set_index("task_id")
    # t1: both read (0.2), b's patch has the higher probability and is right.
    assert group.at["t1", "cost_usd"] == pytest.approx(3.2) and bool(group.at["t1", "success"])
    # t2: an identical patch is read once.
    assert group.at["t2", "cost_usd"] == pytest.approx(3.1)
    # t3: only a passed the free checks; its wrong patch is accepted.
    assert group.at["t3", "cost_usd"] == pytest.approx(3.1) and not bool(group.at["t3", "success"])
    assert members["a+b"] == ("a", "b")
    assert {(row["task_id"], row["passed"]) for row in checks if row["model"] == "a+b"} == {
        ("t1", True), ("t2", True), ("t3", True)}


def test_build_stages_singles_keep_unknown_checks():
    free = {**FREE, ("t1", "a"): None}
    frame, checks, _ = designs.build_stages(outcomes(), free, CASE_OF, VERDICTS, "front", sizes=(1,))
    passed = {(row["task_id"], row["model"]): row["passed"] for row in checks}
    assert passed[("t1", "a")] == "" and passed[("t3", "b")] is False
    # The frontier's own answers are never judged or charged.
    assert set(frame[frame.model == "front"].cost_usd) == {100.0}


def test_paid_once_pays_a_shared_member_and_a_judged_patch_once():
    frame, checks, members = designs.build_stages(outcomes(), FREE, CASE_OF, VERDICTS, "front", sizes=(1, 2))
    _, tasks, _ = designs.complete_tasks(frame, "front")
    ids = list(tasks.task_id)
    own = {model: np.array([OWN[model]] * len(ids)) for model in OWN}
    spend = designs.Spend(ids, own, FREE, CASE_OF, VERDICTS)
    models = sorted(frame.model.unique())
    table = {m: {"passed": np.ones(3, dtype=bool) if m != "a" else np.zeros(3, dtype=bool),
                 "right": np.ones(3, dtype=bool), "cost": np.full(3, OWN.get(m, 0.0))} for m in models}
    rows = np.ones(3, dtype=bool)
    once = designs.chain_totals(["a", "a+b"], table, members, spend, "front", rows, pay_once=True)
    every = designs.chain_totals(["a", "a+b"], table, members, spend, "front", rows, pay_once=False)
    # Every task fails stage a and passes a+b. Paid once: a (1) + its judge read, then b (2) + new reads.
    # Judge: t1 x1 then y1, t2 x2 then nothing new, t3 x3 then nothing new.
    once_cost = 3 * 3.0 + 0.1 * 4
    every_cost = 3 * 1.0 + 0.3 + 3 * 3.0 + (0.2 + 0.1 + 0.1)
    assert once["savings"] == pytest.approx(1 - once_cost / 300)
    assert every["savings"] == pytest.approx(1 - every_cost / 300)
    assert once["judge_calls_per_100"] == pytest.approx(100 * 4 / 3)


def test_search_charged_like_the_cascade_matches_spend_charged_every_stage():
    # Twenty copies of the three tasks, so every part of every split has tasks.
    copies = range(20)
    many = pd.concat([outcomes().assign(task_id=lambda f, i=i: f.task_id + f"-{i}",
                                         prompt=lambda f, i=i: f.prompt + f"-{i}") for i in copies])
    free = {(f"{t}-{i}", m): v for (t, m), v in FREE.items() for i in copies}
    case_of = {(f"{t}-{i}", m): f"{c}-{i}" for (t, m), c in CASE_OF.items() for i in copies}
    judge = {f"{c}-{i}": v for c, v in VERDICTS.items() for i in copies}
    frame, checks, members = designs.build_stages(many, free, case_of, judge, "front", sizes=(1, 2))
    _, tasks, _ = designs.complete_tasks(frame, "front")
    ids = list(tasks.task_id)
    spend = designs.Spend(ids, {m: np.full(len(ids), c) for m, c in OWN.items()}, free, case_of, judge)
    options = {"fractions": (0.4, 0.3, 0.3), "seeds": range(3), "epsilons": (0.5,)}
    plain = designs.search(frame, checks, members, None, "front", pay_once=False, **options)
    every = designs.search(frame, checks, members, spend, "front", pay_once=False, **options)
    for left, right in zip(plain, every):
        assert left["chain"] == right["chain"] and left["savings"] == pytest.approx(right["savings"])


def test_task_folds_keep_every_task_in_one_fold():
    groups = ["t1", "t1", "t2", "t3", "t3", "t3", "t4", "t5"]
    folds = designs.task_folds(groups, 2, seed=0)
    for task in set(groups):
        assert len({folds[i] for i, g in enumerate(groups) if g == task}) == 1
    sizes = np.bincount([folds[groups.index(task)] for task in sorted(set(groups))])
    assert abs(sizes[0] - sizes[1]) <= 1
    assert list(folds) == list(designs.task_folds(groups, 2, seed=0))


def test_isotonic_cv_never_fits_on_the_held_out_task():
    rng = np.random.default_rng(1)
    groups = [f"t{i // 3}" for i in range(90)]
    p = rng.uniform(size=90)
    y = rng.uniform(size=90) < p
    p[5] = np.nan
    calibrated, table = designs.isotonic_cv(p, y, groups, folds=3, seed=0, targets=(0.1,))
    assert np.isnan(calibrated[5]) and not np.isnan(np.delete(calibrated, 5)).any()
    # Changing the labels of one task changes only the predictions outside its fold.
    fold = designs.task_folds(groups, 3, seed=0)
    flipped = y.copy()
    flipped[[0, 1, 2]] = ~flipped[[0, 1, 2]]
    again, _ = designs.isotonic_cv(p, flipped, groups, folds=3, seed=0, targets=(0.1,))
    same_fold = fold == fold[0]
    assert np.allclose(calibrated[same_fold], again[same_fold], equal_nan=True)
    assert {row["scores"] for row in table} == {"raw", "isotonic-cv"}


def test_pick_threshold_meets_the_false_reject_target():
    p = np.array([0.1, 0.2, 0.5, 0.6, 0.9, 0.95, np.nan])
    y = np.array([False, True, False, True, True, True, True])
    # Right patches: 0.2, 0.6, 0.9, 0.95, and one with no probability (always rejected).
    # At 0.2 only the reply with no probability is a false reject (1 of 5).
    assert designs.pick_threshold(p, y, 0.2) == pytest.approx(0.2)
    assert designs.pick_threshold(p, y, 0.4) == pytest.approx(0.6)
    catch, false_reject = designs.rates(p, y, 0.6)
    assert catch == 1.0 and false_reject == pytest.approx(2 / 5)


def test_reliability_brier_and_wilson():
    p, y = [0.05, 0.05, 0.95, 0.95], [False, True, True, True]
    rows = designs.reliability(p, y)
    assert [row["cases"] for row in rows] == [2, 2]
    assert rows[1]["bucket"].endswith("1.00]")
    assert designs.brier(p, y) == pytest.approx((0.05 ** 2 + 0.95 ** 2 + 2 * 0.05 ** 2) / 4)
    low, high = designs.wilson(5, 10)
    assert low < 0.5 < high


def test_changed_lines_ignore_headers():
    patch = "--- a/x\n+++ b/x\n@@\n-old\n+new\n+more\n context\n"
    assert designs.changed_lines(patch) == 3


def test_sample_failures_is_seeded_and_disjoint():
    key = {f"c{i}": {"right": i % 2 == 0} for i in range(40)}
    table = {f"c{i}": verdict("accept" if i % 4 < 2 else "reject", 0.5, 0.0) for i in range(40)}
    first = designs.sample_failures(key, table, 5, seed=3)
    assert first == designs.sample_failures(key, table, 5, seed=3)
    assert all(not key[c]["right"] and table[c]["verdict"] == "accept" for c in first["false-accept"])
    assert all(key[c]["right"] and table[c]["verdict"] == "reject" for c in first["false-reject"])
