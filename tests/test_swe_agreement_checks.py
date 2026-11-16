import csv
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(f"tine_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


swe = load("swe_agreement_checks")
judge_patches = load("judge_patches")

COSTS = {"a": 1.0, "b": 2.0, "c": 3.0, "big": 100.0}
TASKS = ["t1", "t2", "t3", "t4", "t5"]
# Which patch each model wrote per task; equal case ids are identical patches.
CASES = {("t1", "a"): "x1", ("t1", "b"): "x1", ("t1", "c"): "y1",
         ("t2", "a"): "x2", ("t2", "b"): "y2", ("t2", "c"): "z2",
         ("t3", "b"): "y3", ("t3", "c"): "z3",
         ("t4", "a"): "x4", ("t4", "b"): "x4",
         ("t5", "a"): "x5", ("t5", "big"): "x5"}
FREE = {("t1", "a"): True, ("t1", "b"): True, ("t1", "c"): True,
        ("t2", "a"): True, ("t2", "b"): True, ("t2", "c"): True,
        ("t3", "a"): False, ("t3", "b"): True, ("t3", "c"): True,
        ("t4", "a"): True, ("t4", "b"): False, ("t4", "c"): False,
        ("t5", "a"): True, ("t5", "b"): False, ("t5", "c"): None, ("t5", "big"): True}
RIGHT = {"x1", "y2", "z3", "x5"}
JUDGE = {"x1": (False, 0.5), "y1": (True, 0.5), "x2": (False, 0.5), "y2": (True, 0.5), "z2": (True, 0.5),
         "y3": (True, 0.5), "z3": (True, 0.5), "x4": (True, 0.5), "x5": (True, 0.5)}
LINES = {"x2": frozenset("pqrs"), "y2": frozenset("pqrst"), "z2": frozenset("pq"), "y3": frozenset("p"),
         "z3": frozenset("q")}


def outcomes_frame():
    rows = []
    for model, cost in COSTS.items():
        for task in TASKS:
            case = CASES.get((task, model))
            rows.append({"tenant": "t", "task_id": task, "prompt": f"Fix {task}", "task_type": "code", "model": model,
                         "success": case in RIGHT or (model == "big" and task != "t2"), "cost_usd": cost})
    return pd.DataFrame(rows)


def build(**options):
    options.setdefault("lines", LINES)
    judge = None if options.get("mode") == "agree-only" else JUDGE
    frame, checks, detail, report = swe.build_groups(outcomes_frame(), FREE, CASES, judge, "big", **options)
    by = {(row["task_id"], row["model"]): row for row in detail}
    cost = frame.set_index(["task_id", "model"]).cost_usd
    success = frame.set_index(["task_id", "model"]).success
    return by, cost, success, checks, report


def test_changed_lines_and_jaccard_ignore_headers_and_blank_changes():
    patch = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n context\n-old line\n+new line\n+\n"
    assert swe.changed_line_set(patch) == {"old line", "new line"}
    assert swe.jaccard(frozenset("ab"), frozenset("bc")) == pytest.approx(1 / 3)
    assert swe.jaccard(frozenset(), frozenset()) == 0.0


def test_decide_accepts_agreement_without_the_judge_and_otherwise_asks_it():
    agreed = swe.decide(["a", "b", "c"], {"a": "p", "b": "q", "c": "q"}, JUDGE, mode="agree-or-judge")
    assert agreed == {"passed": True, "picked": "b", "agreed": True, "judge_calls": 0, "judge_cost_usd": 0.0}
    judged = swe.decide(["a", "b"], {"a": "x2", "b": "y2"}, JUDGE, mode="agree-or-judge")
    assert judged == {"passed": False, "picked": "a", "agreed": False, "judge_calls": 1, "judge_cost_usd": 0.5}
    # The control ignores agreement; agree-only never calls the judge; no usable patch fails.
    assert swe.decide(["a", "b"], {"a": "x1", "b": "x1"}, JUDGE, mode="judge-only")["judge_calls"] == 1
    assert swe.decide(["a", "b"], {"a": "x2", "b": "y2"}, None, mode="agree-only")["passed"] is False
    assert swe.decide(["a", "b"], {}, JUDGE, mode="agree-or-judge")["picked"] is None
    # A patch the judge never read is rejected at no cost.
    unread = swe.decide(["a"], {"a": "missing"}, JUDGE, mode="agree-or-judge")
    assert unread["passed"] is False and unread["judge_calls"] == 0
    with pytest.raises(ValueError, match="needs judge"):
        swe.decide(["a", "b"], {"a": "x2", "b": "y2"}, None, mode="agree-or-judge")


def test_agree_and_judge_needs_both_checks():
    both = swe.decide(["a", "b", "c"], {"a": "x2", "b": "y1", "c": "y1"}, JUDGE, mode="agree-and-judge")
    assert both == {"passed": True, "picked": "b", "agreed": False, "judge_calls": 1, "judge_cost_usd": 0.5}
    # Agreed but the judge rejects; and no agreement means no judge call at all.
    assert swe.decide(["a", "b"], {"a": "x1", "b": "x1"}, JUDGE, mode="agree-and-judge")["passed"] is False
    alone = swe.decide(["a", "b"], {"a": "x4", "b": "y2"}, JUDGE, mode="agree-and-judge")
    assert alone["passed"] is False and alone["judge_calls"] == 0 and alone["picked"] is None


def test_soft_agreement_uses_the_similarity_threshold():
    usable = {"a": "x2", "b": "y2", "c": "z2"}  # a-b overlap 0.8, a-c 0.5, b-c 0.4
    assert swe.decide(["a", "b", "c"], usable, JUDGE, mode="agree-or-judge", lines=LINES)["agreed"] is False
    at_08 = swe.decide(["a", "b", "c"], usable, JUDGE, mode="agree-or-judge", lines=LINES, similarity=0.8)
    assert at_08["agreed"] and at_08["picked"] == "a" and at_08["judge_calls"] == 0
    assert not swe.decide(["a", "c"], {"a": "x2", "c": "z2"}, JUDGE, mode="agree-only", lines=LINES,
                          similarity=0.8)["agreed"]
    assert swe.decide(["a", "c"], {"a": "x2", "c": "z2"}, JUDGE, mode="agree-only", lines=LINES,
                      similarity=0.5)["agreed"]


def test_pairs_charge_every_member_and_the_judge_only_where_consulted():
    by, cost, success, _, report = build(group_sizes=(2,))
    assert report["candidates"] == 3 and report["cheap_models"] == ["a", "b", "c"]
    # t1: a and b wrote the same patch, so it is accepted although the judge would reject it.
    assert by[("t1", "a+b")]["agreed"] and by[("t1", "a+b")]["passed"] is True
    assert cost[("t1", "a+b")] == 3.0 and bool(success[("t1", "a+b")])
    # t2: no agreement, the judge reads a's patch (the cheapest) and rejects it.
    assert by[("t2", "a+b")]["passed"] is False and cost[("t2", "a+b")] == 3.5
    # t3: a failed the free checks, so the judge reads b's patch; b's grade is the group's.
    assert by[("t3", "a+b")]["picked"] == "b" and by[("t3", "a+b")]["passed"] is True
    assert not success[("t3", "a+b")] and bool(success[("t3", "b+c")]) is False
    # t4: b wrote the same patch as a but failed the free checks: no agreement, one judge call.
    assert not by[("t4", "a+b")]["agreed"] and by[("t4", "a+b")]["judge_calls"] == 1
    # t4, b+c: nobody passed the free checks. Both are still paid for and no judge is called.
    assert by[("t4", "b+c")]["passed"] is False and cost[("t4", "b+c")] == 5.0
    assert report["groups"]["a+b"] == {"tasks": 5, "passed": 4, "agreed": 1, "judge_calls": 4}


def test_the_frontier_is_never_an_agreement_partner_or_judged():
    by, cost, _, checks, _ = build(group_sizes=(1, 2))
    # t5: only the frontier wrote a's patch too. That is not agreement.
    assert not by[("t5", "a+b")]["agreed"] and by[("t5", "a+b")]["judge_calls"] == 1
    assert all("big" not in row["model"].split("+") for row in checks)
    assert cost[("t5", "big")] == 100.0


def test_modes_differ_only_in_how_a_group_accepts():
    control, control_cost, _, _, _ = build(group_sizes=(2,), mode="judge-only")
    assert control[("t1", "a+b")]["passed"] is False and control_cost[("t1", "a+b")] == 3.5
    only, only_cost, _, _, report = build(group_sizes=(2,), mode="agree-only")
    assert only[("t1", "a+b")]["passed"] is True and only[("t2", "a+b")]["passed"] is False
    assert only_cost[("t2", "a+b")] == 3.0 and report["groups"]["a+b"]["judge_calls"] == 0
    with pytest.raises(ValueError, match="at least two"):
        build(group_sizes=(1,), mode="agree-only")
    with pytest.raises(ValueError, match="mode must be"):
        build(mode="vote")


def test_triples_accept_when_any_two_members_agree():
    by, cost, _, _, _ = build(group_sizes=(3,), similarity=0.8)
    assert by[("t1", "a+b+c")]["agreed"] and cost[("t1", "a+b+c")] == 6.0
    assert by[("t2", "a+b+c")]["agreed"] and by[("t2", "a+b+c")]["picked"] == "a"
    assert not by[("t3", "a+b+c")]["agreed"] and by[("t3", "a+b+c")]["picked"] == "b"


def test_a_group_of_one_matches_judge_patches_checks(tmp_path):
    key = tmp_path / "key.csv"
    merged = {}
    for (task, model), case in CASES.items():
        merged.setdefault((case, task), []).append(model)
    key.write_text("case_id,task_id,instance_id,patch_sha,models,benchmark_right,run_resolved\n" + "".join(
        f"{case},{task},i,s,{';'.join(models)},True,True\n" for (case, task), models in merged.items()))
    verdicts = tmp_path / "verdicts.csv"
    with verdicts.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=judge_patches.VERDICT_FIELDS)
        writer.writeheader()
        for case, (accepted, cost) in JUDGE.items():
            writer.writerow({"case_id": case, "judge": "j", "verdict": "accept" if accepted else "reject",
                             "probability": 0.9, "cost_usd": cost})
    free = tmp_path / "free.csv"
    free.write_text("task_id,model,passed\n" + "".join(
        f"{task},{model},{'' if passed is None else passed}\n" for (task, model), passed in FREE.items()))
    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text("".join(json.dumps(row) + "\n" for row in outcomes_frame().to_dict("records")))
    judge_patches.chain_inputs(key, [verdicts], free, outcomes, tmp_path / "checks.csv", tmp_path / "judged.jsonl")
    theirs = {(row["task_id"], row["model"]): row["passed"] for row in swe.read_csv(tmp_path / "checks.csv")}
    their_cost = {(row["task_id"], row["model"]): row["cost_usd"]
                  for row in map(json.loads, (tmp_path / "judged.jsonl").read_text().splitlines())}

    case_of, _ = swe.load_patches(key)
    assert case_of == CASES and swe.load_free_checks(free) == FREE
    frame, checks, _, _ = swe.build_groups(outcomes_frame(), swe.load_free_checks(free), case_of,
                                           swe.load_judge(verdicts), "big", group_sizes=(1,))
    mine = {(row["task_id"], row["model"]): str(row["passed"]) for row in checks}
    assert mine == {key: value for key, value in theirs.items() if key[1] != "big"}
    assert mine[("t5", "c")] == ""  # a free check that was not run stays unknown
    for row in frame.itertuples():
        if row.model != "big":  # the existing join also charges the judge on frontier answers; this one does not
            assert row.cost_usd == pytest.approx(their_cost[(row.task_id, row.model)])


def test_agreement_rates_count_agreed_tasks_and_how_many_are_right():
    table = swe.agreement_rates(outcomes_frame(), FREE, CASES, LINES, "big", group_sizes=(2,),
                                similarities=(None, 0.8))
    identical, soft = table
    # 15 pair-tasks; only a+b on t1 wrote the same patch, and it is right.
    assert identical["group_tasks"] == 15 and identical["agreed"] == round(1 / 15, 4)
    assert identical["agreed_patch_right"] == 1.0 and identical["best_group"] == "a+b"
    # At 0.8 a+b also agree on t2, where a's patch (the one returned) is wrong.
    assert soft["agreed"] == round(2 / 15, 4) and soft["agreed_patch_right"] == 0.5


def test_walk_chain_counts_judge_calls_agreement_and_frontier_calls():
    frame, checks, detail, _ = swe.build_groups(outcomes_frame(), FREE, CASES, JUDGE, "big", group_sizes=(1, 2))
    by = {(row["task_id"], row["model"]): row for row in detail}
    walked = swe.walk_chain(["a+b", "c", "big"], "big", frame, by)
    # a+b: t1 agreed, t2 rejected, t3/t4/t5 accepted by the judge. c then takes t2 (accepted, wrong).
    assert walked["tasks"] == 5 and walked["frontier_calls_per_100"] == 0
    assert walked["agreed_accepts_per_100"] == 20 and walked["judge_calls_per_100"] == 100
    assert walked["accuracy"] == pytest.approx(2 / 5)  # t1 and t5
    assert walked["savings"] == pytest.approx(1 - (5 * 3 + 4 * 0.5 + 3 + 0.5) / 500)
    assert walked["judge_cost_pct_of_frontier"] == pytest.approx(100 * 2.5 / 500)
    alone = swe.walk_chain(["big"], "big", frame, by)
    assert alone["savings"] == 0 and alone["frontier_calls_per_100"] == 100 and alone["judge_calls_per_100"] == 0


def test_run_certifies_every_variant_through_the_cascade_cli(tmp_path, monkeypatch):
    names = [variant["name"] for variant in swe.variants()]
    assert len(names) == len(set(names)) == 20 and "pairs-agree-or-judge-soft80" in names
    assert swe.SIMILARITIES == (0.8, 0.5)
    assert [v["needs_judge"] for v in swe.variants() if "agree-only" in v["name"]] == [False] * 6

    calls = []

    def fake_cascade(command, **_):
        out = Path(command[command.index("--output") + 1])
        out.mkdir(parents=True)
        calls.append(command)
        chain = pd.read_json(command[command.index("--input") + 1], lines=True).model.iloc[-1] + " > big"
        pd.DataFrame([{"epsilon": 0.04, "method": "certified", "chain": chain}]).to_csv(out / "cascade-runs.csv")
        pd.DataFrame([{"epsilon": 0.04, "method": "certified", "savings_mean": 0.1, "accuracy_mean": 0.5,
                       "frontier_accuracy": 0.5, "degradation_max": 0.0, "over_epsilon_runs": 0,
                       "frontier_only_runs": 0, "most_common_chain": chain}]).to_csv(out / "cascade-summary.csv")

    monkeypatch.setattr(swe.subprocess, "run", fake_cascade)
    every = swe.variants()
    monkeypatch.setattr(swe, "variants", lambda: [v for v in every
                                                  if v["name"] in ("singles", "pairs-agree-only-identical")])
    key = tmp_path / "key.csv"
    key.write_text("case_id,task_id,instance_id,patch_sha,models,benchmark_right,run_resolved\n"
                   "x1,t1,i,s,a;b,True,True\nx2,t2,i,s,a,False,False\n")
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"case_id": "x1", "patch": "+one"}) + "\n" + json.dumps({"case_id": "x2", "patch": "+two"}) + "\n")
    verdicts = tmp_path / "v.csv"
    verdicts.write_text("case_id,judge,verdict,probability,cost_usd\nx1,j,accept,0.9,0.5\nx2,j,reject,0.1,0.5\n")
    free = tmp_path / "free.csv"
    free.write_text("task_id,model,passed\nt1,a,True\nt1,b,True\nt2,a,True\nt2,b,False\n")
    outcomes = tmp_path / "outcomes.jsonl"
    frame = outcomes_frame()
    frame = frame[frame.task_id.isin(["t1", "t2"]) & frame.model.isin(["a", "b", "big"])]
    outcomes.write_text("".join(json.dumps(row) + "\n" for row in frame.to_dict("records")))
    swe.main(["run", "--key", str(key), "--cases", str(cases), "--free-checks", str(free), "--outcomes", str(outcomes),
              "--frontier", "big", "--judge", f"j={verdicts}", "--work", str(tmp_path / "work"),
              "--summary", str(tmp_path / "summary.csv")])
    rows = swe.read_csv(tmp_path / "summary.csv")
    # Two variants on two splits; the agree-only one is run once, with no judge.
    assert [(row["variant"], row["judge"], row["split"]) for row in rows] == [
        ("singles", "j", "d"), ("singles", "j", "c50"),
        ("pairs-agree-only-identical", "none", "d"), ("pairs-agree-only-identical", "none", "c50")]
    assert ["--fractions" in command for command in calls] == [False, True, False, True]
    assert rows[0]["most_common_chain"] == "b > big" and float(rows[0]["judge_calls_per_100"]) == 50.0
    assert rows[2]["most_common_chain"] == "a+b > big" and float(rows[2]["agreed_accepts_per_100"]) == 50.0


def test_case_precision_splits_patches_by_agreement_and_verdict(tmp_path):
    key = tmp_path / "key.csv"
    key.write_text("case_id,task_id,instance_id,patch_sha,models,benchmark_right,run_resolved\n"
                   "p1,t1,i,s,a;b,True,True\n"       # two cheap models, identical
                   "p2,t1,i,s,c,False,False\n"       # nothing like it
                   "p3,t2,i,s,a,True,True\n"         # 0.8 like p4
                   "p4,t2,i,s,b,False,False\n"
                   "p5,t2,i,s,big,True,True\n"       # frontier only: not a cheap patch, not a partner
                   "p6,t3,i,s,a;big,True,False\n")   # shared with the frontier only: no agreement
    lines = {"p1": frozenset("xy"), "p2": frozenset("z"), "p3": frozenset("pqrs"), "p4": frozenset("pqrst"),
             "p5": frozenset("pqrs"), "p6": frozenset("k")}
    judge = {"p1": (True, 1.0), "p2": (True, 1.0), "p3": (True, 1.0), "p4": (False, 1.0), "p6": (False, 1.0)}
    table = swe.case_precision(key, {}, lines, {"j": judge}, "big")
    cell = {(row["judge"], row["agreement"], row["verdict"]): (row["cases"], row["right"]) for row in table}
    assert cell[("none", "all patches", "any")] == (5, 2)
    assert cell[("none", "identical patch", "any")] == (1, 1)
    assert cell[("none", "agreement at 0.8", "any")] == (3, 2)
    assert cell[("none", "no agreement at 0.5", "any")] == (2, 0)
    assert cell[("j", "agreement at 0.8", "accept")] == (2, 2)
    assert cell[("j", "agreement at 0.8", "reject")] == (1, 0)
    assert cell[("j", "no agreement at 0.5", "accept")] == (1, 0)


def test_chain_totals_pay_a_reused_model_once_and_the_judge_every_time():
    import numpy as np

    def stage(passed, right, cost):
        return {"passed": np.array(passed), "right": np.array(right), "cost": np.array(cost, dtype=float)}

    own = {"a": np.array([1.0, 1.0, 1.0]), "b": np.array([2.0, 2.0, 2.0]), "c": np.array([3.0, 3.0, 3.0])}
    # a+b costs 3 plus a judge call of 0.5 on tasks 2 and 3; a+c costs 4 plus 0.5 on task 3.
    table = {"a+b": stage([True, False, False], [True, False, False], [3.0, 3.5, 3.5]),
             "a+c": stage([True, True, False], [True, False, False], [4.0, 4.0, 4.5]),
             "big": stage([True, True, True], [True, True, False], [100.0, 100.0, 100.0])}
    charged = swe.chain_totals(["a+b", "a+c"], table, own, "big", pay_once=False)
    once = swe.chain_totals(["a+b", "a+c"], table, own, "big")
    # Task 1 stops at a+b. Task 2 goes on to a+c, which accepts a wrong patch. Task 3 reaches the frontier.
    assert charged["savings"] == pytest.approx(1 - (3 + (3.5 + 4) + (3.5 + 4.5 + 100)) / 300)
    # Paid once, a+c costs c's answer only, and its judge call on task 3.
    assert once["savings"] == pytest.approx(1 - (3 + (3.5 + 3) + (3.5 + 3.5 + 100)) / 300)
    assert once["accuracy"] == charged["accuracy"] == pytest.approx(1 / 3)
    assert once["degradation_rate"] == pytest.approx(1 / 3)  # task 2: the frontier was right, the chain was not
    # A chain with no shared model costs the same either way, and so does a subset of rows.
    assert swe.chain_totals(["a+b"], table, own, "big") == swe.chain_totals(["a+b"], table, own, "big", pay_once=False)
    rows = np.array([False, True, True])
    assert swe.chain_totals(["a+b", "a+c"], table, own, "big", rows)["savings"] == pytest.approx(
        1 - ((3.5 + 3) + (3.5 + 3.5 + 100)) / 200)
    assert swe.chain_totals([], table, own, "big")["savings"] == 0


def random_inputs(tasks=90, seed=3):
    """Graded answers, free checks, patches and verdicts for three cheap models on many
    tasks, so a search has enough tasks for its three parts."""
    import numpy as np

    rng = np.random.default_rng(seed)
    costs = {"a": 1.0, "b": 2.0, "c": 3.0, "big": 40.0}
    rows, free, cases, judge, right = [], {}, {}, {}, set()
    for number in range(tasks):
        task = f"t{number:03d}"
        shared = rng.random() < 0.3  # a and b sometimes write the same patch
        for model in ("a", "b", "c"):
            free[(task, model)] = bool(rng.random() < 0.7)
            case = f"{task}-ab" if shared and model in "ab" else f"{task}-{model}"
            cases[(task, model)] = case
            if rng.random() < 0.5:
                right.add(case)
            judge[case] = (bool(rng.random() < (0.85 if case in right else 0.25)), 0.25)
        for model, cost in costs.items():
            success = bool(rng.random() < 0.6) if model == "big" else cases[(task, model)] in right
            rows.append({"tenant": "t", "task_id": task, "prompt": f"Fix {task}", "task_type": "code",
                         "model": model, "success": success, "cost_usd": cost * (1 + rng.random())})
    return pd.DataFrame(rows), free, cases, judge


def test_paid_once_search_reproduces_the_cascade_when_charged_the_same_way(tmp_path):
    from tine.cascade import run_cascade_experiment

    own, free, cases, judge = random_inputs()
    frame, checks, _, _ = swe.build_groups(own, free, cases, judge, "big", group_sizes=(1, 2))
    options = {"epsilons": (0.05, 0.2), "seeds": range(4), "fractions": (0.4, 0.3, 0.3)}
    run_cascade_experiment(frame, "big", tmp_path / "cascade", checks=pd.DataFrame(checks), **options)
    theirs = pd.read_csv(tmp_path / "cascade" / "cascade-runs.csv").set_index(["seed", "epsilon", "method"])
    mine = pd.DataFrame(swe.paid_once_search(frame, checks, own, "big", pay_once=False, **options))
    assert len(mine) == len(theirs) == 16
    assert (theirs.chain != "big").any()  # the comparison is not only of empty chains
    for row in mine.itertuples():
        their = theirs.loc[(row.seed, row.epsilon, row.method)]
        assert row.chain == their.chain
        assert row.savings == pytest.approx(their.savings) and row.accuracy == pytest.approx(their.accuracy)
        assert row.degradation_rate == pytest.approx(their.degradation_rate)
    summary = swe.summarise_search(mine.to_dict("records"), "big")
    their_summary = pd.read_csv(tmp_path / "cascade" / "cascade-summary.csv")
    assert summary.savings_mean.to_numpy() == pytest.approx(their_summary.savings_mean.to_numpy())
    assert list(summary.frontier_only_runs) == list(their_summary.frontier_only_runs)
    assert list(summary.over_epsilon_runs) == list(their_summary.over_epsilon_runs)


def test_paid_once_search_never_pays_more_for_the_same_chain_and_needs_member_costs():
    own, free, cases, judge = random_inputs()
    frame, checks, _, _ = swe.build_groups(own, free, cases, judge, "big", group_sizes=(2,))
    options = {"epsilons": (0.2,), "seeds": range(4)}
    once = pd.DataFrame(swe.paid_once_search(frame, checks, own, "big", **options))
    charged = pd.DataFrame(swe.paid_once_search(frame, checks, own, "big", pay_once=False, **options))
    # Three models make three pairs, so any chain of two pairs shares a model.
    assert once.reuses_a_model.any()
    same = once.chain == charged.chain
    assert same.any() and (once.savings[same] >= charged.savings[same] - 1e-12).all()
    assert (once.savings[same & once.reuses_a_model] > charged.savings[same & once.reuses_a_model]).all()
    with pytest.raises(ValueError, match="lack a cost"):
        swe.paid_once_search(frame, checks, own[own.model != "b"], "big", **options)


def test_judge_arguments_need_a_plain_name():
    assert swe.parse_judges(["haiku=v.csv", "sonnet-5=runs/a=b.csv"]) == {"haiku": "v.csv", "sonnet-5": "runs/a=b.csv"}
    for bad in ("v.csv", "../up=v.csv", "=v.csv", "name="):
        with pytest.raises(SystemExit, match="NAME=VERDICTS.csv"):
            swe.parse_judges([bad])
