import importlib.util
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from tine.cascade import run_cascade_experiment
from tine.checks import read_checks
from tine.data import read_outcomes
from tine.policy import canonical_answer

spec = importlib.util.spec_from_file_location(
    "tine_agreement", Path(__file__).parents[1] / "scripts" / "agreement_pairs.py")
agreement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agreement)

# Per task: the right answer, then each model's answer. "big" is the frontier.
ANSWERS = {"cheap": "AABCA", "small": "AABDB", "tiny": "ABBD ", "big": "AABCC"}
TRUTH = "AABCC"
COSTS = {"cheap": 1.0, "small": 2.0, "tiny": 0.5, "big": 10.0}


def outcomes_frame():
    rows = []
    for model, answers in ANSWERS.items():
        for index, answer in enumerate(answers, 1):
            rows.append({"tenant": "t", "task_id": f"quiz:test:{index}", "prompt": f"Question {index}?",
                         "task_type": "quiz", "model": model, "success": answer == TRUTH[index - 1],
                         "cost_usd": COSTS[model], "input_tokens": 10, "output_tokens": 5})
    return pd.DataFrame(rows)


def recorded():
    return {(f"quiz:test:{index}", model): agreement.normalize(answer)
            for model, answers in ANSWERS.items() for index, answer in enumerate(answers, 1)}


def test_normalize_treats_blank_and_placeholder_answers_as_missing():
    assert agreement.normalize(" B \n") == "B" and agreement.normalize([[1, 2]]) == "[[1, 2]]"
    for missing in [None, "", "   ", "None", "[None]"]:
        assert agreement.normalize(missing) is None


def test_pairs_cost_both_calls_and_pass_only_when_answers_match():
    frame, checks, report = agreement.build_pairs(outcomes_frame(), recorded(), "big", max_cost_share=0.5)
    assert report["pairs"] == 3 and set(frame.model) == {"big", "cheap+small", "cheap+tiny", "small+tiny"}
    pair = frame[frame.model == "cheap+small"].set_index("task_id")
    assert (pair.cost_usd == 3.0).all() and (pair.input_tokens == 20).all()
    passed = {(row["task_id"], row["model"]): row["passed"] for row in checks}
    assert [passed[(f"quiz:test:{n}", "cheap+small")] for n in range(1, 6)] == [True, True, True, False, False]
    # tiny gave no answer on task 5: a missing answer never agrees.
    assert passed[("quiz:test:5", "small+tiny")] is False
    # Where the pair agrees its recorded success is the shared one.
    assert pair.success.tolist()[:3] == [True, True, True]
    assert report["agreement_rate"]["cheap+small"] == 0.6


def test_triples_require_every_member_and_respect_the_cost_cap():
    frame, checks, report = agreement.build_pairs(outcomes_frame(), recorded(), "big", group_size=3)
    assert report["pairs"] == 1 and "cheap+small+tiny" in set(frame.model)
    passed = [row["passed"] for row in checks]
    assert passed == [True, False, True, False, False]
    assert (frame[frame.model == "cheap+small+tiny"].cost_usd == 3.5).all()
    with pytest.raises(ValueError, match="cheap enough"):
        agreement.build_pairs(outcomes_frame(), recorded(), "big", max_cost_share=0.1)
    for bad in [{"frontier": "nobody"}, {"max_cost_share": 0}, {"group_size": 1}]:
        with pytest.raises(ValueError):
            agreement.build_pairs(outcomes_frame(), recorded(), **({"frontier": "big"} | bad))


def test_agreement_builder_rejects_task_ids_shared_across_tenants():
    first = outcomes_frame()
    second = first.assign(tenant="another-tenant")
    with pytest.raises(ValueError, match="one tenant"):
        agreement.build_pairs(pd.concat([first, second]), recorded(), "big")


def test_agreeing_answers_with_conflicting_grades_are_rejected():
    frame = outcomes_frame()
    frame.loc[(frame.model == "cheap") & (frame.task_id == "quiz:test:1"), "success"] = False
    with pytest.raises(ValueError, match="Conflicting grades"):
        agreement.build_pairs(frame, recorded(), "big")


def test_check_rows_carry_the_evaluated_tenant():
    _, checks, _ = agreement.build_pairs(outcomes_frame(), recorded(), "big")
    assert {row["tenant"] for row in checks} == {"t"}


def test_pending_group_grade_is_not_filled_from_another_model():
    frame = outcomes_frame()
    frame["success"] = frame.success.astype(object)
    frame.loc[(frame.model == "cheap") & (frame.task_id == "quiz:test:1"), "success"] = None
    paired, checks, _ = agreement.build_pairs(frame, recorded(), "big")
    row = paired[(paired.model == "cheap+small") & (paired.task_id == "quiz:test:1")].iloc[0]
    assert pd.isna(row.success)
    assert next(row["passed"] for row in checks if row["task_id"] == "quiz:test:1" and row["model"] == "cheap+small")


def test_recorded_answers_must_match_the_graded_source_hash(tmp_path):
    folder = tmp_path / "cheap"
    folder.mkdir()
    path = folder / "run.json"
    data = {"dataset_name": "quiz", "split": "test", "model_name": "cheap",
            "records": [{"index": 1, "prediction": "A"}]}
    path.write_text(json.dumps(data))
    frame = outcomes_frame()
    frame["source_sha256"] = None
    frame.loc[frame.model == "cheap", "source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert agreement.recorded_answers(tmp_path, outcomes=frame)[("quiz:test:1", "cheap")] == "A"
    data["records"][0]["prediction"] = "B"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source_sha256"):
        agreement.recorded_answers(tmp_path, outcomes=frame)


def test_duplicate_recorded_answers_cannot_overwrite_each_other(tmp_path):
    folder = tmp_path / "cheap"
    folder.mkdir()
    (folder / "run.json").write_text(json.dumps({
        "dataset_name": "quiz", "split": "test", "model_name": "cheap",
        "records": [{"index": 1, "prediction": "A"}, {"index": 1, "prediction": "B"}]}))
    with pytest.raises(ValueError, match="Duplicate recorded answer"):
        agreement.recorded_answers(tmp_path)


@pytest.mark.parametrize("contents", [
    "task_id,model,answer\nquiz:1,cheap,A\nquiz:1,cheap,B\n",
    "task_id,model,answer,answer\nquiz:1,cheap,A,B\n",
    "task_id,model,answer\nquiz:1,cheap,A,extra\n",
    "task_id,model,answer\nquiz:1,cheap\n",
    "task_id,model,answer\n,cheap,A\n",
    "task_id,model,answer\nquiz:1, ,A\n",
    "task_id,model\nquiz:1,cheap\n",
])
def test_csv_answers_reject_ambiguous_or_incomplete_evidence(tmp_path, contents):
    path = tmp_path / "answers.csv"
    path.write_text(contents)
    with pytest.raises(ValueError):
        agreement.answers_from_csv(path)


@pytest.mark.parametrize("output", ["--output", "--checks"])
def test_cli_refuses_to_overwrite_csv_answers(tmp_path, output):
    source = tmp_path / "answers.csv"
    source.write_text("preserve source")
    paths = {"--output": tmp_path / "groups.jsonl", "--checks": tmp_path / "checks.csv"}
    paths[output] = source
    with pytest.raises(SystemExit) as error:
        agreement.main(["--input", str(tmp_path / "missing.jsonl"), "--answers", str(source),
                        "--frontier", "big", "--output", str(paths["--output"]),
                        "--checks", str(paths["--checks"])])
    assert error.value.code == 2
    assert source.read_text() == "preserve source"


def test_csv_answers_reject_multiple_tenants_even_when_keys_differ(tmp_path):
    path = tmp_path / "answers.csv"
    path.write_text("tenant,task_id,model,answer\na,one,cheap,A\nb,two,cheap,B\n")
    with pytest.raises(ValueError, match="tenant"):
        agreement.answers_from_csv(path)


def test_csv_answers_preserve_literal_text_quoted_answers_and_missing_values(tmp_path):
    path = tmp_path / "answers.csv"
    path.write_text('\ufeff\n \ntenant,task_id,model,answer\nt,NA,001," A,\nB "\nt,002,null,\n')
    assert agreement.answers_from_csv(path, tenant="t") == {("NA", "001"): "A, B", ("002", "null"): None}


@pytest.mark.parametrize("tenant", ["other-tenant", ""])
def test_cli_csv_tenant_must_match_outcomes_before_outputs_are_written(tmp_path, tenant):
    source = tmp_path / "outcomes.jsonl"
    outcomes_frame().to_json(source, orient="records", lines=True)
    answers = tmp_path / "answers.csv"
    answers.write_text(f"tenant,task_id,model,answer\n{tenant},quiz:test:1,cheap,A\n")
    output, checks = tmp_path / "groups.jsonl", tmp_path / "checks.csv"
    output.write_text("previous groups")
    checks.write_text("previous checks")
    with pytest.raises(ValueError, match="tenant"):
        agreement.main(["--input", str(source), "--answers", str(answers), "--frontier", "big",
                        "--output", str(output), "--checks", str(checks)])
    assert output.read_text() == "previous groups"
    assert checks.read_text() == "previous checks"


@pytest.mark.parametrize("collision", ["frontier", "another-group"])
def test_generated_group_identifiers_must_not_collide(collision):
    frame = outcomes_frame()
    frontier = "big"
    if collision == "frontier":
        frontier = "cheap+small"
        frame["model"] = frame.model.replace({"big": frontier})
    else:
        extra = frame.loc[frame.model == "tiny"].assign(model="c")
        frame["model"] = frame.model.replace({"cheap": "a", "small": "b+c", "tiny": "a+b"})
        frame = pd.concat([frame, extra])
    with pytest.raises(ValueError, match="collid"):
        agreement.build_pairs(frame, {}, frontier)


@pytest.mark.parametrize("pair", [("input", "output"), ("input", "checks"), ("output", "checks")])
def test_cli_refuses_output_paths_that_overwrite_inputs_or_each_other(tmp_path, pair):
    paths = {name: tmp_path / f"{name}.jsonl" for name in ["input", "output", "checks"]}
    paths[pair[1]] = paths[pair[0]]
    paths["input"].write_text("preserve source")
    with pytest.raises(SystemExit) as error:
        agreement.main(["--input", str(paths["input"]), "--bench-dir", str(tmp_path / "bench"),
                        "--frontier", "big", "--output", str(paths["output"]),
                        "--checks", str(paths["checks"])])
    assert error.value.code == 2
    assert paths["input"].read_text() == "preserve source"


def test_cli_output_feeds_the_cascade(tmp_path):
    rng_frame = []
    import numpy as np
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 4, 300)
    bench = tmp_path / "bench"
    for model, accuracy, cost in [("a", 0.8, 1.0), ("b", 0.8, 1.0), ("big", 0.95, 10.0)]:
        answers = np.where(rng.random(300) < accuracy, truth, (truth + 1 + rng.integers(0, 3, 300)) % 4)
        (bench / model).mkdir(parents=True)
        (bench / model / "run.json").write_text(json.dumps({
            "dataset_name": "quiz", "split": "test", "model_name": model,
            "records": [{"index": n + 1, "prediction": "ABCD"[answer]} for n, answer in enumerate(answers)]}))
        for n, answer in enumerate(answers):
            rng_frame.append({"tenant": "t", "task_id": f"quiz:test:{n + 1}", "prompt": f"Question {n}?",
                              "task_type": "quiz", "model": model, "success": bool(answer == truth[n]),
                              "cost_usd": cost, "input_tokens": 10, "output_tokens": 5})
    source = tmp_path / "outcomes.jsonl"
    pd.DataFrame(rng_frame).to_json(source, orient="records", lines=True)
    agreement.main(["--input", str(source), "--bench-dir", str(bench), "--frontier", "big",
                    "--output", str(tmp_path / "pairs.jsonl"), "--checks", str(tmp_path / "agree.csv")])
    result = run_cascade_experiment(read_outcomes(tmp_path / "pairs.jsonl"), "big", tmp_path / "run",
                                    checks=read_checks(tmp_path / "agree.csv"), seeds=range(2), epsilons=(0.05,))
    assert result["check_mode"] == "measured" and result["models"] == ["a+b", "big"]
    assert result["check_coverage"]["known_checks"] == 300


def test_singles_are_offered_unchecked_and_answers_can_come_from_a_csv(tmp_path):
    frame, checks, report = agreement.build_pairs(outcomes_frame(), recorded(), "big", max_cost_share=0.3,
                                                  include_singles=True)
    assert report["include_singles"] and {"cheap", "small", "tiny"} <= set(frame.model)
    alone = [row for row in checks if row["model"] == "tiny"]
    assert len(alone) == 5 and all(row["passed"] for row in alone)
    path = tmp_path / "answers.csv"
    pd.DataFrame([{"task_id": task_id, "model": model, "answer": answer or ""}
                  for (task_id, model), answer in recorded().items()]).to_csv(path, index=False)
    assert agreement.answers_from_csv(path) == recorded()


def test_singles_are_still_offered_when_no_pair_is_cheap_enough(tmp_path, capsys):
    # The cheapest pair, cheap+tiny, costs 1.5 of the frontier's 10: over a 10% share.
    with pytest.raises(ValueError, match=r"No pair of models is cheap enough \(the cheapest costs 15\.0% of the "
                                         r"frontier, over max_cost_share 10\.0%\)"):
        agreement.build_pairs(outcomes_frame(), recorded(), "big", max_cost_share=0.1)
    frame, checks, report = agreement.build_pairs(outcomes_frame(), recorded(), "big", max_cost_share=0.1,
                                                  include_singles=True)
    assert set(frame.model) == {"big", "cheap", "tiny"} and report["pairs"] == 0 and report["agreement_rate"] == {}
    assert report["cheapest_pair_cost_share"] == 0.15 and "15.0% of the frontier" in report["no_pair_cheap_enough"]
    assert len(checks) == 10 and all(row["passed"] is True for row in checks)
    # The single models' rows still say whether each answer equals the frontier's.
    assert report["frontier_answers"] > 0 and all(isinstance(row["matches_frontier"], bool) for row in checks)
    # With no single model under the share either there is nothing to offer.
    with pytest.raises(ValueError, match="cheap enough"):
        agreement.build_pairs(outcomes_frame(), recorded(), "big", max_cost_share=0.01, include_singles=True)
    # When a pair is cheap enough the report says nothing about it, as before.
    usual = agreement.build_pairs(outcomes_frame(), recorded(), "big", include_singles=True)[2]
    assert "no_pair_cheap_enough" not in usual and "cheapest_pair_cost_share" not in usual
    # The command says so in the report it prints, and writes inputs the cascade reads.
    outcomes_frame().to_json(tmp_path / "outcomes.jsonl", orient="records", lines=True)
    pd.DataFrame([{"task_id": task_id, "model": model, "answer": answer or ""}
                  for (task_id, model), answer in recorded().items()]).to_csv(tmp_path / "answers.csv", index=False)
    agreement.main(["--input", str(tmp_path / "outcomes.jsonl"), "--answers", str(tmp_path / "answers.csv"),
                    "--frontier", "big", "--output", str(tmp_path / "singles.jsonl"),
                    "--checks", str(tmp_path / "singles.csv"), "--max-cost-share", "0.1", "--include-singles"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["pairs"] == 0 and printed["cheapest_pair_cost_share"] == 0.15
    assert set(read_outcomes(tmp_path / "singles.jsonl").model) == {"big", "cheap", "tiny"}
    assert len(read_checks(tmp_path / "singles.csv")) == 10
    assert printed["frontier_answers"] == report["frontier_answers"]
    assert pd.read_csv(tmp_path / "singles.csv").matches_frontier.notna().all()


def test_math_comparison_agrees_by_value_and_the_default_keeps_every_result(tmp_path):
    """The same answers compared exactly and by value: exact reproduces what this
    script always did, math also accepts one value written two ways."""
    written = {"cheap": [r"14.5\%", r"\dfrac{1}{2}", "7", "3", "1,234"],
               "small": ["14.50", r"\frac{1}{2}", "7", "4", "1234"],
               "tiny": ["14.5", "0.5", "8", "3", "None"],
               "big": ["14.5", r"\frac{1}{2}", "7", "3", "1234"]}
    answers = {(f"quiz:test:{index}", model): agreement.normalize(answer)
               for model, values in written.items() for index, answer in enumerate(values, 1)}
    # Graded by value against the frontier, which is right on every task here.
    frame = pd.DataFrame([
        {"tenant": "t", "task_id": f"quiz:test:{index}", "prompt": f"Question {index}?", "task_type": "quiz",
         "model": model, "success": canonical_answer(answer, "math") == canonical_answer(written["big"][index - 1],
                                                                                         "math"),
         "cost_usd": COSTS[model], "input_tokens": 10, "output_tokens": 5}
        for model, values in written.items() for index, answer in enumerate(values, 1)])
    exact_frame, exact_checks, exact = agreement.build_pairs(frame, answers, "big")
    default_frame, default_checks, _ = agreement.build_pairs(frame, answers, "big", answer_compare="exact")
    assert exact["answer_compare"] == "exact" and exact_checks == default_checks
    assert exact_frame.equals(default_frame)
    passed = {(row["task_id"], row["model"]): row["passed"] for row in exact_checks}
    assert [passed[(f"quiz:test:{n}", "cheap+small")] for n in range(1, 6)] == [False, False, True, False, False]
    _, math_checks, by_value = agreement.build_pairs(frame, answers, "big", answer_compare="math")
    passed = {(row["task_id"], row["model"]): row["passed"] for row in math_checks}
    assert [passed[(f"quiz:test:{n}", "cheap+small")] for n in range(1, 6)] == [True, True, True, False, True]
    # A fraction and a decimal are not read as one value; a missing answer never agrees.
    assert passed[("quiz:test:2", "cheap+tiny")] is False and passed[("quiz:test:5", "small+tiny")] is False
    assert by_value["answer_compare"] == "math" and by_value["agreement_rate"]["cheap+small"] == 0.8
    with pytest.raises(ValueError, match="answer_compare"):
        agreement.build_pairs(frame, answers, "big", answer_compare="numeric")
    # The command line takes the same choice.
    source, csv_answers = tmp_path / "outcomes.jsonl", tmp_path / "answers.csv"
    frame.to_json(source, orient="records", lines=True)
    pd.DataFrame([{"task_id": task_id, "model": model, "answer": answer or ""}
                  for (task_id, model), answer in answers.items()]).to_csv(csv_answers, index=False)
    common = ["--input", str(source), "--answers", str(csv_answers), "--frontier", "big"]
    for mode, expected in (("exact", 1), ("math", 4)):
        checks = tmp_path / f"{mode}.csv"
        agreement.main([*common, "--output", str(tmp_path / f"{mode}.jsonl"), "--checks", str(checks),
                        "--answer-compare", mode])
        rows = pd.read_csv(checks)
        assert rows[rows.model == "cheap+small"].passed.sum() == expected
    agreement.main([*common, "--output", str(tmp_path / "default.jsonl"), "--checks", str(tmp_path / "default.csv")])
    assert (tmp_path / "default.csv").read_text() == (tmp_path / "exact.csv").read_text()
    with pytest.raises(SystemExit):
        agreement.main([*common, "--output", str(tmp_path / "bad.jsonl"), "--checks", str(tmp_path / "bad.csv"),
                        "--answer-compare", "numeric"])


def test_grade_conflicts_can_be_counted_as_wrong_instead_of_refused():
    frame = outcomes_frame()
    # cheap and small both answer "A" on task 1; make their grades disagree.
    frame.loc[(frame.model == "small") & (frame.task_id == "quiz:test:1"), "success"] = False
    with pytest.raises(ValueError, match="Conflicting grades"):
        agreement.build_pairs(frame, recorded(), "big", max_cost_share=0.5)
    pairs, checks, report = agreement.build_pairs(frame, recorded(), "big", max_cost_share=0.5,
                                                  grade_conflicts="wrong")
    assert report["grade_conflicts_counted_wrong"] >= 1
    row = pairs[(pairs.model == "cheap+small") & (pairs.task_id == "quiz:test:1")].iloc[0]
    assert not row.success and all("tenant" in check for check in checks)


def test_check_rows_say_whether_the_candidates_answer_equals_the_frontiers(tmp_path):
    """For the drift baseline: a group's answer is its first member's, and it is compared
    with the frontier's as shadow mode compares an accepted answer."""
    def matches(rows, model):
        found = {row["task_id"]: row["matches_frontier"] for row in rows if row["model"] == model}
        return [found[f"quiz:test:{index}"] for index in range(1, 6)]

    frame, checks, report = agreement.build_pairs(outcomes_frame(), recorded(), "big", max_cost_share=0.3,
                                                  include_singles=True)
    assert report["frontier_answers"] == 5 and all(set(row) == {
        "tenant", "task_id", "model", "passed", "matches_frontier"} for row in checks)
    # cheap answers AABCA, small AABDB, tiny ABBD and nothing; the frontier AABCC.
    assert matches(checks, "cheap+small") == matches(checks, "cheap+tiny") == [True, True, True, True, False]
    assert matches(checks, "small+tiny") == matches(checks, "small") == [True, True, True, False, False]
    assert matches(checks, "tiny") == [True, False, True, False, False]  # no answer is not the frontier's answer
    # A frontier answer that could not be read differs, as it does in shadow mode.
    unread = {**recorded(), ("quiz:test:1", "big"): None}
    _, checks, report = agreement.build_pairs(outcomes_frame(), unread, "big")
    assert report["frontier_answers"] == 4 and matches(checks, "cheap+small") == [False, True, True, True, False]
    # With no frontier answer at all nothing can be said, and nothing is.
    without = {key: answer for key, answer in recorded().items() if key[1] != "big"}
    with_frame, with_checks, _ = agreement.build_pairs(outcomes_frame(), recorded(), "big")
    without_frame, checks, report = agreement.build_pairs(outcomes_frame(), without, "big")
    assert report["frontier_answers"] == 0 and {row["matches_frontier"] for row in checks} == {None}
    # Whether the answers match decides nothing else: the groups and what passed are the same.
    assert without_frame.equals(with_frame)
    assert [row["passed"] for row in checks] == [row["passed"] for row in with_checks]
    # By value, one value written two ways is the frontier's answer.
    written = {"cheap": r"14.5\%", "small": "14.50", "big": "14.5"}
    answers = {("quiz:test:1", model): answer for model, answer in written.items()}
    one = outcomes_frame()
    one = one[(one.task_id == "quiz:test:1") & (one.model != "tiny")]
    assert agreement.build_pairs(one, answers, "big")[1][0]["matches_frontier"] is False
    assert agreement.build_pairs(one, answers, "big", answer_compare="math")[1][0]["matches_frontier"] is True


def test_the_checks_file_carries_the_match_column_and_leaves_unknown_blank(tmp_path):
    source, answers = tmp_path / "outcomes.jsonl", tmp_path / "answers.csv"
    outcomes_frame().to_json(source, orient="records", lines=True)
    rows = [{"task_id": task_id, "model": model, "answer": answer or ""}
            for (task_id, model), answer in recorded().items()]
    pd.DataFrame(rows).to_csv(answers, index=False)
    common = ["--input", str(source), "--frontier", "big", "--output", str(tmp_path / "pairs.jsonl")]
    agreement.main([*common, "--answers", str(answers), "--checks", str(tmp_path / "checks.csv")])
    text = (tmp_path / "checks.csv").read_text()
    assert text.splitlines()[0] == "tenant,task_id,model,passed,matches_frontier"
    assert "t,quiz:test:5,cheap+small,False,False" in text.splitlines()
    table = read_checks(tmp_path / "checks.csv")
    assert set(table.matches_frontier) == {"True", "False"}
    # The columns that were always written are written as before.
    kept = [",".join(line.split(",")[:4]) for line in text.splitlines()]
    assert kept[0] == "tenant,task_id,model,passed" and kept[1] == "t,quiz:test:1,cheap+small,True"
    pd.DataFrame([row for row in rows if row["model"] != "big"]).to_csv(answers, index=False)
    agreement.main([*common, "--answers", str(answers), "--checks", str(tmp_path / "blank.csv")])
    blank = read_checks(tmp_path / "blank.csv")
    assert set(blank.matches_frontier) == {""} and blank.passed.tolist() == table.passed.tolist()
