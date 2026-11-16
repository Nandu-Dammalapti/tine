import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("tine_agreement_datasets", SCRIPTS / "agreement_datasets.py")
datasets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(datasets)

# Per task: the right answer, then each model's answer. "big" and "twin" are equally
# accurate; "free" has no recorded cost and is dropped on import.
TRUTH = "AABCCABDAB"
ANSWERS = {"big": "AABCCABAAB", "twin": "AABCCABAAB", "mid": "AABCDBBDAB", "cheap": "AABDCABDBB",
           "small": "AABDBABDAC", "tiny": "ABBD ABDAB", "free": "AABCCABDAB"}
COSTS = {"big": 10.0, "twin": 12.0, "mid": 4.0, "cheap": 1.0, "small": 2.0, "tiny": 0.5, "free": 0.0}


def outcomes_frame(models=("big", "twin", "mid", "cheap", "small", "tiny")):
    rows = []
    for model in models:
        for index, answer in enumerate(ANSWERS[model], 1):
            rows.append({"tenant": "t", "task_id": f"quiz:test:{index}", "prompt": f"Question {index}?",
                         "task_type": "quiz", "model": model, "success": answer == TRUTH[index - 1],
                         "cost_usd": COSTS[model], "input_tokens": 10, "output_tokens": 5})
    return pd.DataFrame(rows)


def recorded(models=tuple(ANSWERS)):
    return {(f"quiz:test:{index}", model): datasets.agreement_pairs.normalize(answer)
            for model in models for index, answer in enumerate(ANSWERS[model], 1)}


def test_frontier_is_the_most_accurate_model_and_ties_go_to_the_cheaper():
    assert datasets.pick_frontier(outcomes_frame()) == "big"
    assert datasets.pick_frontier(outcomes_frame(("mid", "cheap", "tiny"))) == "cheap"


def test_sources_cover_single_datasets_pools_and_the_reference_split():
    assert datasets.sources_for("gpqa") == {"gpqa": "test"}
    assert datasets.sources_for("pool-wide-small") == {"gpqa": "test", "aime": "hybrid", "livemathbench": "test"}
    assert datasets.sources_for("mmlupro-3000") == {"mmlupro": "test_3000"}
    assert all(dataset in datasets.DATASETS for pool in datasets.POOLS.values() for dataset in pool)


def test_models_are_picked_by_cost_alone():
    frame = outcomes_frame()
    # A quarter of the frontier's cost is 2.5: mid (4.0) is too dear although it is accurate.
    assert datasets.pick_models(frame, "big") == ["tiny", "cheap", "small"]
    assert datasets.pick_models(frame, "big", max_models=2) == ["tiny", "cheap"]
    assert datasets.pick_models(frame, "big", cost_share=0.05) == ["tiny"]
    assert "big" not in datasets.pick_models(frame, "big", cost_share=1.0)


def test_group_rates_count_agreement_losses_and_gains_against_the_frontier():
    rates = datasets.group_rates(outcomes_frame(), recorded(), "big", [("cheap", "small"), ("cheap", "tiny")])
    pair = rates.set_index("group").loc["cheap+small"]
    # cheap and small differ on tasks 5, 9 and 10; their agreed answer is wrong on task 4.
    assert pair.agreement_rate == pytest.approx(7 / 10)
    assert pair.accuracy_when_agree == pytest.approx(6 / 7)
    assert pair.degradation_rate == pytest.approx(1 / 10)  # task 4: frontier right, group wrong
    assert pair.gain_rate == pytest.approx(1 / 10)  # task 8: frontier wrong, group right
    assert pair.cost_share_of_frontier == pytest.approx(0.3)
    assert pair.grade_conflicts == 0
    # Both are right on six tasks and give the same string on each of them.
    assert pair.all_right_rate == pytest.approx(6 / 10) and pair.agree_when_all_right == pytest.approx(1.0)
    # A blank answer never agrees.
    assert rates.set_index("group").loc["cheap+tiny"].agreement_rate == pytest.approx(7 / 10)


def test_group_rates_count_differently_graded_agreeing_answers_as_wrong():
    frame = outcomes_frame()
    frame.loc[(frame.model == "small") & (frame.task_id == "quiz:test:1"), "success"] = False
    pair = datasets.group_rates(frame, recorded(), "big", [("cheap", "small")]).iloc[0]
    assert pair.grade_conflicts == 1
    assert pair.grade_conflict_share_of_agreed == pytest.approx(1 / 7)
    assert pair.accuracy_when_agree == pytest.approx(5 / 7)
    assert pair.degradation_rate == pytest.approx(2 / 10)


def test_passable_loss_rate_shows_how_much_room_a_certify_set_leaves():
    assert datasets.passable_loss_rate(50, 0.04) is None  # zero losses in 50 tasks bound only 5.8%
    assert datasets.passable_loss_rate(100, 0.04) == 0.0  # zero losses pass, one does not
    assert 0.025 < datasets.passable_loss_rate(900, 0.04) < 0.035
    assert datasets.passable_loss_rate(900, 0.02) < datasets.passable_loss_rate(900, 0.04)
    assert datasets.passable_loss_rate(300, 0.04) < datasets.passable_loss_rate(900, 0.04)


def test_sample_tasks_keeps_every_model_for_a_seeded_subset_of_tasks():
    frame = outcomes_frame()
    assert datasets.sample_tasks(frame, None) is frame and datasets.sample_tasks(frame, 10) is frame
    sample = datasets.sample_tasks(frame, 4)
    assert sample.task_id.nunique() == 4 and set(sample.groupby("task_id").model.nunique()) == {6}
    assert sample.equals(datasets.sample_tasks(frame, 4))
    assert set(sample.task_id) != set(datasets.sample_tasks(frame, 4, seed=1).task_id)


def test_certify_room_compares_dataset_size_with_the_lowest_loss_rate():
    rates = datasets.group_rates(outcomes_frame(), recorded(), "big",
                                 [("cheap", "small"), ("cheap", "tiny"), ("cheap", "small", "tiny")])
    room = datasets.certify_room(3000, rates)
    assert room["certify_tasks"] == 900
    assert room["passable_loss_rate_0.02"] < room["passable_loss_rate_0.04"]
    assert room["lowest_pair_loss_rate"] == pytest.approx(1 / 10)
    assert room["highest_pair_agreement_rate"] == pytest.approx(7 / 10)
    assert room["lowest_triple_loss_rate"] == pytest.approx(1 / 10)
    empty = datasets.certify_room(100, pd.DataFrame())
    assert empty["lowest_pair_loss_rate"] is None and empty["passable_loss_rate_0.02"] is None


def test_group_rates_show_right_answers_spelled_differently():
    answers = recorded()
    answers[("quiz:test:1", "small")] = "option A"  # graded right, spelled another way
    pair = datasets.group_rates(outcomes_frame(), answers, "big", [("cheap", "small")]).iloc[0]
    assert pair.agreement_rate == pytest.approx(6 / 10)
    assert pair.agree_when_all_right == pytest.approx(5 / 6)


def test_build_retries_conflicting_grades_as_wrong_and_says_so(tmp_path):
    frame = outcomes_frame()
    _, _, report, _ = datasets.build(frame, recorded(), "big", ["cheap", "small"], tmp_path, "clean")
    assert "grade_conflicts_mode" not in report
    frame.loc[(frame.model == "small") & (frame.task_id == "quiz:test:1"), "success"] = False
    groups, _, report, paths = datasets.build(frame, recorded(), "big", ["cheap", "small"], tmp_path, "quiz")
    assert report["grade_conflicts_mode"] == "wrong" and report["grade_conflicts_counted_wrong"] == 1
    group = groups[groups.model == "cheap+small"].set_index("task_id")
    assert not group.at["quiz:test:1", "success"]
    assert all(path.exists() for path in paths)


def test_singles_only_keeps_single_models_and_the_frontier(tmp_path):
    frame, checks, _, _ = datasets.build(outcomes_frame(), recorded(), "big", ["cheap", "small"], tmp_path, "quiz",
                                         include_singles=True)
    single, single_checks = datasets.singles_only(frame, checks)
    assert set(single.model) == {"big", "cheap", "small"}
    assert {row["model"] for row in single_checks} == {"cheap", "small"}
    assert all(row["passed"] for row in single_checks)


def write_archive(root: Path, dataset: str = "quiz", costs=COSTS) -> None:
    """A tiny archive in the LLMRouterBench layout: one dataset, one split."""
    for model, answers in ANSWERS.items():
        records = [{"index": index, "origin_query": f"Question {index}?", "prompt": f"Question {index}?",
                    "prompt_tokens": 10, "completion_tokens": 5, "cost": costs[model],
                    "score": float(answer == TRUTH[(index - 1) % len(TRUTH)]), "prediction": answer.strip(),
                    "ground_truth": TRUTH[(index - 1) % len(TRUTH)], "raw_output": answer}
                   for index, answer in enumerate(answers * 5, 1)]
        folder = root / dataset / "test" / model
        folder.mkdir(parents=True)
        (folder / "run.json").write_text(json.dumps({
            "model_name": model, "dataset_name": dataset, "split": "test", "records": records,
            "performance": 0.0, "time_taken": 1.0, "prompt_tokens": 10, "completion_tokens": 5,
            "cost": costs[model] * len(records), "counts": len(records)}))


def test_run_dataset_end_to_end_on_a_tiny_archive(tmp_path, monkeypatch):
    write_archive(tmp_path / "bench")
    monkeypatch.setitem(datasets.DATASETS, "quiz", "test")
    done = datasets.run_dataset("quiz", {"quiz": "test"}, tmp_path / "bench", tmp_path / "work", seeds=2)
    info = done["info"]
    assert info["frontier"] == "big" and info["tasks"] == 50
    assert info["priced_models"] == 6  # "free" is unpriced and dropped
    assert info["models_paired"] == "tiny cheap small"
    assert info["pairs"] == 3 and info["triples"] == 1 and info["grade_conflict_mode"] == "none"
    certified = done["certified"]
    assert set(certified.kind) == {"pairs", "triples", "pairs+singles", "singles"}
    assert set(certified.epsilon) == {0.02, 0.04} and set(certified.tasks_used) == {50}
    assert (certified.savings_mean >= 0).all()
    summary = datasets.rate_summary(done["rates"], certified)
    assert list(summary["size"]) == [2, 3] and list(summary.groups) == [3, 1]
    models = done["models"].set_index("model")
    assert models.frontier.sum() == 1 and models.paired.sum() == 3
    assert not (tmp_path / "work" / "quiz-quads.jsonl").exists()


def test_a_pool_combines_datasets_and_reads_answers_from_each(tmp_path):
    write_archive(tmp_path / "bench", "quiz")
    write_archive(tmp_path / "bench", "exam")
    done = datasets.run_dataset("pool", {"quiz": "test", "exam": "test"}, tmp_path / "bench", tmp_path / "work",
                                seeds=2)
    assert done["info"]["tasks"] == 100 and done["info"]["split"] == "quiz/test exam/test"
    # Agreement is found in both halves, so answers were read from both folders.
    rates = done["rates"].set_index("group")
    assert rates.loc["cheap+small", "agreement_rate"] == pytest.approx(7 / 10)
    # The cascade keeps one task per identical prompt: the two halves ask the same questions.
    assert set(done["certified"].tasks_used) == {50}


def test_main_writes_only_small_summary_tables(tmp_path, monkeypatch):
    write_archive(tmp_path / "bench")
    monkeypatch.setitem(datasets.DATASETS, "quiz", "test")
    datasets.main(["--bench", str(tmp_path / "bench"), "--work", str(tmp_path / "work"),
                   "--results", str(tmp_path / "results"), "--datasets", "quiz", "--seeds", "2"])
    written = sorted(path.name for path in (tmp_path / "results").iterdir())
    assert written == ["agreement-rates.csv", "certified.csv", "datasets.csv", "models.csv"]
    for name in written:
        table = pd.read_csv(tmp_path / "results" / name)
        assert "task_id" not in table and "prompt" not in table and len(table) <= 10
    datasets.main(["--bench", str(tmp_path / "bench"), "--work", str(tmp_path / "work"), "--brief",
                   "--results", str(tmp_path / "brief"), "--datasets", "quiz", "--seeds", "2", "--label", "cut",
                   "--max-tasks", "30"])
    assert sorted(path.name for path in (tmp_path / "brief").iterdir()) == ["cut-certified.csv", "cut-datasets.csv"]
    assert pd.read_csv(tmp_path / "brief" / "cut-datasets.csv").tasks.tolist() == [30]


def test_run_label_names_a_sampled_run_after_its_cut():
    assert datasets.run_label("gpqa") == "gpqa"
    assert datasets.run_label("mmlupro-3000", 1000, 3) == "mmlupro-3000-1000-sample3"


def spoil_record(root: Path, model: str, index: int, **changes) -> None:
    """Change one record of one model's run in a tiny archive."""
    path = root / "quiz" / "test" / model / "run.json"
    data = json.loads(path.read_text())
    data["records"][index - 1].update(changes)
    path.write_text(json.dumps(data))


def test_tasks_the_importer_refuses_are_left_out_for_every_model(tmp_path):
    bench = tmp_path / "bench"
    write_archive(bench)
    assert datasets.refused_tasks(bench / "quiz" / "test") == set()
    spoil_record(bench, "big", 7, completion_tokens=-52, cost=0.0, score=0.0)
    spoil_record(bench, "tiny", 9, prompt_tokens="-3")
    spoil_record(bench, "mid", 12, origin_query="Question  12?")  # another spelling of the same prompt
    assert datasets.refused_tasks(bench / "quiz" / "test") == {"7", "9", "12"}
    done = datasets.run_dataset("quiz", {"quiz": "test"}, bench, tmp_path / "work", seeds=2)
    assert done["info"]["tasks"] == 47 and done["info"]["tasks_left_out"] == 3
    # The copy holds every model's run without those tasks; the archive is untouched.
    copy = json.loads((tmp_path / "work" / "clean" / "quiz" / "test" / "cheap" / "run.json").read_text())
    assert len(copy["records"]) == 47
    assert {"7", "9", "12"}.isdisjoint(str(record["index"]) for record in copy["records"])
    assert len(json.loads((bench / "quiz" / "test" / "cheap" / "run.json").read_text())["records"]) == 50
    # Answers are read from the same copy, so agreement is still found.
    assert done["rates"].set_index("group").loc["cheap+small", "agreement_rate"] > 0.5


def test_conflicting_grades_are_counted_in_the_info_row(tmp_path):
    bench = tmp_path / "bench"
    write_archive(bench)
    # On task 1 cheap, small and tiny all answer "A". Grade small's "A" wrong.
    spoil_record(bench, "small", 1, score=0.0)
    info = datasets.run_dataset("quiz", {"quiz": "test"}, bench, tmp_path / "work", seeds=2)["info"]
    assert info["grade_conflict_mode"] == "wrong"
    # cheap+small and small+tiny each hold the conflict; so does the one triple.
    assert info["pairs_grade_conflicts"] == 2 and info["triples_grade_conflicts"] == 1


def test_every_group_adds_a_certification_over_all_groups(tmp_path):
    write_archive(tmp_path / "bench")
    done = datasets.run_dataset("quiz", {"quiz": "test"}, tmp_path / "bench", tmp_path / "work", seeds=2,
                                every_group=True)
    certified = done["certified"]
    assert {"pairs-every-group", "triples-every-group"} <= set(certified.kind)
    assert set(certified[certified.kind == "pairs-every-group"].epsilon) == {0.02, 0.04}
    # One-group chains only: a certified pick names one group and the frontier.
    chains = certified[certified.kind.str.endswith("every-group")].most_common_chain
    assert all(chain.count(" > ") <= 1 for chain in chains)


def test_main_reports_each_sample_seed_and_a_dataset_with_nothing_to_pair(tmp_path, monkeypatch):
    write_archive(tmp_path / "bench")
    # In "flat" the most accurate model is also the cheapest, so no group can be built.
    write_archive(tmp_path / "bench", "flat", costs={**COSTS, "big": 0.1, "twin": 0.2})
    monkeypatch.setitem(datasets.DATASETS, "quiz", "test")
    monkeypatch.setitem(datasets.DATASETS, "flat", "test")
    common = ["--bench", str(tmp_path / "bench"), "--work", str(tmp_path / "work"), "--seeds", "2"]
    datasets.main([*common, "--results", str(tmp_path / "flat"), "--datasets", "flat"])
    info = pd.read_csv(tmp_path / "flat" / "datasets.csv")
    assert info.frontier.tolist() == ["big"] and info.pairs.tolist() == [0]
    assert pd.read_csv(tmp_path / "flat" / "certified.csv").empty
    datasets.main([*common, "--results", str(tmp_path / "cuts"), "--datasets", "quiz", "--brief",
                   "--max-tasks", "30", "--sample-seeds", "0", "1"])
    info = pd.read_csv(tmp_path / "cuts" / "datasets.csv")
    assert info.dataset.tolist() == ["quiz-30-sample0", "quiz-30-sample1"] and info.tasks.tolist() == [30, 30]
    assert set(pd.read_csv(tmp_path / "cuts" / "certified.csv").dataset) == set(info.dataset)
