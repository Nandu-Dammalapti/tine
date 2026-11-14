import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "tine_train_verifier", Path(__file__).parents[1] / "scripts" / "train_verifier.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)

PATCH = "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1,2 +1,2 @@\n context\n-old = 1\n+{line}\n"


def write_cases(tmp_path, tasks=30):
    """Each task has two patches: the right one adds ``fixed_value``, the wrong one ``guess``."""
    cases, key = tmp_path / "cases.jsonl", tmp_path / "key.csv"
    with cases.open("w") as stream, key.open("w", newline="") as key_stream:
        writer = csv.writer(key_stream)
        writer.writerow(["case_id", "task_id", "instance_id", "patch_sha", "models", "benchmark_right", "run_resolved"])
        for task in range(tasks):
            for right in (True, False):
                case_id = f"c{task}-{int(right)}"
                line = f"fixed_value = {task}" if right else f"guess = {task}"
                stream.write(json.dumps({"case_id": case_id, "repository": "o/r", "issue": f"issue number {task}",
                                         "patch": PATCH.format(line=line)}) + "\n")
                writer.writerow([case_id, f"t:{task}", f"task-{task}", "sha", "a;b" if right else "c", right, right])
    return cases, key


def test_changed_lines_and_structure_features_read_the_patch():
    added, removed = verifier.changed_lines(PATCH.format(line="new = 2"))
    assert added == ["new = 2"] and removed == ["old = 1"]
    features = verifier.structure_features({"patch": PATCH.format(line="new = 2"), "issue": "x"})
    assert len(features) == 10 and features[6] == 0.0


def test_agreement_features_skip_patches_only_the_excluded_model_wrote():
    rows = [{"task": "t", "patch": PATCH.format(line="a = 1"), "models": ["cheap"]},
            {"task": "t", "patch": PATCH.format(line="a = 1"), "models": ["frontier"]},
            {"task": "u", "patch": PATCH.format(line="b = 1"), "models": ["cheap", "other"]}]
    assert verifier.agreement_features(rows)[0][2] == 1.0
    excluded = verifier.agreement_features(rows, frozenset({"frontier"}))
    assert excluded[0][1:] == [0.0, 0.0, 0.0]
    assert excluded[2][0] > 0  # another cheap model wrote the same patch


def test_grouped_folds_keep_a_task_together_and_threshold_respects_the_limit():
    tasks = ["a", "a", "b", "b", "c", "d"]
    folds = verifier.grouped_folds(tasks, 2, 0)
    assert folds[0] == folds[1] and folds[2] == folds[3] and set(folds) == {0, 1}
    probabilities, right = [0.1, 0.2, 0.3, 0.9, 0.05], [True, True, True, True, False]
    assert verifier.reject_threshold(probabilities, right, 0.25) == 0.2  # rejects one of four right patches
    assert verifier.reject_threshold(probabilities, right, 0.0) == 0.1


def test_train_learns_a_separable_signal_on_held_out_tasks(tmp_path):
    cases, key = write_cases(tmp_path)
    output = tmp_path / "verdicts.csv"
    report = verifier.train(cases, key, output, groups=["structure", "text"], folds=3, c=[0.1, 10.0],
                            subset_key=key)
    assert report["all"]["auc"] > 0.95 and report["subset"]["cases"] == 60 and report["tasks"] == 30
    rows = list(csv.DictReader(output.open()))
    assert len(rows) == 60 and {row["verdict"] for row in rows} <= {"accept", "reject"}
    assert rows[0]["judge"] == "trained:structure+text"
    assert json.loads(Path(str(output) + ".json").read_text())["grade"] == "run_resolved"


def test_encode_saves_one_row_per_case_in_case_order_and_train_reads_it(tmp_path):
    cases, key = write_cases(tmp_path, tasks=6)
    output = tmp_path / "hidden.npz"

    def states(texts):
        # A vector that identifies the case, so the saved order can be checked.
        marks = np.array([[float(text.split("issue number ")[1].split()[0]), float("fixed_value" in text)]
                          for text in texts])
        return {"last_1": marks, "mean_1": marks * 2}

    facts = verifier.encode(cases, output, model="fake", revision="0", layers=[1], max_tokens=100,
                            batch_size=5, device="cpu", states=states)
    assert facts["cases"] == 12 and facts["dimension"] == 2 and facts["arrays"] == ["last_1", "mean_1"]
    with np.load(output) as saved:
        assert saved["case_ids"].tolist()[:2] == ["c0-1", "c0-0"]
        assert saved["last_1"][:3].tolist() == [[0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]
    report = verifier.train(cases, key, tmp_path / "v.csv", groups=["hidden:last_1"], embeddings=output, folds=2)
    assert report["all"]["cases"] == 12 and report["all"]["auc"] == 1.0


def test_compare_writes_one_row_per_feature_set_and_subset(tmp_path):
    cases, key = write_cases(tmp_path)
    output = tmp_path / "summary.csv"
    verifier.main(["compare", "--cases", str(cases), "--key", str(key), "--output", str(output),
                   "--feature-sets", "structure+text", "agreement", "--seeds", "2", "--folds", "3",
                   "--subset-key", str(key), "--agreement-exclude", "b"])
    rows = list(csv.DictReader(output.open()))
    assert [(row["features"], row["cases_scored"]) for row in rows] == [
        ("structure+text", "all"), ("structure+text", "cheap"), ("structure+text", "subset"),
        ("agreement", "all"), ("agreement", "cheap"), ("agreement", "subset")]
    assert rows[1]["cases"] == "60" and float(rows[0]["auc"]) > 0.95
