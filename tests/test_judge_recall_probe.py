import csv
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "tine_judge_recall_probe", Path(__file__).parents[1] / "scripts" / "judge_recall_probe.py")
recall = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recall)

REFERENCE = "--- a/x.py\n+++ b/x.py\n@@ -1,3 +1,3 @@\n def f():\n-    return  1\n+    return 2\n+    )\n"


def test_overlap_counts_reference_lines_reproduced_ignoring_spacing_and_headers():
    assert recall.changed_lines(REFERENCE) == {"-return 1", "+return 2"}
    assert recall.overlap("+    return 2\n-return 1", REFERENCE) == 1.0
    assert recall.overlap("+    return 3\n-return 1", REFERENCE) == 0.5
    assert recall.overlap("nothing", "no changes") == 0.0
    assert recall.extract_diff("text\n```diff\n+a\n```\nmore\n```\n+b\n```") == "+b\n"


def test_score_splits_judge_accuracy_by_recall(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("".join(json.dumps({"instance_id": iid, "patch": REFERENCE, "difficulty": "easy"}) + "\n"
                               for iid in ("t1", "t2")))
    replies = tmp_path / "replies.jsonl"
    replies.write_text(json.dumps({"instance_id": "t1", "reply": "+return 2\n-return 1"}) + "\n"
                       + json.dumps({"instance_id": "t2", "reply": "+something else"}) + "\n")
    key = tmp_path / "key.csv"
    with key.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case_id", "instance_id", "run_resolved"])
        writer.writerows([["a", "t1", "True"], ["b", "t1", "False"], ["c", "t2", "True"], ["d", "t2", "False"]])
    verdicts = tmp_path / "verdicts.csv"
    verdicts.write_text("case_id,verdict,probability\na,accept,0.9\nb,reject,0.1\nc,reject,0.2\nd,accept,0.8\n")
    result = recall.score(dataset, replies, key, verdicts)
    assert result["tasks"] == 2 and result["tasks_recalled"] == 1
    rows = {(row["difficulty"], row["group"]): row for row in result["table"]}
    assert rows[("all", "recalled")]["catch_rate"] == 1.0 and rows[("all", "recalled")]["false_reject_rate"] == 0.0
    assert rows[("all", "not recalled")]["catch_rate"] == 0.0 and rows[("all", "not recalled")]["false_reject_rate"] == 1.0
