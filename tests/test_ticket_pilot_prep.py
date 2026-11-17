import csv
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "ticket_pilot_prep", Path(__file__).resolve().parents[1] / "scripts" / "ticket_pilot_prep.py")
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)

CATEGORIES = ["Fileservice", "O365", "Software"]


def write_data(folder: Path) -> None:
    tickets = [(index, f"request {index % 20} about {CATEGORIES[index % 3]}", CATEGORIES[index % 3])
               for index in range(1, 91)]
    for part, rows in (("train", tickets[:60]), ("test", tickets[60:])):
        with (folder / f"X_{part}.csv").open("w", newline="") as stream:
            csv.writer(stream).writerows([("id", "text"), *((i, text) for i, text, _ in rows)])
        with (folder / f"y_{part}.csv").open("w", newline="") as stream:
            csv.writer(stream).writerows([("id", "category_truth"), *((i, label) for i, _, label in rows)])


def test_sets_are_disjoint_and_examples_are_used_nowhere_else(tmp_path):
    write_data(tmp_path)
    counts = prep.prepare(tmp_path, tmp_path / "sets", sizes={"calibrate": 40, "shadow": 20, "live": 15},
                          examples_per_category=2)
    assert (counts["examples"], counts["calibrate"], counts["shadow"], counts["live"], counts["unused"]) == (6, 40, 20, 15, 9)
    assert counts["letters"] == {"Fileservice": "A", "O365": "B", "Software": "C"}
    sets = {name: [json.loads(line) for line in (tmp_path / "sets" / f"tickets-{name}.jsonl").read_text().splitlines()]
            for name in ("calibrate", "shadow", "live")}
    ids = [row["task_id"] for rows in sets.values() for row in rows]
    prompts = [row["prompt"] for rows in sets.values() for row in rows]
    assert len(set(ids)) == len(ids) == 75 and len(set(prompts)) == 75
    row = sets["shadow"][0]
    assert row["dataset"] == "tickets" and row["ground_truth"] == counts["letters"][row["category"]]
    assert row["prompt"].count("Category: ") == 6 and row["prompt"].rstrip().endswith("is one of ABC.")
    assert f"Ticket {row['task_id'].split(':')[1]}: " in row["prompt"]
    # With 20 distinct texts over 90 tickets, most new tickets repeat a text already seen.
    assert counts["shadow_text_seen_before"] > 0 and all(not r["text_seen_before"] for r in sets["calibrate"])
