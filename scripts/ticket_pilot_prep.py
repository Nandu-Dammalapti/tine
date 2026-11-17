#!/usr/bin/env python3
"""Question sets for a live pilot on one company's real IT support tickets.

The data is Zenodo record 7648117, "Classification of IT Support Tickets" (Leonardo
Santiago Benitez Pereira, CC BY 4.0): 2,229 tickets from one IT support company, each
given one of 7 categories by its IT staff, with personal details replaced by tags. Only
the four CSV files are read. The record's pickle and model files are never opened.

Each question is the prompt a company's triage step would send: the categories as
lettered options, a few past tickets per category as examples, and the new ticket. The
examples are tickets set aside for the prompt and used in no other set. The ticket's id
is in the prompt, as a real ticket's would be, which also keeps every prompt distinct
although many tickets share their text once names are replaced by tags.

Standard library only, so it can be run with ``python -I`` on the downloaded files.
The sets go under artifacts/; only summary tables are written for results.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

LETTERS = "ABCDEFG"
MAX_EXAMPLE_CHARS = 300
INSTRUCTIONS = """You route IT support tickets for a managed IT services company. Assign the new ticket below to exactly one category.

Categories:
{categories}

Past tickets and the category the support team gave them:

{examples}

The new ticket follows. Personal details in tickets were replaced by tags such as [NAME].

Ticket {ticket_id}: {text}

Reply with one line only, in the format 'Answer: $LETTER' (without quotes) where LETTER is one of {letters}."""


def read_tickets(folder: Path) -> list[dict]:
    tickets = []
    for part in ("train", "test"):
        with (folder / f"X_{part}.csv").open(newline="", encoding="utf-8") as stream:
            texts = {row["id"]: row["text"] for row in csv.DictReader(stream)}
        with (folder / f"y_{part}.csv").open(newline="", encoding="utf-8") as stream:
            labels = {row["id"]: row["category_truth"] for row in csv.DictReader(stream)}
        if texts.keys() != labels.keys():
            raise SystemExit(f"{part}: ticket ids of texts and labels differ")
        tickets += [{"id": key, "text": texts[key], "category": labels[key]} for key in sorted(texts, key=int)]
    return tickets


def prepare(folder: Path, output: Path, *, sizes: dict[str, int], examples_per_category: int = 5,
            seed: int = 0) -> dict:
    tickets = read_tickets(folder)
    categories = sorted({ticket["category"] for ticket in tickets})
    if len(categories) > len(LETTERS):
        raise SystemExit("more categories than letters")
    letter = {category: LETTERS[index] for index, category in enumerate(categories)}
    rng = random.Random(seed)
    rng.shuffle(tickets)
    chosen, rest, taken = [], [], Counter()
    for ticket in tickets:
        if taken[ticket["category"]] < examples_per_category:
            taken[ticket["category"]] += 1
            chosen.append(ticket)
        else:
            rest.append(ticket)
    rng.shuffle(chosen)
    example_text = "\n\n".join(
        f"Ticket: {' '.join(ticket['text'].split())[:MAX_EXAMPLE_CHARS]}\nCategory: {letter[ticket['category']]}"
        for ticket in chosen)
    category_text = "\n".join(f"{letter[category]}. {category}" for category in categories)

    def question(ticket: dict, seen: set[str]) -> dict:
        prompt = INSTRUCTIONS.format(categories=category_text, examples=example_text, ticket_id=ticket["id"],
                                     text=ticket["text"].strip(), letters=LETTERS[:len(categories)])
        return {"task_id": f"tickets:{ticket['id']}", "prompt": prompt, "ground_truth": letter[ticket["category"]],
                "dataset": "tickets", "category": ticket["category"],
                # Whether this exact text is also an example or in the graded history.
                "text_seen_before": ticket["text"] in seen}

    output.mkdir(parents=True, exist_ok=True)
    counts, start = {"examples": len(chosen)}, 0
    seen = {ticket["text"] for ticket in chosen}
    for name, size in sizes.items():
        part = rest[start:start + size]
        start += size
        rows = [question(ticket, set() if name == "calibrate" else seen) for ticket in part]
        (output / f"tickets-{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        counts[name] = len(rows)
        if name == "calibrate":
            seen |= {ticket["text"] for ticket in part}
        else:
            counts[f"{name}_text_seen_before"] = sum(row["text_seen_before"] for row in rows)
    counts["unused"] = len(rest) - start
    counts["letters"] = letter
    counts["prompt_chars_median"] = sorted(len(question(t, set())["prompt"]) for t in rest)[len(rest) // 2]
    return counts


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True, help="folder holding X_train.csv, y_train.csv, ...")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibrate", type=int, default=1200)
    parser.add_argument("--shadow", type=int, default=300)
    parser.add_argument("--live", type=int, default=400)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.data_dir, args.output,
                             sizes={"calibrate": args.calibrate, "shadow": args.shadow, "live": args.live}), indent=1))


if __name__ == "__main__":
    main()
