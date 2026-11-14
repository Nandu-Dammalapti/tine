#!/usr/bin/env python3
"""Train a small verifier on graded patches, and measure it on tasks it never saw.

Off-the-shelf open models up to 32B could not tell right patches from wrong ones
(results/2026-11-11-judge-pilot). This script tests the other route to a verifier a
customer can run: a small classifier trained on verification outcomes. It reads the
cases and key written by ``scripts/judge_patches.py prepare`` (an issue, a patch that
already passed the free checks, and whether the patch resolved the issue when the
tests were run). Nothing is executed and no model is called by ``train``.

``encode``  Read every case once with a frozen language model and keep hidden states
            (the last position and the mean over positions, at several layers) as an
            NPZ file keyed by case id. Needs a GPU. Nothing is generated or trained.
``compare`` Score several feature sets over several seeds of the grouped split and
            write one summary table: the comparison to quote.
``train``   Fit a logistic regression on chosen feature groups and score it by
            cross-validation grouped by task, so every prediction is for a task whose
            patches were all held out. Writes the held-out predictions in the verdict
            format of ``judge_patches.py``, so ``judge_patches.py score`` compares the
            trained verifier with every other judge on the same cases.

Feature groups for ``train --features``:

``structure``  sizes and counts read from the patch and the issue.
``text``       TF-IDF of the issue and of the lines the patch adds and removes.
``agreement``  how similar the patch is to other models' accepted patches for the same
               task. Using this at run time means calling several cheap models per
               request, which is routing cost.
``hidden:NAME`` an array from the ``encode`` output, e.g. ``hidden:last_21``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import judge_patches  # noqa: E402

ENCODE_INSTRUCTIONS = ("Below are a GitHub issue from the {repository} repository and a patch that claims to "
                       "resolve it. Decide whether the patch correctly and completely resolves the issue.")
ENCODE_QUESTION = "Does the patch correctly and completely resolve the issue? Answer yes or no."
DEFAULT_LAYERS = (12, 16, 21, 28)
VERSION = "2026-11-13"


def load_cases(cases_path: Path, key_path: Path, grade: str = "run_resolved") -> list[dict]:
    """Cases joined with their grade and task, in the order of the cases file."""
    key = {row["case_id"]: row for row in judge_patches.read_csv(key_path)}
    rows = []
    for case in judge_patches.read_jsonl(cases_path):
        entry = key[case["case_id"]]
        rows.append({**case, "task": entry["instance_id"], "right": entry[grade] == "True",
                     "models": [name for name in entry["models"].split(";") if name]})
    return rows


def changed_lines(patch: str) -> tuple[list[str], list[str]]:
    """Lines a unified diff adds and removes, without the markers and file headers."""
    added, removed = [], []
    for line in patch.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("+"):
            added.append(line[1:].strip())
        elif line.startswith("-"):
            removed.append(line[1:].strip())
    return [line for line in added if line], [line for line in removed if line]


def structure_features(case: dict) -> list[float]:
    from math import log1p

    patch, issue = case["patch"], case["issue"]
    added, removed = changed_lines(patch)
    files = re.findall(r"^\+\+\+ (?:b/)?(\S+)", patch, flags=re.MULTILINE)
    return [log1p(len(patch)), log1p(len(added)), log1p(len(removed)), log1p(len(files)),
            log1p(patch.count("\n@@")), log1p(len(issue)), float(any("test" in name for name in files)),
            float(len(added) == 0), log1p(sum(len(line) for line in added)),
            float("```" in issue or "Traceback" in issue)]


def agreement_features(rows: list[dict], exclude: frozenset[str] = frozenset()) -> list[list[float]]:
    """Similarity of each patch's changed lines to the other accepted patches for its task.
    Patches written only by models in ``exclude`` are not compared against: pass the
    frontier model, whose answer a chain does not have when it checks a cheaper one."""
    from math import log1p

    lines = [set(sum(changed_lines(row["patch"]), [])) for row in rows]
    by_task: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_task.setdefault(row["task"], []).append(index)
    features = []
    for index, row in enumerate(rows):
        overlaps = []
        for other in by_task[row["task"]]:
            if other != index and set(rows[other]["models"]) - exclude:
                union = lines[index] | lines[other]
                overlaps.append(len(lines[index] & lines[other]) / len(union) if union else 0.0)
        # Other cheap models that wrote exactly this patch.
        same = max(len(set(row["models"]) - exclude) - 1, 0)
        features.append([log1p(same), log1p(len(overlaps)), max(overlaps, default=0.0),
                         sum(overlaps) / len(overlaps) if overlaps else 0.0])
    return features


def text_documents(case: dict) -> tuple[str, str]:
    added, removed = changed_lines(case["patch"])
    return case["issue"], "\n".join(added) + "\n" + "\n".join(removed)


class Verifier:
    """Logistic regression over the chosen feature groups. TF-IDF is fitted on the
    training cases only; dense features are standardised on them."""

    def __init__(self, groups: list[str], hidden: dict | None = None, c: float = 1.0, seed: int = 42):
        self.groups, self.hidden, self.c, self.seed = groups, hidden or {}, c, seed

    def _dense(self, rows, dense_index):
        import numpy as np

        blocks = []
        if "structure" in self.groups:
            blocks.append(np.array([structure_features(row) for row in rows], dtype=float))
        if "agreement" in self.groups:
            blocks.append(np.array([row["agreement"] for row in rows], dtype=float))
        for group in self.groups:
            if group.startswith("hidden:"):
                blocks.append(self.hidden[group.split(":", 1)[1]][dense_index])
        return np.hstack(blocks) if blocks else None

    def _matrix(self, rows, dense_index, fit: bool):
        from scipy import sparse
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import StandardScaler

        parts = []
        dense = self._dense(rows, dense_index)
        if dense is not None:
            if fit:
                self.scaler = StandardScaler().fit(dense)
            if "text" not in self.groups:
                return self.scaler.transform(dense)
            parts.append(sparse.csr_matrix(self.scaler.transform(dense)))
        if "text" in self.groups:
            documents = [text_documents(row) for row in rows]
            if fit:
                self.vectorizers = [TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=20000,
                                                    token_pattern=r"[A-Za-z_][A-Za-z_0-9]+")
                                    for _ in range(2)]
                for position, vectorizer in enumerate(self.vectorizers):
                    vectorizer.fit([document[position] for document in documents])
            parts.extend(vectorizer.transform([document[position] for document in documents])
                         for position, vectorizer in enumerate(self.vectorizers))
        return sparse.hstack(parts).tocsr()

    def fit(self, rows, dense_index):
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(C=self.c, max_iter=2000, random_state=self.seed)
        self.model.fit(self._matrix(rows, dense_index, True), [row["right"] for row in rows])
        return self

    def predict(self, rows, dense_index):
        return self.model.predict_proba(self._matrix(rows, dense_index, False))[:, 1]


def grouped_folds(tasks: list[str], folds: int, seed: int) -> list[int]:
    """A fold number per case, with every task's cases in one fold."""
    import random

    names = sorted(set(tasks))
    random.Random(seed).shuffle(names)
    fold = {name: position % folds for position, name in enumerate(names)}
    return [fold[task] for task in tasks]


def reject_threshold(probabilities, right, max_false_reject: float) -> float:
    """The highest threshold whose share of right patches rejected stays within the
    limit. A patch is rejected when its probability is below the threshold."""
    kept = sorted(p for p, ok in zip(probabilities, right) if ok)
    if not kept:
        return 0.0
    allowed = int(max_false_reject * len(kept))
    return float(kept[allowed])


def held_out_predictions(rows, groups, hidden, *, folds, seed, c, max_false_reject):
    """Predictions for every case from a model that saw none of its task's cases.
    Each fold's regularisation strength (when ``c`` lists several) and threshold come
    from predictions inside its own training cases."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    strengths = list(c) if isinstance(c, (list, tuple)) else [c]

    index = np.arange(len(rows))
    outer = np.array(grouped_folds([row["task"] for row in rows], folds, seed))
    probability, verdict = np.zeros(len(rows)), np.zeros(len(rows), dtype=bool)
    for fold in range(folds):
        train, test = index[outer != fold], index[outer == fold]
        train_rows = [rows[i] for i in train]
        inner = np.array(grouped_folds([row["task"] for row in train_rows], folds, seed + 1))
        train_right = [row["right"] for row in train_rows]
        best = None
        for strength in strengths:
            inner_probability = np.zeros(len(train))
            for inner_fold in range(folds):
                fit_part, check_part = np.where(inner != inner_fold)[0], np.where(inner == inner_fold)[0]
                model = Verifier(groups, hidden, strength, seed).fit([train_rows[i] for i in fit_part], train[fit_part])
                inner_probability[check_part] = model.predict([train_rows[i] for i in check_part], train[check_part])
            score = roc_auc_score(train_right, inner_probability)
            if best is None or score > best[0]:
                best = (score, strength, inner_probability)
        threshold = reject_threshold(best[2], train_right, max_false_reject)
        model = Verifier(groups, hidden, best[1], seed).fit(train_rows, train)
        probability[test] = model.predict([rows[i] for i in test], test)
        verdict[test] = probability[test] >= threshold
    return probability, verdict


def metrics(rows, probability, verdict, subset=None) -> dict:
    from sklearn.metrics import roc_auc_score

    chosen = [i for i, row in enumerate(rows) if subset is None or row["case_id"] in subset]
    right = [rows[i]["right"] for i in chosen]
    accepted = [bool(verdict[i]) for i in chosen]
    wrong_count, right_count = right.count(False), right.count(True)
    accepted_right = sum(a and r for a, r in zip(accepted, right))
    return {"cases": len(chosen), "right": right_count, "wrong": wrong_count,
            "auc": round(float(roc_auc_score(right, [probability[i] for i in chosen])), 4),
            "catch_rate": round(sum((not a) and (not r) for a, r in zip(accepted, right)) / wrong_count, 4),
            "false_reject_rate": round(sum((not a) and r for a, r in zip(accepted, right)) / right_count, 4),
            "accepted_share": round(sum(accepted) / len(chosen), 4),
            "accuracy_when_accepted": round(accepted_right / sum(accepted), 4) if sum(accepted) else None}


def train(cases_path, key_path, output, *, groups, embeddings=None, grade="run_resolved", folds=5, seed=42,
          c=1.0, max_false_reject=0.1, subset_key=None, name=None, agreement_exclude=()) -> dict:
    rows = load_cases(cases_path, key_path, grade)
    for row, features in zip(rows, agreement_features(rows, frozenset(agreement_exclude))):
        row["agreement"] = features
    names = [group.split(":", 1)[1] for group in groups if group.startswith("hidden:")]
    hidden = load_hidden(embeddings, rows, names) if names else {}
    probability, verdict = held_out_predictions(rows, groups, hidden, folds=folds, seed=seed, c=c,
                                                max_false_reject=max_false_reject)
    name = name or "trained:" + "+".join(groups)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=judge_patches.VERDICT_FIELDS)
        writer.writeheader()
        for row, p, accepted in zip(rows, probability, verdict):
            writer.writerow({"case_id": row["case_id"], "judge": name, "verdict": "accept" if accepted else "reject",
                             "probability": round(float(p), 4), "reason": "", "prompt_tokens": "",
                             "completion_tokens": "", "tokens_estimated": "", "seconds": "",
                             "prompt_version": "trained-" + VERSION})
    report = {"judge": name, "features": groups, "grade": grade, "folds": folds, "seed": seed, "c": c,
              "max_false_reject": max_false_reject, "agreement_exclude": sorted(agreement_exclude), "tasks": len({row["task"] for row in rows}),
              "all": metrics(rows, probability, verdict)}
    if subset_key:
        subset = {row["case_id"] for row in judge_patches.read_csv(subset_key)}
        report["subset"] = metrics(rows, probability, verdict, subset)
    Path(str(output) + ".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_hidden(embeddings, rows, names) -> dict:
    import numpy as np

    with np.load(embeddings, allow_pickle=False) as saved:
        position = {case_id: i for i, case_id in enumerate(saved["case_ids"].tolist())}
        order = [position[row["case_id"]] for row in rows]
        return {name: saved[name][order].astype(float) for name in names}


def compare(cases_path, key_path, output, *, feature_sets, embeddings=None, grade="run_resolved", folds=5,
            seeds=5, c=(1.0,), max_false_reject=0.1, subset_key=None, agreement_exclude=()) -> list[dict]:
    """Mean held-out metrics per feature set over ``seeds`` grouped splits, on all cases,
    on cases written by a model outside ``agreement_exclude`` (``cheap``), and on the
    subset in ``subset_key``."""
    from statistics import mean, pstdev

    exclude = frozenset(agreement_exclude)
    rows = load_cases(cases_path, key_path, grade)
    for row, features in zip(rows, agreement_features(rows, exclude)):
        row["agreement"] = features
    names = sorted({group.split(":", 1)[1] for groups in feature_sets for group in groups if group.startswith("hidden:")})
    hidden = load_hidden(embeddings, rows, names) if names else {}
    subsets = {"all": None}
    if exclude:
        subsets["cheap"] = {row["case_id"] for row in rows if set(row["models"]) - exclude}
    if subset_key:
        subsets["subset"] = {row["case_id"] for row in judge_patches.read_csv(subset_key)}
    table = []
    for groups in feature_sets:
        runs = [held_out_predictions(rows, groups, hidden, folds=folds, seed=seed, c=list(c),
                                     max_false_reject=max_false_reject) for seed in range(seeds)]
        for label, subset in subsets.items():
            scored = [metrics(rows, probability, verdict, subset) for probability, verdict in runs]
            entry = {"features": "+".join(groups), "cases_scored": label, "cases": scored[0]["cases"],
                     "right": scored[0]["right"], "wrong": scored[0]["wrong"], "seeds": seeds,
                     "auc": round(mean(m["auc"] for m in scored), 3), "auc_sd": round(pstdev(m["auc"] for m in scored), 3)}
            for field in ("catch_rate", "false_reject_rate", "accepted_share"):
                entry[field] = round(mean(m[field] for m in scored), 3)
            accuracies = [m["accuracy_when_accepted"] for m in scored if m["accuracy_when_accepted"] is not None]
            entry["accuracy_when_accepted"] = round(mean(accuracies), 3) if accuracies else ""
            table.append(entry)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    return table


def encode_text(case: dict) -> str:
    return (ENCODE_INSTRUCTIONS.format(repository=case["repository"]) + "\n\n" + judge_patches.case_text(case)
            + "\n\n" + ENCODE_QUESTION)


def layer_states(encoder, texts: list[str], layers) -> dict:
    """One forward pass: ``last_L`` and ``mean_L`` arrays for each layer ``L``."""
    torch = encoder.torch
    batch = encoder.tokenizer([encoder._text(text) for text in texts], return_tensors="pt", padding=True,
                              add_special_tokens=False)
    batch = {key: value.to(encoder.device) for key, value in batch.items()}
    with torch.no_grad():
        states = encoder.model(**batch, output_hidden_states=True).hidden_states
    mask = batch["attention_mask"].unsqueeze(-1).to(states[0].dtype)
    result = {}
    for layer in layers:
        result[f"last_{layer}"] = states[layer][:, -1, :].float().cpu().numpy()
        result[f"mean_{layer}"] = ((states[layer] * mask).sum(1) / mask.sum(1)).float().cpu().numpy()
    return result


def encode(cases_path, output, *, model, revision, layers, max_tokens, batch_size, device, cache_dir=None,
           limit=None, states=None) -> dict:
    """Hidden states of a frozen model for every case: ``last_L`` and ``mean_L`` per layer.
    ``states`` replaces the model in tests: a function from texts to those arrays."""
    import numpy as np

    cases = judge_patches.read_jsonl(cases_path)[:limit]
    if states is None:
        import hidden_states

        encoder = hidden_states.HiddenStateEncoder(model, revision, layer=max(layers), max_tokens=max_tokens,
                                                   device=device, chat_template=True, cache_dir=cache_dir)
        states = lambda texts: layer_states(encoder, texts, layers)  # noqa: E731
    arrays: dict[str, list] = {}
    # Long cases first, so a batch that does not fit fails at the start.
    order = sorted(range(len(cases)), key=lambda i: -len(cases[i]["issue"]) - len(cases[i]["patch"]))
    for start in range(0, len(order), batch_size):
        for name, values in states([encode_text(cases[i]) for i in order[start:start + batch_size]]).items():
            arrays.setdefault(name, []).append(values)
    inverse = np.argsort(order)
    saved = {name: np.concatenate(parts, axis=0)[inverse].astype(np.float32) for name, parts in arrays.items()}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, case_ids=np.array([case["case_id"] for case in cases]), **saved)
    facts = {"cases": len(cases), "model": model, "revision": revision, "layers": list(layers),
             "max_tokens": max_tokens, "arrays": sorted(saved), "dimension": int(next(iter(saved.values())).shape[1]),
             "version": VERSION}
    Path(str(output) + ".json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    return facts


def main(argv=None) -> None:
    import hidden_states

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    enc = commands.add_parser("encode", help="hidden states of a frozen model for every case (GPU)")
    enc.add_argument("--cases", type=Path, required=True)
    enc.add_argument("--output", type=Path, required=True)
    enc.add_argument("--model", default=hidden_states.DEFAULT_MODEL)
    enc.add_argument("--revision", default=hidden_states.DEFAULT_REVISION, help="full Hugging Face commit SHA")
    enc.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    enc.add_argument("--max-seq-length", type=int, default=10000, help="tokens kept per case, from the start")
    enc.add_argument("--batch-size", type=int, default=2)
    enc.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    enc.add_argument("--cache-dir")
    enc.add_argument("--limit", type=int)
    fit = commands.add_parser("train", help="cross-validated verifier on chosen feature groups (CPU)")
    fit.add_argument("--cases", type=Path, required=True)
    fit.add_argument("--key", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True, help="held-out verdicts, in judge_patches format")
    fit.add_argument("--features", nargs="+", default=["structure", "text"])
    fit.add_argument("--embeddings", type=Path, help="NPZ from encode, for hidden:NAME features")
    fit.add_argument("--grade", choices=("benchmark_right", "run_resolved"), default="run_resolved")
    fit.add_argument("--folds", type=int, default=5)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--c", type=float, nargs="+", default=[1.0],
                     help="inverse regularisation strength; with several, each fold picks one on its training cases")
    fit.add_argument("--max-false-reject", type=float, default=0.1,
                     help="share of right patches the threshold may reject, set on training cases")
    fit.add_argument("--subset-key", type=Path, help="also report on the cases in this key file")
    fit.add_argument("--agreement-exclude", nargs="*", default=[],
                     help="models whose patches the agreement features may not compare against (the frontier)")
    fit.add_argument("--judge-name")
    many = commands.add_parser("compare", help="several feature sets over several seeds, one summary table")
    many.add_argument("--cases", type=Path, required=True)
    many.add_argument("--key", type=Path, required=True)
    many.add_argument("--output", type=Path, required=True, help="summary CSV")
    many.add_argument("--feature-sets", nargs="+", required=True,
                      help="each a +-joined list of groups, e.g. structure+text hidden:last_21")
    many.add_argument("--embeddings", type=Path)
    many.add_argument("--grade", choices=("benchmark_right", "run_resolved"), default="run_resolved")
    many.add_argument("--folds", type=int, default=5)
    many.add_argument("--seeds", type=int, default=5)
    many.add_argument("--c", type=float, nargs="+", default=[1.0])
    many.add_argument("--max-false-reject", type=float, default=0.1)
    many.add_argument("--subset-key", type=Path)
    many.add_argument("--agreement-exclude", nargs="*", default=[])
    args = parser.parse_args(argv)
    known = {"structure", "text", "agreement"}

    def check(groups):
        for group in groups:
            if group not in known and not group.startswith("hidden:"):
                raise SystemExit(f"Unknown feature group {group!r}")
        if any(group.startswith("hidden:") for group in groups) and not args.embeddings:
            raise SystemExit("hidden:NAME features need --embeddings")

    if args.command == "compare":
        sets = [item.split("+") for item in args.feature_sets]
        for groups in sets:
            check(groups)
        table = compare(args.cases, args.key, args.output, feature_sets=sets, embeddings=args.embeddings,
                        grade=args.grade, folds=args.folds, seeds=args.seeds, c=args.c,
                        max_false_reject=args.max_false_reject, subset_key=args.subset_key,
                        agreement_exclude=args.agreement_exclude)
        for entry in table:
            print(json.dumps(entry))
    elif args.command == "encode":
        print(json.dumps(encode(args.cases, args.output, model=args.model, revision=args.revision,
                                layers=args.layers, max_tokens=args.max_seq_length, batch_size=args.batch_size,
                                device=args.device, cache_dir=args.cache_dir, limit=args.limit)))
    else:
        check(args.features)
        print(json.dumps(train(args.cases, args.key, args.output, groups=args.features,
                               embeddings=args.embeddings, grade=args.grade, folds=args.folds, seed=args.seed,
                               c=args.c, max_false_reject=args.max_false_reject, subset_key=args.subset_key,
                               name=args.judge_name, agreement_exclude=args.agreement_exclude), indent=2))


if __name__ == "__main__":
    main()
