"""Reproducible offline evaluation on *existing* graded model outcomes.

Training, probability calibration, risk calibration, and final evaluation use
disjoint prompt groups. Actual test costs are used only for evaluation; routing
uses estimated costs learned from training rows. Guarantees concern the
nonnegative degradation event on IID tasks from the calibration
distribution, not unseen providers, distribution shifts, or arbitrary tenants.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tine.calibration import calibrate, evaluate
from tine.data import outcome_json_lines, validate_outcomes
from tine.routing import Router, route_indices

METHODS = ("logistic", "knn", "cluster")
SPLITS = ("train", "score_calibration", "risk_calibration", "test")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def split_tasks(tasks: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Group identical prompts within a tenant, including different task IDs."""
    tasks = tasks.copy()
    tasks["prompt_hash"] = tasks.prompt.map(prompt_hash)
    tasks["split"] = ""
    for tenant, group in tasks.groupby("tenant", sort=True):
        hashes = sorted(group.prompt_hash.unique())
        if len(hashes) < 8:
            raise ValueError(f"Tenant {tenant!r} needs at least 8 distinct prompts for four disjoint splits")
        rng = np.random.default_rng(seed)
        hashes = rng.permutation(hashes).tolist()
        n = len(hashes)
        counts = [max(1, int(n * fraction)) for fraction in (0.5, 0.15, 0.2)]
        counts.append(n - sum(counts))
        mapping = {}
        offset = 0
        for name, count in zip(SPLITS, counts, strict=True):
            mapping.update(dict.fromkeys(hashes[offset:offset + count], name))
            offset += count
        tasks.loc[group.index, "split"] = group.prompt_hash.map(mapping)
    return tasks


def complete_tasks(frame: pd.DataFrame, frontier: str):
    """Drop entire incomplete tasks, report exclusions; never impute outcomes."""
    frame = validate_outcomes(frame)
    tasks = frame.drop_duplicates(["tenant", "task_id"])[
        ["tenant", "task_id", "prompt", "task_type"]
    ].copy()
    accepted, excluded = [], []
    for tenant, rows in frame.groupby("tenant", sort=True):
        models = sorted(rows.model.unique())
        if frontier not in models:
            raise ValueError(f"Frontier {frontier!r} is absent from tenant {tenant!r}")
        for task_id, task_rows in rows.groupby("task_id", sort=True):
            missing = sorted(set(models) - set(task_rows.model))
            if missing or task_rows.success.isna().any():
                excluded.append({"tenant": tenant, "task_id": task_id,
                                 "missing_models": missing,
                                 "pending_outcomes": int(task_rows.success.isna().sum())})
            else:
                accepted.append((tenant, task_id))
    keys = set(accepted)
    keep = [(tenant, task_id) in keys for tenant, task_id in zip(tasks.tenant, tasks.task_id)]
    tasks = tasks.loc[keep].sort_values(["tenant", "task_id"]).reset_index(drop=True)
    if tasks.empty:
        raise ValueError("No complete graded tasks remain; supply overlapping known outcomes for every model")
    # Copied prompts must not inflate the number of independent observations.
    # The representative is selected by ID, without reading outcomes or costs.
    if tasks.groupby(["tenant", "prompt"]).task_type.nunique().gt(1).any():
        raise ValueError("Identical tenant prompts have conflicting task types")
    copied = tasks.duplicated(["tenant", "prompt"], keep="first")
    for row in tasks.loc[copied].itertuples():
        excluded.append({"tenant": row.tenant, "task_id": row.task_id,
                         "reason": "duplicate_prompt", "missing_models": [], "pending_outcomes": 0})
    tasks = tasks.loc[~copied].reset_index(drop=True)
    frame = frame.merge(tasks[["tenant", "task_id"]], on=["tenant", "task_id"], validate="many_to_one")
    return frame, tasks, excluded


def load_embeddings(path: str | Path, tasks: pd.DataFrame) -> dict:
    """Load data-only NPZ; bind every vector to tenant/task and prompt digest."""
    with np.load(path, allow_pickle=False) as blob:
        tenants = blob["tenants"].astype(str)
        ids = blob["task_ids"].astype(str)
        hashes = blob["prompt_hashes"].astype(str)
        vectors = blob["embeddings"]
        if vectors.ndim != 2 or not np.isfinite(vectors).all() or vectors.shape[1] == 0:
            raise ValueError("Embeddings must be a finite, nonempty 2D numeric array")
        if not (len(tenants) == len(ids) == len(hashes) == len(vectors)):
            raise ValueError("Embedding metadata lengths differ")
        lookup, digest = {}, {}
        for tenant, task_id, sha, vector in zip(tenants, ids, hashes, vectors, strict=True):
            key = (tenant, task_id)
            if key in lookup:
                raise ValueError(f"Duplicate embedding key {key}")
            lookup[key], digest[key] = vector.copy(), sha
    for row in tasks.itertuples():
        key = (str(row.tenant), str(row.task_id))
        if key not in lookup or digest[key] != prompt_hash(row.prompt):
            raise ValueError(f"Missing or stale prompt embedding for {key}")
    return lookup


def matrix(frame, tasks, models, column, *, boolean=False):
    pivot = frame.pivot(index=["tenant", "task_id"], columns="model", values=column)
    keys = pd.MultiIndex.from_frame(tasks[["tenant", "task_id"]])
    values = pivot.reindex(index=keys, columns=models)
    if values.isna().any().any():
        raise ValueError(f"Missing {column} in selected task matrix")
    return values.to_numpy(dtype=bool if boolean else float)


def _json(value):
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_experiment(frame, frontier, output, *, epsilon=0.02, delta=0.05, seed=42,
                   embeddings=None, thresholds=None, synthetic=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    thresholds = list(np.linspace(0, 1, 21)) if thresholds is None else list(thresholds)
    frame, tasks, excluded = complete_tasks(frame, frontier)
    tasks = split_tasks(tasks, seed)
    lookup = load_embeddings(embeddings, tasks) if embeddings else None
    group_count = tasks[["tenant", "task_type"]].drop_duplicates().shape[0]
    family_size = group_count * len(METHODS)
    all_metrics, group_metrics, calibrations, decisions, curves = [], [], [], [], []
    split_rows = frame.merge(tasks[["tenant", "task_id", "split"]], on=["tenant", "task_id"])
    for tenant, tenant_tasks in tasks.groupby("tenant", sort=True):
        partitions = {name: tenant_tasks.loc[tenant_tasks.split == name].copy() for name in SPLITS}
        rows = split_rows.loc[split_rows.tenant == tenant]
        router = Router(frontier, seed=seed, embedding_lookup=lookup).fit(
            rows.loc[rows.split == "train"], rows.loc[rows.split == "score_calibration"])
        models = router.model_names
        frontier_index = models.index(frontier)
        risk_tasks, test_tasks = partitions["risk_calibration"], partitions["test"]
        risk_outcomes = matrix(rows, risk_tasks, models, "success", boolean=True)
        test_outcomes = matrix(rows, test_tasks, models, "success", boolean=True)
        observed_costs = matrix(rows, test_tasks, models, "cost_usd")
        risk_costs, test_costs = router.estimated_costs(risk_tasks), router.estimated_costs(test_tasks)
        choices = {"frontier": np.full(len(test_tasks), frontier_index),
                   "random": np.random.default_rng(seed).integers(len(models), size=len(test_tasks))}
        oracle_costs = np.where(test_outcomes, observed_costs, np.inf)
        oracle = oracle_costs.argmin(axis=1)
        oracle[~test_outcomes.any(axis=1)] = frontier_index
        choices["oracle"] = oracle
        method_timings = {}
        # Independent test labels/costs are never used in policy selection.
        for method in METHODS:
            risk_scores = router.predict_scores(risk_tasks, method=method)
            started = time.perf_counter()
            test_scores = router.predict_scores(test_tasks, method=method)
            method_timings[method] = (time.perf_counter() - started) / len(test_tasks) * 1000
            chosen = np.full(len(test_tasks), frontier_index)
            for task_type in sorted(tenant_tasks.task_type.unique()):
                risk_mask = (risk_tasks.task_type == task_type).to_numpy()
                test_mask = (test_tasks.task_type == task_type).to_numpy()
                result = calibrate(risk_scores[risk_mask], risk_costs[risk_mask],
                                   risk_outcomes[risk_mask], frontier_index,
                                   thresholds, epsilon, delta, family_size)
                calibrations.append({"tenant": tenant, "task_type": task_type,
                                     "method": method, **result})
                if test_mask.any():
                    chosen[test_mask] = route_indices(
                        test_scores[test_mask], test_costs[test_mask],
                        1.0 if result["tau"] is None else result["tau"], frontier_index,
                        force_frontier=result["force_frontier"])
            choices[method] = chosen
            for tau in thresholds:
                policy = route_indices(test_scores, test_costs, tau, frontier_index)
                curves.append({"tenant": tenant, "method": method, "tau": tau,
                               **evaluate(policy, test_outcomes, observed_costs, frontier_index)})
        for method, policy in choices.items():
            metrics = evaluate(policy, test_outcomes, observed_costs, frontier_index)
            all_metrics.append({"tenant": tenant, "method": method,
                                "batch_prediction_ms_per_task": method_timings.get(method), **metrics})
            for task_type in sorted(test_tasks.task_type.unique()):
                mask = (test_tasks.task_type == task_type).to_numpy()
                group_metrics.append({"tenant": tenant, "task_type": task_type, "method": method,
                                      **evaluate(policy[mask], test_outcomes[mask],
                                                 observed_costs[mask], frontier_index)})
            for i, row in enumerate(test_tasks.itertuples()):
                decisions.append({"tenant": tenant, "task_id": row.task_id,
                                  "task_type": row.task_type, "method": method,
                                  "model": models[policy[i]],
                                  "success": bool(test_outcomes[i, policy[i]]),
                                  "cost_usd": float(observed_costs[i, policy[i]]),
                                  "frontier_success": bool(test_outcomes[i, frontier_index])})
    provenance = hashlib.sha256()
    for line in outcome_json_lines(frame.sort_values(["tenant", "task_id", "model"])):
        provenance.update(line.encode("utf-8"))
    summary = {"schema_version": 1, "synthetic": synthetic, "seed": seed,
               "frontier": frontier, "epsilon": epsilon, "delta": delta,
               "feature_source": "supplied_embeddings" if lookup else "train_only_tfidf",
               "dataset_sha256": provenance.hexdigest(), "complete_tasks": len(tasks),
               "excluded_tasks": len(excluded), "family_size": family_size,
               "metrics": all_metrics, "group_metrics": group_metrics,
               "split_counts": tasks.groupby(["tenant", "split"]).size().reset_index(name="tasks").to_dict(orient="records"),
               "software_versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")},
               "scope": "Offline benchmark only; no production or distribution-shift guarantee.",
               "baseline_note": "cluster is an Avengers-style heuristic, not a reproduction of Avengers-Pro; oracle uses hindsight.",
               "latency_note": "batch prediction time divided by tasks; excludes network and is not p50/p99 request latency."}
    (output / "summary.json").write_text(json.dumps(_json(summary), indent=2, allow_nan=False) + "\n")
    (output / "calibration.json").write_text(json.dumps(_json(calibrations), indent=2, allow_nan=False) + "\n")
    (output / "excluded.json").write_text(json.dumps(_json(excluded), indent=2) + "\n")
    tasks.drop(columns="prompt").to_csv(output / "splits.csv", index=False)
    pd.DataFrame(decisions).to_csv(output / "decisions.csv", index=False)
    pd.DataFrame(curves).to_csv(output / "curves.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(group_metrics).to_csv(output / "group-metrics.csv", index=False)
    _plot(output, all_metrics, synthetic, curves)
    report = ["# Tine offline report", "", "SYNTHETIC FIXTURE — not benchmark evidence." if synthetic else "Existing graded outcomes; held-out test split.", "",
              f"Complete tasks: {len(tasks)}. Excluded incomplete/pending tasks: {len(excluded)}.",
              f"Frontier: `{frontier}`. Epsilon: {epsilon}. Delta: {delta}.", "",
              "See metrics.csv, calibration.json, decisions.csv, and cost-quality.png.", "",
              "Calibration bounds the event 'frontier succeeded and routed answer failed' per tenant/task type, with a Bonferroni correction over the fixed threshold grid and all methods/groups. It requires independently sampled (IID), representative tasks within each group and independent training/calibration/test splits. Exact duplicate prompts are deduplicated by a label-independent task ID ordering; semantic duplicates and upstream contamination require a separate audit.", "",
              "Threshold curves use test outcomes for reporting only. Do not select a new policy on this test set. Unknown/missing outcomes are excluded, which may bias results; exclusions are recorded. Costs are recorded per completed task and must include all retries in the source data. No model output was generated by this experiment.", "",
              summary["baseline_note"], "", summary["latency_note"], ""]
    (output / "report.md").write_text("\n".join(report))
    return _json(summary)


def _plot(output, metrics, synthetic, curves):
    fig, ax = plt.subplots(figsize=(8, 5))
    curve_frame = pd.DataFrame(curves)
    for (tenant, method), curve in curve_frame.groupby(["tenant", "method"]):
        ax.plot(curve.mean_cost_usd, curve.accuracy, alpha=0.5,
                label=f"{tenant}/{method} thresholds (test diagnostics)")
    for metric in metrics:
        # Use raw observed task cost and accuracy without manipulating epsilon.
        cost = metric.get("mean_cost_usd", metric.get("mean_cost"))
        accuracy = metric.get("accuracy")
        if cost is not None and accuracy is not None:
            ax.scatter(cost, accuracy)
            ax.annotate(f'{metric["tenant"]}/{metric["method"]}', (cost, accuracy), fontsize=8)
    ax.set(xlabel="Observed mean cost per task (USD)", ylabel="Held-out success rate",
           title="Synthetic pipeline check" if synthetic else "Offline cost-quality comparison")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "cost-quality.png", dpi=140)
    plt.close(fig)
