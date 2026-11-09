"""Descriptive single-request CPU timings for an already fitted routing policy."""

import time

import numpy as np

from tine.routing import route_indices


LATENCY_SCOPE = (
    "Warm single-request CPU routing, including task validation, feature lookup/"
    "transformation, calibrated scoring, estimated costs, and policy selection. "
    "Excludes input DataFrame construction, embedding generation, verified-cache "
    "lookup, network, and provider latency. Sample percentiles are descriptive, "
    "not production latency guarantees."
)


def benchmark_routing(router, tasks, policies, method, *, samples, seed=42):
    """Time a fixed policy on a reproducible sample of distinct tasks.

    Warm up on three calls before timing. Sample without replacement, capped by
    the available tasks. Only task inputs reach the router; labels and observed
    costs cannot affect these routing decisions.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("latency samples must be a positive integer")
    tasks = router._prediction_tasks(tasks).sort_values("task_id").reset_index(drop=True)
    if set(tasks.task_type) - set(policies):
        raise ValueError("Every latency task type needs a frozen calibration policy")
    indices = np.random.default_rng(seed).choice(len(tasks), min(samples, len(tasks)), replace=False)

    def decide(task):
        scores = router.predict_scores(task, method=method)
        costs = router.estimated_costs(task)
        policy = policies[task.task_type.iloc[0]]
        return int(route_indices(
            scores, costs, 1.0 if policy["tau"] is None else policy["tau"],
            router.frontier_index, force_frontier=policy["force_frontier"],
        )[0])

    warmup = tasks.iloc[[indices[0]]]
    for _ in range(3):
        decide(warmup)
    observations = []
    for index in indices:
        task = tasks.iloc[[index]]
        started = time.perf_counter_ns()
        selected = decide(task)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        observations.append({
            "task_id": task.task_id.iloc[0], "task_type": task.task_type.iloc[0],
            "selected_model": router.model_names[selected], "latency_ms": elapsed_ms,
        })
    durations = [item["latency_ms"] for item in observations]
    return {
        "tenant": tasks.tenant.iloc[0], "method": method,
        "requested_samples": samples, "samples": len(observations), "warmup_calls": 3,
        "seed": seed, "p50_ms": float(np.percentile(durations, 50)),
        "p99_ms": float(np.percentile(durations, 99)), "max_ms": max(durations),
        "observations": observations,
    }
