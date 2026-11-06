"""Metrics and cost-quality charts for routing decisions on a test set."""

import numpy as np

from tine.baselines import oracle_choices
from tine.calibration import regret


def routing_metrics(choices, success, cost, frontier, models=None):
    """Accuracy, cost and savings of one set of routing decisions.

    Cost is the logged cost per task of the chosen model (cost per completed task,
    including that model's retries and output length), not per-token prices.
    """
    rows = np.arange(len(choices))
    accuracy = float(success[rows, choices].mean())
    total_cost = float(cost[rows, choices].sum())
    frontier_accuracy = float(success[:, frontier].mean())
    frontier_cost = float(cost[:, frontier].sum())
    oracle = oracle_choices(success, cost)
    oracle_cost = float(cost[rows, oracle].sum())
    saved = frontier_cost - total_cost
    out = {
        "accuracy": accuracy,
        "cost_usd": total_cost,
        "cost_per_task_usd": total_cost / len(choices),
        "cost_vs_frontier": total_cost / frontier_cost,
        "savings_pct": 100.0 * saved / frontier_cost,
        "quality_loss": frontier_accuracy - accuracy,
        "regret_rate": float(regret(choices, success, frontier).mean()),
        "oracle_gap_closed": saved / (frontier_cost - oracle_cost)
        if frontier_cost > oracle_cost else float("nan"),
    }
    if models is not None:
        counts = np.bincount(choices, minlength=len(models)) / len(choices)
        out["traffic_share"] = {m: float(s) for m, s in zip(models, counts)}
    return out


def best_savings_within(curve, epsilon):
    """Best savings among curve points whose test quality loss <= epsilon.

    Picked on the test set itself, so it flatters the baseline: an upper bound on
    what that baseline could claim, not a certified number.
    """
    ok = [p for p in curve if p["quality_loss"] <= epsilon]
    return max(ok, key=lambda p: p["savings_pct"]) if ok else None


def plot_cost_quality(curves, points, path, title="Cost vs quality (test set)"):
    """curves: {name: [metrics...]}; points: {name: metrics}. Writes a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, curve in curves.items():
        curve = sorted(curve, key=lambda p: p["cost_vs_frontier"])
        ax.plot([100 * p["cost_vs_frontier"] for p in curve],
                [100 * p["accuracy"] for p in curve], marker=".", label=name, linewidth=1.5)
    styles = iter([("*", "black"), ("s", "dimgray"), ("D", "goldenrod"), ("^", "purple")])
    for name, p in points.items():
        marker, color = next(styles)
        ax.scatter(100 * p["cost_vs_frontier"], 100 * p["accuracy"], s=120, marker=marker,
                   color=color, zorder=5, label=name)
    ax.set_xlabel("Cost (% of always-frontier)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
