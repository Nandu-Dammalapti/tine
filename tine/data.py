"""Load graded results into a tasks x models table, and split it for training.

Sources:
- any CSV/JSONL with the columns in `tine.schema.RESULT_COLUMNS`;
- LLMRouterBench result JSON files (`load_llmrouterbench`);
- invented data for tests and the demo (`make_synthetic_results`).
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from tine.schema import RESULT_COLUMNS, validate_results


@dataclass(frozen=True)
class RoutingTable:
    """One tenant's graded outcomes. Row i is a task, column j is a model.

    `success[i, j]` is 1.0, 0.0, or NaN when unknown/pending.
    `cost[i, j]` is the logged cost in USD of model j on task i (retries included).
    """

    tenant: str
    task_ids: np.ndarray
    prompts: np.ndarray
    task_types: np.ndarray
    models: list
    success: np.ndarray
    cost: np.ndarray

    def __len__(self):
        return len(self.task_ids)

    def subset(self, index) -> "RoutingTable":
        index = np.asarray(index)
        return replace(
            self,
            task_ids=self.task_ids[index],
            prompts=self.prompts[index],
            task_types=self.task_types[index],
            success=self.success[index],
            cost=self.cost[index],
        )

    def complete_index(self) -> np.ndarray:
        """Rows where every model has a known outcome and cost.

        Evaluation and calibration need every model's outcome on the same task.
        """
        known = ~np.isnan(self.success).any(axis=1) & ~np.isnan(self.cost).any(axis=1)
        return np.flatnonzero(known)

    def complete(self) -> "RoutingTable":
        return self.subset(self.complete_index())

    def model_index(self, name: str) -> int:
        if name not in self.models:
            raise ValueError(f"Unknown model {name!r}; have {self.models}")
        return self.models.index(name)

    @classmethod
    def from_results(cls, frame: pd.DataFrame, tenant: str | None = None) -> "RoutingTable":
        """Pivot long result rows into a table for exactly one tenant.

        Tenants are never mixed: pass `tenant` when the frame holds several.
        """
        frame = validate_results(frame)
        tenants = sorted(frame["tenant"].unique())
        if tenant is None:
            if len(tenants) != 1:
                raise ValueError(f"Results hold several tenants {tenants}; pass tenant=")
            tenant = tenants[0]
        frame = frame[frame["tenant"] == tenant]
        if frame.empty:
            raise ValueError(f"No rows for tenant {tenant!r}")
        tasks = frame.drop_duplicates("task_id").set_index("task_id")
        task_ids = tasks.index.to_numpy()
        models = sorted(frame["model"].unique())
        success = frame.pivot(index="task_id", columns="model", values="success")
        cost = frame.pivot(index="task_id", columns="model", values="cost_usd")
        success = success.reindex(index=task_ids, columns=models)
        cost = cost.reindex(index=task_ids, columns=models)
        return cls(
            tenant=tenant,
            task_ids=task_ids.astype(str),
            prompts=tasks["prompt"].to_numpy().astype(str),
            task_types=tasks["task_type"].to_numpy().astype(str),
            models=models,
            success=success.to_numpy(dtype=float),
            cost=cost.to_numpy(dtype=float),
        )


def read_results(path) -> pd.DataFrame:
    """Read result rows from .csv, .jsonl or .json."""
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, keep_default_na=True)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported results file {path}")


def split_indices(task_types, fractions=(0.6, 0.2, 0.2), seed=0):
    """Split tasks into train / calibration / test, stratified by task type.

    The calibration set is never used for training: it only certifies tau.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    task_types = np.asarray(task_types)
    parts = ([], [], [])
    for task_type in np.unique(task_types):
        idx = rng.permutation(np.flatnonzero(task_types == task_type))
        cut1 = int(round(fractions[0] * len(idx)))
        cut2 = cut1 + int(round(fractions[1] * len(idx)))
        for part, chunk in zip(parts, (idx[:cut1], idx[cut1:cut2], idx[cut2:])):
            part.extend(chunk.tolist())
    return tuple(np.sort(np.array(p, dtype=int)) for p in parts)


# --- LLMRouterBench -------------------------------------------------------------------------

def load_llmrouterbench(root, datasets=None, models=None, success_threshold=1.0,
                        tenant="public-llmrouterbench") -> pd.DataFrame:
    """Read LLMRouterBench results laid out as `<root>/<dataset>/<split>/<model>/<time>.json`.

    Layout and field names follow the project README (MIT license), checked
    November 2026: each file has `records` with `index`, `prompt`, `origin_query`,
    `score`, `prompt_tokens`, `completion_tokens` and `cost`. When a model has several
    runs, the newest file name wins. `success` is `score >= success_threshold`;
    check each dataset's score scale (judge-graded sets may not be 0/1) before use.
    This adapter was written against the documented layout and a local fixture,
    not the full release.
    """
    root = Path(root)
    rows = []
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if datasets and dataset_dir.name not in datasets:
            continue
        for split_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            for model_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                if models and model_dir.name not in models:
                    continue
                runs = sorted(model_dir.glob("*.json"))
                if not runs:
                    continue
                with open(runs[-1], encoding="utf-8") as handle:
                    payload = json.load(handle)
                for position, record in enumerate(payload.get("records", [])):
                    prompt = record.get("prompt", record.get("origin_query", ""))
                    if not isinstance(prompt, str):
                        prompt = json.dumps(prompt, ensure_ascii=False)
                    score = record.get("score")
                    success = np.nan if score is None else float(float(score) >= success_threshold)
                    rows.append({
                        "tenant": tenant,
                        "task_id": f"{dataset_dir.name}/{split_dir.name}/"
                                   f"{record.get('index', position)}",
                        "prompt": prompt,
                        "task_type": dataset_dir.name,
                        "model": model_dir.name,
                        "success": success,
                        "cost_usd": record.get("cost", np.nan),
                        "input_tokens": record.get("prompt_tokens", np.nan),
                        "output_tokens": record.get("completion_tokens", np.nan),
                        "cached_tokens": np.nan,
                        "latency_ms": np.nan,
                        "timestamp": runs[-1].stem,
                    })
    if not rows:
        raise ValueError(f"No LLMRouterBench records found under {root}")
    return pd.DataFrame(rows, columns=list(RESULT_COLUMNS))


# --- Synthetic data -------------------------------------------------------------------------

SYNTHETIC_MODELS = {
    # name: (ability, $ per 1M input tokens, $ per 1M output tokens, typical output tokens)
    "open-small": (-0.6, 0.05, 0.10, 350),
    "open-medium": (0.3, 0.20, 0.60, 400),
    "api-mid": (0.9, 1.00, 4.00, 450),
    "frontier": (1.8, 3.00, 15.00, 550),
}
_TYPE_BONUS = {
    "coding": {"open-small": -0.2, "open-medium": 0.2, "api-mid": 0.0, "frontier": 0.0},
    "math": {"open-small": -0.5, "open-medium": -0.2, "api-mid": 0.2, "frontier": 0.0},
    "writing": {"open-small": 0.5, "open-medium": 0.3, "api-mid": 0.0, "frontier": 0.0},
}
_TYPE_WORDS = {
    "coding": "python function bug test refactor api endpoint class compile stacktrace",
    "math": "integral proof equation solve probability matrix derivative theorem sum",
    "writing": "email summary rewrite tone paragraph draft blog headline outline",
}
_EASY_WORDS = "simple quick small basic short typo rename list format".split()
_HARD_WORDS = "concurrency distributed optimize proof edge-case migration security "\
              "recursive multi-step ambiguous".split()
_FILLER_WORDS = "please the a for with this our team need can you make it into and".split()


def make_synthetic_results(n_tasks=6000, seed=0, tenant="synthetic-tenant") -> pd.DataFrame:
    """Invent graded results with a learnable link between prompt words and success.

    For tests and the demo only. Numbers from it are not evidence about real models.
    """
    rng = np.random.default_rng(seed)
    types = np.array(list(_TYPE_WORDS))
    rows = []
    for task in range(n_tasks):
        task_type = types[rng.integers(len(types))]
        difficulty = rng.normal(0.0, 1.2)
        n_hard = rng.poisson(np.exp(0.8 * difficulty))
        n_easy = rng.poisson(np.exp(-0.8 * difficulty))
        words = (
            list(rng.choice(_TYPE_WORDS[task_type].split(), size=3))
            + list(rng.choice(_HARD_WORDS, size=n_hard))
            + list(rng.choice(_EASY_WORDS, size=n_easy))
            + list(rng.choice(_FILLER_WORDS, size=rng.integers(3, 10)))
        )
        rng.shuffle(words)
        prompt = " ".join(words)
        input_tokens = 150 + 2 * len(words)
        for model, (ability, price_in, price_out, out_tokens) in SYNTHETIC_MODELS.items():
            logit = 2.5 * (ability + _TYPE_BONUS[task_type][model] - difficulty)
            success = rng.random() < 1.0 / (1.0 + np.exp(-logit))
            output_tokens = max(20, int(out_tokens * (1 + 0.3 * difficulty) * rng.lognormal(0, 0.2)))
            cost = (input_tokens * price_in + output_tokens * price_out) / 1e6
            rows.append({
                "tenant": tenant,
                "task_id": f"syn-{task:06d}",
                "prompt": prompt,
                "task_type": task_type,
                "model": model,
                "success": float(success),
                "cost_usd": cost,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_tokens": 0,
                "latency_ms": float(output_tokens) * 10,
                "timestamp": "2026-11-06T00:00:00Z",
            })
    return pd.DataFrame(rows, columns=list(RESULT_COLUMNS))
