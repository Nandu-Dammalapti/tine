"""Validate measured check evidence before using it to simulate escalation."""

import numpy as np
import pandas as pd

from tine.data import _success, _unique_csv_header


def read_checks(path):
    """Preserve literal task/model IDs and distinguish empty checks from failures."""
    with open(path, encoding="utf-8-sig", newline="") as stream:
        _unique_csv_header(stream)
        return pd.read_csv(stream, dtype=str, keep_default_na=False)


def prepare_checks(checks, tasks, models):
    """Return a tenant-scoped Boolean pass matrix and explicit evidence coverage.

    Missing/unknown results conservatively pass without evidence, as no check
    demonstrated a reason to escalate. Duplicate measurements require an
    explicit choice of run by the caller, rather than aggregation.
    """
    if not isinstance(checks, pd.DataFrame) or not checks.columns.is_unique:
        raise ValueError("checks must be a table with unique columns")
    required = {"task_id", "model", "passed"}
    if required - set(checks.columns):
        raise ValueError("checks require task_id, model, and passed columns")
    if tasks.empty or tasks.tenant.nunique() != 1:
        raise ValueError("Check alignment requires one nonempty tenant task set")
    if tasks.task_id.duplicated().any() or len(set(models)) != len(models):
        raise ValueError("Check alignment requires unique task IDs and model names")
    tenant = tasks.tenant.iloc[0]
    selected = checks.copy()
    scoped = "tenant" in selected
    for column in ["task_id", "model", *(["tenant"] if scoped else [])]:
        if not selected[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Check {column} must contain nonempty text identifiers")
    if scoped:
        selected = selected.loc[selected.tenant == tenant]
    selected = selected.loc[selected.task_id.isin(tasks.task_id) & selected.model.isin(models)]
    if selected.duplicated(["task_id", "model"]).any():
        raise ValueError("Duplicate check results for tenant/task_id/model; choose one check run explicitly")

    def outcome(value):
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            if value in {"0.0", "1.0"}:
                value = float(value)
        try:
            return _success(value)
        except (TypeError, ValueError):
            raise ValueError("passed must contain Boolean/zero-one check results or an empty unknown") from None

    selected = selected.assign(passed=selected.passed.map(outcome))
    pivot = selected.pivot(index="task_id", columns="model", values="passed")
    aligned = pivot.reindex(index=tasks.task_id.to_numpy(), columns=models)
    known = aligned.notna().to_numpy()
    passes = np.where(known, aligned.to_numpy(), True).astype(bool)
    report = {
        "tenant": tenant, "identity_scope": "tenant/task/model" if scoped else "single-tenant task/model",
        "source_rows": len(checks), "matched_rows": len(selected),
        "ignored_rows": len(checks) - len(selected), "expected_checks": int(known.size),
        "known_checks": int(known.sum()), "unknown_checks": int((~known).sum()),
        "unknown_policy": "pass_without_evidence",
        "per_model": {model: {"known": int(known[:, column].sum()),
                              "expected": len(tasks)} for column, model in enumerate(models)},
    }
    return passes, report
