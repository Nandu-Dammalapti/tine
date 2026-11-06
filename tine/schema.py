"""The result-row schema every loader produces (one row per task x model attempt)."""

import numpy as np
import pandas as pd

RESULT_COLUMNS = (
    "tenant",
    "task_id",
    "prompt",
    "task_type",
    "model",
    "success",
    "cost_usd",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "latency_ms",
    "timestamp",
)

_SUCCESS_VALUES = {
    "true": 1.0,
    "false": 0.0,
    "success": 1.0,
    "failure": 0.0,
    "pending": np.nan,
    "unknown": np.nan,
    "": np.nan,
}


def _normalize_success(value):
    """Map an outcome to 1.0 (success), 0.0 (failure) or NaN (pending/unknown)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _SUCCESS_VALUES:
            return _SUCCESS_VALUES[key]
        try:
            value = float(key)  # CSVs with mixed values read numbers as text, e.g. "1.0"
        except ValueError:
            raise ValueError(f"Unrecognized success value {value!r}") from None
    value = float(value)
    if value not in (0.0, 1.0):
        raise ValueError(f"success must be 0/1, got {value!r}")
    return value


def validate_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Check columns and types; return a copy with `success` as 1.0 / 0.0 / NaN.

    Pending outcomes stay NaN, never failure: production signals such as a ticket
    staying closed can arrive days later.
    """
    missing = [c for c in RESULT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing result columns: {missing}")
    out = frame.loc[:, list(RESULT_COLUMNS)].copy()
    out["success"] = out["success"].map(_normalize_success).astype(float)
    for col in ("cost_usd", "input_tokens", "output_tokens", "cached_tokens", "latency_ms"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    if (out["cost_usd"] < 0).any():
        raise ValueError("cost_usd must be non-negative")
    for col in ("tenant", "task_id", "task_type", "model"):
        out[col] = out[col].astype(str)
    out["prompt"] = out["prompt"].fillna("").astype(str)
    dupes = out.duplicated(["tenant", "task_id", "model"])
    if dupes.any():
        example = out.loc[dupes, ["tenant", "task_id", "model"]].iloc[0].tolist()
        raise ValueError(
            "One row per (tenant, task_id, model) is required; fold retries into one row "
            f"with their summed cost. First duplicate: {example}"
        )
    return out
