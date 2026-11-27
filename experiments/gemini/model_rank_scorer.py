# Ranks Colab experiment runs by a weighted, min-max normalized composite of higher- and lower-is-better metrics.
# Source: Gemini scheduled action, 2026-11-26 14:17
# Experimental; not imported by the tine package.
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union
import numpy as np
import pandas as pd


def score_and_rank_runs(
    df: pd.DataFrame,
    maximize_cols: Sequence[str] = ("accuracy", "f1_score", "roc_auc", "r2_score"),
    minimize_cols: Sequence[str] = ("loss", "val_loss", "rmse", "mse", "mae"),
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute min-max normalized composite rank scores and rank positions across rows."""
    if df.empty:
        return df.copy()

    out = df.copy()
    active_max = [c for c in maximize_cols if c in out.columns]
    active_min = [c for c in minimize_cols if c in out.columns]
    metric_cols = active_max + active_min

    if not metric_cols:
        raise ValueError("None of the specified maximize_cols or minimize_cols exist in DataFrame.")

    weight_map: Dict[str, float] = {
        col: float((weights or {}).get(col, 1.0)) for col in metric_cols
    }
    total_weight = sum(abs(w) for w in weight_map.values())
    if total_weight <= 0.0:
        raise ValueError("Total metric weight must be greater than zero.")

    weighted_sum = np.zeros(len(out), dtype=float)

    for col in metric_cols:
        series = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(series)
        if not np.any(finite_mask):
            continue

        min_v = float(np.min(series[finite_mask]))
        max_v = float(np.max(series[finite_mask]))
        span = max_v - min_v

        if span > 0.0:
            norm_vals = np.where(finite_mask, (series - min_v) / span, 0.0)
        else:
            norm_vals = np.where(finite_mask, 1.0, 0.0)

        if col in active_min:
            norm_vals = 1.0 - norm_vals

        weighted_sum += norm_vals * weight_map[col]

    composite_score = np.round(weighted_sum / total_weight, 6)
    out["rank_score"] = composite_score
    out["rank_position"] = (
        pd.Series(composite_score, index=out.index)
        .rank(ascending=False, method="min")
        .astype(int)
    )
    out["is_best_model"] = out["rank_position"] == 1

    return out.sort_values(by=["rank_position"], kind="mergesort").reset_index(drop=True)


def format_colab_model_rankings(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    maximize_cols: Sequence[str] = ("accuracy", "f1_score", "roc_auc", "r2_score"),
    minimize_cols: Sequence[str] = ("loss", "val_loss", "rmse", "mse", "mae"),
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Load a Colab CSV or JSON experiment table and export rows sorted by composite rank."""
    in_path = Path(input_path)
    out_path = (
        Path(output_path)
        if output_path
        else in_path.with_name(f"{in_path.stem}_ranked.csv")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = in_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(in_path)
    elif suffix == ".json":
        payload = json.loads(in_path.read_text(encoding="utf-8"))
        df = pd.DataFrame(payload if isinstance(payload, list) else [payload])
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    ranked_df = score_and_rank_runs(
        df,
        maximize_cols=maximize_cols,
        minimize_cols=minimize_cols,
        weights=weights,
    )

    if out_path.suffix.lower() == ".json":
        out_path.write_text(
            json.dumps(ranked_df.to_dict(orient="records"), indent=2), encoding="utf-8"
        )
    else:
        ranked_df.to_csv(out_path, index=False)

    best_row: Optional[Dict[str, Any]] = (
        ranked_df.iloc[0].to_dict() if not ranked_df.empty else None
    )
    return {
        "source_file": in_path.name,
        "output_file": out_path.name,
        "row_count": int(len(ranked_df)),
        "best_rank_score": best_row["rank_score"] if best_row else None,
    }
