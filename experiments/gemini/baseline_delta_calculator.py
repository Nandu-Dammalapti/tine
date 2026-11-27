# Adds absolute delta and percentage uplift columns against a baseline row in Colab experiment tables.
# Source: Gemini scheduled action, 2026-11-24 08:36
# Experimental; not imported by the tine package.
import json
from pathlib import Path
from typing import Optional, Sequence, Union
import numpy as np
import pandas as pd


def compute_baseline_deltas(
    df: pd.DataFrame,
    baseline_filter_col: str = "model_name",
    baseline_value: str = "baseline",
    metric_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Append delta and relative percentage change columns against a reference baseline row."""
    out = df.copy()
    if baseline_filter_col not in out.columns:
        raise KeyError(f"Baseline column '{baseline_filter_col}' not found.")

    base_rows = out[out[baseline_filter_col].astype(str) == str(baseline_value)]
    if base_rows.empty:
        raise ValueError(
            f"No row found where {baseline_filter_col} == '{baseline_value}'."
        )

    base_row = base_rows.iloc[0]
    if metric_cols:
        targets = [c for c in metric_cols if c in out.columns]
    else:
        targets = [
            c
            for c in out.select_dtypes(include="number").columns
            if not str(c).lower().startswith(("seed", "fold", "epoch", "step", "id"))
        ]

    for col in targets:
        base_num = pd.to_numeric(base_row[col], errors="coerce")
        if pd.isna(base_num):
            continue
        b_val = float(base_num)
        col_vals = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)

        deltas = np.where(np.isfinite(col_vals), np.round(col_vals - b_val, 6), np.nan)
        # Element-wise `&` (Gemini's version used `and`, which raises on arrays longer than 1).
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_pct = np.where(
                np.isfinite(col_vals) & (abs(b_val) > 0),
                np.round(((col_vals - b_val) / abs(b_val)) * 100.0, 4),
                np.nan,
            )

        out[f"{col}_delta_base"] = deltas
        out[f"{col}_uplift_pct"] = rel_pct

    return out


def format_colab_baseline_deltas(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    baseline_filter_col: str = "model_name",
    baseline_value: str = "baseline",
    metric_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load a Colab CSV or JSON experiment table and export rows enriched with baseline deltas."""
    in_path = Path(input_path)
    out_path = (
        Path(output_path)
        if output_path
        else in_path.with_name(f"{in_path.stem}_with_deltas.csv")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = in_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(in_path)
    elif suffix == ".json":
        df = pd.DataFrame(json.loads(in_path.read_text(encoding="utf-8")))
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    enriched = compute_baseline_deltas(
        df,
        baseline_filter_col=baseline_filter_col,
        baseline_value=baseline_value,
        metric_cols=metric_cols,
    )
    if out_path.suffix.lower() == ".json":
        out_path.write_text(
            json.dumps(enriched.to_dict(orient="records"), indent=2), encoding="utf-8"
        )
    else:
        enriched.to_csv(out_path, index=False)

    return enriched
