# Buckets Colab score/latency distributions into quantile bins and reports q05-q95 and IQR.
# Source: Gemini scheduled action, 2026-11-24 14:32
# Experimental; not imported by the tine package.
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd


def compute_quantile_buckets(
    values: Sequence[float],
    n_quantiles: int = 10,
) -> Dict[str, Any]:
    """Partition finite numeric values into quantile buckets and compute key percentiles."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0 or n_quantiles < 1:
        raise ValueError("values must contain at least one finite number and n_quantiles >= 1.")

    percentiles = {
        "q05": float(np.percentile(finite, 5.0)),
        "q25": float(np.percentile(finite, 25.0)),
        "q50": float(np.percentile(finite, 50.0)),
        "q75": float(np.percentile(finite, 75.0)),
        "q95": float(np.percentile(finite, 95.0)),
    }
    percentiles["iqr"] = round(percentiles["q75"] - percentiles["q25"], 6)
    summary = {
        "n_samples": int(finite.size),
        "n_quantiles": int(n_quantiles),
        "min": round(float(np.min(finite)), 6),
        "max": round(float(np.max(finite)), 6),
        "mean": round(float(np.mean(finite)), 6),
        **{k: round(v, 6) for k, v in percentiles.items()},
    }

    quant_edges = np.linspace(0.0, 100.0, n_quantiles + 1)
    edges = np.percentile(finite, quant_edges)
    buckets: List[Dict[str, Any]] = []

    for idx in range(n_quantiles):
        low = float(edges[idx])
        high = float(edges[idx + 1])
        if idx == n_quantiles - 1:
            mask = (finite >= low) & (finite <= high)
        else:
            mask = (finite >= low) & (finite < high)
        slice_vals = finite[mask]
        cnt = int(slice_vals.size)

        buckets.append(
            {
                "quantile_rank": idx + 1,
                "bucket_label": f"q{idx + 1}_of_{n_quantiles}",
                "lower_bound": round(low, 6),
                "upper_bound": round(high, 6),
                "sample_count": cnt,
                "sample_share": round(cnt / float(finite.size), 6),
                "mean_value": round(float(np.mean(slice_vals)), 6) if cnt > 0 else None,
                "median_value": round(float(np.median(slice_vals)), 6) if cnt > 0 else None,
            }
        )

    return {"summary": summary, "buckets": buckets}


def format_colab_quantile_bins(
    input_path: Union[str, Path],
    value_col: str = "score",
    n_quantiles: int = 10,
    output_csv: Optional[Union[str, Path]] = None,
    summary_json: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load a Colab CSV or JSON file and export quantile bucket rows + percentile summary."""
    in_path = Path(input_path)
    suffix = in_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(in_path)
        target_col = value_col if value_col in df.columns else next(
            (c for c in df.select_dtypes(include="number").columns), None
        )
        if not target_col:
            raise KeyError(f"No numeric column '{value_col}' found in {in_path.name}")
        raw_vals = df[target_col].tolist()
    elif suffix == ".json":
        payload = json.loads(in_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and value_col in payload and isinstance(payload[value_col], list):
            raw_vals = payload[value_col]
        elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
            df_p = pd.DataFrame(payload)
            if value_col not in df_p.columns:
                raise KeyError(f"Key '{value_col}' not found in {in_path.name}")
            raw_vals = df_p[value_col].tolist()
        elif isinstance(payload, list):
            raw_vals = payload
        else:
            raise ValueError(f"Unsupported JSON structure in {in_path.name}")
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    result = compute_quantile_buckets(raw_vals, n_quantiles=n_quantiles)

    if output_csv is None and summary_json is None:
        output_csv = in_path.with_name(f"{in_path.stem}_quantile_bins.csv")
        summary_json = in_path.with_name(f"{in_path.stem}_quantile_summary.json")

    if output_csv:
        c_path = Path(output_csv)
        c_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result["buckets"]).to_csv(c_path, index=False)

    if summary_json:
        s_path = Path(summary_json)
        s_path.parent.mkdir(parents=True, exist_ok=True)
        s_path.write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")

    return result
