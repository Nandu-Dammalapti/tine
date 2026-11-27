# Adds rolling mean, rolling std and EMA columns to numeric training/eval curves in Colab exports.
# Source: Gemini scheduled action, 2026-11-26 11:29
# Experimental; not imported by the tine package.
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd


def compute_smoothed_series(
    series: pd.Series,
    window: int = 5,
    ema_alpha: Optional[float] = None,
    decimals: int = 6,
) -> Dict[str, List[Optional[float]]]:
    """Compute rolling mean, rolling std, and exponential moving average for a numeric Series."""
    num = pd.to_numeric(series, errors="coerce").astype(float)
    win = max(1, int(window))
    alpha = ema_alpha if ema_alpha is not None else (2.0 / (win + 1.0))

    roll_mean = num.rolling(window=win, min_periods=1).mean()
    roll_std = num.rolling(window=win, min_periods=1).std(ddof=0).fillna(0.0)
    ema = num.ewm(alpha=alpha, adjust=False).mean()

    def _to_clean_list(s: pd.Series) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        for val in s.to_numpy(dtype=float):
            if np.isfinite(val):
                out.append(round(float(val), decimals))
            else:
                out.append(None)
        return out

    return {
        "rolling_mean": _to_clean_list(roll_mean),
        "rolling_std": _to_clean_list(roll_std),
        "ema": _to_clean_list(ema),
    }


def smooth_colab_series_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    columns: Optional[Sequence[str]] = None,
    window: int = 5,
    ema_alpha: Optional[float] = None,
    decimals: int = 6,
) -> Dict[str, Any]:
    """Append rolling and EMA columns for numeric series in a Colab CSV or JSON export."""
    in_path = Path(input_path)
    out_path = (
        Path(output_path)
        if output_path
        else in_path.with_name(f"{in_path.stem}_smoothed.csv")
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

    if columns:
        target_cols = [c for c in columns if c in df.columns]
    else:
        target_cols = [
            c
            for c in df.select_dtypes(include="number").columns
            if str(c).lower() not in {"step", "epoch", "iter", "iteration", "seed", "fold", "id"}
        ]

    smoothed_cols: List[str] = []
    for col in target_cols:
        curves = compute_smoothed_series(
            df[col], window=window, ema_alpha=ema_alpha, decimals=decimals
        )
        df[f"{col}_roll{window}_mean"] = curves["rolling_mean"]
        df[f"{col}_roll{window}_std"] = curves["rolling_std"]
        df[f"{col}_ema"] = curves["ema"]
        smoothed_cols.append(str(col))

    if out_path.suffix.lower() == ".json":
        out_path.write_text(
            json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8"
        )
    else:
        df.to_csv(out_path, index=False)

    return {
        "source_file": in_path.name,
        "output_file": out_path.name,
        "window": int(window),
        "smoothed_columns": smoothed_cols,
    }
