# Adds ms_per_sample, samples_per_sec and tokens_per_sec columns to Colab benchmark timing tables.
# Source: Gemini scheduled action, 2026-11-26 08:35
# Experimental; not imported by the tine package.
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd


def compute_throughput_columns(
    df: pd.DataFrame,
    elapsed_sec_col: str = "elapsed_sec",
    sample_count_col: str = "num_samples",
    token_count_col: Optional[str] = "num_tokens",
) -> pd.DataFrame:
    """Append ms_per_sample, samples_per_sec, and optional tokens_per_sec columns."""
    out = df.copy()
    if elapsed_sec_col not in out.columns or sample_count_col not in out.columns:
        raise KeyError(
            f"Required columns '{elapsed_sec_col}' and '{sample_count_col}' not found."
        )

    elapsed = pd.to_numeric(out[elapsed_sec_col], errors="coerce").to_numpy(dtype=float)
    samples = pd.to_numeric(out[sample_count_col], errors="coerce").to_numpy(dtype=float)

    valid_time = np.isfinite(elapsed) & (elapsed > 0.0)
    valid_samples = np.isfinite(samples) & (samples > 0.0)
    both_valid = valid_time & valid_samples

    ms_per_sample = np.where(
        both_valid,
        np.round((elapsed * 1000.0) / np.where(valid_samples, samples, 1.0), 6),
        np.nan,
    )
    samples_per_sec = np.where(
        both_valid,
        np.round(samples / np.where(valid_time, elapsed, 1.0), 4),
        np.nan,
    )

    out["ms_per_sample"] = ms_per_sample
    out["samples_per_sec"] = samples_per_sec

    if token_count_col and token_count_col in out.columns:
        tokens = pd.to_numeric(out[token_count_col], errors="coerce").to_numpy(dtype=float)
        valid_tokens = valid_time & np.isfinite(tokens) & (tokens >= 0.0)
        out["tokens_per_sec"] = np.where(
            valid_tokens,
            np.round(tokens / np.where(valid_time, elapsed, 1.0), 4),
            np.nan,
        )

    return out


def format_colab_latency_table(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    elapsed_sec_col: str = "elapsed_sec",
    sample_count_col: str = "num_samples",
    token_count_col: Optional[str] = "num_tokens",
) -> Dict[str, Any]:
    """Load a Colab CSV or JSON benchmark table and export normalized throughput metrics."""
    in_path = Path(input_path)
    out_path = (
        Path(output_path)
        if output_path
        else in_path.with_name(f"{in_path.stem}_throughput.csv")
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

    enriched = compute_throughput_columns(
        df,
        elapsed_sec_col=elapsed_sec_col,
        sample_count_col=sample_count_col,
        token_count_col=token_count_col,
    )

    if out_path.suffix.lower() == ".json":
        out_path.write_text(
            json.dumps(enriched.to_dict(orient="records"), indent=2), encoding="utf-8"
        )
    else:
        enriched.to_csv(out_path, index=False)

    mean_sps = (
        round(float(np.nanmean(enriched["samples_per_sec"])), 4)
        if not enriched.empty
        else None
    )
    return {
        "source_file": in_path.name,
        "output_file": out_path.name,
        "row_count": int(len(enriched)),
        "mean_samples_per_sec": mean_sps,
    }
