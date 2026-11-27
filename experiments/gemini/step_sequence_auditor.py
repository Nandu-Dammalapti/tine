# Audits Colab step/epoch logs for out-of-order, duplicate and missing steps, and can sort or dedupe them.
# Source: Gemini scheduled action, 2026-11-25 08:34
# Experimental; not imported by the tine package.
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


def audit_and_order_steps(
    input_path: Union[str, Path],
    step_col: str = "step",
    output_path: Optional[Union[str, Path]] = None,
    sort_steps: bool = True,
    drop_duplicate_steps: bool = False,
    report_json: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Check step monotonicity, duplicates, and gaps in Colab logs and optionally sort."""
    in_path = Path(input_path)
    out_path = Path(output_path) if output_path else in_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = in_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(in_path)
    elif suffix == ".json":
        payload = json.loads(in_path.read_text(encoding="utf-8"))
        df = pd.DataFrame(payload if isinstance(payload, list) else [payload])
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    if step_col not in df.columns:
        raise KeyError(f"Step column '{step_col}' not found in {in_path.name}")

    steps_num = pd.to_numeric(df[step_col], errors="coerce")
    valid_steps = steps_num.dropna().to_numpy(dtype=float)

    diffs = np.diff(valid_steps) if valid_steps.size > 1 else np.array([], dtype=float)
    is_monotonic_non_decreasing = bool(np.all(diffs >= 0)) if diffs.size > 0 else True
    duplicate_count = int(steps_num.duplicated(keep="first").sum())

    gaps: List[Dict[str, int]] = []
    if valid_steps.size > 1:
        sorted_ints = np.sort(valid_steps.astype(int))
        for i in range(len(sorted_ints) - 1):
            curr_s, next_s = int(sorted_ints[i]), int(sorted_ints[i + 1])
            if next_s - curr_s > 1:
                gaps.append({"from_step": curr_s, "to_step": next_s, "missing_steps": next_s - curr_s - 1})

    working_df = df.copy()
    working_df["_step_sort_key"] = steps_num
    if sort_steps:
        working_df = working_df.sort_values(by="_step_sort_key", kind="mergesort", na_position="last")
    if drop_duplicate_steps:
        working_df = working_df.drop_duplicates(subset=[step_col], keep="last")
    working_df = working_df.drop(columns=["_step_sort_key"]).reset_index(drop=True)

    if out_path.suffix.lower() == ".json":
        out_path.write_text(
            json.dumps(working_df.to_dict(orient="records"), indent=2), encoding="utf-8"
        )
    else:
        working_df.to_csv(out_path, index=False)

    report: Dict[str, Any] = {
        "source_file": in_path.name,
        "output_file": out_path.name,
        "step_column": step_col,
        "row_count": int(len(working_df)),
        "is_monotonic_non_decreasing": is_monotonic_non_decreasing,
        "duplicate_step_count": duplicate_count,
        "gap_count": len(gaps),
        "gaps": gaps[:25],
    }

    if report_json:
        r_path = Path(report_json)
        r_path.parent.mkdir(parents=True, exist_ok=True)
        r_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report
