# Removes exact or key-based duplicate rows from Colab CSV/JSON tables and reports what was dropped.
# Source: Gemini scheduled action, 2026-11-27 08:27
# Experimental; not imported by the tine package.
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import pandas as pd


def deduplicate_colab_table(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    subset_cols: Optional[Sequence[str]] = None,
    keep: str = "last",
    report_json: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Deduplicate rows in a Colab CSV or JSON export and write an audit summary."""
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

    rows_before = int(len(df))
    valid_subset: Optional[List[str]] = None
    if subset_cols:
        present = [str(c) for c in subset_cols if c in df.columns]
        if present:
            valid_subset = present

    keep_arg: Union[str, bool] = keep if keep in ("first", "last") else False
    dedup_df = df.drop_duplicates(subset=valid_subset, keep=keep_arg).reset_index(drop=True)
    rows_after = int(len(dedup_df))

    if out_path.suffix.lower() == ".json":
        out_path.write_text(
            json.dumps(dedup_df.to_dict(orient="records"), indent=2), encoding="utf-8"
        )
    else:
        dedup_df.to_csv(out_path, index=False)

    report: Dict[str, Any] = {
        "source_file": in_path.name,
        "output_file": out_path.name,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicates_removed": max(0, rows_before - rows_after),
        "subset_columns": valid_subset or [],
        "keep": keep,
    }

    if report_json:
        r_path = Path(report_json)
        r_path.parent.mkdir(parents=True, exist_ok=True)
        r_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report
