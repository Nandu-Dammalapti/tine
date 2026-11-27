# Melts wide fold/seed/epoch-suffixed metric columns (acc_fold0, acc_fold1, ...) in Colab tables into tidy long rows.
# Source: Gemini scheduled action, 2026-11-27 08:27
# Experimental; not imported by the tine package.
import json
import re
from pathlib import Path
from typing import List, Optional, Sequence, Union
import pandas as pd

SLICE_SUFFIX_RE = re.compile(
    r"^(?P<base>.+?)[_\-](?P<kind>fold|seed|epoch|split|run)[_\-]?(?P<idx>\d+)$",
    re.IGNORECASE,
)


def melt_sliced_metric_columns(
    df: pd.DataFrame,
    id_vars: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Convert wide slice-suffixed columns (e.g., acc_fold0, acc_fold1) into tidy rows."""
    matched_cols: List[tuple[str, str, str, int]] = []
    for col in df.columns:
        col_str = str(col)
        m = SLICE_SUFFIX_RE.match(col_str)
        if m:
            matched_cols.append(
                (
                    col_str,
                    m.group("base").strip("_").lower(),
                    m.group("kind").lower(),
                    int(m.group("idx")),
                )
            )

    if not matched_cols:
        raise ValueError("No fold/seed/epoch/split/run suffixed columns found to melt.")

    slice_col_names = [item[0] for item in matched_cols]
    meta_lookup = {
        item[0]: (item[1], item[2], item[3]) for item in matched_cols
    }

    if id_vars:
        keep_ids = [c for c in id_vars if c in df.columns]
    else:
        keep_ids = [c for c in df.columns if str(c) not in set(slice_col_names)]

    working = df.copy()
    if not keep_ids:
        working["_row_id"] = range(len(working))
        keep_ids = ["_row_id"]

    melted = working.melt(
        id_vars=keep_ids,
        value_vars=slice_col_names,
        var_name="_raw_col",
        value_name="value",
    )

    melted["metric_name"] = melted["_raw_col"].map(lambda c: meta_lookup[c][0])
    melted["slice_type"] = melted["_raw_col"].map(lambda c: meta_lookup[c][1])
    melted["slice_index"] = melted["_raw_col"].map(lambda c: meta_lookup[c][2])
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    melted = melted.drop(columns=["_raw_col"]).reset_index(drop=True)

    ordered_cols = keep_ids + ["metric_name", "slice_type", "slice_index", "value"]
    return melted[ordered_cols]


def format_colab_wide_metrics(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    id_vars: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load a wide Colab CSV or JSON metric file and export a tidy long-form table."""
    in_path = Path(input_path)
    out_path = (
        Path(output_path)
        if output_path
        else in_path.with_name(f"{in_path.stem}_long.csv")
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

    long_df = melt_sliced_metric_columns(df, id_vars=id_vars)
    if out_path.suffix.lower() == ".json":
        out_path.write_text(
            json.dumps(long_df.to_dict(orient="records"), indent=2), encoding="utf-8"
        )
    else:
        long_df.to_csv(out_path, index=False)

    return long_df
