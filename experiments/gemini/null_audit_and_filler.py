# Audits missing values per column in Colab CSV/JSON exports and optionally fills them (mean/median/mode/constant).
# Source: Gemini scheduled action, 2026-11-24 08:36
# Experimental; not imported by the tine package.
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd


def audit_and_fill_missing(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    strategy: str = "none",
    fill_constant: Any = 0,
    report_json: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Audit null counts per column and optionally impute missing values in Colab exports."""
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

    # Treat whitespace-only strings as missing
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].replace(r"^\s*$", pd.NA, regex=True)

    total_rows = int(len(df))
    col_stats: List[Dict[str, Any]] = []
    mode_choice = strategy.strip().lower()

    for col in df.columns:
        null_cnt = int(df[col].isna().sum())
        null_ratio = round(null_cnt / float(total_rows), 6) if total_rows > 0 else 0.0

        if null_cnt > 0 and mode_choice != "none":
            if mode_choice == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            elif mode_choice == "median" and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            elif mode_choice == "mode":
                modes = df[col].mode(dropna=True)
                if not modes.empty:
                    df[col] = df[col].fillna(modes.iloc[0])
            elif mode_choice == "constant":
                df[col] = df[col].fillna(fill_constant)

        col_stats.append(
            {
                "column": str(col),
                "missing_before": null_cnt,
                "missing_ratio_before": null_ratio,
                "missing_after": int(df[col].isna().sum()),
            }
        )

    if out_path.suffix.lower() == ".json":
        out_path.write_text(
            json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8"
        )
    else:
        df.to_csv(out_path, index=False)

    report: Dict[str, Any] = {
        "source_file": in_path.name,
        "output_file": out_path.name,
        "row_count": total_rows,
        "imputation_strategy": mode_choice,
        "columns": col_stats,
    }

    if report_json:
        r_path = Path(report_json)
        r_path.parent.mkdir(parents=True, exist_ok=True)
        r_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report
