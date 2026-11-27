# Merges every per-run JSON file in a Colab results folder into one CSV (and optional JSON) with source columns.
# Source: Gemini scheduled action, 2026-11-25 08:34
# Experimental; not imported by the tine package.
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd


def _flatten_shallow(obj: Dict[str, Any], prefix: str = "", sep: str = "_") -> Dict[str, Any]:
    """Flatten nested single-level dicts so record rows stay tabular."""
    flat: Dict[str, Any] = {}
    for k, v in obj.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(_flatten_shallow(v, prefix=key, sep=sep))
        else:
            flat[key] = v
    return flat


def consolidate_colab_json_folder(
    input_dir: Union[str, Path],
    output_csv: Union[str, Path] = "colab/consolidated_results.csv",
    output_json: Optional[Union[str, Path]] = None,
    pattern: str = "*.json",
    exclude_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Merge matching JSON result files in a Colab folder into a single CSV/JSON table."""
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    excluded = set(exclude_names or ["manifest.json", "consolidated_results.json"])
    records: List[Dict[str, Any]] = []

    for json_path in sorted(root.glob(pattern)):
        if not json_path.is_file() or json_path.name in excluded:
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else [payload]

        for item in items:
            row = (
                _flatten_shallow(item)
                if isinstance(item, dict)
                else {"value": item}
            )
            row["source_file"] = json_path.name
            row["run_stem"] = json_path.stem
            records.append(row)

    df = pd.DataFrame(records)
    # Place run identifiers up front for easier inspection
    front_cols = [c for c in ("run_stem", "source_file") if c in df.columns]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]

    c_out = Path(output_csv)
    c_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(c_out, index=False)

    if output_json:
        j_out = Path(output_json)
        j_out.parent.mkdir(parents=True, exist_ok=True)
        j_out.write_text(
            json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8"
        )

    return df
