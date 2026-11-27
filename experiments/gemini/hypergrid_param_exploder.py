# Parses stringified list/dict hyperparameter cells in Colab tables and adds item-count and min/max helper columns.
# Source: Gemini scheduled action, 2026-11-26 08:35
# Experimental; not imported by the tine package.
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


def _try_parse_literal(val: Any) -> Any:
    """Safely parse stringified Python/JSON lists or dicts into native containers."""
    if isinstance(val, (list, tuple, dict)):
        return val
    if not isinstance(val, str):
        return val
    text = val.strip()
    if not text or text[0] not in ("[", "{", "("):
        return val
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return val


def format_colab_hypergrid_columns(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Standardize list/dict hyperparameter cells in Colab CSV or JSON exports."""
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

    enriched_cols: List[str] = []
    for col in list(df.columns):
        parsed_series = df[col].map(_try_parse_literal)
        has_containers = parsed_series.map(
            lambda x: isinstance(x, (list, tuple, dict))
        ).any()
        if not has_containers:
            continue

        enriched_cols.append(str(col))
        df[col] = parsed_series.map(
            lambda x: json.dumps(list(x) if isinstance(x, tuple) else x)
            if isinstance(x, (list, tuple, dict))
            else x
        )
        df[f"{col}_item_count"] = parsed_series.map(
            lambda x: len(x) if isinstance(x, (list, tuple, dict)) else None
        )

        # Extract numeric summary helpers when cells are non-empty numeric sequences
        def _num_stats(item: Any) -> tuple[Optional[float], Optional[float]]:
            if not isinstance(item, (list, tuple)) or not item:
                return None, None
            nums = []
            for el in item:
                if isinstance(el, (int, float)) and not isinstance(el, bool) and np.isfinite(el):
                    nums.append(float(el))
            if len(nums) != len(item):
                return None, None
            return round(float(np.min(nums)), 8), round(float(np.max(nums)), 8)

        stats_pairs = parsed_series.map(_num_stats)
        if stats_pairs.map(lambda p: p[0] is not None).any():
            df[f"{col}_min"] = stats_pairs.map(lambda p: p[0])
            df[f"{col}_max"] = stats_pairs.map(lambda p: p[1])

    if out_path.suffix.lower() == ".json":
        out_path.write_text(
            json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8"
        )
    else:
        df.to_csv(out_path, index=False)

    return {
        "source_file": in_path.name,
        "output_file": out_path.name,
        "formatted_container_columns": enriched_cols,
    }
