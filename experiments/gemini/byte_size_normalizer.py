# Converts size strings in Colab logs ("512 MB", "256 KiB") into numeric _bytes and _megabytes columns.
# Source: Gemini scheduled action, 2026-11-25 08:34
# Experimental; not imported by the tine package.
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd

SIZE_RE = re.compile(
    r"^\s*([-+]?\d+(?:\.\d+)?)\s*(b|bytes?|kb|kib|mb|mib|gb|gib|tb|tib)\s*$",
    re.IGNORECASE,
)

UNIT_BYTES: Dict[str, float] = {
    "b": 1.0,
    "byte": 1.0,
    "bytes": 1.0,
    "kb": 1000.0,
    "kib": 1024.0,
    "mb": 1_000_000.0,
    "mib": 1024.0 ** 2,
    "gb": 1_000_000_000.0,
    "gib": 1024.0 ** 3,
    "tb": 1_000_000_000_000.0,
    "tib": 1024.0 ** 4,
}


def parse_byte_size(val: Any) -> Optional[float]:
    """Parse a human-readable size string into total bytes as a float."""
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, (int, float)):
        num = float(val)
        return num if math.isfinite(num) else None
    if not isinstance(val, str):
        return None

    match = SIZE_RE.match(val.strip())
    if not match:
        return None

    magnitude = float(match.group(1))
    unit_key = match.group(2).lower()
    multiplier = UNIT_BYTES.get(unit_key)
    if multiplier is None:
        return None
    return round(magnitude * multiplier, 4)


def is_size_column(series: pd.Series, min_ratio: float = 0.8) -> bool:
    """Check whether a Series primarily contains byte/memory size strings."""
    non_null = series.dropna()
    if non_null.empty:
        return False
    parsed = non_null.map(parse_byte_size)
    return float(parsed.notna().mean()) >= min_ratio


def normalize_colab_byte_sizes(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Add numeric _bytes and _megabytes fields for size strings in Colab CSV or JSON."""
    in_path = Path(input_path)
    out_path = Path(output_path) if output_path else in_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = in_path.suffix.lower()
    converted: list[str] = []

    if suffix == ".csv":
        df = pd.read_csv(in_path)
        for col in list(df.columns):
            if is_size_column(df[col]):
                byte_vals = df[col].map(parse_byte_size)
                df[f"{col}_bytes"] = byte_vals
                df[f"{col}_megabytes"] = byte_vals.map(
                    lambda b: round(b / (1024.0 ** 2), 6) if b is not None else None
                )
                converted.append(str(col))
        df.to_csv(out_path, index=False)

    elif suffix == ".json":
        payload = json.loads(in_path.read_text(encoding="utf-8"))

        def _enrich_node(node: Any, path_key: str = "") -> Any:
            if isinstance(node, dict):
                out: Dict[str, Any] = {}
                for k, v in node.items():
                    child_key = f"{path_key}.{k}" if path_key else str(k)
                    out[k] = _enrich_node(v, child_key)
                    b_val = parse_byte_size(v)
                    if b_val is not None and isinstance(v, str):
                        out[f"{k}_bytes"] = b_val
                        out[f"{k}_megabytes"] = round(b_val / (1024.0 ** 2), 6)
                        if child_key not in converted:
                            converted.append(child_key)
                return out
            if isinstance(node, list):
                return [_enrich_node(item, path_key) for item in node]
            return node

        cleaned = _enrich_node(payload)
        out_path.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return {
        "source_file": in_path.name,
        "output_file": out_path.name,
        "converted_fields": converted,
    }
