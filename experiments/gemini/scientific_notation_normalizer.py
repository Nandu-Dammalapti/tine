# Parses scientific-notation strings ("1.25e-04", "2.5 x 10^-3") in Colab tables into floats, with optional log10 columns.
# Source: Gemini scheduled action, 2026-11-25 08:34
# Experimental; not imported by the tine package.
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd

SCI_TEXT_RE = re.compile(
    r"^\s*([-+]?\d+(?:\.\d+)?)\s*(?:[eE]|(?:\s*[x×*]\s*10\s*\^\s*))\s*([-+]?\d+)\s*$"
)


def parse_scientific_value(val: Any) -> Optional[float]:
    """Convert standard or math-notation scientific strings into float values."""
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) if math.isfinite(float(val)) else None
    if not isinstance(val, str):
        return None

    text = val.strip()
    if not text:
        return None

    match = SCI_TEXT_RE.match(text)
    if match:
        mantissa = float(match.group(1))
        exponent = int(match.group(2))
        result = mantissa * (10.0 ** exponent)
        return result if math.isfinite(result) else None

    try:
        parsed = float(text)
        return parsed if math.isfinite(parsed) else None
    except ValueError:
        return None


def is_scientific_column(series: pd.Series, min_ratio: float = 0.8) -> bool:
    """Detect string columns primarily encoding scientific or exponential numbers."""
    non_null = series.dropna()
    if non_null.empty or not pd.api.types.is_object_dtype(non_null) and not pd.api.types.is_string_dtype(non_null):
        return False
    has_exp_token = non_null.astype(str).str.contains(
        r"(?:[eE][-+]?\d+|10\s*\^)", regex=True
    )
    if not bool(has_exp_token.any()):
        return False
    parsed = non_null.map(parse_scientific_value)
    return float(parsed.notna().mean()) >= min_ratio


def normalize_colab_scientific_numbers(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    include_log10: bool = False,
) -> Dict[str, Any]:
    """Normalize scientific-notation strings in a Colab CSV or JSON into numeric floats."""
    in_path = Path(input_path)
    out_path = Path(output_path) if output_path else in_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = in_path.suffix.lower()
    converted: list[str] = []

    if suffix == ".csv":
        df = pd.read_csv(in_path)
        for col in list(df.columns):
            if is_scientific_column(df[col]):
                nums = df[col].map(parse_scientific_value)
                df[col] = nums
                converted.append(str(col))
                if include_log10:
                    df[f"{col}_log10"] = nums.map(
                        lambda x: round(math.log10(x), 6) if (x is not None and x > 0.0) else None
                    )
        df.to_csv(out_path, index=False)

    elif suffix == ".json":
        payload = json.loads(in_path.read_text(encoding="utf-8"))

        def _clean_node(node: Any, path_key: str = "") -> Any:
            if isinstance(node, dict):
                out: Dict[str, Any] = {}
                for k, v in node.items():
                    child_key = f"{path_key}.{k}" if path_key else str(k)
                    out[k] = _clean_node(v, child_key)
                    if isinstance(v, str) and ("e" in v.lower() or "10^" in v.lower()):
                        parsed = parse_scientific_value(v)
                        if parsed is not None:
                            out[k] = parsed
                            if child_key not in converted:
                                converted.append(child_key)
                            if include_log10 and parsed > 0.0:
                                out[f"{k}_log10"] = round(math.log10(parsed), 6)
                return out
            if isinstance(node, list):
                return [_clean_node(item, path_key) for item in node]
            return node

        cleaned = _clean_node(payload)
        out_path.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return {
        "source_file": in_path.name,
        "output_file": out_path.name,
        "converted_fields": converted,
    }
