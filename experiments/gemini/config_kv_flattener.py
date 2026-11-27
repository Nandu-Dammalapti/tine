# Flattens a nested Colab config JSON into a typed key-value table (CSV) and a flat path-to-value JSON map.
# Source: Gemini scheduled action, 2026-11-27 08:27
# Experimental; not imported by the tine package.
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd


def flatten_config_entries(
    config_obj: Any,
    prefix: str = "",
    sep: str = ".",
) -> List[Dict[str, Any]]:
    """Flatten nested configuration dicts/lists into typed key-value rows."""
    rows: List[Dict[str, Any]] = []

    if isinstance(config_obj, dict):
        for k, v in config_obj.items():
            key_path = f"{prefix}{sep}{k}" if prefix else str(k)
            rows.extend(flatten_config_entries(v, prefix=key_path, sep=sep))
    elif isinstance(config_obj, list) and any(isinstance(el, (dict, list)) for el in config_obj):
        for idx, el in enumerate(config_obj):
            key_path = f"{prefix}[{idx}]"
            rows.extend(flatten_config_entries(el, prefix=key_path, sep=sep))
    else:
        leaf_name = prefix.split(sep)[-1] if prefix else "value"
        if "[" in leaf_name:
            leaf_name = leaf_name.split("[", 1)[0]
        rows.append(
            {
                "param_path": prefix,
                "param_leaf": leaf_name,
                "param_value": json.dumps(config_obj) if isinstance(config_obj, list) else config_obj,
                "value_type": type(config_obj).__name__ if config_obj is not None else "null",
            }
        )

    return rows


def format_colab_config_dump(
    input_json: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    flat_json: Optional[Union[str, Path]] = None,
    sep: str = ".",
) -> Dict[str, Any]:
    """Convert a Colab nested config JSON file into a flat KV CSV and flat JSON map."""
    in_path = Path(input_json)
    payload = json.loads(in_path.read_text(encoding="utf-8"))

    rows = flatten_config_entries(payload, sep=sep)
    flat_map: Dict[str, Any] = {r["param_path"]: r["param_value"] for r in rows}

    if output_csv is None and flat_json is None:
        output_csv = in_path.with_name(f"{in_path.stem}_kv.csv")
        flat_json = in_path.with_name(f"{in_path.stem}_flat.json")

    if output_csv:
        c_path = Path(output_csv)
        c_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(c_path, index=False)

    if flat_json:
        j_path = Path(flat_json)
        j_path.parent.mkdir(parents=True, exist_ok=True)
        j_path.write_text(json.dumps(flat_map, indent=2), encoding="utf-8")

    return {"param_count": len(rows), "entries": rows, "flat_map": flat_map}
