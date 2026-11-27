# Cleans Colab .jsonl exports: skips blank/corrupt lines, flattens nested dicts, writes clean JSONL and an optional CSV.
# Source: Gemini scheduled action, 2026-11-24 08:36
# Experimental; not imported by the tine package.
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd


def _flatten_record(obj: Any, prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dictionaries inside a JSONL record."""
    if not isinstance(obj, dict):
        return {"value": obj}
    flat: Dict[str, Any] = {}
    for k, v in obj.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(_flatten_record(v, prefix=key, sep=sep))
        else:
            flat[key] = v
    return flat


def normalize_jsonl_stream(
    input_jsonl: Union[str, Path],
    output_jsonl: Optional[Union[str, Path]] = None,
    output_csv: Optional[Union[str, Path]] = None,
    flatten: bool = True,
) -> Dict[str, Any]:
    """Clean a Colab .jsonl file, drop corrupt lines, and write clean JSONL/CSV."""
    in_path = Path(input_jsonl)
    out_jsonl = (
        Path(output_jsonl)
        if output_jsonl
        else in_path.with_name(f"{in_path.stem}_clean.jsonl")
    )
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    valid_records: List[Dict[str, Any]] = []
    skipped_lines = 0

    for raw_line in in_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            skipped_lines += 1
            continue

        rec = _flatten_record(parsed) if flatten and isinstance(parsed, dict) else (
            parsed if isinstance(parsed, dict) else {"value": parsed}
        )
        valid_records.append(rec)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in valid_records:
            f.write(json.dumps(rec) + "\n")

    if output_csv:
        c_path = Path(output_csv)
        c_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(valid_records).to_csv(c_path, index=False)

    return {
        "source_file": in_path.name,
        "output_jsonl": out_jsonl.name,
        "valid_records": len(valid_records),
        "skipped_corrupt_lines": skipped_lines,
    }
