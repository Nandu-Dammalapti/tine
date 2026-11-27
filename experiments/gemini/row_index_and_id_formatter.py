# Adds a row_index and a deterministic 12-character SHA-256 record_uid to each row of a Colab export.
# Source: Gemini scheduled action, 2026-11-26 08:35
# Experimental; not imported by the tine package.
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import pandas as pd


def compute_row_uid(row_dict: Dict[str, Any], key_cols: Optional[Sequence[str]] = None) -> str:
    """Generate a deterministic 12-char SHA-256 UID from selected keys or full row."""
    if key_cols:
        payload = {k: row_dict.get(k) for k in key_cols if k in row_dict}
    else:
        payload = {
            k: v
            for k, v in row_dict.items()
            if k not in {"row_index", "record_uid"}
        }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def add_row_index_and_uids(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    key_cols: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Attach row_index and record_uid columns to a Colab CSV or JSON export."""
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

    records: List[Dict[str, Any]] = df.to_dict(orient="records")
    uids = [compute_row_uid(rec, key_cols=key_cols) for rec in records]

    df_out = df.copy()
    df_out.insert(0, "row_index", range(len(df_out)))
    df_out.insert(1, "record_uid", uids)

    if out_path.suffix.lower() == ".json":
        out_path.write_text(
            json.dumps(df_out.to_dict(orient="records"), indent=2), encoding="utf-8"
        )
    else:
        df_out.to_csv(out_path, index=False)

    return {
        "source_file": in_path.name,
        "output_file": out_path.name,
        "row_count": int(len(df_out)),
        "unique_uids": len(set(uids)),
    }
