# Maps messy status labels in Colab exports ("PASS", "succeeded", "timeout") to canonical pass/warn/fail/skip with a severity code.
# Source: Gemini scheduled action, 2026-11-27 08:27
# Experimental; not imported by the tine package.
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd

DEFAULT_STATUS_MAP: Dict[str, str] = {
    "pass": "pass",
    "passed": "pass",
    "ok": "pass",
    "success": "pass",
    "succeeded": "pass",
    "completed": "pass",
    "done": "pass",
    "warn": "warn",
    "warning": "warn",
    "partial": "warn",
    "degraded": "warn",
    "fail": "fail",
    "failed": "fail",
    "error": "fail",
    "err": "fail",
    "exception": "fail",
    "timeout": "fail",
    "oom": "fail",
    "skip": "skip",
    "skipped": "skip",
    "ignored": "skip",
}

STATUS_SEVERITY_CODE: Dict[str, int] = {
    "pass": 0,
    "skip": 1,
    "warn": 2,
    "unknown": 3,
    "fail": 4,
}


def normalize_status_label(
    val: Any,
    alias_map: Dict[str, str] = DEFAULT_STATUS_MAP,
) -> Dict[str, Any]:
    """Map a status token to a canonical label and numeric severity code."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        canonical = "unknown"
    else:
        token = str(val).strip().lower()
        canonical = alias_map.get(token, "unknown")

    return {
        "canonical_status": canonical,
        "status_code": STATUS_SEVERITY_CODE.get(canonical, 3),
        "is_success": canonical == "pass",
    }


def normalize_colab_status_column(
    input_path: Union[str, Path],
    status_col: str = "status",
    output_path: Optional[Union[str, Path]] = None,
    alias_map: Dict[str, str] = DEFAULT_STATUS_MAP,
) -> Path:
    """Enrich a Colab CSV or JSON export with canonical status and severity columns."""
    in_path = Path(input_path)
    out_path = Path(output_path) if output_path else in_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = in_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(in_path)
        target_col = status_col if status_col in df.columns else next(
            (c for c in df.columns if "status" in str(c).lower() or "state" in str(c).lower()),
            None,
        )
        if target_col:
            mapped = pd.DataFrame([normalize_status_label(v, alias_map) for v in df[target_col]])
            df[f"{target_col}_canonical"] = mapped["canonical_status"]
            df[f"{target_col}_code"] = mapped["status_code"]
            df[f"{target_col}_is_success"] = mapped["is_success"]
        df.to_csv(out_path, index=False)

    elif suffix == ".json":
        payload = json.loads(in_path.read_text(encoding="utf-8"))

        def _enrich_record(rec: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(rec)
            for k, v in rec.items():
                if k == status_col or "status" in str(k).lower() or "state" in str(k).lower():
                    info = normalize_status_label(v, alias_map)
                    out[f"{k}_canonical"] = info["canonical_status"]
                    out[f"{k}_code"] = info["status_code"]
                    out[f"{k}_is_success"] = info["is_success"]
            return out

        if isinstance(payload, list):
            cleaned = [_enrich_record(r) if isinstance(r, dict) else r for r in payload]
        elif isinstance(payload, dict):
            cleaned = _enrich_record(payload)
        else:
            cleaned = payload
        out_path.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return out_path
