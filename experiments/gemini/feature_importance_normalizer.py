# Ranks Colab feature importances by magnitude with normalized and cumulative shares and a top-k summary.
# Source: Gemini scheduled action, 2026-11-25 14:38
# Experimental; not imported by the tine package.
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd


def normalize_feature_importances(
    features: Sequence[str],
    importances: Sequence[float],
    top_k: int = 10,
) -> Dict[str, Any]:
    """Rank features by magnitude, compute normalized and cumulative shares, and summarize."""
    if len(features) != len(importances) or len(features) == 0:
        raise ValueError("features and importances must be non-empty sequences of equal length.")

    raw_vals = np.asarray(importances, dtype=float)
    abs_vals = np.where(np.isfinite(raw_vals), np.abs(raw_vals), 0.0)
    total_abs = float(np.sum(abs_vals))

    order = np.argsort(-abs_vals, kind="mergesort")
    ranked_rows: List[Dict[str, Any]] = []
    running_share = 0.0

    for rank_idx, idx in enumerate(order, start=1):
        feat_name = str(features[idx]).strip()
        val_f = float(raw_vals[idx]) if np.isfinite(raw_vals[idx]) else None
        abs_f = float(abs_vals[idx])
        share = (abs_f / total_abs) if total_abs > 0.0 else 0.0
        running_share = min(1.0, running_share + share)

        ranked_rows.append(
            {
                "rank": rank_idx,
                "feature": feat_name,
                "importance": round(val_f, 8) if val_f is not None else None,
                "abs_importance": round(abs_f, 8),
                "normalized_share": round(share, 6),
                "cumulative_share": round(running_share, 6),
                "is_top_k": rank_idx <= top_k,
            }
        )

    top_features = [row["feature"] for row in ranked_rows[:top_k]]
    summary: Dict[str, Any] = {
        "total_features": len(ranked_rows),
        "top_k": min(top_k, len(ranked_rows)),
        "total_abs_importance": round(total_abs, 8),
        "top_k_features": top_features,
    }
    return {"summary": summary, "features": ranked_rows}


def format_colab_feature_importances(
    input_path: Union[str, Path],
    feature_col: str = "feature",
    importance_col: str = "importance",
    top_k: int = 10,
    output_csv: Optional[Union[str, Path]] = None,
    summary_json: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load a Colab feature-importance JSON or CSV and export ranked CSV + summary JSON."""
    in_path = Path(input_path)
    suffix = in_path.suffix.lower()

    if suffix == ".json":
        payload = json.loads(in_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "features" in payload and "importances" in payload:
            feat_list = payload["features"]
            imp_list = payload["importances"]
        elif isinstance(payload, dict) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in payload.values()
        ):
            feat_list = list(payload.keys())
            imp_list = list(payload.values())
        elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
            df_p = pd.DataFrame(payload)
            feat_list = df_p[feature_col].astype(str).tolist()
            imp_list = df_p[importance_col].tolist()
        else:
            raise ValueError(f"Unsupported JSON layout in {in_path.name}")

    elif suffix == ".csv":
        df_c = pd.read_csv(in_path)
        f_col = feature_col if feature_col in df_c.columns else next(
            (c for c in ("feature", "name", "column") if c in df_c.columns), None
        )
        i_col = importance_col if importance_col in df_c.columns else next(
            (c for c in ("importance", "weight", "score", "shap_value") if c in df_c.columns),
            None,
        )
        if not f_col or not i_col:
            raise KeyError(f"Could not locate feature/importance columns in {in_path.name}")
        feat_list = df_c[f_col].astype(str).tolist()
        imp_list = df_c[i_col].tolist()
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    result = normalize_feature_importances(feat_list, imp_list, top_k=top_k)

    if output_csv is None and summary_json is None:
        output_csv = in_path.with_name(f"{in_path.stem}_feature_ranks.csv")
        summary_json = in_path.with_name(f"{in_path.stem}_feature_summary.json")

    if output_csv:
        c_path = Path(output_csv)
        c_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result["features"]).to_csv(c_path, index=False)

    if summary_json:
        s_path = Path(summary_json)
        s_path.parent.mkdir(parents=True, exist_ok=True)
        s_path.write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")

    return result
