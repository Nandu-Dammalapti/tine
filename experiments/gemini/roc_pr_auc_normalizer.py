# Normalizes ROC/PR curve exports from Colab into a point table and computes ROC-AUC and PR-AUC.
# Source: Gemini scheduled action, 2026-11-25 11:31
# Experimental; not imported by the tine package. Needs numpy >= 2.0 (np.trapezoid).
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd


def _pad_thresholds(thresholds: Optional[Sequence[float]], target_len: int) -> List[Optional[float]]:
    """Align threshold arrays that often contain one fewer element than FPR/TPR/precision/recall."""
    if thresholds is None:
        return [None] * target_len
    arr = np.asarray(thresholds, dtype=float)
    out: List[Optional[float]] = []
    for i in range(target_len):
        if i < len(arr) and np.isfinite(arr[i]):
            out.append(round(float(arr[i]), 6))
        else:
            out.append(None)
    return out


def normalize_curve_data(
    fpr: Optional[Sequence[float]] = None,
    tpr: Optional[Sequence[float]] = None,
    precision: Optional[Sequence[float]] = None,
    recall: Optional[Sequence[float]] = None,
    thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Build a uniform point table and compute ROC-AUC and PR-AUC via trapezoidal integration."""
    lengths = [
        len(seq)
        for seq in (fpr, tpr, precision, recall)
        if seq is not None and len(seq) > 0
    ]
    if not lengths:
        raise ValueError("At least one curve sequence (fpr/tpr or precision/recall) must be non-empty.")

    n_points = max(lengths)

    def _to_float_array(seq: Optional[Sequence[float]]) -> np.ndarray:
        if seq is None or len(seq) == 0:
            return np.full(n_points, np.nan, dtype=float)
        arr = np.asarray(seq, dtype=float)
        if len(arr) == n_points:
            return arr
        padded = np.full(n_points, np.nan, dtype=float)
        padded[: min(len(arr), n_points)] = arr[: min(len(arr), n_points)]
        return padded

    fpr_arr = _to_float_array(fpr)
    tpr_arr = _to_float_array(tpr)
    prec_arr = _to_float_array(precision)
    rec_arr = _to_float_array(recall)
    thresh_list = _pad_thresholds(thresholds, n_points)

    roc_auc: Optional[float] = None
    valid_roc = np.isfinite(fpr_arr) & np.isfinite(tpr_arr)
    if int(np.sum(valid_roc)) >= 2:
        order = np.argsort(fpr_arr[valid_roc])
        roc_auc = round(float(np.trapezoid(tpr_arr[valid_roc][order], fpr_arr[valid_roc][order])), 6)

    pr_auc: Optional[float] = None
    if precision is not None and recall is not None:
        mask_pr = np.isfinite(rec_arr) & np.isfinite(prec_arr)
        if int(np.sum(mask_pr)) >= 2:
            order_pr = np.argsort(rec_arr[mask_pr])
            pr_auc = round(float(np.trapezoid(prec_arr[mask_pr][order_pr], rec_arr[mask_pr][order_pr])), 6)

    points: List[Dict[str, Any]] = []
    for i in range(n_points):
        p_val = float(prec_arr[i]) if np.isfinite(prec_arr[i]) else None
        r_val = float(rec_arr[i]) if np.isfinite(rec_arr[i]) else None
        f1_val: Optional[float] = None
        if p_val is not None and r_val is not None and (p_val + r_val) > 0.0:
            f1_val = round(2.0 * p_val * r_val / (p_val + r_val), 6)

        points.append(
            {
                "point_index": i,
                "fpr": round(float(fpr_arr[i]), 6) if np.isfinite(fpr_arr[i]) else None,
                "tpr": round(float(tpr_arr[i]), 6) if np.isfinite(tpr_arr[i]) else None,
                "precision": round(p_val, 6) if p_val is not None else None,
                "recall": round(r_val, 6) if r_val is not None else None,
                "f1_at_point": f1_val,
                "threshold": thresh_list[i],
            }
        )

    summary = {
        "n_points": n_points,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }
    return {"summary": summary, "points": points}


def format_colab_curve_file(
    input_path: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    summary_json: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load a Colab ROC/PR JSON or CSV export and write tidy curve CSV + AUC summary JSON."""
    in_path = Path(input_path)
    suffix = in_path.suffix.lower()

    if suffix == ".json":
        payload = json.loads(in_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            result = normalize_curve_data(
                fpr=payload.get("fpr"),
                tpr=payload.get("tpr"),
                precision=payload.get("precision"),
                recall=payload.get("recall"),
                thresholds=payload.get("thresholds"),
            )
        else:
            df_p = pd.DataFrame(payload)
            result = normalize_curve_data(
                fpr=df_p["fpr"].tolist() if "fpr" in df_p else None,
                tpr=df_p["tpr"].tolist() if "tpr" in df_p else None,
                precision=df_p["precision"].tolist() if "precision" in df_p else None,
                recall=df_p["recall"].tolist() if "recall" in df_p else None,
                thresholds=df_p["threshold"].tolist() if "threshold" in df_p else None,
            )
    elif suffix == ".csv":
        df_c = pd.read_csv(in_path)
        result = normalize_curve_data(
            fpr=df_c["fpr"].tolist() if "fpr" in df_c else None,
            tpr=df_c["tpr"].tolist() if "tpr" in df_c else None,
            precision=df_c["precision"].tolist() if "precision" in df_c else None,
            recall=df_c["recall"].tolist() if "recall" in df_c else None,
            thresholds=df_c["threshold"].tolist() if "threshold" in df_c else None,
        )
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    if output_csv is None and summary_json is None:
        output_csv = in_path.with_name(f"{in_path.stem}_curve_points.csv")
        summary_json = in_path.with_name(f"{in_path.stem}_auc_summary.json")

    if output_csv:
        c_path = Path(output_csv)
        c_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result["points"]).to_csv(c_path, index=False)

    if summary_json:
        s_path = Path(summary_json)
        s_path.parent.mkdir(parents=True, exist_ok=True)
        s_path.write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")

    return result
