# Turns Colab confusion matrices (.csv/.npy/.json) into a tidy cell table and accuracy/precision/recall/F1/MCC metrics.
# Source: Gemini scheduled action, 2026-11-24 08:36
# Experimental; not imported by the tine package.
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd


def summarize_confusion_matrix(
    matrix: Union[np.ndarray, Sequence[Sequence[int]]],
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute tidy cell records and binary/macro metrics from a square confusion matrix."""
    cm = np.asarray(matrix, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or cm.shape[0] == 0:
        raise ValueError(f"Expected non-empty square matrix, got shape {cm.shape}")

    n_classes = cm.shape[0]
    class_labels = (
        [str(lbl) for lbl in labels]
        if labels and len(labels) == n_classes
        else [f"class_{i}" for i in range(n_classes)]
    )

    total = float(np.sum(cm))
    row_sums = np.sum(cm, axis=1, keepdims=True)
    row_shares = np.where(row_sums > 0, cm / np.where(row_sums > 0, row_sums, 1.0), 0.0)

    cells: List[Dict[str, Any]] = []
    for r in range(n_classes):
        for c in range(n_classes):
            cnt = int(round(float(cm[r, c])))
            cells.append(
                {
                    "true_label": class_labels[r],
                    "pred_label": class_labels[c],
                    "count": cnt,
                    "row_share": round(float(row_shares[r, c]), 6),
                    "total_share": round(float(cm[r, c]) / total, 6) if total > 0 else 0.0,
                    "is_diagonal": r == c,
                }
            )

    accuracy = round(float(np.trace(cm)) / total, 6) if total > 0 else 0.0
    metrics: Dict[str, Any] = {"total_samples": int(round(total)), "accuracy": accuracy}

    if n_classes == 2:
        tn, fp, fn, tp = float(cm[0, 0]), float(cm[0, 1]), float(cm[1, 0]), float(cm[1, 1])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
        metrics.update(
            {
                "true_positives": int(round(tp)),
                "false_positives": int(round(fp)),
                "true_negatives": int(round(tn)),
                "false_negatives": int(round(fn)),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "specificity": round(specificity, 6),
                "f1_score": round(f1, 6),
                "mcc": round(mcc, 6),
            }
        )

    return {"metrics": metrics, "cells": cells}


def format_colab_confusion_matrix(
    input_path: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    metrics_json: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load a Colab confusion matrix (.csv, .npy, .json) and export cell CSV + metrics JSON."""
    in_path = Path(input_path)
    suffix = in_path.suffix.lower()
    labels: Optional[List[str]] = None

    if suffix == ".csv":
        df_cm = pd.read_csv(in_path, index_col=0)
        labels = [str(c) for c in df_cm.columns]
        mat = df_cm.to_numpy(dtype=float)
    elif suffix == ".npy":
        mat = np.load(in_path)
    elif suffix == ".json":
        payload = json.loads(in_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "matrix" in payload:
            mat = np.asarray(payload["matrix"], dtype=float)
            labels = payload.get("labels")
        else:
            df_cm = pd.DataFrame(payload)
            labels = [str(c) for c in df_cm.columns]
            mat = df_cm.to_numpy(dtype=float)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    summary = summarize_confusion_matrix(mat, labels=labels)

    if output_csv is None and metrics_json is None:
        output_csv = in_path.with_name(f"{in_path.stem}_cm_cells.csv")
        metrics_json = in_path.with_name(f"{in_path.stem}_cm_metrics.json")

    if output_csv:
        c_path = Path(output_csv)
        c_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary["cells"]).to_csv(c_path, index=False)

    if metrics_json:
        m_path = Path(metrics_json)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        m_path.write_text(json.dumps(summary["metrics"], indent=2), encoding="utf-8")

    return summary
