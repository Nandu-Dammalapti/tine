# Builds a class-distribution table and imbalance/entropy summary from Colab label counts or label arrays.
# Source: Gemini scheduled action, 2026-11-24 11:10
# Experimental; not imported by the tine package.
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd


def compute_class_balance(
    counts_or_labels: Union[Dict[str, int], Sequence[Any]],
    minority_share_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Compute class proportions, imbalance ratios, entropy, and per-class records."""
    if isinstance(counts_or_labels, dict):
        counter: Counter[str] = Counter()
        for k, v in counts_or_labels.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v):
                counter[str(k)] += max(0, int(round(float(v))))
    else:
        filtered = [
            str(x)
            for x in counts_or_labels
            if x is not None and not (isinstance(x, float) and math.isnan(x))
        ]
        counter = Counter(filtered)

    if not counter:
        raise ValueError("No valid class labels or counts found.")

    total_samples = int(sum(counter.values()))
    majority_label, majority_count = max(counter.items(), key=lambda item: item[1])
    minority_label, minority_count = min(counter.items(), key=lambda item: item[1])

    entropy = 0.0
    rows: List[Dict[str, Any]] = []

    for label, cnt in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        share = cnt / float(total_samples) if total_samples > 0 else 0.0
        if share > 0.0:
            entropy -= share * math.log2(share)

        ratio_to_majority = (
            round(cnt / float(majority_count), 6) if majority_count > 0 else 0.0
        )
        rows.append(
            {
                "class_label": label,
                "sample_count": int(cnt),
                "class_share": round(share, 6),
                "ratio_to_majority": ratio_to_majority,
                "is_minority_class": share < minority_share_threshold,
            }
        )

    imbalance_ratio = (
        round(float(majority_count) / float(minority_count), 6)
        if minority_count > 0
        else None
    )

    summary: Dict[str, Any] = {
        "total_samples": total_samples,
        "n_classes": len(rows),
        "majority_class": majority_label,
        "majority_count": int(majority_count),
        "minority_class": minority_label,
        "minority_count": int(minority_count),
        "imbalance_ratio": imbalance_ratio,
        "shannon_entropy_bits": round(entropy, 6),
        "rows": rows,
    }
    return summary


def format_colab_class_balance(
    input_path: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    summary_json: Optional[Union[str, Path]] = None,
    label_col: str = "label",
    count_col: str = "count",
    minority_share_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Load a Colab class distribution JSON or CSV and export clean balance rows + summary."""
    in_path = Path(input_path)
    suffix = in_path.suffix.lower()

    source_data: Union[Dict[str, int], List[Any]]
    if suffix == ".json":
        payload = json.loads(in_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "class_counts" in payload and isinstance(payload["class_counts"], dict):
            source_data = payload["class_counts"]
        elif isinstance(payload, dict) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in payload.values()
        ):
            source_data = payload
        elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
            df_p = pd.DataFrame(payload)
            if label_col in df_p.columns and count_col in df_p.columns:
                source_data = dict(zip(df_p[label_col].astype(str), df_p[count_col]))
            elif label_col in df_p.columns:
                source_data = df_p[label_col].tolist()
            else:
                raise KeyError(f"Column '{label_col}' not found in {in_path.name}")
        elif isinstance(payload, list):
            source_data = payload
        else:
            raise ValueError(f"Unsupported JSON layout in {in_path.name}")

    elif suffix == ".csv":
        df_csv = pd.read_csv(in_path)
        if label_col in df_csv.columns and count_col in df_csv.columns:
            source_data = dict(zip(df_csv[label_col].astype(str), df_csv[count_col]))
        elif label_col in df_csv.columns:
            source_data = df_csv[label_col].tolist()
        else:
            raise KeyError(f"Column '{label_col}' not found in {in_path.name}")
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    result = compute_class_balance(
        source_data, minority_share_threshold=minority_share_threshold
    )

    if output_csv is None and summary_json is None:
        output_csv = in_path.with_name(f"{in_path.stem}_class_balance.csv")
        summary_json = in_path.with_name(f"{in_path.stem}_class_balance_summary.json")

    if output_csv:
        c_path = Path(output_csv)
        c_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result["rows"]).to_csv(c_path, index=False)

    if summary_json:
        s_path = Path(summary_json)
        s_path.parent.mkdir(parents=True, exist_ok=True)
        meta_only = {k: v for k, v in result.items() if k != "rows"}
        s_path.write_text(json.dumps(meta_only, indent=2), encoding="utf-8")

    return result
