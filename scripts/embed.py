#!/usr/bin/env python3
"""Create resumable, tenant-keyed prompt embeddings on Colab.

No model dependency is imported until a non-completed batch is requested. The
encoder is fixed at an explicit Hugging Face commit; it is never fine-tuned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tine.data import read_outcomes  # noqa: E402


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def prepare_tasks(outcomes: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    """Return stable task order, rejecting conflicting text for a task key."""
    columns = ["tenant", "task_id", "prompt"]
    if any(column not in outcomes for column in columns):
        raise ValueError("Outcomes must contain tenant, task_id, and prompt")
    tasks = outcomes[columns].copy()
    if tasks.isna().any().any():
        raise ValueError("Task keys and prompts cannot be missing")
    for column in columns:
        if not tasks[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{column} must contain strings")
    conflicts = tasks.groupby(["tenant", "task_id"], sort=False)["prompt"].nunique()
    if (conflicts > 1).any():
        raise ValueError("A tenant/task_id has conflicting prompts")
    tasks = tasks.drop_duplicates(["tenant", "task_id"]).reset_index(drop=True)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        tasks = tasks.head(limit).copy()
    if tasks.empty:
        raise ValueError("No tasks to embed")
    tasks["prompt_hash"] = tasks["prompt"].map(prompt_hash)
    return tasks


def _fingerprint(tasks: pd.DataFrame) -> str:
    payload = tasks[["tenant", "task_id", "prompt_hash"]].values.tolist()
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checked_vectors(vectors: Any, rows: int, dimension: int | None = None) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != rows or vectors.shape[1] < 1:
        raise ValueError("Encoder returned an invalid embedding shape")
    if dimension is not None and vectors.shape[1] != dimension:
        raise ValueError("Embedding dimension changed between batches")
    if not np.isfinite(vectors).all():
        raise ValueError("Encoder returned non-finite embeddings")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Encoder must return nonzero normalized embeddings")
    return vectors


def embed_tasks(
    tasks: pd.DataFrame,
    output: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    revision: str,
    batch_size: int = 64,
    device: str = "cpu",
    max_seq_length: int = 256,
    cache_dir: str | None = None,
    encoder: Any = None,
) -> dict[str, Any]:
    """Encode batches with atomic checkpoints; injectable encoder enables CPU tests.

    Resume requires matching ordered task keys and prompt hashes, model revision,
    batch size, and truncation. GPU/CPU device may differ after a disconnect.
    Never run concurrent writers against the same output/checkpoint directory.
    """
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("revision must be a full 40-character Hugging Face commit SHA")
    if batch_size < 1 or max_seq_length < 1:
        raise ValueError("batch_size and max_seq_length must be positive")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be cpu, cuda, or mps")
    tasks = prepare_tasks(tasks)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output.with_name(output.name + ".checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1,
        "model": model_name,
        "revision": revision.lower(),
        "input_fingerprint": _fingerprint(tasks),
        "n_tasks": len(tasks),
        "batch_size": batch_size,
        "max_seq_length": max_seq_length,
        "normalize_embeddings": True,
        "prompt_handling": "Full prompt hashed; encoder truncates at max_seq_length tokens",
    }
    manifest_path = checkpoint_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise ValueError("Checkpoint configuration/input mismatch; use a new output path")
    elif list(checkpoint_dir.iterdir()):
        raise ValueError("Checkpoint directory has shards but no manifest; use a new output path")
    else:
        _atomic_json(manifest_path, metadata)

    batches = []
    dimension = None
    for start in range(0, len(tasks), batch_size):
        end = min(start + batch_size, len(tasks))
        batch_path = checkpoint_dir / f"batch-{start:09d}.npz"
        if batch_path.exists():
            with np.load(batch_path, allow_pickle=False) as saved:
                if int(saved["start"]) != start or str(saved["fingerprint"]) != metadata["input_fingerprint"]:
                    raise ValueError(f"Invalid checkpoint {batch_path}")
                vectors = _checked_vectors(saved["embeddings"], end - start, dimension)
        else:
            if encoder is None:
                # Colab-only optional dependency; never trust remote Python code.
                from sentence_transformers import SentenceTransformer

                encoder = SentenceTransformer(
                    model_name,
                    revision=revision,
                    device=device,
                    cache_folder=cache_dir,
                    trust_remote_code=False,
                )
                encoder.max_seq_length = max_seq_length
            vectors = _checked_vectors(
                encoder.encode(
                    tasks["prompt"].iloc[start:end].tolist(),
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ),
                end - start,
                dimension,
            )
            _atomic_npz(
                batch_path,
                start=np.array(start),
                fingerprint=np.array(metadata["input_fingerprint"]),
                embeddings=vectors,
            )
        dimension = vectors.shape[1]
        batches.append(vectors)
        print(f"Checkpointed {end}/{len(tasks)} prompts", flush=True)

    metadata["dimension"] = dimension
    _atomic_npz(
        output,
        tenants=tasks["tenant"].to_numpy(dtype=str),
        task_ids=tasks["task_id"].to_numpy(dtype=str),
        prompt_hashes=tasks["prompt_hash"].to_numpy(dtype=str),
        embeddings=np.concatenate(batches, axis=0),
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )
    _atomic_json(output.with_suffix(output.suffix + ".json"), metadata)
    print(f"Saved {output}", flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", required=True, help="Full verified Hugging Face commit SHA")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--max-seq-length", default=256, type=int)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--cache-dir")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    tasks = prepare_tasks(read_outcomes(args.input), args.limit)
    embed_tasks(
        tasks,
        args.output,
        model_name=args.model,
        revision=args.revision,
        batch_size=args.batch_size,
        device=args.device,
        max_seq_length=args.max_seq_length,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
