"""Prompt embeddings.

- `TfidfEmbedder`: CPU-only stand-in, no downloads. Good enough to run and test
  the pipeline; real runs should use sentence-transformer embeddings.
- `embed_with_sentence_transformers`: the real embedder. Runs on Colab
  (see colab/01_embed_prompts.ipynb), saved with `save_embeddings`, and read back
  on CPU with `load_embeddings`.
"""

from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TfidfEmbedder:
    """TF-IDF (word 1-2 grams) -> truncated SVD -> unit length."""

    def __init__(self, n_components=128, seed=0):
        self.n_components = n_components
        self.seed = seed
        self._pipeline = None

    def fit(self, prompts):
        vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                                     max_features=50_000)
        n_terms = len(vectorizer.fit(prompts).vocabulary_)
        components = max(2, min(self.n_components, n_terms - 1))
        self._pipeline = make_pipeline(
            vectorizer,
            TruncatedSVD(n_components=components, random_state=self.seed),
            Normalizer(),
        )
        self._pipeline.fit(prompts)
        return self

    def transform(self, prompts) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("Call fit first")
        return self._pipeline.transform(prompts).astype(np.float32)


def embed_with_sentence_transformers(prompts, model_name=DEFAULT_ST_MODEL, batch_size=128,
                                     device=None) -> np.ndarray:
    """Embed prompts with sentence-transformers (downloads the model). Meant for Colab."""
    from sentence_transformers import SentenceTransformer  # imported lazily: not a CPU dep

    model = SentenceTransformer(model_name, device=device)
    vectors = model.encode(list(prompts), batch_size=batch_size, show_progress_bar=True,
                           normalize_embeddings=True, convert_to_numpy=True)
    return vectors.astype(np.float32)


def save_embeddings(path, task_ids, vectors, model_name):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, task_ids=np.asarray(task_ids, dtype=str),
                        vectors=np.asarray(vectors, dtype=np.float32),
                        model_name=np.array(model_name))


def load_embeddings(path, task_ids) -> np.ndarray:
    """Load vectors saved by `save_embeddings`, in the order of `task_ids`."""
    with np.load(path, allow_pickle=False) as archive:
        saved_ids = archive["task_ids"].astype(str)
        vectors = archive["vectors"]
    position = {task_id: i for i, task_id in enumerate(saved_ids)}
    missing = [t for t in task_ids if t not in position]
    if missing:
        raise ValueError(f"{len(missing)} task_ids have no embedding, e.g. {missing[:3]}")
    out = vectors[[position[t] for t in task_ids]].astype(np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-12)
