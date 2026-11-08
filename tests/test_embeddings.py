import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tine.data import synthetic_fixture
from tine.pipeline import load_embeddings

spec = importlib.util.spec_from_file_location("tine_embed", Path(__file__).parents[1] / "scripts" / "embed.py")
embedding = importlib.util.module_from_spec(spec)
spec.loader.exec_module(embedding)


class FakeEncoder:
    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after

    def encode(self, prompts, **kwargs):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("Simulated runtime disconnect")
        vectors = np.array([[len(prompt), 1.0] for prompt in prompts])
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_interrupted_embedding_resumes_and_pipeline_validates(tmp_path):
    tasks = embedding.prepare_tasks(synthetic_fixture(n_tasks=9))
    output = tmp_path / "embeddings.npz"
    with pytest.raises(RuntimeError, match="disconnect"):
        embedding.embed_tasks(tasks, output, revision="a" * 40, batch_size=3, encoder=FakeEncoder(fail_after=1))
    assert not output.exists()
    resumed = FakeEncoder()
    embedding.embed_tasks(tasks, output, revision="a" * 40, batch_size=3, encoder=resumed)
    assert resumed.calls == 2
    lookup = load_embeddings(output, tasks)
    assert len(lookup) == 9
    assert all(np.isclose(np.linalg.norm(vector), 1) for vector in lookup.values())
    completed = FakeEncoder(fail_after=0)
    embedding.embed_tasks(tasks, output, revision="a" * 40, batch_size=3, encoder=completed)
    assert completed.calls == 0


def test_embedding_resume_rejects_different_input_and_revision(tmp_path):
    tasks = embedding.prepare_tasks(synthetic_fixture(n_tasks=3))
    output = tmp_path / "embeddings.npz"
    embedding.embed_tasks(tasks, output, revision="b" * 40, encoder=FakeEncoder())
    with pytest.raises(ValueError, match="mismatch"):
        embedding.embed_tasks(tasks, output, revision="c" * 40, encoder=FakeEncoder())
    tasks.loc[0, "prompt"] += " changed"
    with pytest.raises(ValueError, match="mismatch"):
        embedding.embed_tasks(tasks, output, revision="b" * 40, encoder=FakeEncoder())
