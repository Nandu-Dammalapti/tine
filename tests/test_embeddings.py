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


class HalfPrecisionEncoder:
    """Mimics a bfloat16 encoder: unnormalized, low-precision output."""

    def encode(self, prompts, **kwargs):
        assert kwargs["normalize_embeddings"] is False
        vectors = np.array([[3.0 * len(prompt), 4.0, 0.001] for prompt in prompts], dtype=np.float16)
        return vectors


def test_half_precision_encoders_are_normalized_in_float32(tmp_path):
    tasks = synthetic_fixture(n_tasks=5)
    output = tmp_path / "half.npz"
    embedding.embed_tasks(tasks, output, revision="d" * 40, batch_size=2, encoder=HalfPrecisionEncoder())
    with np.load(output, allow_pickle=False) as saved:
        vectors = saved["embeddings"]
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


class ZeroEncoder:
    def encode(self, prompts, **kwargs):
        return np.zeros((len(prompts), 3))


def test_zero_vectors_are_still_rejected(tmp_path):
    with pytest.raises(ValueError, match="non-finite"):
        embedding.embed_tasks(synthetic_fixture(n_tasks=3), tmp_path / "zero.npz", revision="e" * 40,
                              encoder=ZeroEncoder())


def test_issue_view_extracts_one_issue_block_or_keeps_the_prompt():
    assert embedding.issue_view("Intro\n<issue>\nBug here\n</issue>\n<code>x</code>") == "Bug here"
    for whole in ["plain task", "<issue> </issue> code", "<issue>a</issue><issue>b</issue>"]:
        assert embedding.issue_view(whole) == whole


def test_issue_view_is_encoded_but_full_prompt_is_hashed(tmp_path):
    tasks = embedding.prepare_tasks(synthetic_fixture(n_tasks=3))
    tasks["prompt"] = [f"Intro <issue>{'x' * (index + 1)}</issue> " + "code " * 50 for index in range(3)]
    tasks["prompt_hash"] = tasks["prompt"].map(embedding.prompt_hash)
    output = tmp_path / "issue.npz"
    metadata = embedding.embed_tasks(tasks, output, revision="a" * 40, encoder=FakeEncoder(),
                                     prompt_view="issue")
    assert metadata["prompt_view"] == "issue"
    lookup = load_embeddings(output, tasks)  # bound to the full prompts
    lengths = [vector[0] / vector[1] for vector in lookup.values()]
    assert np.allclose(sorted(lengths), [1, 2, 3], rtol=1e-4)
    with pytest.raises(ValueError, match="mismatch"):
        embedding.embed_tasks(tasks, output, revision="a" * 40, encoder=FakeEncoder())
    with pytest.raises(ValueError, match="prompt_view"):
        embedding.embed_tasks(tasks, tmp_path / "other.npz", revision="a" * 40,
                              encoder=FakeEncoder(), prompt_view="code")
