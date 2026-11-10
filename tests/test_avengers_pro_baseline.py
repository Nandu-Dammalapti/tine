import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tine.data import synthetic_fixture
from tine.pipeline import complete_tasks, prompt_hash

spec = importlib.util.spec_from_file_location(
    "tine_avengers", Path(__file__).parents[1] / "scripts" / "avengers_pro_baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)

# A stand-in with the official code's interface. It routes every test task to the
# cheapest model below a performance weight of 0.5 and to the most accurate above.
FAKE_OFFICIAL = {
    "config.py": '''
from dataclasses import dataclass

@dataclass
class SimpleClusterConfig:
    data_path: str
    n_clusters: int = 32
    train_ratio: float = 0.7
    seed: int = 42
    max_router: int = 1
    top_k: int = 1
    beta: float = 9.0
    performance_weight: float = 0.7
    cost_sensitivity: float = 0.3
    embedding_api_key: str = ""
    embedding_model: str = ""
    max_tokens: int = 7500
''',
    "embedding_cache.py": '''
class EmbeddingCache:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("The embedding service must not be contacted")
''',
    "simple_cluster_router.py": '''
import json
import numpy as np
from embedding_cache import EmbeddingCache

class tiktoken:
    @staticmethod
    def encoding_for_model(name):
        raise RuntimeError("The tokenizer download must not be attempted")

class SimpleClusterRouter:
    def __init__(self, config):
        self.config = config
        self.embedder = EmbeddingCache()
        self.tokenizer = tiktoken.encoding_for_model(config.embedding_model)

    def load_and_split_data(self):
        rows = [json.loads(line) for line in open(self.config.data_path, encoding="utf-8")]
        order = np.random.default_rng(self.config.seed).permutation(len(rows))
        cut = int(len(rows) * self.config.train_ratio)
        return [rows[i] for i in order[:cut]], [rows[i] for i in order[cut:]]

    def _generate_embeddings_concurrent(self, queries):
        raise RuntimeError("The embedding service must not be contacted")
''',
    "balance_cluster_router.py": '''
import numpy as np
from simple_cluster_router import SimpleClusterRouter

class BalanceClusterRouter(SimpleClusterRouter):
    def build_cluster_model(self, train):
        assert len(self._generate_embeddings_concurrent([row["query"] for row in train])) == len(train)
        models = sorted(train[0]["records"])
        accuracy = {m: np.mean([row["records"][m] for row in train]) for m in models}
        cost = {m: np.mean([row["usages"][m]["cost"] for row in train]) for m in models}
        self.choice = (max(models, key=accuracy.get) if self.config.performance_weight >= 0.5
                       else min(models, key=cost.get))

    def evaluate_routing(self, test):
        print("official progress output")
        self._generate_embeddings_concurrent([row["query"] for row in test])
        return {"routing_details": [{"index": row["index"], "selected_models": [self.choice]}
                                    for row in test]}
''',
}


@pytest.fixture
def fake_official(tmp_path):
    directory = tmp_path / "official"
    directory.mkdir()
    for name, source in FAKE_OFFICIAL.items():
        (directory / name).write_text(source)
    names = [name[:-3] for name in FAKE_OFFICIAL]
    saved = {name: sys.modules.pop(name, None) for name in names}
    path = list(sys.path)
    yield directory
    sys.path[:] = path
    for name in names:
        sys.modules.pop(name, None)
        if saved[name] is not None:
            sys.modules[name] = saved[name]


def embeddings_file(frame, path):
    tasks = frame.drop_duplicates(["tenant", "task_id"])
    np.savez(path, tenants=tasks.tenant.to_numpy(dtype=str), task_ids=tasks.task_id.to_numpy(dtype=str),
             prompt_hashes=[prompt_hash(prompt) for prompt in tasks.prompt],
             embeddings=np.random.default_rng(0).normal(size=(len(tasks), 6)))
    return path


def test_official_rows_carry_every_models_outcome_and_cost():
    frame, tasks, _ = complete_tasks(synthetic_fixture(n_tasks=5), "frontier")
    models = sorted(frame.model.unique())
    rows = baseline.official_rows(frame, tasks, models)
    assert [row["index"] for row in rows] == list(range(5))
    first = frame[frame.task_id == tasks.task_id[0]].set_index("model")
    assert rows[0]["query"] == tasks.prompt[0] and rows[0]["dataset"] == tasks.task_type[0]
    for model in models:
        assert rows[0]["records"][model] == float(first.success[model])
        assert rows[0]["usages"][model]["cost"] == pytest.approx(first.cost_usd[model])


def test_baseline_scores_official_choices_without_an_embedding_service(tmp_path, fake_official, capsys):
    frame = synthetic_fixture(n_tasks=200, seed=4)
    vectors = embeddings_file(frame, tmp_path / "vectors.npz")
    result = baseline.run_baseline(frame, "frontier", vectors, fake_official, tmp_path / "out",
                                   weights=(0.0, 1.0), seeds=range(3), clusters=4, top_k=2,
                                   avengers_commit="abc123")
    assert capsys.readouterr().out == ""  # the official code's printing is suppressed
    runs = pd.read_csv(tmp_path / "out" / "avengers-runs.csv")
    assert len(runs) == 6 and (runs.test_tasks == 60).all()
    cheap, accurate = runs[runs.performance_weight == 0.0], runs[runs.performance_weight == 1.0]
    assert (cheap.frontier_share == 0).all() and (cheap.savings > 0.5).all()
    assert (cheap.degradation_rate > 0.02).all()
    assert (accurate.frontier_share == 1).all() and (accurate.savings == 0).all()
    picks = pd.read_csv(tmp_path / "out" / "avengers-picks.csv")
    # The cheapest model alone breaks epsilon on the choosing half, so no weight is picked.
    assert picks.picked_weight.isna().all() and (picks.savings == 0).all() and (picks.eval_tasks == 30).all()
    summary = json.loads((tmp_path / "out" / "avengers-summary.json").read_text())
    assert summary["official_code"]["commit"] == "abc123" and summary["picked"]["frontier_only_runs"] == 3
    assert result["complete_tasks"] == 200 and len(result["curve"]) == 2


def test_baseline_rejects_bad_settings_and_missing_code(tmp_path, fake_official):
    frame = synthetic_fixture(n_tasks=50, seed=4)
    vectors = embeddings_file(frame, tmp_path / "vectors.npz")
    for bad in [{"weights": ()}, {"weights": (1.5,)}, {"seeds": ()}, {"epsilon": 2}]:
        with pytest.raises(ValueError):
            baseline.run_baseline(frame, "frontier", vectors, fake_official, tmp_path / "out", **bad)
    with pytest.raises(FileNotFoundError):
        baseline.run_baseline(frame, "frontier", vectors, tmp_path / "nowhere", tmp_path / "out")
    with pytest.raises(ValueError, match="embedding"):
        baseline.run_baseline(synthetic_fixture(n_tasks=60, seed=9), "frontier", vectors,
                              fake_official, tmp_path / "out")
