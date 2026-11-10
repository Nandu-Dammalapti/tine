# Avengers-Pro baseline

Avengers-Pro is the routing baseline AGENTS.md asks us to compare against
(*Beyond GPT-5: Making LLMs Cheaper and Better via Performance-Efficiency Optimized
Routing*, [arXiv:2508.12631](https://arxiv.org/abs/2508.12631)). This page records
what its official code does, how we run it on our data, and how to read the output.
Tine's own `cluster` method in `tine/routing.py` is only an Avengers-style heuristic;
the numbers from this page are the real head-to-head.

## The official code

- Repository: <https://github.com/ZhangYiqun018/AvengersPro> (MIT licence).
- Commit read and run for the 2026-11-10 results: `dfb60b156c4b6011b89edfc8516aa16d8e4f4d38`.
- It is **not vendored** in this repository. Download it next to your data and pass
  its path with `--avengers-dir`. Read third-party code before running it.

What `balance_cluster_router.py` and `simple_cluster_router.py` do, as read from that
commit:

1. **Split.** Tasks are grouped by `dataset`, sorted by `index`, and split 70 / 30
   into train and test with Hugging Face `datasets.train_test_split(seed=...)`.
2. **Embed and cluster.** Each query's embedding is L2-normalised. K-means
   (`n_init=10`, `random_state=seed`) is fitted on the train embeddings.
3. **Score models per cluster.** For each cluster and model: accuracy = mean score
   of the cluster's train tasks, cost = mean cost. Then
   `score = performance_weight * (accuracy - min) / (max - min)
   + cost_sensitivity * (1 - cost / max_cost)`, with min and max taken over the
   models in that cluster. Models are ranked by this score.
4. **Route.** A test query's distance to each cluster centre is `1 - dot product`.
   Its `top_k` nearest clusters get weights `softmax(-beta * distance)`. Each model
   collects `weight / (rank + 1)` from each of those clusters, and the model with
   the highest total is chosen.

Differences between the code and the paper, which matter when quoting either:

| | Paper | Code at this commit |
| --- | --- | --- |
| Clusters | k = 60 | 25 in `config/balance_config.json`, 32 as the CLI default |
| Nearest clusters | p = 4 (the text also says "three") | `top_k` = 3 in the config file |
| Cost in the score | total cluster cost, min-max normalised | mean cost divided by the cluster maximum |
| Aggregation over clusters | sum of the scores | distance-weighted sum of reciprocal ranks |
| Trade-off | one alpha | `performance_weight` and `cost_sensitivity` set separately |
| Embedding model | Qwen3-Embedding-8B | whatever the embedding endpoint serves |

Avengers-Pro makes **no statement about quality loss**. The weight moves a policy
along a cost-accuracy curve; nothing bounds how much worse than the frontier model
the result can be. That is the main difference from Gacrux.

## How we run it

`scripts/avengers_pro_baseline.py` runs the official classes unmodified except for
one thing: the embedding client. The official code calls an OpenAI-style embedding
endpoint and truncates text with `tiktoken`. The script subclasses
`BalanceClusterRouter` and overrides `_generate_embeddings_concurrent` to return
vectors from an NPZ in `scripts/embed.py` format, and replaces the two imports that
would open a network connection. Splitting, clustering, scoring and routing are the
official code paths.

Steps the script takes:

1. Read canonical outcomes, keep complete tasks (`tine.pipeline.complete_tasks`, the
   same filter as `tine run` and `tine cascade`) and check every task has an
   embedding bound to its prompt hash.
2. Write the tasks to a temporary JSONL in the official format (`query`, `records`,
   `usages`, `dataset` = our task type, `index` = task row).
3. For each seed and each performance weight (`cost_sensitivity = 1 - weight`), let
   the official code split, cluster and route, and read back which model it chose
   for each test task.
4. Score those choices with `tine.calibration.evaluate`, the same function that
   scores Tine's routers: savings against always using the frontier, accuracy, and
   degradation rate (frontier solved the task, chosen model did not).

Because Avengers-Pro has no certification step, the script reports two things:

- **The curve**: every weight on the official 30% test split, averaged over seeds.
  Picking the best weight from this curve uses test labels, so it is optimistic.
- **A held-out pick**: for each seed, half of the official test split chooses the
  weight with the most savings among those with degradation <= epsilon on that
  half; the other half scores it. This is the fair number to put beside Tine's
  `uncertified` picks. It is still not comparable to `certified` picks, which also
  pass a confidence bound.

## Running it

The official code needs packages that are not in `requirements-dev.lock`
(`datasets`, `tiktoken`, `loguru`, `openai`, `tqdm`). Install them in a separate
environment so the locked development environment stays as it is.

```bash
python scripts/avengers_pro_baseline.py \
  --avengers-dir PATH/TO/AvengersPro --avengers-commit dfb60b156c4b6011b89edfc8516aa16d8e4f4d38 \
  --input DATA/lcb.jsonl --embeddings EMB/lcb-swe-qwen3-0.6b-issue-2048.npz \
  --frontier gpt-5 --clusters 60 --top-k 4 --output RUNS/avengers-lcb-paper
```

`--clusters 60 --top-k 4` are the paper's settings and the defaults;
`--clusters 25 --top-k 3` are the repository's config file. `--weights` sets the
performance weights (default 0.0 to 1.0 in steps of 0.1), `--seeds` the number of
seeds (default 10), `--epsilon` the limit used for the held-out pick.

## Output files

| File | Contents |
| --- | --- |
| `avengers-runs.csv` | One row per seed and weight: savings, accuracy, frontier accuracy, degradation rate and the share of tasks sent to the frontier, on the official test split |
| `avengers-curve.csv` | The same per weight, averaged over seeds, with the number of runs over epsilon |
| `avengers-picks.csv` | One row per seed: the weight picked on half of the test split and its results on the other half |
| `avengers-summary.json` | Settings, official commit, embedding fingerprint, the curve and the held-out pick summary |

## Limits

- The embedding model matters to Avengers-Pro and the paper used Qwen3-Embedding-8B.
  State which embeddings a result used.
- With about 740 training tasks, 60 clusters hold about 12 tasks each, so per-cluster
  accuracies are noisy. The paper's setting was chosen for a larger, mixed benchmark.
- The script evaluates one tenant and treats each task type as one official "dataset".
- The official split differs from Tine's splits, so compare methods on savings and
  degradation rates, not task by task.
