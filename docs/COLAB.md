# Tine Colab handoff

Open [`notebooks/tine_colab.ipynb`](../notebooks/tine_colab.ipynb) in Google Colab:

<https://colab.research.google.com/github/Nandu-Dammalapti/tine/blob/main/notebooks/tine_colab.ipynb>

Use a **CPU runtime**. Run the two setup code cells, Section 1 to fetch and import the graded LiveCodeBench outcomes, and Section 2 to inspect coverage. **Stop after Section 2 for Step 1.** It saves `outcomes.jsonl`, `step1-preflight.json`, and the archive manifest to Drive. Sections 3–5 are for later CPU evaluation and optional embeddings.

## What you need to change

Only one value is required: set `FRONTIER` to the exact name of your intended default frontier model from the model names the notebook prints. This is the baseline against which savings are measured. Do not choose it by searching the test set for the highest accuracy.

The standard output location is `/content/drive/MyDrive/tine`. If you already have canonical Tine outcomes, skip the archive download/import cells and set `DATASET` to that JSONL or CSV file instead.

The notebook keeps `EPSILON = 0.02`, `DELTA = 0.05`, and `SEED = 42`. Thresholds are tested in a fixed sequence (cautious to aggressive, stopping at the first failure), so only the three routing methods share the error budget. One tenant and task type then needs at least 203 risk-calibration tasks even with zero observed degradations. LiveCodeBench alone (1,055 tasks) gives 211, which is barely enough. Step 2 pools more coding datasets.

## Step 2: pool the coding datasets

Run [`notebooks/tine_colab_step2_coding.ipynb`](../notebooks/tine_colab_step2_coding.ipynb) on a CPU runtime after Step 1. It imports the graded HumanEval, MBPP and SWE-bench Verified results from the archive already on Drive. It pools them with LiveCodeBench into one `coding` group with `tine combine-outcomes`, which keeps only models present in every dataset, drops models whose recorded cost is zero on every row, and excludes `openrouter` (a router, not a model). It then evaluates against `gpt-5` and `claude-sonnet-4`. The pooled guarantee covers tasks drawn like that mix, not each dataset on its own.

## Repository tests on SWE-bench patches

[`notebooks/tine_colab_swe_repo_tests.ipynb`](../notebooks/tine_colab_swe_repo_tests.ipynb) runs each
repository's tests on the recorded model patches, on a CPU runtime. **It executes model-written code**:
use a fresh runtime and delete it afterwards. Read [SWE_CHECKS.md](SWE_CHECKS.md) first. It has not
been run yet; its first cells are a smoke test for that reason.

## Cells and outputs

| Stage | What to run | What it saves |
| --- | --- | --- |
| Mount and install | First two code cells | Current repository revision is displayed |
| Fetch existing data | `scripts/fetch_benchmark.py` cell | Pinned archive, safe extraction, and SHA-256 manifest in `tine/benchmark/` |
| Import | `import-llmrouterbench` cell | `tine/outcomes.jsonl` using existing binary labels and recorded costs |
| Inspect and choose baseline | Data summary, then `FRONTIER` and preflight cell | `tine/run-config.json`, `tine/step1-preflight.json` |
| CPU evaluation | `python -m tine run` without embeddings | `tine/cpu-lexical/` report artifacts |
| Pin encoder | Optional dependency/model metadata cell | `tine/encoder-config.json`, `tine/colab-packages.txt` |
| Debug encoder | `scripts/embed.py --limit 50 --device cpu` cell | `tine/embeddings-smoke.npz` and batch checkpoints |
| Full encoder | Same helper without `--limit` | `tine/embeddings.npz`, metadata, and resumable batch checkpoints |
| Embedded evaluation | `python -m tine run --embeddings ...` | `tine/cpu-embedded/` report artifacts |

All paths in this table are relative to `MyDrive`. Read the reports' dataset exclusions, calibration fallback, test metrics, and cost-quality curves. These are offline results on observed benchmark outcomes; they do not establish production savings, latency, or a guarantee under arbitrary distribution shifts.

## Resume after a disconnect

Mount Drive and rerun setup/config cells. The existing encoder config is reused, and the embedding helper resumes completed batch shards. Keep the same dataset contents and order, model revision, batch size, and `max_seq_length`. It rejects incompatible checkpoints rather than silently mixing embeddings. For another configuration, use another output filename. Do not run concurrent writers against one output/checkpoint directory.

The default encoder is `sentence-transformers/all-MiniLM-L6-v2`. Its full commit SHA is obtained from Hugging Face's official metadata and pinned in the saved config. The helper uses `trust_remote_code=False`, normalized embeddings, and a 256-token truncation limit. It hashes each full prompt and keys embeddings by both tenant and task ID so the evaluator can verify input identity. Truncation is a baseline tradeoff; it does not preserve every part of long coding prompts.

## Boundaries and troubleshooting

- No LLM answer generation or paid provider APIs are used. The optional encoder stage downloads model weights only when you run it in Colab.
- The pinned [benchmark archive](https://huggingface.co/datasets/NPULH/LLMRouterBench/blob/b95870f9e60c589e82f7972c91ffa8de9ca89f85/bench-release.tar.gz) is about 1.28 GB compressed. Drive holds the archive and extracted results. The helper limits extraction to 20 GB and verifies the published LFS checksum when available.
- Missing costs, fractional/ambiguous grades, incompatible schemas, and conflicting runs produce errors. Do not turn missing values into zero or choose an arbitrary success threshold to force an import.
- If Git clone cannot read a private repository, upload and extract your authorized repository ZIP so `/content/tine/pyproject.toml` exists, then rerun setup. ZIP sources are recorded without a Git revision; never paste tokens into saved cells.
- CPU modules and local helper behavior are tested in the development environment. The real Hugging Face download and Colab encoder path still need validation when this notebook is run; they are not described as already executed.

Keep the report directories, data/source manifests, encoder config, and embedding files/checkpoints on Drive. After saving the outputs, choose **Runtime → Disconnect and delete runtime**. Do not host Tine's proxy on Colab.
