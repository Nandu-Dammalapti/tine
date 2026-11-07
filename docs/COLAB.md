# Tine Colab handoff

Open [`notebooks/tine_colab.ipynb`](../notebooks/tine_colab.ipynb) in Google Colab:

<https://colab.research.google.com/github/Nandu-Dammalapti/tine/blob/main/notebooks/tine_colab.ipynb>

The notebook contains the exact Python cells. Run them from top to bottom. Start with a **CPU runtime**, not a GPU. It first downloads existing graded benchmark results, imports LiveCodeBench outcomes, and runs the model-free CPU baseline. It then offers a small fixed sentence encoder, with a 50-prompt CPU check before full embedding. T4 acceleration is optional; the small encoder can run on CPU too.

## What you need to change

Only one value is required: set `FRONTIER` to the exact name of your intended default frontier model from the model names the notebook prints. This is the baseline against which savings are measured. Do not choose it by searching the test set for the highest accuracy.

The standard output location is `/content/drive/MyDrive/tine`. If you already have canonical Tine outcomes, skip the archive download/import cells and set `DATASET` to that JSONL or CSV file instead.

The notebook keeps `EPSILON = 0.02`, `DELTA = 0.05`, and `SEED = 42`. Do not loosen the quality requirement to manufacture savings. A frontier-only routing result can be the correct answer when calibration data or predictive quality is insufficient.

## Cells and outputs

| Stage | What to run | What it saves |
| --- | --- | --- |
| Mount and install | First two code cells | Current repository revision is displayed |
| Fetch existing data | `scripts/fetch_benchmark.py` cell | Archive, safe extraction, source revision and SHA-256 manifest in `tine/benchmark/` |
| Import | `import-llmrouterbench` cell | `tine/outcomes.jsonl` using existing binary labels and recorded costs |
| Inspect and choose baseline | Data summary, then `FRONTIER` cell | `tine/run-config.json` |
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
- The benchmark archive can be large. The fetch cell reports metadata and limits extraction to 20 GB; ensure sufficient runtime and Drive storage. Do not increase the limit without reviewing the archive size.
- Missing costs, fractional/ambiguous grades, incompatible schemas, and conflicting runs produce errors. Do not turn missing values into zero or choose an arbitrary success threshold to force an import.
- If Git clone cannot read a private repository, upload and extract your authorized repository ZIP so `/content/tine/pyproject.toml` exists, then rerun setup. ZIP sources are recorded without a Git revision; never paste tokens into saved cells.
- CPU modules and local helper behavior are tested in the development environment. The real Hugging Face download and Colab encoder path still need validation when this notebook is run; they are not described as already executed.

Keep the report directories, data/source manifests, encoder config, and embedding files/checkpoints on Drive. After saving the outputs, choose **Runtime → Disconnect and delete runtime**. Do not host Tine's proxy on Colab.
