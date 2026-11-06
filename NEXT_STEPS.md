# Status and next steps

Last updated 2026-11-06. Read AGENTS.md first for the full plan.

## What exists

Gacrux router v1, CPU evaluation path (build steps 1, 6, 8, 10), in `tine/`:

| File | What it does |
| --- | --- |
| `tine/schema.py` | The result-row schema; pending outcomes stay unknown, never "failure" |
| `tine/data.py` | `RoutingTable` (tasks × models, one tenant only), CSV/JSONL loader, LLMRouterBench loader, stratified train/calibration/test split, synthetic data |
| `tine/features.py` | TF-IDF stand-in embedder (CPU, no downloads); save/load of Colab embeddings; sentence-transformers wrapper used on Colab |
| `tine/predictors.py` | P(success) per model: kNN retrieval memory, logistic regression, boosted trees; isotonic calibration on out-of-fold scores |
| `tine/routing.py` | Rule: cheapest model with P ≥ τ, else frontier. `FRONTIER_ONLY` (not τ = 1.0) when nothing is certified. Cost estimate per task type |
| `tine/calibration.py` | Certifies τ: Learn-then-Test fixed-sequence testing with exact binomial (Clopper–Pearson) p-values; per task type optional |
| `tine/baselines.py` | Best single, random, kNN router, Avengers-Pro style, oracle |
| `tine/evaluate.py` | Savings, quality loss, regret, oracle gap closed, cost-quality chart |
| `tine/pipeline.py`, `tine/__main__.py` | End-to-end run and CLI |
| `tine/generation.py` | vLLM answer generation for Colab (untested: needs a GPU) |
| `colab/` | Notebooks for the GPU work; see `colab/README.md` |
| `tests/` | 28 tests, including a simulation that checks the guarantee's failure rate stays ≤ δ |

Run it:

```bash
bash scripts/setup.sh
source .tine-dev/venv/bin/activate
bash scripts/check.sh
python -m tine demo --predictor logreg
```

Outputs (`summary.json`, `calibration_steps.csv`, `cost_quality.png`) go to `artifacts/`, which Git ignores.

## What the guarantee means here

Loss per task = 1 when the frontier model was right and the routed model was wrong
("regret"). Its average bounds the accuracy drop from above. τ is certified when, scanning
from cautious to aggressive, the one-sided Clopper–Pearson bound stays ≤ ε; testing stops at
the first failure, which is what makes the many-thresholds search valid without extra
correction. It assumes calibration and future tasks come from the same distribution; with
`--per-task-type`, each type's statement holds separately. At least
`ceil(ln δ / ln(1 − ε))` calibration tasks are needed (149 for ε = 0.02, δ = 0.05).

## Not verified yet

- Every number so far is from **synthetic data**. It is not evidence about real models.
- The LLMRouterBench loader follows the README's documented layout and a small local
  fixture. It has not been run on the real release. Check per-dataset score scales
  (judge-graded sets may not be 0/1) and prompt formats.
- `Avengers-Pro style` is rebuilt from the paper's description. Compare with the authors'
  code before calling it a reproduction.
- `tine/generation.py` and both notebooks have not been run.

## Next steps, in order

1. **Get LLMRouterBench** (run the download cell in `colab/01_embed_prompts.ipynb`, or
   download locally) and run on CPU with the TF-IDF stand-in:
   `python -m tine run --llmrouterbench <results/bench> --datasets humaneval mbpp --per-task-type`.
   Fix the loader against what the real files hold.
2. **Choose the frontier model** explicitly with `--frontier` (default is the most accurate
   model in training data) and the candidate set with `--models`.
3. **Embeddings on Colab**: run `colab/01_embed_prompts.ipynb` (DEBUG first), then rerun
   step 1 with `--embeddings`.
4. **Better predictor = more savings at the same ε.** Compare `--predictor knn|logreg|gbt`;
   consider LightGBM (needs a lockfile update) and combining kNN + classifier scores.
5. **Check the guarantee over many splits** (rerun with several `--seed` values; held-out
   regret should exceed ε in at most about δ of runs, allowing for test-set noise).
6. **Headline chart** for the YC application: % saved at ≤ ε loss vs all baselines.
7. Only then: FastAPI proxy + logger + shadow mode + savings report on a small VM.

Gaps in the router itself, for later: no verified-cache stage, no local LLM judge stage,
cost estimates are per task type means and ignore cached tokens, no nightly retrain job.
