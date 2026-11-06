# Colab work

Everything in this folder needs a GPU or a big download, so it runs on Google Colab.
Everything else (loading results, training the router, calibrating τ, charts) runs on a
normal CPU from the repo root with `python -m tine`.

Notebooks stay thin: they import code from the `tine/` package and only handle Drive,
runtime setup and checkpoints.

| Build step | Notebook | Runtime | Status |
| --- | --- | --- | --- |
| 5. Embed prompts once, save to Drive | `01_embed_prompts.ipynb` | T4 / L4 | Ready; not run yet |
| 2. Fill gaps with open-model answers (vLLM) | `02_generate_open_model_answers.ipynb` | A100 / G4 | Template; not run yet |
| 7. Fine-tuned encoder router (v2) | not written | L4, or TPU with JAX | Later |
| 9. Distillation / LoRA on open-weight outputs | not written | A100 / G4 / TPU with JAX | Later; check provider terms first |

Steps that are **not** Colab work: 1 (load data), 3 (frontier answers via official APIs),
4 (grading), 6 (router training), 8 (calibration), 10 (charts) and the proxy demo, which
goes on a small VM and never on Colab.

## How to open a notebook

In Colab: File → Open notebook → GitHub tab → `Nandu-Dammalapti/tine`, pick the branch,
pick the notebook. The repo is private, so the setup cell clones it with a read-only
fine-grained GitHub token stored as the Colab secret `GITHUB_TOKEN` (key icon in the left bar).

## Rules (from AGENTS.md)

- Debug first with `DEBUG = True` (about 50 rows, CPU runtime). Only then pick a GPU.
- Save outputs and checkpoints to Drive under `MyDrive/tine/`.
- Run the last cell (`runtime.unassign()`) as soon as a job ends; idle runtimes burn units.
- Use vLLM batching, never a per-prompt generation loop.
- Use TPUs only with JAX.
- GPU time and paid API calls need the team's go-ahead.

## Hand-off back to CPU

`01_embed_prompts` writes `MyDrive/tine/embeddings/llmrouterbench-minilm.npz`
(`task_ids`, `vectors`, `model_name`). Download it next to the extracted results and run:

```bash
python -m tine run --llmrouterbench path/to/results/bench --embeddings path/to/llmrouterbench-minilm.npz --per-task-type
```
