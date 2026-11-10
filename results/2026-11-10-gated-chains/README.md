# Predictor-gated escalation chains, 2026-11-10

Offline results on already-graded LLMRouterBench outcomes. No model was called and
no model-written code was run. Small summary tables only; per-task outputs and the
benchmark files stay out of Git. Same data, frontier and pool as
[2026-11-09](../2026-11-09-coding-benchmark/README.md): `lcb`, LiveCodeBench,
1,055 tasks, 28 models, frontier `gpt-5`, delta = 0.05.

## Question

The 2026-11-09 results showed that escalation chains need a check that catches about
95% of wrong answers, and that public samples catch only 38-74% for the cheap models
worth trying. Does the router's predictor close that gap? A gated chain tries a
cheaper model only where its predicted success is at least a threshold, then still
checks the answer (`tine cascade --gate`, see [docs/CHECKS.md](../../docs/CHECKS.md)).

## Setup

- 10 seeds. Each splits tasks 35 / 10 / 20 / 20 / 15 into train / score-calibration /
  choose / certify / test parts. The predictor (TF-IDF features) is fitted on the
  first two parts only.
- `certified` = plain chain, `gated_certified` = the same certified search over every
  chain with no gate or a gate at 0.5-0.9. Both use the same parts.
- Checks are simulated. `catch` 1.0 / 0.95 / 0.90 is one rate for every model.
  `per-model` gives each model the public-sample catch rate measured on 2026-11-09
  (`lcb-catch-rates.csv`), applied independently of the task. Per-answer check results
  were not available for this run, so this cannot show whether the check and the
  predictor miss the same answers.
- The certify part has about 211 tasks, so nothing can be certified at epsilon 0.01
  here (zero degradations give a bound of 0.014). The 2026-11-09 run used 30% of tasks
  to certify; its numbers are not directly comparable.

## Results

Mean held-out savings vs gpt-5 over 10 seeds, certified picks
(`gated-lcb-*-summary.csv`). gpt-5 accuracy on the test parts: 88.4%.

| catch | epsilon | plain chain | gated: logistic | gated: knn | gated: cluster |
| --- | --- | --- | --- | --- | --- |
| 1.00 | 0.02 | 45.7% | 45.7% | 45.8% | 45.7% |
| 0.95 | 0.02 | 5.0% | 8.1% | 13.7% | 9.7% |
| 0.90 | 0.02 | 0% | 0% | 0% | 1.3% |
| per-model | 0.02 | 0% | 1.4% | 1.1% | 0.1% |
| 0.95 | 0.05 | 44.1% | 40.3% | 45.4% | 48.4% |
| 0.90 | 0.05 | 26.6% | 26.6% | 22.1% | 19.3% |
| per-model | 0.05 | 3.2% | 18.1% | 15.2% | 12.6% |

- At epsilon 0.02 with realistic checks (`per-model`), gating moves certified savings
  from 0% to about 1%. In 6 to 8 of 10 seeds the pick is still gpt-5 only.
- With a 95% check, gating raises savings from 5% to 8-14%, by certifying a chain in
  2-3 of 10 seeds where the plain chain certifies in 1.
- At epsilon 0.05 with `per-model` checks, gating helps more (3% to 13-18%), at
  1.2-1.5 points lower accuracy than gpt-5. Epsilon 0.05 is the proposed setting for
  low-stakes internal work, not for coding.
- Runs over epsilon on the test part: 0 of 10 everywhere except four gated settings
  at epsilon 0.05 with 1 of 10 each (logistic and cluster, at `per-model` and at
  catch 0.90). The test part is about 158 tasks, so a single run is noisy.
- With a perfect check the gate is not chosen: skipping a cheap attempt saves little.

**Conclusion:** the gate works as built, but the TF-IDF predictors are too weak for
it to rescue weak checks at coding-level epsilon. This matches the 2026-11-09 finding
that the predictors cannot tell from the prompt which tasks a cheaper model solves.

## Qwen3 embeddings for the gate

Added later on 2026-11-10 (`gated-lcb-qwen3-*-summary.csv`). The same runs with the
gate predictor fed Qwen3-Embedding-0.6B vectors (1,024-d, revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, full prompt, 2,048 tokens, computed
locally) through `--gate-embeddings`, in place of TF-IDF. Same seeds and parts, so the
plain-chain column is identical.

| catch | epsilon | plain chain | TF-IDF gate: logistic / knn / cluster | Qwen3 gate: logistic / knn / cluster |
| --- | --- | --- | --- | --- |
| 0.95 | 0.02 | 5.0% | 8.1% / 13.7% / 9.7% | 12.3% / 8.8% / 5.2% |
| per-model | 0.02 | 0% | 1.4% / 1.1% / 0.1% | 1.4% / 1.2% / 0.9% |
| 0.90 | 0.05 | 26.6% | 26.6% / 22.1% / 19.3% | 27.5% / 30.3% / 31.4% |
| per-model | 0.05 | 3.2% | 18.1% / 15.2% / 12.6% | 11.0% / 16.1% / 14.0% |

A stronger encoder does not make the gate better: the differences go both ways and
are within the spread between predictors. Runs over epsilon: none at epsilon 0.02;
at epsilon 0.05, 2 of 10 for the Qwen3 cluster gate and 1 of 10 for the Qwen3
logistic gate on `per-model`, and 1 of 10 for the Qwen3 knn gate at catch 0.90. This
matches the router result in the 2026-11-09 folder: Qwen3 embeddings did not help
there either. The 8B encoder was not tried for the gate.

Reproduce: add `--gate-embeddings EMB/lcb-qwen3-0.6b-2048.npz` to the command below
(the embedding command is in [avengers-pro](../2026-11-10-avengers-pro/README.md)).

## Next

1. Rerun with per-answer check results (`--checks`) from `scripts/lcb_sample_checks.py`
   instead of simulated per-model rates.
2. The gate with Qwen3 0.6B embeddings is done (above). `lcb-swe` with the issue-only
   view is still to run.
3. Stronger checks remain the main lever: differential testing between models, or a
   judge model, measured the same way.

## Reproduce

After the 2026-11-09 steps have produced `DATA/lcb.jsonl`:

```bash
for g in logistic knn cluster; do
  python -m tine cascade --input DATA/lcb.jsonl --frontier gpt-5 --gate $g \
    --catch 1.0 0.95 0.9 --catch-rates results/2026-11-10-gated-chains/lcb-catch-rates.csv \
    --epsilon 0.02 0.01 0.05 --output RUNS/gated-lcb-$g
done
```
