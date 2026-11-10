# Avengers-Pro head-to-head on LiveCodeBench, 2026-11-10

Offline results on already-graded LLMRouterBench outcomes. No model was called.
Small summary tables only. Read [docs/AVENGERS_PRO.md](../../docs/AVENGERS_PRO.md)
first: it explains what the official code does and how we run it.

## Question

AGENTS.md names Avengers-Pro as the main routing baseline. Tine's `cluster` method is
only an Avengers-style heuristic. How does the official code do on the `lcb` pool,
and can it hold a 2% quality limit?

## Setup

- Official code: <https://github.com/ZhangYiqun018/AvengersPro> at commit
  `dfb60b156c4b6011b89edfc8516aa16d8e4f4d38`, `BalanceClusterRouter`. Its split,
  clustering, scoring and routing ran unmodified through
  `scripts/avengers_pro_baseline.py`. Only the embedding client was replaced.
- Data: `lcb`, 1,055 LiveCodeBench tasks, 28 models, frontier `gpt-5` (same pool as
  the other results folders). The official 70 / 30 split gives 738 train and 317
  test tasks per seed. 10 seeds.
- Embeddings: Qwen3-Embedding-0.6B (1,024-d, revision
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`), full prompt, 2,048-token truncation,
  computed locally in float32 on CPU (feature fingerprint
  `60a57594f65762ed0497742498c056eb4a6dc3575c08a99e83c73c8e627aa013`). **The paper used Qwen3-Embedding-8B.** The 8B
  vectors exist on Nandu's Drive and were not available for this run.
- Two settings: the paper's (60 clusters, 4 nearest) and the repository's config
  file (25 clusters, 3 nearest). `beta` = 9 in both.
- Performance weight from 0.0 to 1.0 in steps of 0.1, with cost sensitivity =
  1 - weight.
- Scoring: `tine.calibration.evaluate`, the same function as for Tine's routers.
  Savings are against sending every task to gpt-5. Degradation = gpt-5 solved the
  task and the chosen model did not.

## Result 1: the cost-accuracy curve

Mean over 10 seeds on the official test split (`avengers-lcb-*-curve.csv`). gpt-5
accuracy on these test tasks: 86.3%.

| performance weight | paper setting: saved / accuracy / degradation | repository setting: saved / accuracy / degradation |
| --- | --- | --- |
| 1.0 (accuracy only) | 7.6% / 82.7% / 5.2% | -1.0% / 84.4% / 2.6% |
| 0.9 | 28.3% / 82.1% / 5.6% | 19.5% / 84.2% / 2.8% |
| 0.8 | 40.7% / 80.3% / 7.8% | 40.4% / 81.9% / 5.8% |
| 0.7 | 61.8% / 77.4% / 11.5% | 70.9% / 78.5% / 10.6% |
| 0.5 | 90.5% / 72.8% / 16.3% | 90.4% / 75.9% / 13.6% |
| 0.0 (cost only) | 99.9% / 16.4% / 69.9% | 100.0% / 15.9% / 70.4% |

- **No weight keeps mean degradation at or below 2%.** The lowest is 2.6%
  (repository setting, weight 1.0), and that policy costs 1% more than gpt-5.
- Even when told to maximise accuracy, Avengers-Pro is less accurate than always
  using gpt-5 (84.4% or 82.7% against 86.3%). It sends 73% (repository) or 58%
  (paper) of test tasks to gpt-5 and the rest to whichever model looked best in a
  small cluster. With 60 clusters each holds about 12 training tasks, so "best in
  cluster" is often luck. This is why the paper's setting does worse here.
- It is a cost-accuracy dial. Large savings are available, at a visible cost in
  accuracy: 40% saved costs 4.5 to 6.1 points.

## Result 2: picking a weight under a quality limit

For each seed, half of the official test split picks the weight with the most
savings among those with degradation <= epsilon on that half; the other half
(159 tasks) scores it (`avengers-lcb-*-picks-eps*.csv`). This pick is
**uncertified**: Avengers-Pro has no confidence bound.

| setting | epsilon | mean saved | accuracy vs gpt-5 | runs over epsilon | runs that fell back to gpt-5 only |
| --- | --- | --- | --- | --- | --- |
| paper | 0.02 | 0% | 86.5% vs 86.5% | 0 of 10 | 10 |
| paper | 0.05 | 14.3% | 84.0% vs 86.5% | 4 of 10 | 4 |
| repository | 0.02 | 18.2% | 84.2% vs 86.5% | 6 of 10 | 3 |
| repository | 0.05 | 34.2% | 82.1% vs 86.5% | 5 of 10 | 0 |

A weight that looks safe on 158 tasks breaks the limit on the next 159 about half
the time. Where it saves money it gives up 2.3 to 4.5 points of accuracy.

## Beside Tine's methods on the same pool

Different splits were used (see Limits), so compare the pattern, not single digits.

| method | guarantee | epsilon 0.02: saved / runs over | epsilon 0.05: saved / runs over | accuracy change at 0.05 |
| --- | --- | --- | --- | --- |
| Avengers-Pro, repository setting, held-out pick | none | 18.2% / 6 of 10 | 34.2% / 5 of 10 | -4.5 points |
| Avengers-Pro, paper setting, held-out pick | none | 0% / 0 | 14.3% / 4 of 10 | -2.5 points |
| Tine prompt-only router (`tine run`) | certified | 0% / 0 | not run | 0 |
| Tine chain + response check, no other check | certified | 3.6% / 1 of 10 | 14.3% / 1 of 10 | -1.4 points |
| Tine chain + response check + sample checks (simulated at measured per-model rates) | certified | 6.2% / 0 | 31.8% / 0 | +0.1 points |

Sources for the Tine rows: [response-check](../2026-11-10-response-check/README.md)
and [coding-benchmark](../2026-11-09-coding-benchmark/README.md).

- Avengers-Pro's savings are real but come with an accuracy loss it cannot bound
  and, when one tries to bound it by picking a weight, the limit fails in about half
  of the runs.
- Tine's certified chains save less at epsilon 0.02, hold the limit, and at epsilon
  0.05 with sample checks save about as much as Avengers-Pro's repository setting
  with no loss of accuracy. That last row still contains a simulated check.
- Both prompt-only approaches fail at a 2% limit on this pool: Tine's router
  certifies nothing, and Avengers-Pro cannot stay within it. The difference is that
  Tine's router reports that it cannot, and falls back to gpt-5.

## Results on `lcb-swe`

Added later on 2026-11-10 (`avengers-lcb-swe-*`). 1,555 tasks (LiveCodeBench +
SWE-bench Verified), 13 models, 1,088 train and 467 test tasks per seed. Embeddings:
the same encoder with the issue-only view for SWE-bench prompts
(`scripts/embed.py --prompt-view issue`). gpt-5 accuracy on the test tasks: 64.0%.

| performance weight | paper setting: saved / accuracy / degradation | repository setting: saved / accuracy / degradation |
| --- | --- | --- |
| 1.0 | -21.7% / 67.1% / 5.3% | -34.5% / 68.4% / 3.9% |
| 0.8 | 10.3% / 66.0% / 5.7% | 4.9% / 67.5% / 4.4% |
| 0.7 | 27.3% / 65.1% / 6.6% | 18.9% / 66.6% / 5.5% |
| 0.6 | 47.1% / 63.3% / 8.4% | 41.8% / 64.6% / 7.7% |
| 0.5 | 66.1% / 60.7% / 11.0% | 62.3% / 61.8% / 10.4% |

This pool behaves differently, and the reason matters for Gacrux:

- **Avengers-Pro beats gpt-5's accuracy here while spending less**: repository
  setting, weight 0.6: 41.8% saved at 64.6% against 64.0%. That is the kind of result
  the paper reports.
- **gpt-5 is not the best model on SWE-bench in this benchmark.** It solves 15.8% of
  the SWE-bench tasks; claude-sonnet-4 and gemini-2.5-pro solve 34.6%, and
  deepseek-v3.1-terminus solves 25.4% at a tenth of gpt-5's cost. gpt-5 is the most
  accurate model on the pool only because of LiveCodeBench (86.4%). Clustering
  separates the two kinds of task and sends SWE-bench elsewhere.
- **The degradation rate is still 3.9% or more at every weight**, because other
  models fail some SWE-bench tasks that gpt-5 solves, even while solving more overall.
  So under Tine's guarantee, which counts only losses, none of these policies passes
  at epsilon 0.02, and the held-out pick falls back to gpt-5 in 9 or 10 of 10 seeds
  (table below).
- A hindsight rule shows the size of this: "send every SWE-bench task to
  deepseek-v3.1-terminus, everything else to gpt-5" saves 27.7% and is 3.1 points
  more accurate than gpt-5 (66.8% against 63.7% on all 1,555 tasks), with a
  degradation rate of 3.1%.

Held-out picks (234 evaluation tasks per seed):

| setting | epsilon | mean saved | accuracy vs gpt-5 | runs over epsilon | runs that fell back to gpt-5 only |
| --- | --- | --- | --- | --- | --- |
| paper | 0.02 | 0% | 64.4% vs 64.4% | 0 of 10 | 10 |
| paper | 0.05 | 5.7% | 64.5% vs 64.4% | 2 of 10 | 8 |
| repository | 0.02 | 1.2% | 64.5% vs 64.4% | 1 of 10 | 9 |
| repository | 0.05 | 7.0% | 65.4% vs 64.4% | 1 of 10 | 7 |

Two consequences for Gacrux, both open decisions (see the results index):

1. Tine's router rule "cheapest model whose predicted success is at least tau, else
   the frontier" can never leave the frontier on tasks where every model's predicted
   success is low, even when the frontier is the worst choice there.
2. A guarantee that counts only losses rejects a policy that is cheaper and more
   accurate than the frontier.

## Limits

- 0.6B embeddings, not the paper's 8B. On the 2026-11-09 router runs the 8B encoder
  did not change Tine's results, but Avengers-Pro has not been run with it.
- One benchmark, one task type. The paper reports results on six mixed benchmarks,
  where clusters separate kinds of task. Within one coding benchmark there is less
  for clustering to find.
- The official split (70 / 30) differs from Tine's splits, and the held-out pick
  uses 158 + 159 tasks, so single runs are noisy.
- The weight grid is 0.1 wide. A finer grid near 0.9-1.0 might find a slightly
  better point in the repository setting. It would not create a guarantee.

## Next

1. Rerun with the Qwen3-Embedding-8B vectors from Nandu's Drive.
2. Run on a pool with more kinds of task (for example `lcb` + MMLU-Pro), which is
   closer to the paper's setting.

## Reproduce

After the 2026-11-09 steps have produced `DATA/lcb.jsonl`, with the official code
downloaded to `AP/` and an embeddings NPZ covering the `lcb` tasks:

```bash
python scripts/embed.py --input DATA/lcb.jsonl --output EMB/lcb-qwen3-0.6b-2048.npz \
  --model Qwen/Qwen3-Embedding-0.6B --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 \
  --max-seq-length 2048 --batch-size 8
for eps in 0.02 0.05; do
  python scripts/avengers_pro_baseline.py --avengers-dir AP --avengers-commit dfb60b156c4b6011b89edfc8516aa16d8e4f4d38 \
    --input DATA/lcb.jsonl --embeddings EMB/lcb-qwen3-0.6b-2048.npz --frontier gpt-5 \
    --clusters 60 --top-k 4 --epsilon $eps --output RUNS/avengers-lcb-paper-eps$eps
  python scripts/avengers_pro_baseline.py --avengers-dir AP --avengers-commit dfb60b156c4b6011b89edfc8516aa16d8e4f4d38 \
    --input DATA/lcb.jsonl --embeddings EMB/lcb-qwen3-0.6b-2048.npz --frontier gpt-5 \
    --clusters 25 --top-k 3 --epsilon $eps --output RUNS/avengers-lcb-repo-eps$eps
done
```

The embeddings used here were produced as the first 1,055 rows of an `lcb-swe` run
with `--prompt-view issue`; LiveCodeBench prompts have no issue block, so they are
encoded whole and the vectors are the same as from the command above.
