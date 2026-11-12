# Router rule and guarantee options, 2026-11-11

Offline results on already-graded LLMRouterBench outcomes. No model was called.
Small summary table only (`router-rules-summary.csv`).

## Question

The prompt-only router certifies about 0% savings
([coding-benchmark](../2026-11-09-coding-benchmark/README.md)). The
[Avengers-Pro run](../2026-11-10-avengers-pro/README.md) pointed at two parts of the
Gacrux definition that might be holding it back. Do they?

1. **The rule.** `absolute` (AGENTS.md): cheapest model whose predicted success is at
   least tau, else the frontier. `relative` (`tine run --rule relative`): cheapest
   model whose predicted success is at least the frontier's own on that request, with
   a margin set by the threshold.
2. **The guarantee.** `degradation`: bounds the rate of "frontier right, chosen model
   wrong". `net` (`tine run --loss net`): bounds net accuracy loss, so tasks gained
   offset tasks lost.

It also records the comparison of quality limits the team asked for: epsilon 0.02
against 0.04, with 0.05 for reference.

## Setup

- `tine run`, 10 seeds, delta 0.05, frontier `gpt-5`, the three predictors.
- `lcb`: 1,055 LiveCodeBench tasks, TF-IDF features. `lcb-swe`: 1,555 tasks
  (LiveCodeBench + SWE-bench Verified), Qwen3-Embedding-0.6B features with the
  issue-only view, as in the earlier `lcb-swe` runs.
- All four combinations of rule and guarantee at epsilon 0.02, 0.04, 0.05: 240 runs.
- The net guarantee is an upper bound on the degradation rate minus a lower bound on
  the gain rate, each at half the error level, so it is not automatically looser than
  the degradation guarantee.

## Results

Mean certified savings vs gpt-5 over 10 seeds; the best of the three predictors in
each cell, with its accuracy change against gpt-5 in points.

`lcb`:

| epsilon | absolute, degradation (current) | relative, degradation | relative, net |
| --- | --- | --- | --- |
| 0.02 | 0.1% | 1.1% (+0.3) | 0% |
| 0.04 | 2.1% (-0.4) | 2.9% (-0.1) | 2.9% (-0.1) |
| 0.05 | 3.7% (-0.7) | 4.7% (-0.5) | 4.7% (-0.5) |

`lcb-swe`:

| epsilon | absolute, degradation (current) | relative, degradation | relative, net |
| --- | --- | --- | --- |
| 0.02 | 0.6% (-0.2) | 2.8% (+0.6) | 2.9% (+0.2) |
| 0.04 | 4.3% (-1.1) | 8.6% (+0.7) | 17.8% (+2.3) |
| 0.05 | 4.8% (-1.2) | 12.5% (+0.6) | 24.3% (+0.6) |

- **On `lcb` neither option matters.** gpt-5 is the most accurate model on every kind
  of task there, so there is nothing for a relative rule to find and few gains to
  offset losses. The router stays under 5% at any limit tried.
- **On `lcb-swe` both help, and together they take the router from about 4-5% to
  18-24% at epsilon 0.04-0.05, with accuracy above gpt-5's.** The relative rule lets
  the router leave gpt-5 on the SWE-bench tasks; the net guarantee accepts that it
  loses some tasks there while gaining more.
- **At epsilon 0.02 nothing gets past 3%.**
- Runs over the limit on the test split: at most 1 of 10 in any cell.
- `absolute` with `net` is never better than the current setting.

## Epsilon 0.02 against 0.04

With the current rule and guarantee the router goes from 0.1% to 2.1% on `lcb` and
from 0.6% to 4.3% on `lcb-swe`. Raising the limit alone does little for the
prompt-only router. Chains benefit far more from a 4% limit: see
[agreement-check](../2026-11-11-agreement-check/README.md) (64% on MMLU-Pro) and
[response-check](../2026-11-10-response-check/README.md).

## Read this before quoting the `lcb-swe` numbers

The gain there comes from gpt-5 scoring 15.8% on SWE-bench in this benchmark. Later
work showed that score is mostly a formatting failure: 70% of gpt-5's SWE-bench
answers are patches that do not apply or contain no usable patch, and the benchmark
under-credits some patches that do work
([swe-static-checks](../2026-11-10-swe-static-checks/README.md)). A router that sends
SWE-bench tasks away from gpt-5 is exploiting that. It is real evidence that the
relative rule can find "the frontier is the wrong choice here" when it is true, and
that the net guarantee then lets the saving through. It is not evidence that a
customer would see 20%.

## What this means for the design

- The relative rule costs nothing where the frontier is best everywhere and helps
  where it is not. There is no result here against making it the default; that is a
  team decision, and it changes the objective in AGENTS.md section 2.
- The net guarantee is a different promise to the customer: accuracy overall will
  not fall by more than epsilon, but individual tasks may regress. Offer it as a
  customer choice, not a silent default.
- Negative savings appear for the cluster predictor in a few runs (as low as -7.9%):
  the router chooses by estimated cost per task type, and on SWE-bench some models
  cost more than gpt-5 while their pooled estimate says less. Cost estimates need to
  be per kind of task, not per pooled task type.

## Reproduce

```bash
for eps in 0.02 0.04 0.05; do for seed in 0 1 2 3 4 5 6 7 8 9; do
  for rule in absolute relative; do for loss in degradation net; do
    python -m tine run --input DATA/lcb.jsonl --frontier gpt-5 --epsilon $eps --seed $seed \
      --rule $rule --loss $loss --output RUNS/rules/lcb-$rule-$loss-$eps-$seed
    python -m tine run --input DATA/lcb-swe.jsonl --frontier gpt-5 --epsilon $eps --seed $seed \
      --rule $rule --loss $loss --embeddings EMB/lcb-swe-qwen3-0.6b-issue-2048.npz \
      --output RUNS/rules/lcbswe-$rule-$loss-$eps-$seed
done; done; done; done
```
