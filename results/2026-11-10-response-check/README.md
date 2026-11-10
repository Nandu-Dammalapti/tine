# Response check for escalation chains, 2026-11-10

Offline results on already-graded LLMRouterBench outcomes. No model was called and
no model-written code was run. Small summary tables only. Same data and frontier
(`gpt-5`) as [2026-11-09](../2026-11-09-coding-benchmark/README.md); delta = 0.05.

## Question

Escalation chains need a check that catches most wrong answers. Public samples catch
38-74% for the cheap models worth trying, and gating by the prompt-only predictor
barely helped ([gated chains](../2026-11-10-gated-chains/README.md)). Does the
response itself say whether the answer is right?

## The check

`tine/acceptance.py`, `tine cascade --response-check`
([docs/CHECKS.md](../../docs/CHECKS.md)). For each cheaper model, a small classifier
scores an answer from two numbers the ledger already records: prompt tokens and
response tokens. The answer is accepted only if the score reaches a threshold;
otherwise the chain escalates. It runs no tests and calls no model.

It works because a long response usually means the model struggled. On tasks the
classifier never saw, the score separates right from wrong answers with an AUC of
0.76-0.99 across the 27 cheaper `lcb` models (mean over 10 seeds,
`response-check-auc.json`): for example 0.93 for Qwen3-8B, 0.88 for intern-s1-new,
0.87 for qwen3-235b-a22b-thinking, 0.80 for qwen3-235b-a22b-2507. In the 2026-11-09
results the prompt alone could not make this distinction.

## Setup

- 10 seeds. Each splits tasks 20 / 10 / 15 / 40 / 15 into train / score-calibration /
  choose / certify / test parts. The classifier is fitted on the first two only.
  `lcb` has about 422 certify and 158 test tasks per seed.
- `certified` = plain chain. `response_certified` = the same certified search over
  every chain with the setting's check alone or combined with the response check at
  a threshold of 0.5-0.95.
- Check settings: `none` (`--catch 0`: the response check is the only check, so
  nothing in that row is simulated); `per-model` (each model's measured public-sample
  catch rate from 2026-11-09, simulated per answer); and one simulated rate for all
  models. A simulated check and the response check are combined as if their misses
  were unrelated. Per-answer sample-check results were not available for this run.

## Results on `lcb`

Mean held-out savings vs gpt-5 over 10 seeds, certified picks
(`response-lcb-summary.csv`). gpt-5 accuracy on the test parts: 88.4%.

| other check | epsilon | plain chain | with response check | accuracy | runs over epsilon |
| --- | --- | --- | --- | --- | --- |
| none | 0.02 | 0% | 3.6% | 88.1% | 1 of 10 |
| per-model samples | 0.02 | 0% | 6.2% | 88.8% | 0 |
| 90% catch | 0.02 | 0% | 10.0% | 89.1% | 0 |
| 95% catch | 0.02 | 12.8% | 12.7% | 88.8% | 0 |
| none | 0.05 | 0% | 14.3% | 87.0% | 1 of 10 |
| per-model samples | 0.05 | 0% | 31.8% | 88.5% | 0 |
| 90% catch | 0.05 | 42.0% | 49.4% | 89.2% | 1 of 10 |
| 95% catch | 0.05 | 46.9% | 51.0% | 90.0% | 0 |

- With realistic sample checks at epsilon 0.02, certified savings go from 0% to 6.2%,
  against about 1% for the prompt-only gate. In 4 of 10 seeds the pick is still
  gpt-5 only.
- At epsilon 0.05 with realistic sample checks: 31.8% saved at the same accuracy as
  gpt-5 (88.5% vs 88.4%), certified in all 10 seeds. The usual pick is
  `Qwen3-8B > gpt-5`. The chain loses some tasks gpt-5 solves and gains some gpt-5
  fails; the guarantee counts only the losses.
- With no other check at all, the response check certifies 3.6% at epsilon 0.02 and
  14.3% at epsilon 0.05.
- At epsilon 0.01 almost nothing certifies (0.5-1.1%).
- Epsilon 0.02 is limited by sample size as much as by the check: with 422 certify
  tasks a policy passes only with at most 3 degradations (0.7%).

## Results on `lcb-swe`

1,555 tasks, 13 models, gpt-5 accuracy on the test parts 64.4%
(`response-lcb-swe-summary.csv`). No measured sample-check rates exist for SWE-bench.

| other check | epsilon | plain chain | with response check | runs over epsilon |
| --- | --- | --- | --- | --- |
| none | 0.02 | 0% | 0.7% | 1 of 10 |
| 90% catch | 0.02 | 3.3% | 9.6% | 1 of 10 (plain: 1 of 10) |
| 95% catch | 0.02 | 9.5% | 12.0% | 0 |
| none | 0.05 | 0% | 6.8% | 2 of 10 |

The AUC here (0.78-0.92) is partly the classifier telling long SWE-bench prompts from
short LiveCodeBench ones, so it overstates the signal within one kind of task.

## Limits

- Runs over epsilon are expected in up to 5% of runs per setting (delta = 0.05) and
  the test parts are small. `none` at epsilon 0.05 on `lcb-swe` had 2 of 10.
- Token counts must mean the same in production as in the fitting data (reasoning
  tokens counted or not, same model version). The check must be refitted per model.
- These are benchmark runs by one provider setup. A model whose response length does
  not track difficulty gets no benefit.
- Escalated answers are still paid for: the `lcb` picks average 1.2-1.8 calls per task.

## Next

1. Rerun with per-answer sample-check results (`--checks`) so the combination of
   sample check and response check is measured, not assumed independent.
2. Log response tokens in shadow mode and fit the check on customer traffic.
3. Decide whether the guarantee should bound net accuracy loss instead of only
   counting losses: at epsilon 0.05 the certified chain matches gpt-5's accuracy.

## Reproduce

After the 2026-11-09 steps have produced `DATA/lcb.jsonl` and `DATA/lcb-swe.jsonl`:

```bash
python -m tine cascade --input DATA/lcb.jsonl --frontier gpt-5 --response-check \
  --catch 0 0.9 0.95 1.0 --catch-rates results/2026-11-10-gated-chains/lcb-catch-rates.csv \
  --epsilon 0.02 0.01 0.05 --output RUNS/response-lcb
python -m tine cascade --input DATA/lcb-swe.jsonl --frontier gpt-5 --response-check \
  --catch 0 0.9 0.95 1.0 --epsilon 0.02 0.01 0.05 --output RUNS/response-lcb-swe
```
