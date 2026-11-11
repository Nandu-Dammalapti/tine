# Agreement check on MMLU-Pro, 2026-11-11

Offline results on already-graded LLMRouterBench outcomes and the answers recorded
with them. No model was called. Nothing here is simulated: the check, the costs and
the certification all come from recorded data. Small summary tables only.

## Question

Chains need a check that catches about 95% of wrong answers. For tasks with an exact
answer there is a check that needs no tests and no judge: ask two (or three) cheaper
models and accept their answer only when they agree. Does it certify savings once
the extra calls are paid for?

## Setup

- Data: MMLU-Pro `test_3000`, 3,000 multiple-choice tasks, frontier `gpt-5` (87.4%
  accurate, the most accurate model). 13 cheaper models with recorded cost.
- `scripts/agreement_pairs.py` builds one candidate per group of models. A group
  costs the sum of its members' recorded costs on every task, because all are always
  called. Its check passes on a task when all members recorded the same answer.
  Groups costing more than half of gpt-5 are dropped: 25 pairs; 29 triples from the
  seven cheapest useful models.
- `tine cascade --checks` then does what it does for any measured check: 10 seeds,
  tasks split 40 / 30 / 30 into choose / certify / test (900 certify tasks), chains of
  up to three groups ending in gpt-5, certified with delta 0.05.

## Results

Mean held-out savings vs gpt-5 over 10 seeds, certified picks. Savings include every
call made. gpt-5 accuracy on the test parts: 87.3%.

| epsilon | pairs: saved / accuracy / runs over | triples: saved / accuracy / runs over |
| --- | --- | --- |
| 0.01 | 0% / 87.3% / 0 | 3.9% / 87.3% / 1 of 10 |
| 0.02 | 11.1% / 87.2% / 1 of 10 | **32.3% / 87.6% / 0** |
| 0.04 | **63.8% / 86.9% / 0** | 42.6% / 87.5% / 0 |
| 0.05 | 70.3% / 86.5% / 0 | not run |

`agreement-pairs-mmlupro-summary.csv`, `agreement-triples-mmlupro-summary.csv`.

- **At the 2% limit, three cheap models that agree save 32% and are slightly more
  accurate than gpt-5.** It certified in 9 of 10 seeds. This is the first result in
  this repository with large certified savings at epsilon 0.02 from measured data only.
- **At 4%, two models that agree save 64%**, at 0.4 points below gpt-5.
- Pairs barely certify at 2% (2 of 10 seeds): their degradation rate, 1.3% for the
  best pair, is too close to the limit for 900 certify tasks.
- For comparison, the response check alone certified 3.4% at epsilon 0.02 and 19.4% at
  0.05 on this dataset ([response-check](../2026-11-10-response-check/README.md)).

Why it works (`agreement-*-mmlupro-rates.csv`, all 3,000 tasks):

| | best pair | best triple |
| --- | --- | --- |
| members | deepseek-v3.1-terminus + intern-s1-new | deepseek-v3.1-terminus + intern-s1-new + kimi-k2-0905 |
| they agree on | 85.1% of tasks | 76.4% |
| accuracy when they agree | 91.6% | 93.5% |
| degradation (they agree, are wrong, gpt-5 was right) | 1.3% | 0.7% |
| gain (they agree, are right, gpt-5 was wrong) | 1.9% | 1.2% |
| cost, as a share of gpt-5 | 18.6% | 29.3% |

Agreement is a strong signal: when two cheap models give the same answer they are
right more often than gpt-5 is overall, and they gain more tasks than they lose.

## Limits

- One dataset, multiple choice with up to ten options. Agreement by chance is rare
  here; with two or four options it would be more common and the check weaker.
- Exact-answer tasks only. Free text and code cannot be compared this way without
  running the code or a judge.
- Models that share training data make the same mistakes. Agreement between closely
  related models (two versions of one family) says less than between unrelated ones.
- A chain of two groups that share a member is charged for that member twice, so
  savings for multi-group chains are slightly understated.
- Latency: two or three calls in parallel, then a frontier call when they disagree
  (14-31% of tasks at epsilon 0.02-0.04).
- Benchmark runs, not customer traffic.

## Next

1. Other exact-answer datasets (GPQA and the maths sets are too small to certify alone;
   pool them).
2. Agreement plus the response check on the tasks where the group disagrees.
3. For code: agreement on behaviour (two patches pass the same tests), which needs
   execution.

## Reproduce

After importing MMLU-Pro (see [response-check](../2026-11-10-response-check/README.md)):

```bash
python scripts/agreement_pairs.py --input DATA/mmlupro.jsonl --frontier gpt-5 \
  --bench-dir BENCH/extracted/bench-release/mmlupro/test_3000 \
  --output DATA/mmlupro-pairs.jsonl --checks RUNS/mmlupro-agreement.csv
python -m tine cascade --input DATA/mmlupro-pairs.jsonl --frontier gpt-5 \
  --checks RUNS/mmlupro-agreement.csv --epsilon 0.02 0.01 0.04 0.05 --output RUNS/agreement-mmlupro
python scripts/agreement_pairs.py --input DATA/mmlupro.jsonl --frontier gpt-5 --group-size 3 \
  --models deepseek-v3.1-terminus intern-s1-new intern-s1 qwen3-235b-a22b-2507 kimi-k2-0905 deepseek-v3-0324 gemini-2.5-flash \
  --bench-dir BENCH/extracted/bench-release/mmlupro/test_3000 \
  --output DATA/mmlupro-triples.jsonl --checks RUNS/mmlupro-agreement3.csv
python -m tine cascade --input DATA/mmlupro-triples.jsonl --frontier gpt-5 \
  --checks RUNS/mmlupro-agreement3.csv --epsilon 0.02 0.01 0.04 --output RUNS/agreement3-mmlupro
```
