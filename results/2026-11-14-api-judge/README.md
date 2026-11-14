# A pinned strong judge for code, and the first chains with it, 2026-11-14

**This folder's judge run called a paid model.** `claude-sonnet-5-5` through the
Anthropic API, 1,110 requests, **$8.85** in total as reported by the API's usage
figures, authorised by the team with a $50 cap. No patch was executed here: the grades
come from the Colab test run (`results/2026-11-10-swe-static-checks`). Small summary
tables only; verdicts, issue text and patches stay on the machine that ran it.

## Question

Free checks (patch applies, files parse, existing tests pass) catch about two thirds
of wrong patches and certify nothing. Open models up to 32B and trained classifiers
cannot tell the rest apart (`results/2026-11-11-judge-pilot`,
`results/2026-11-13-trained-verifier`). Does a named, repeatable strong model do it,
what does a check cost, and does a chain with it save money once the judge is paid for?

## Setup

- Judge: `claude-sonnet-5-5`, effort `medium`, one request per case, the same
  instructions as every other judge (`PROMPT_VERSION` 2026-11-11), no fallback model.
  `scripts/judge_patches.py judge --provider anthropic`.
- Cases: every patch that passed the free checks on a validated task in the 277-task
  run, identical patches for one task merged: **1,110 cases on 256 tasks** (856
  django, 254 sympy). Label: whether the patch resolved the issue when the tests were
  run. 644 right, 466 wrong.
- Chains: the 274 valid tasks, eight models, frontier `claude-opus-4.1`, this run's
  grades, `tine cascade --checks`, 10 seeds, delta 0.05. An answer passes when it
  passed the free checks **and** the judge accepted it. **The judge's cost is added to
  the cost of every answer it read**, so savings are net of it.

## Result 1: the judge

`judge-sonnet55-1110-own-grades.csv`, `judges-pilot-150-own-grades.csv`.

| cases | wrong patches rejected | right patches rejected | accepted patches that are right | AUC |
| --- | --- | --- | --- | --- |
| 150 pilot cases (django) | 88.9% | 5.7% | 92.1% | 0.96 |
| all 1,110 | 79.4% | 9.0% | 85.9% | 0.92 |
| django, 856 | 81.5% | 8.9% | 89.0% | 0.93 |
| sympy, 254 | 74.5% | 9.7% | 73.9% | 0.90 |
| without a judge, all 1,110 | 0% | 0% | 58.0% | - |

- **A pinned strong model does verify patches**, far better than anything open that
  was tried (best open result: AUC 0.62 from a model, 0.76 from agreement).
- **The pilot was optimistic.** On all 1,110 cases it catches 79% of wrong patches,
  not 89%, and rejects 9% of right ones. The 150 pilot cases are django only and were
  the ones earlier choices were made on; quote the 1,110-case numbers.
- At probability 0.9 or higher it catches 89.9% and rejects 23.8% of right patches.
- 18 replies (1.6%) had no usable verdict and count as rejections.
- **Cost: $0.0080 per check** (1,015 tokens in, 595 out including the model's
  reasoning), 2.4% of the mean cost of a `claude-opus-4.1` answer on these tasks.
- Against the benchmark's own grades the same verdicts score 68.4% and 9.0%; those
  grades under-credit patches that need fuzzy application (see the grade dispute in
  the static-checks folder).

## Result 2: chains with the judge

Frontier accuracy on the test parts: 49-50%.

| split of the 274 tasks (choose / certify / test) | epsilon | certified: saved / accuracy / runs over / runs with nothing certified | uncertified: saved / accuracy / runs over |
| --- | --- | --- | --- |
| 40 / 30 / 30 (the default) | 0.02 | 0% / 50.3% / 0 / 10 | 33.8% / 56.4% / 1 of 10 |
| | 0.04 | 0% / 50.3% / 0 / 10 | 34.0% / 56.1% / 0 |
| | 0.05 | 3.1% / 50.4% / 0 / 9 | 34.0% / 56.1% / 0 |
| 30 / 50 / 20 | 0.02 | 0% / 49.2% / 0 / 10 | 34.8% / 55.1% / 0 |
| | 0.04 | **22.2% / 52.7% / 0 / 3** | 34.7% / 55.1% / 0 |
| | 0.05 | **30.4% / 54.8% / 0 / 1** | 34.7% / 55.1% / 0 |

`cascade-judge-verdict-split-*-summary.csv`; `cascade-judge-p90-*` is the same with the
stricter judge threshold (certifies 7.7% and 19.7% at epsilon 0.04 on the two splits).

- **With the judge, the chain is cheaper and more accurate than the frontier, and it
  stays within the limit.** The usual chain tries `deepseek-v3.1-terminus`, then
  `deepseek-v3-0324`, then `deepseek-r1-0528`, then `claude-opus-4.1`. On held-out
  tasks it costs about a third less, including every judge call, and solves 55-56% of
  tasks against the frontier's 49-50%: the cheap models solve tasks the frontier does
  not, and the judge lets those answers through. It loses under 1% of the tasks the
  frontier solves on average (worst run 2.6%).
- **It is the first chain on code that certifies anything: 22% at a 4% limit and 30%
  at 5%**, never over the limit. This needed a larger certify part.
- **Certification is limited by the number of tasks, not by the chain.** With the
  default split the certify part has about 80 tasks, where even zero failures gives a
  bound of 3.6%, and a single failure fails the 4% limit. Giving half the tasks to
  certification (137) is what lets it pass. **The second split was chosen after
  seeing the first result**; it is a statement about sample size, reported next to
  the default, not a tuned headline. Nothing certifies at 2% with either split: that
  needs about 150 certify tasks with zero failures, or more with any.
- On all 274 tasks at once (no split, so only descriptive): that chain saves 36.5%,
  solves 55.5% against 48.2%, and loses 2 of the 274 tasks the frontier solves.

## What this means

For code, Gacrux is: a cheap model writes the patch, the free checks run, a strong
model reads what passes, and the frontier writes the patch only when those fail. On
this data that is about a third cheaper and several points more accurate, and the
judge costs about 2% of what it guards. The remaining work is evidence, not design:
more graded tasks so that the certify part is large enough for a 2% limit.

## Limits

- **Memory.** These are public django and sympy issues; a strong model may have seen
  their fixes. The lower numbers on sympy and outside the pilot are consistent with
  some of that, or with those cases being harder. A customer's private code is the
  real test.
- 274 tasks from two repositories, single-shot patches from this benchmark. The
  frontier solves only half of them, so there is unusual room for cheap models to add
  accuracy; on tasks the frontier nearly always solves the accuracy gain would vanish
  and only the saving would remain.
- One model, one effort setting, one prompt. Cheaper and stronger models were not run.
- Prices are those of 2026-11-14. Recorded model costs are the benchmark's.
- Judge calls add latency (5.6 seconds each on average); not counted here.

## Next

1. More graded tasks: the 29 not yet run, then other SWE-bench repositories, to
   certify at 2%.
2. The same judge at lower effort, and one cheaper and one stronger model, on the
   pilot cases: find the cheapest judge that holds.
3. A judge stage type in `tine.policy`, so a frozen policy can carry this chain.

## Reproduce

```bash
python scripts/judge_patches.py prepare --bench-dir BENCH/extracted/bench-release/swe-bench/verified \
  --static-checks RUNS/static-checks-linux.csv --repo-tests RUNS/repo-tests.csv \
  --dataset DATA/swe-bench-verified.jsonl --cases RUNS/judge/cases.jsonl --key RUNS/judge/key.csv
# paid: needs authorisation and ANTHROPIC_API_KEY in the environment
python scripts/judge_patches.py judge --provider anthropic --model claude-sonnet-5-5 \
  --cases RUNS/judge/cases.jsonl --output RUNS/judge/verdicts.csv --max-cost-usd 12
python scripts/judge_patches.py score --key RUNS/judge/key.csv --verdicts RUNS/judge/verdicts.csv --grade run_resolved
```

The checks file for `tine cascade` marks an answer passed when its free check passed
and its judge verdict was `accept`, and the outcomes file adds each case's `cost_usd`
to the answers that case covers. That join was done in a notebook-style script and is
not yet a command.
