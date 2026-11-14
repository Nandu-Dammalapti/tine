# A pinned strong judge for code, and the first chains with it, 2026-11-14

**This folder's judge runs called paid models** through the Anthropic API:
`claude-sonnet-5-5` on 1,110 cases ($8.85), `claude-haiku-5-5` on 1,110 ($0.59), and
on the 150 pilot cases `claude-sonnet-5-5` at low effort ($1.13) and `claude-opus-5-5`
($2.12). **$12.69 in total** as reported by the API's usage figures, authorised by
the team with a $50 cap. No patch was executed here: the grades
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

## Result 3: cheaper and stronger judges

`judges-pilot-150-own-grades.csv`, `judge-haiku55-1110-own-grades.csv`,
`cascade-judge-variants.csv`.

On the 150 pilot cases (optimistic for every judge, see Result 1):

| judge | wrong rejected | right rejected | AUC | cost per check |
| --- | --- | --- | --- | --- |
| `claude-haiku-5-5` | 76.2% | 4.6% | 0.92 | $0.0005 |
| `claude-sonnet-5-5`, low effort | 87.3% | 8.0% | 0.96 | $0.0075 |
| `claude-sonnet-5-5`, medium effort | 88.9% | 5.7% | 0.96 | $0.0078 |
| `claude-opus-5-5` | 92.1% | 4.6% | 0.98 | $0.0141 |

- Lower effort on Sonnet saves almost nothing (4%) and is no better. Opus is the
  best judge and costs 1.8 times Sonnet. **Haiku costs a sixteenth of Sonnet** and
  gives up about 12 points of catch rate.

Haiku on all 1,110 cases: rejects 71.7% of wrong patches and 9.5% of right ones
(AUC 0.90), against Sonnet's 79.4% and 9.0%. 16 replies had no usable verdict.

Chains on the 274 tasks with each judge (default split unless stated; `haiku, then
sonnet` means Sonnet reads only what Haiku accepted and both must accept):

| judge in the chain | uncertified at 4%: saved / accuracy vs 50.3% / worst run | runs over 2% / 3% / 4% | certified at 4%, half the tasks certifying |
| --- | --- | --- | --- |
| `claude-haiku-5-5` | **39.3%** / 54.8% / 2.6% | 2 / 0 / 0 of 10 | 8.5% |
| `claude-sonnet-5-5` | 34.0% / 56.1% / 2.6% | 1 / 0 / 0 | 22.2% |
| haiku, then sonnet | 30.1% / 56.1% / 2.6% | 0 / 0 / 0 | 22.9% |
| `claude-sonnet-5-5`, probability >= 0.9 | 21.8% / 54.5% / 1.3% | 0 / 0 / 0 | 19.7% |

- **The cheapest judge gives the cheapest chain.** With Haiku the chain saves 39%
  against Sonnet's 34%, at about one point less accuracy, and never lost more than
  2.6% of the frontier's tasks in a held-out run. Its check costs 0.16% of a frontier
  answer.
- **At the 2%, 3% and 4% limits the picture is the same as before: the chains hold
  the limit on held-out tasks and cannot yet prove it.** No judge certifies anything
  at 2%. At 3% only the strictest setting does (Sonnet at probability >= 0.9: 11.5%
  with half the tasks certifying, in 4 of 10 runs). At 4% the best is 22-23%.
  A stricter or stronger judge lowers the failure rate, but with 80 to 137 certify
  tasks the bound cannot go below 2.2-3.6% even with zero failures.
- So the way to certified savings at 2-4% is more graded tasks, not a better judge:
  about 150 certify tasks with no failures for 2%, about 500 graded tasks in all.
  Running Opus on all cases (about $14) was not done for that reason.

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
- One prompt. The stronger judge was run on the pilot cases only.
- Prices are those of 2026-11-14. Recorded model costs are the benchmark's.
- Judge calls add latency (5.6 seconds each on average); not counted here.

## Next

1. More graded tasks: the 29 not yet run, then other SWE-bench repositories, to
   certify at 2%.
2. A judge stage type in `tine.policy`, so a frozen policy can carry this chain.

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

```bash
python scripts/judge_patches.py checks --key RUNS/judge/key.csv --verdicts RUNS/judge/verdicts.csv \
  --free-checks RUNS/repo-test-checks.csv --outcomes DATA/swe-valid-own-grades.jsonl \
  --checks-output RUNS/judged-checks.csv --outcomes-output RUNS/judged.jsonl
python -m tine cascade --input RUNS/judged.jsonl --frontier claude-opus-4.1 --checks RUNS/judged-checks.csv \
  --epsilon 0.02 0.03 0.04 --fractions 0.3 0.5 0.2 --output RUNS/judge-chain
```

`checks` marks an answer passed when its free check passed and every judge accepted
it, and adds each judge's recorded cost to the answers it read. The outcomes file
here is the 274 valid tasks with this run's grades in place of the benchmark's.
