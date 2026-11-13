# A trained verifier for patches, 2026-11-13

Offline results on recorded SWE-bench Verified answers. No model was called and no
patch was executed in this folder's runs: the grades come from the earlier Colab test
run (`results/2026-11-10-swe-static-checks`). Small summary tables only.

**Status: first half.** The features that need no GPU are measured below. The
hidden-state features need `notebooks/tine_colab_train_verifier.ipynb`, which has not
been run yet.

## Question

A chain needs a check on a cheap model's patch after the free checks (patch applies,
files parse, existing tests pass). A strong model reading the patch separates right
from wrong almost perfectly; open models of 14B and 32B do not
(`results/2026-11-11-judge-pilot`). Can a small classifier trained on graded patches
do it, at no model cost per request?

## Setup

- Cases: every patch that passed the free checks on a validated task in the 222-task
  test run, identical patches for one task merged: **861 cases on 204 tasks** (645
  django, 216 sympy). 473 resolved their issue when the tests were run, 388 did not.
  That grade (`run_resolved`) is the label.
- `scripts/train_verifier.py`. Logistic regression on a feature group, scored by
  5-fold cross-validation grouped by task, so every prediction is for a task none of
  whose patches were in training. Each fold picks its regularisation strength (0.01,
  0.1, 1) and its accept threshold on its own training cases; the threshold aims to
  reject at most 10% of right patches. Mean over 5 seeds of the split.
- Feature groups:
  - `structure`: ten sizes and counts from the patch and issue.
  - `text`: TF-IDF of the issue and of the lines the patch adds and removes.
  - `agreement`: how similar the patch's changed lines are to the other accepted
    patches for the same task from other models, and how many there are. Patches
    written only by the frontier (`claude-opus-4.1`) are not compared against, since a
    chain does not have the frontier's answer when it checks a cheaper one.
- Rows `cheap` are the 747 cases written by at least one model other than the
  frontier, which is what a chain would check. Rows `subset` are the 150 pilot cases,
  for comparison with the judges measured before.

## Result

`cpu-features-5-seeds.csv`. AUC is how well the score ranks right patches above wrong
ones: 0.5 is chance, 1.0 is perfect.

| verifier | AUC, 747 cheap-model cases | AUC, 150 pilot cases | wrong patches rejected | right patches rejected | accepted patches that are right |
| --- | --- | --- | --- | --- | --- |
| no check | - | - | 0% | 0% | 53.4% |
| trained on `structure` | 0.63 | 0.70 | 22.2% | 9.4% | 57.2% |
| trained on `text` | 0.58 | 0.54 | 19.3% | 10.5% | 55.9% |
| trained on `structure` + `text` | 0.62 | 0.66 | 22.8% | 8.8% | 57.5% |
| trained on `agreement` | **0.76** | **0.79** | 41.8% | 10.3% | 63.9% |
| trained on all three | 0.76 | 0.80 | 37.8% | 10.3% | 62.3% |
| qwen2.5-coder:32b, reason first (pilot only) | - | 0.62 | 68.3% | 49.4% | 68.8% |
| Claude Code session (pilot only) | - | 0.99 | 95.2% | 2.3% | 96.6% |

Rates in the last three columns are on the cheap-model cases for the trained rows and
on the pilot cases for the two judges.

- **Reading the patch's surface does little.** Sizes, counts and words give an AUC
  of about 0.6. A trained classifier on these is no better than the 32B model.
- **Agreement between cheap models is the best signal found that needs no strong
  model**: AUC 0.76-0.79, above the 32B model's 0.62. A patch that resembles what
  other cheap models wrote for the same issue is more often right. Exact matches are
  rare (43 of 861 cases) but 91% of them are right.
- **It is still far from the ceiling.** Rejecting 10% of right patches, it catches
  42% of the wrong ones and lifts the share of accepted patches that are right from
  53% to 64%. The session judge reaches 97%.
- Agreement is not free. It needs several cheap models' answers per request, and
  their cost is routing cost. Here those answers already exist in the benchmark.

## Limits

- 204 tasks from two repositories. The 150 pilot cases were also used when choosing
  what to try next, so the `cheap` column is the cleaner one.
- The agreement features use every other model with an accepted patch (up to six
  cheap models and `claude-sonnet-4`). With two or three models the signal will be
  weaker; not measured.
- The threshold aims at 10% of right patches rejected and lands at 9-11% on held-out
  tasks. Nothing here is certified.
- Single-shot patches from this benchmark, as in the other SWE-bench folders.

## Next

1. Run `notebooks/tine_colab_train_verifier.ipynb`: the same training on hidden
   states of a frozen 7B model reading the issue and patch, alone and with agreement.
2. If a combination reaches a useful operating point, put it into a chain with
   `tine cascade --checks` and net the cost of the extra cheap calls.
3. Otherwise the remaining route for code is a pinned strong judge through an API.

## Reproduce

```bash
python scripts/judge_patches.py prepare --bench-dir BENCH/extracted/bench-release/swe-bench/verified \
  --static-checks RUNS/static-checks-linux.csv --repo-tests RUNS/repo-tests.csv \
  --dataset DATA/swe-bench-verified.jsonl --cases RUNS/judge/cases-222.jsonl --key RUNS/judge/key-222.csv
python scripts/train_verifier.py compare --cases RUNS/judge/cases-222.jsonl --key RUNS/judge/key-222.csv \
  --subset-key RUNS/judge/key-pilot.csv --agreement-exclude claude-opus-4.1 --c 0.01 0.1 1 \
  --feature-sets structure text structure+text agreement structure+text+agreement \
  --output RUNS/judge/cpu-features-5-seeds.csv
```
