# Static patch checks on SWE-bench, 2026-11-10

Offline results on already-graded LLMRouterBench outcomes. No model was called and
**no model-written code was executed**: patches were applied to scratch copies of
the files they name and the patched Python files were parsed. Small summary tables
only.

## Question

Escalation chains need a check on each cheap answer. For a code change, the cheapest
possible checks need only the repository: does the patch apply, and do the files it
touches still parse? How many wrong SWE-bench answers do these catch, and is that
enough to certify savings?

This is the first of two steps. The second, running each repository's existing tests
on the patches that pass these checks, executes model-written code and needs an
isolated Linux machine. It has not been run.

## Setup

- Data: the SWE-bench Verified results in LLMRouterBench (same archive as the other
  folders): 500 tasks, 14 models (`openrouter` excluded), 7,000 recorded answers.
  Each record carries the model's patch and the benchmark's grade.
- Task metadata (repository, base commit): `princeton-nlp/SWE-bench_Verified` on
  Hugging Face. Repositories: bare clones of the 12 projects.
- `scripts/swe_patch_checks.py`. An answer passes if it contains a patch, the patch
  names files inside the repository, it applies at the task's base commit, and every
  Python file it touches still parses (unless that file did not parse before).
  "Applies" follows the SWE-bench harness: `git apply`, then `patch --batch --fuzz=5`.
- **Run on macOS with Apple's `patch` 2.0.** The harness uses GNU patch, which accepts
  some hunks Apple's rejects. This shows up as false rejects: 3.0% of answers the
  benchmark graded right fail here because their patch does not apply with these
  tools. Rerun on Linux before quoting these rates as final.

## Result 1: what the checks catch

`static-check-rates-swe.csv`. Catch rate = share of wrong answers that fail the
check. False reject = share of right answers that fail it.

| model | accuracy | passes the check | catch rate | false reject | accuracy among answers that pass |
| --- | --- | --- | --- | --- | --- |
| all 14 models | 23.1% | 50.5% | 63.5% | 3.0% | 44.4% |
| claude-opus-4.1 | 41.6% | 65.4% | 58.6% | 1.0% | 63.0% |
| claude-sonnet-4 | 34.6% | 61.4% | 58.4% | 1.2% | 55.7% |
| gemini-2.5-pro | 34.6% | 56.2% | 66.4% | 1.2% | 60.9% |
| deepseek-r1-0528 | 28.6% | 59.0% | 56.3% | 2.8% | 47.1% |
| deepseek-v3.1-terminus | 25.4% | 59.4% | 53.1% | 3.9% | 41.1% |
| kimi-k2-0905 | 23.4% | 51.4% | 62.9% | 1.7% | 44.7% |
| glm-4.6 | 21.8% | 47.0% | 67.5% | 0.9% | 46.0% |
| qwen3-235b-a22b-thinking-2507 | 21.4% | 39.6% | 76.1% | 2.8% | 52.5% |
| gpt-5 | 15.8% | 24.0% | 87.9% | 12.7% | 57.5% |

- **Half of all recorded answers fail before any test is run**: 42.0% have a patch
  that does not apply, 6.6% contain no patch or name no file, 1.0% break Python syntax.
- These checks catch 53-76% of wrong answers from the cheaper models, at no cost.
- An answer that passes is still wrong more often than not for the cheap models
  (41-53% right). Applying and parsing say nothing about whether the issue is fixed.
- **gpt-5's low SWE-bench score here is mostly a formatting failure.** 76% of its
  answers fail these checks (21.8% no usable patch, 54.0% a patch that does not
  apply). Among its answers that pass, 57.5% are right, close to the best models.
  This benchmark asks for a patch in one shot with no tools; an agent harness that
  edits files directly would not fail this way. The earlier finding that "gpt-5 is
  weak on SWE-bench in this benchmark" should be read with this in mind.

## Result 2: chains with these checks

SWE-bench only: 500 tasks, 14 models. Frontier `claude-opus-4.1`, the most accurate
(41.6%) and most expensive ($0.37 per task, 80 times deepseek-v3.1-terminus). 10
seeds, delta 0.05, measured checks (`tine cascade --checks`).

| check | epsilon 0.02 | epsilon 0.04 | epsilon 0.05 |
| --- | --- | --- | --- |
| static checks (`cascade-swe-static-summary.csv`) | 0% | 0% | 0% |
| static checks + response check (`cascade-swe-static-response-summary.csv`) | 0.0% | 1.5% | 3.0% (1 of 10 runs over) |
| response check only (`cascade-swe-response-only-summary.csv`) | 0% | 0.7% (1 of 10 over) | 0.6% |

Certified savings are near zero. Two reasons:

- The check is too weak alone: about half of the cheap answers that pass are wrong.
- The pool is small. With 500 tasks the certify part has 150-200, so a policy must
  show zero degradations to pass at epsilon 0.02.

Without certification the picture is different and shows what is at stake: a chain
picked on 75 tasks with the static checks alone (`uncertified` in
`cascade-swe-static-response-summary.csv`) saves 34.7% at epsilon 0.04 and 43.3% at
0.05 on held-out tasks, but goes over the limit in 7 of 10 runs and loses about 3
points of accuracy. The price gap on this pool is large enough that a stronger
check would pay off.

## What this means

Static checks are worth having: they are free and remove half the answers. They are
not enough. The next measurement is the one that decides the business case: how many
of the remaining wrong patches a repository's existing tests catch.

## Limits

- Apple `patch`, not GNU patch (see Setup).
- Single-shot patch generation. Applies-or-not failures are common here and would be
  rare with an agent harness, so these catch rates do not transfer to agent traffic.
- 500 tasks from 12 Python repositories.
- The benchmark's grades come from its own run of the SWE-bench harness and were not
  re-derived.

## Next

1. Rerun `scripts/swe_patch_checks.py` on Linux (GNU patch).
2. Run each repository's existing tests on the 3,534 answers that pass, on an
   isolated Linux machine, and feed the result to `tine cascade --checks`.

## Reproduce

```bash
# metadata: instance_id, repo, base_commit for every task, one JSON object per line,
# from princeton-nlp/SWE-bench_Verified (data/test-00000-of-00001.parquet)
for r in django/django sympy/sympy sphinx-doc/sphinx matplotlib/matplotlib scikit-learn/scikit-learn \
         astropy/astropy pydata/xarray pytest-dev/pytest pylint-dev/pylint psf/requests mwaskom/seaborn pallets/flask; do
  git clone --bare https://github.com/$r.git REPOS/$(echo $r | sed 's#/#__#').git
done
python scripts/swe_patch_checks.py --bench-dir BENCH/extracted/bench-release/swe-bench/verified \
  --metadata META.jsonl --repos-dir REPOS --output RUNS/swe-patch-checks.csv --workers 8
python -m tine import-llmrouterbench --input BENCH/extracted/bench-release/swe-bench --output DATA/outcomes-swe-bench.jsonl --dataset swe-bench --run-selection latest
python -m tine combine-outcomes --input DATA/outcomes-swe-bench.jsonl --output DATA/swe.jsonl --task-type coding --exclude-model openrouter
python -m tine cascade --input DATA/swe.jsonl --frontier claude-opus-4.1 --checks RUNS/swe-patch-checks.csv \
  --epsilon 0.02 0.04 0.05 --output RUNS/swe-static
python -m tine cascade --input DATA/swe.jsonl --frontier claude-opus-4.1 --checks RUNS/swe-patch-checks.csv \
  --response-check --epsilon 0.02 0.04 0.05 --output RUNS/swe-static-response
python -m tine cascade --input DATA/swe.jsonl --frontier claude-opus-4.1 --response-check --catch 0 \
  --epsilon 0.02 0.04 0.05 --output RUNS/swe-response-only
```
