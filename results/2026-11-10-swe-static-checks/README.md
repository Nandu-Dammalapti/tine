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

## Linux rerun of the static checks

Added later on 2026-11-10 (`static-check-rates-swe-linux.csv`), from the Colab
notebook, with GNU patch as in the harness. GNU patch applies about 400 more patches
than Apple's, and more of those leave a file that no longer parses.

| | macOS (Apple patch) | Linux (GNU patch) |
| --- | --- | --- |
| answers that pass, of 7,000 | 50.5% | 54.7% |
| catch rate, all models | 63.5% | 58.2% |
| catch rate, cheaper models in the table above | 53-76% | 47-66% |
| false rejects | 3.0% | 2.6% |
| accuracy among answers that pass | 44.4% | 41.1% |

Use the Linux numbers. The false rejects that remain are answers the benchmark
graded right whose patch does not apply here with either tool.

## Existing tests: partial run on django

`repo-test-rates-django-partial.csv`, `repo-test-summary-django-partial.json`.
`scripts/swe_repo_tests.py` on a 2-core Colab CPU runtime (see
[docs/SWE_CHECKS.md](../../docs/SWE_CHECKS.md)). **This executed model-written code,
on Colab only.** The run was stopped by hand after 137 of the planned 303 tasks, all
django; sympy has not run. Eight models: claude-opus-4.1, claude-sonnet-4,
deepseek-r1-0528, deepseek-v3.1-terminus, deepseek-v3-0324, kimi-k2-0905, glm-4.6,
qwen3-235b-a22b-2507.

Is the Docker-free environment trustworthy?

- 135 of 137 tasks resolved with their reference patch. The other two need Python
  3.5, whose dependencies no longer build.
- Over 683 model patches run, this run's "resolved" agrees with the benchmark's
  grade 93.9% of the time. The 42 disagreements are almost all one way: 41 patches
  resolve here that the benchmark graded wrong, 1 the reverse. They cluster on a few
  tasks (six models on `django__django-12419`, four on `django__django-11119`). The numbers
  below use the benchmark's grades.
- **Checked by reading the patches (2026-11-11): the benchmark is the one that is
  wrong on these.** 15 of the 41 make exactly the same edit as the task's reference
  patch (same added and removed lines), so they are certainly right; for example on
  `django__django-12419` six models change `SECURE_REFERRER_POLICY = None` to
  `'same-origin'`, as the reference does, and the benchmark credits two of them. 38
  of the 41 are patches that `git apply` rejects and GNU `patch --fuzz` applies
  (wrong line counts in the hunk header). Among such patches this run resolves 66
  and the benchmark credits 28. The benchmark's own run evidently did not apply
  these patches. Its SWE-bench grades therefore undercount models that write sloppy
  diffs, by about 6% of the patches run here. Prefer this run's grades where they exist.

What the checks catch, over 1,065 answers on the 135 valid tasks (311 right, 754 wrong):

| check | wrong answers caught | right answers rejected | accepted answers that are right |
| --- | --- | --- | --- |
| static only | 49.3% | - | - |
| static + existing tests still pass | 65.4% | 1.3% | 54.0% |

Per model the combined catch rate is 54-75%.

Of the 375 wrong patches that applied and ran:

- 114 (30%) broke a test that passed before. This is what a customer's existing
  test suite would catch.
- 220 (59%) broke nothing and failed the tests written for the issue. Only a test
  aimed at the change catches these.
- 41 (11%) passed every test here and are the disputed grades above. If this run is
  right about them, the combined catch rate is 69% and 61% of accepted answers are right.

**Reading:** on django, a repository's existing tests add 16 points to the static
checks, and the two together catch about two thirds of wrong patches. The 95% that
escalation chains need (see the results index) is not reachable with checks that only
ask "did anything break". Most wrong patches are wrong quietly: they apply, break
nothing, and do not fix the issue.

This sample is too small to certify a chain (135 tasks), so no savings figure is given.

## Existing tests: larger partial run, django and sympy (2026-11-12)

`repo-test-rates-partial-222.csv`, `repo-test-summary-partial-222.json`. The same Colab run,
resumed: 222 of the planned 303 tasks, now including sympy. 219 are valid (160 django, 59 sympy);
two django tasks need Python 3.5 and on one the reference patch did not resolve here. Same eight
models, 1,734 answers. **Model-written code was executed on Colab only.**

| grades used | check | wrong answers caught | right answers rejected | accepted answers that are right |
| --- | --- | --- | --- | --- |
| benchmark's | static only | 49.6% | - | - |
| benchmark's | static + existing tests | 64.3% | 1.1% | 49.8% |
| this run's | static only | 52.2% | - | - |
| this run's | static + existing tests | 67.8% | 0% (by construction) | 56.7% |

- The django-only numbers above hold on the larger sample: the two free checks catch about two
  thirds of wrong patches (per model 55-71% on the benchmark's grades, 56-74% on this run's).
- sympy is like django for catching (63.0% against 64.8%, benchmark's grades) but its answers are
  worse to begin with: 37.6% of accepted sympy answers are right against 53.7% for django.
- Agreement with the benchmark's grades is 94.1% over 1,080 patches run. All but one of the 64
  disagreements are patches that resolve here and the benchmark graded wrong, the pattern
  explained in the section above.
- Of the 627 wrong patches (benchmark's grades) that ran: 171 (27%) broke an existing test, 393
  (63%) broke nothing and failed the issue's own tests, 63 (10%) are the disputed grades.

Still to run: 81 tasks. 219 tasks is still too few to certify a chain, so no savings figure.

## What this means

Static checks are worth having: they are free and remove about half the answers.
They are not enough, and on the django sample neither are existing tests. What is
missing is a check aimed at the change itself.

## Limits

- Apple `patch`, not GNU patch (see Setup).
- Single-shot patch generation. Applies-or-not failures are common here and would be
  rare with an agent harness, so these catch rates do not transfer to agent traffic.
- 500 tasks from 12 Python repositories.
- The benchmark's grades come from its own run of the SWE-bench harness and were not
  re-derived.

## Next

1. Finish the test run (the rest of django, then sympy) and run `tine cascade
   --checks` on the result. It resumes from `MyDrive/tine/swe/repo-tests.csv`.
2. Measure a check aimed at the change: tests written for the issue by a model, or
   a stronger model judging the patch. The hidden issue tests catch 220 of the 261
   wrong patches that existing tests miss, which is the ceiling such a check aims at.

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
