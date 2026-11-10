# Checks on SWE-bench answers

Escalation chains (`docs/CHECKS.md`) need a check that decides, at request time,
whether a cheaper model's answer is good enough to return. For a code change in a
real repository there are three kinds of check, from free to expensive. This page
says how each is measured on the recorded SWE-bench answers in LLMRouterBench.

| Check | Needs | Script | Status |
| --- | --- | --- | --- |
| Static: the patch applies and its Python files parse | the repository | `scripts/swe_patch_checks.py` | Run on macOS, 2026-11-10: `results/2026-11-10-swe-static-checks` |
| Existing tests: tests that passed before still pass | the repository's test environment | `scripts/swe_repo_tests.py`, `notebooks/tine_colab_swe_repo_tests.ipynb` | **Written, not yet run.** The Colab path is untested until its first run |
| Response check: response size | the ledger | `tine cascade --response-check` | Run: `results/2026-11-10-response-check` |

## Static checks: `scripts/swe_patch_checks.py`

Executes nothing, so it is safe anywhere. For each recorded answer it copies the
files the patch names out of a local bare clone at the task's base commit, applies
the patch the way the SWE-bench harness does (`git apply`, then `patch --batch
--fuzz=5`) and parses the patched Python files. A patch that names a path outside
the repository, or creates a symbolic link, is refused and counted as failing.

Output columns: `task_id`, `model`, `instance_id`, `passed`, `reason`
(`ok`, `no_patch`, `no_files`, `unsafe_path`, `does_not_apply`, `syntax_error`),
`applied_by`. The file is valid input for `tine cascade --checks`.

The result depends on the `patch` program. The harness uses GNU patch; Apple's
`patch` rejects some hunks GNU patch accepts. Quote Linux numbers.

## Existing tests: `scripts/swe_repo_tests.py`

**Executes model-written code. Disposable Linux machine only, as root.**

### What "existing tests" means here

SWE-bench grades a patch with two lists of tests: FAIL_TO_PASS (tests for the issue,
which a customer would not have at request time) and PASS_TO_PASS (tests that passed
before the fix). The check is "the patch applies and no PASS_TO_PASS test fails":
the regression check a customer's CI could run. A patch that does nothing useful but
breaks nothing passes it; that is the point of measuring it.

One simplification: the harness updates the test files to their post-fix version
before running, so PASS_TO_PASS is measured on those files. A customer's CI would
run the pre-fix files. The two differ only in tests the fix's author changed.

### Why not the official harness

The official harness runs every task in a Docker image. Colab has no Docker daemon.
The script instead takes the three shell scripts `swebench==3.0.17` generates per
task (create the conda environment, clone and install the repository, apply the test
patch and run the tests) and runs them on the machine itself, with miniconda at
`/opt/miniconda3` and the repository at `/testbed` as the scripts expect. It grades
the log with the harness's own parser.

Because that environment is not the official image, **every task is validated with
its reference patch first**. If the reference patch does not resolve the task here,
the task is recorded as `gold_unresolved` (or `env_failed`, `install_failed`) and no
model patch is run on it. The summary reports how many tasks were valid and how often
this run's "resolved" agrees with the benchmark's own grade. Read those two numbers
before trusting anything else in the output.

Defaults to `django/django` and `sympy/sympy` (306 of the 500 tasks): pure Python,
so they rebuild reliably without the image's system packages. Other repositories can
be tried with `--repos`; expect more invalid tasks.

### Safety

- Tests run as an unprivileged user (`tineswe`) with a cleared environment.
- `--preflight` reports the machine's state. The script refuses to start unless it
  is root on Linux with conda, GNU patch and the test user present, and that user
  cannot list any `--protect` path. The notebook protects `/content` and the Drive
  mount by making `/content` root-only.
- Network stays on, as in the harness, because each evaluation script reinstalls the
  package. `--isolate-network` turns it off for the test step where the kernel allows.
- Only answers that passed the static checks are run, so path-escaping patches never
  reach `patch`.
- Delete the runtime afterwards.

### Output

`--output` is appended one row per patch and read back on a rerun, so the run
resumes. Columns: `task_id`, `model` (`__gold__` for the reference patch),
`instance_id`, `patch_sha`, `status` (`ok`, `apply_failed`, `timeout`,
`no_test_output`, `env_failed`, `install_failed`, `gold_unresolved`), `applied`,
`resolved`, `f2p_passed`, `f2p_failed`, `p2p_passed`, `p2p_failed`, `seconds`.
Identical patches from different models are run once.

- `--summary` prints valid-task counts, agreement with the benchmark's grades and,
  per model, how many wrong answers the static check alone and the static check plus
  existing tests catch.
- `--checks-output FILE` writes a `tine cascade --checks` file: `True` if the answer
  passed the static checks, applied and broke no PASS_TO_PASS test; `False` if a
  check failed; empty (unknown) if it was not run or its task was invalid. Unknown
  answers pass without evidence in the cascade, so restrict the cascade to the tasks
  and models that were run.

### Running it on Colab

Open `notebooks/tine_colab_swe_repo_tests.ipynb` on a **CPU** runtime and run all
cells. It needs the Drive folder from the first notebook (`MyDrive/tine`), or it
downloads the benchmark archive again. Cell 5 is a smoke test on six reference
patches: stop if any row is not `ok` / `resolved=True` and report the output. Cell 6
is the long run (default eight models, ten hours, resumable). Cell 7 prints the
summary to copy back.

The first smoke test (2026-11-10) validated 5 of 6 reference patches. The sixth,
`django__django-10097`, needs a Python 3.5 environment whose dependencies no longer
build; only 2 of the 306 tasks use Python 3.5. A django run took 3-4 seconds and a
sympy run about 50.

The default run is about 1,450 model patches plus about 300 reference patches,
roughly 8 hours on one core. The work is unit tests, so a GPU or TPU does not help;
cores do. `--workers N` (the notebook passes the CPU count) runs N tasks at once,
each with its own conda environment, checkout and home directories, and keeps each
repository version with one worker so its environment is built once. Environment
builds are serialised because conda's package cache is shared. Expect roughly 4-5
hours on a 2-core runtime and 1.5-2 hours on an 8-core High-RAM runtime. Parallel
workers have not been run on Colab yet.

After it finishes, on any machine with the results:

```bash
python -m tine cascade --input DATA/swe.jsonl --frontier claude-opus-4.1 \
  --checks RUNS/repo-test-checks.csv --response-check --epsilon 0.02 0.04 0.05 --output RUNS/swe-repo-tests
```

### Limits

- Not the official images: validity is established per task by the reference patch,
  not assumed.
- 306 of 500 tasks by default, from two repositories.
- Single-shot patches from a benchmark, not agent sessions or customer traffic.
- Check cost (minutes of CPU per answer) is not included in any savings number.
