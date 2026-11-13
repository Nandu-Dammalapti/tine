# Judge check

Gacrux stage 4 is a small judge model on the customer's own hardware (AGENTS.md
section 3). The results so far say where it is needed: after a cheaper model has
answered. Free checks on a code change (the patch applies, files parse, existing
tests still pass) catch about two thirds of wrong patches on django; most of the rest
apply cleanly, break nothing and do not fix the issue
(`results/2026-11-10-swe-static-checks`). A judge reads the issue and the patch and
says whether the patch resolves the issue. This page says how a judge is measured.

`scripts/judge_patches.py` has three steps. Nothing it does executes a patch.

## 1. Prepare cases

```bash
python scripts/judge_patches.py prepare \
  --bench-dir BENCH/extracted/bench-release/swe-bench/verified \
  --static-checks RUNS/static-checks-linux.csv --repo-tests RUNS/repo-tests.csv \
  --dataset DATA/swe-bench-verified.jsonl \
  --cases RUNS/judge/cases.jsonl --key RUNS/judge/key.csv --sample 150 --seed 42
```

It keeps only answers that already passed the static checks and the existing tests,
on tasks whose reference patch validated: these are the answers a chain would
otherwise accept. Identical patches for one task become one case. Each case holds an
opaque id, the repository name, the issue text and the patch. The grades (the
benchmark's, and whether our own test run resolved the task) go to a separate key
file that no judge is given. `--models` limits the answers; `--sample` draws a seeded
random subset.

## 2. Get verdicts

Every judge sees the same instructions (`INSTRUCTIONS` in the script, versioned by
`PROMPT_VERSION`) and returns, per case, `accept` or `reject`, a probability that the
patch resolves the issue, and a short reason.

**A model behind an OpenAI-compatible endpoint** (the deployment case: an open model
on customer hardware behind vLLM, Ollama or LM Studio; also any provider API):

```bash
python scripts/judge_patches.py judge --cases RUNS/judge/cases.jsonl \
  --output RUNS/judge/verdicts-local.csv \
  --base-url http://127.0.0.1:11434/v1 --model MODEL_NAME
```

One request per case, temperature 0. It records prompt and completion tokens from
the endpoint's `usage`, and seconds per request, and resumes if rerun. A reply with
no usable verdict is recorded as `unparsed` and scored as a rejection, since a chain
would escalate. For an endpoint that needs a key, pass `--api-key-env NAME`; the key
is read from that environment variable and never written anywhere. Calling a paid
model needs the task's authorization (AGENTS.md).

**A Claude Code session** (used for the first pilot, as a ceiling): sub-agents were
each given ten cases from different tasks and the same instructions, told to read
nothing else, and their verdicts were written in the same CSV format with judge name
`claude-session`. Token counts for this judge are estimates (characters / 4): the
session does not report per-call usage, and reasoning tokens are not included. It is
not a pinned model with fixed settings, so treat it as an upper reference, not a
reproducible baseline.

## 3. Score

```bash
python scripts/judge_patches.py score --key RUNS/judge/key.csv \
  --verdicts RUNS/judge/verdicts-local.csv RUNS/judge/verdicts-session.csv \
  --output RUNS/judge/scores.csv
```

Per judge, at the judge's own verdict and at probability thresholds 0.5, 0.7, 0.9:

- `catch_rate`: share of wrong patches rejected.
- `false_reject_rate`: share of right patches rejected. These escalate needlessly.
- `accepted_share` and `accuracy_when_accepted`.
- mean prompt and completion tokens and seconds per case.

`--grade run_resolved` scores against our own test run instead of the benchmark's
grade; the two disagree on about 6% of patches (see the results folder).

## Reading the numbers

- The cases are only the patches the free checks let through. A judge's catch rate
  here is the catch rate on that remainder; combine it with the free checks' catch
  rate for the whole pipeline.
- The judge's tokens are routing cost (AGENTS.md section 2). A judge call on every
  cheap answer must cost less than the savings from the answers it lets through.
  Compare tokens per case with the tokens of the frontier call it replaces.
- 150 cases bound a rate to about +/- 8 points. Enough to see whether a judge works,
  not enough to certify a chain.
- The comparison that matters for deployment is a small open model against the
  session judge on the same cases: how much of the ceiling does the local model keep?

## Running a local judge on a Mac

`--style think` makes the judge reason through the change before its verdict (use
`--max-tokens 900`). It was added for small models, which accept nearly everything
when asked for a bare verdict; on the first one tried it only swapped that for
rejecting nearly everything.

With Ollama (or any runner exposing `/v1/chat/completions`): install it, start it, pull an
open coding model that fits in memory, and run step 2 against
`http://localhost:11434/v1`. Claude Code's sandbox cannot use the Mac's GPU or start
the runner, so start it yourself. From inside the sandbox the request has to go
through the session's proxy: run with `NO_PROXY= no_proxy=` and the host `localhost`.

## Running a larger open model on Colab

[`notebooks/tine_colab_judge.ipynb`](../notebooks/tine_colab_judge.ipynb) runs step 2 with
`qwen2.5-coder:32b` under Ollama on a Colab A100, in both styles, on the 150 pilot cases. Runner,
prompt and script are the same as the 14B run on the Mac, so the comparison isolates model size. It
sets a 16k context so no case is cut off, and saves the verdict files and a run record (model
digest, Ollama version, GPU, repository revision) to `MyDrive/tine/`. The cases file goes on Drive,
never in the repository. No patch is executed. Score afterwards with step 3.

## Training a verifier instead

`scripts/train_verifier.py` takes the same cases and key and trains a logistic regression to predict
the grade, scored by cross-validation grouped by task. `compare` runs several feature groups over
several splits and writes one table; `train` writes held-out predictions in the verdict format, so
step 3 scores a trained verifier next to any judge. `encode` (GPU) saves a frozen language model's
hidden states for each case as extra features;
[`notebooks/tine_colab_train_verifier.ipynb`](../notebooks/tine_colab_train_verifier.ipynb) runs it.
The `agreement` features compare a patch with other models' patches for the same task: using them at
run time means several cheap calls per request, which is routing cost. Pass the frontier model to
`--agreement-exclude`. Results: `results/2026-11-13-trained-verifier`.
