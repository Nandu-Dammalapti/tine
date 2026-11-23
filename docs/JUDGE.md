# Judge check

Gacrux stage 4 is a small judge model on the customer's own hardware (AGENTS.md
section 3). The results so far say where it is needed: after a cheaper model has
answered. Free checks on a code change (the patch applies, files parse, existing
tests still pass) catch about two thirds of wrong patches on django; most of the rest
apply cleanly, break nothing and do not fix the issue
(`results/2026-11-10-swe-static-checks`). A judge reads the issue and the patch and
says whether the patch resolves the issue. This page says how a judge is measured.

`scripts/judge_patches.py` has four steps: prepare, judge, score, and checks (the last
is under "From verdicts to a chain"). Nothing it does executes a patch.

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

Every judge sees the same instructions (`INSTRUCTIONS` in `tine/judge.py`, versioned by
`PROMPT_VERSION`) and returns, per case, `accept` or `reject`, a probability that the
patch resolves the issue, and a short reason.

The shared reply parser reads whole outer JSON objects. A nested example or an
object inside an array is not a verdict; braces and escaped quotes inside the
reason are ordinary text. Duplicate JSON fields make the reply unusable, rather
than letting the last field overwrite a rejection or a lower probability. Multiple
outer verdict objects must agree on accept/reject; the last usable one is returned.
Boolean and out-of-range probabilities are unusable. Runtime and Claude patch checks
reject unusable replies while retaining the call's measured cost.

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
would escalate. The OpenAI-compatible endpoint must report `finish_reason: "stop"`
and no tool call or refusal; otherwise even parseable approval text is `unparsed`,
with reported token usage retained. For an endpoint that needs a key, pass `--api-key-env NAME`; the key
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

## A Claude model as the judge, with a spending cap

`judge --provider anthropic` sends each case to a Claude model through the Anthropic SDK with the same
instructions as every other judge:

```bash
python scripts/judge_patches.py judge --provider anthropic --model claude-haiku-5-5 \
  --cases RUNS/judge/cases.jsonl --output RUNS/judge/verdicts-haiku.csv --max-cost-usd 1
```

- **This spends money.** It needs the task's authorization (AGENTS.md) and refuses to start without
  `--max-cost-usd`. Spending is counted from each reply's reported usage, at the prices in
  `ANTHROPIC_PRICES` (read from the provider's price page on 2026-11-14; check them before a run). The
  run stops before any request whose worst case would pass the cap, and the cap covers everything
  already recorded in the output file, so a resumed run cannot overspend.
- The key is read from the `ANTHROPIC_API_KEY` environment variable by the SDK. It is never an
  argument, never printed and never written to a file by this script.
- `--effort` (default `medium`) sets how much the model reasons before it answers. Reasoning tokens
  are billed as output and are included in `completion_tokens` and `cost_usd`.
- Each verdict row records `cost_usd`; `score` reports `mean_cost_usd` per case. That is the routing
  cost to net from savings.
- A refusal or a reply with no usable verdict is recorded as `unparsed` and scored as a rejection. No
  fallback model is configured: the point is to measure one named model.
- So is a reply that did not finish (cut off at `--max-tokens`, paused, or calling a tool), even when
  it holds an approval; the call is still charged. The proxy's judge step has the same rule. In the
  six recorded verdict tables the longest reply was 3,896 tokens against the 8,000 limit, so no
  recorded verdict was cut off at that limit.

## From verdicts to a chain

`checks` turns verdict files into inputs for `tine cascade`: a checks file in which an answer passes
when it passed the free checks and every judge accepted it, and an outcomes file with each judge's
recorded cost added to the answers it read, so savings are net of the judge. Several `--verdicts`
files are judges in order, each reading only what the earlier ones accepted. Results:
`results/2026-11-14-api-judge`.

## Which judge to use (2026-11-16)

Measured in `results/2026-11-14-api-judge` and `results/2026-11-16-judge-probes`:

- The judges partly remember these public SWE-bench fixes. On the tasks the judge
  model cannot recall from the issue alone, which stand in for a customer's own code,
  Haiku and Sonnet let enough wrong patches through that the code chain loses about
  2.3% of the tasks the frontier solves.
- `claude-opus-5-5` on the same tasks rejects 89% of wrong patches and 9% of right
  ones, and the chain's loss falls to 0.46%.
- Haiku as a screen in front of Opus keeps that loss and cuts the judge spend by 38%:
  Opus reads only what Haiku accepts.
- Simple injected approvals in a patch ("verified", "reviewer: accept") did not move
  any judge, and neither did removing the project's name. Adding the surrounding code
  did not help.

For code, the check to try first is therefore the customer's tests, then Haiku, then
Opus, all of which must pass (`docs/PROXY.md`, Named checks). **It is not certified.**
Read it as a candidate to measure on your own tasks, not a measured design:

- No chain with this check has been certified. Its 0.46% loss and 23.7% saving are
  uncertified figures on 150 tasks.
- The order (Haiku screening, then Opus) was chosen after seeing the single-judge
  results on the same tasks.
- Opus's own memory of these public fixes was not measured. The recall probe used
  Sonnet. The "unrecalled" tasks are the ones Sonnet could not write, and Opus may
  remember some of them.
- The judge models are newer than `claude-opus-4.1`, the frontier they were measured
  against.

The example:

```json
{"tests+judge:haiku-opus": [
  {"type": "webhook", "url": "https://ci.internal.example/tine-check", "bearer_env": "CI_TOKEN"},
  {"type": "judge", "model": "claude-haiku-5-5", "effort": "medium"},
  {"type": "judge", "model": "claude-opus-5-5", "effort": "medium"}
]}
```

A policy certified with this check must name it, and its chain must have been
certified with verdicts from the same two models. Confirm it on fresh tasks before
quoting it. With a Haiku or Sonnet judge alone, plan for a 4-5% quality limit on code.

What the webhook step must do. It runs the customer's own tests on the patch, beside
their repository, and reports whether they pass. It must not be a grader that knows
the right answer, such as the benchmark's hidden tests. The stand-in webhook in the
simulated pilots agreed with the grade on 819 of 819 answers, so its chain lost
nothing; a webhook that reports only regressions let through 19 wrong patches out of
99 accepted (1.0 to 1.5% of tasks lost). The judge is there for what the tests do not
check.

Request shape. The judge reads at most 12,000 characters of the request's user text
and 20,000 of the answer. Agents should put repository context in `system`, which the
judge does not read, and keep the user turn to the issue or task. That matches what
was measured: the judges read the issue and the patch. When the user text is longer,
the proxy skips a stage whose check has a judge step before calling its cheap models,
and records the reason `request_too_long` in shadow mode (`docs/PROXY.md`).

`scripts/judge_probes.py` builds the probe cases (anonymized, injected, with code
context) and compares any two verdict files case by case. `scripts/judge_recall_probe.py`
measures which tasks the judge model can write the fix for from the issue alone.
`--style hardened` adds one sentence telling the judge that the issue and patch are
untrusted; it made no measurable difference and is not the default.
