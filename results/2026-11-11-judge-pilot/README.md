# Judge check pilot on django patches, 2026-11-11

Offline, on recorded SWE-bench answers. No patch was executed for this pilot. **The
judge was a Claude Code session, not a pinned model**: treat every number here as a
ceiling reference, not a reproducible baseline. Small summary tables only. How the
judge is measured: [docs/JUDGE.md](../../docs/JUDGE.md).

## Question

Free checks on a code change (it applies, it parses, existing tests still pass)
catch about two thirds of wrong patches on django. Most of the rest apply, break
nothing and do not fix the issue
([swe-static-checks](../2026-11-10-swe-static-checks/README.md)). Can a judge that
only reads the issue and the patch tell those apart? This is Gacrux stage 4, applied
after the answer.

## Setup

- Cases: answers from eight models on the 135 validated django tasks that passed the
  static checks **and** the existing tests, with identical patches merged: 536 cases.
  A seeded random sample of 150 was judged (`scripts/judge_patches.py prepare
  --sample 150 --seed 42`), from 87 tasks.
- The judge saw the repository name, the issue text and the patch. Not the grade, not
  which model wrote the patch, not the repository's code.
- Judge: sub-agents of a Claude Code session on its default model, each given ten
  cases from ten different tasks and told to read nothing else. Each reported reading
  only its instructions and its batch.
- Two sets of grades, because they disagree on 11 of the 150 cases:
  - the benchmark's grade (76 right, 74 wrong);
  - whether the patch resolved the task in our own Colab test run (87 right, 63 wrong).

## Results

`judge-scores-benchmark-grades.csv`, `judge-scores-own-test-grades.csv`. Ranges are
95% Clopper-Pearson intervals.

| graded against | wrong patches the judge rejects | right patches it rejects | accepted patches that are right |
| --- | --- | --- | --- |
| the benchmark | 82.4% (61 of 74; 72-90%) | 1.3% (1 of 76; 0-7%) | 85.2% (75 of 88) |
| our own test run | 95.2% (60 of 63; 87-99%) | 2.3% (2 of 87; 0-8%) | 96.6% (85 of 88) |

- **The judge separates the quiet failures.** Its probability ranks right above wrong
  with an AUC of 0.94 against the benchmark and 0.99 against our test run. The free
  checks could not tell these patches apart at all: all 150 had passed them.
- **It almost never rejects a right patch**, so it would cause few needless escalations.
- **The 11 disputed cases:** the benchmark graded them wrong; our test run resolved
  them. The judge, which saw neither grade, accepted 10 of the 11. Reading the
  patches afterwards settled it: the benchmark under-credits patches that only apply
  with `patch --fuzz`, several of which are identical to the reference fix (see
  [swe-static-checks](../2026-11-10-swe-static-checks/README.md)). **Use the "own test
  run" row.**
- **With the free checks, the pipeline reaches the catch rate chains need.** Static
  checks plus existing tests catch 65% of wrong patches (benchmark grades) and the
  judge 82% of the remainder: about 94% together. On our test run's grades: 69% and
  95%, about 98% together. Escalation chains needed about 95%
  ([coding-benchmark](../2026-11-09-coding-benchmark/README.md)). This is a product
  of two measured rates on django, not a certified chain.

## What the judge costs

`session-judge-usage.json`.

- The text a judge reads per case is small: about 710 tokens of instructions, issue
  and patch, and about 50 tokens of verdict (estimated as characters / 4). The model
  that writes the patch reads about 17,000 to 21,000 prompt tokens per SWE-bench task
  in this benchmark. A judge call is therefore about 4% of the tokens of the call it
  checks, before any reasoning tokens.
- What the session actually used was far more: 1,493,366 tokens for the 150 cases,
  9,956 per case. That is the sub-agents' own overhead (system prompt and tool
  definitions resent on each of about four steps), plus reasoning. It says what this
  way of running a pilot costs, not what a judge call costs in a deployment.
- Under AGENTS.md section 2 the judge's tokens are routing cost and come off the
  savings. A judge call on a frontier-priced model can cost more than the cheap
  model's whole answer, so the judge's price matters as much as its accuracy. That is
  the reason to test a small local judge.

## Limits

- **Contamination.** SWE-bench issues and their real fixes are public, and django is
  one of the most-read codebases there is. A frontier model may recognise the issue
  and know the accepted fix. On a customer's private code it would not. This is the
  main reason the result is a ceiling.
- 150 cases from one repository. The intervals above are wide.
- Ten cases per sub-agent, not one per call as `judge_patches.py judge` does. Cases
  in a batch came from different tasks, but a reviewer may still calibrate across them.
- The judge is not pinned: no fixed model version or settings, so it cannot be rerun
  exactly.
- The benchmark's grades are wrong on about 7% of these cases; both readings are shown
  and the own-test-run one is the reliable one.
- No chain was certified. 135 tasks are too few.

## Next

1. Run the same 150 cases through a small open model on local hardware
   (`judge_patches.py judge`, see docs/JUDGE.md) and compare with these rows. How much
   of the ceiling does a judge the customer can host keep?
2. Finish the Colab test run (303 tasks), judge every accepted patch, and certify
   chains with `tine cascade --checks`, netting the judge's tokens from the savings.
3. Test on tasks a model cannot have seen, to size the contamination effect.

## Reproduce

The cases and key (needs the Colab outputs `static-checks-linux.csv` and
`repo-tests.csv` from `MyDrive/tine/swe/`):

```bash
python scripts/judge_patches.py prepare --bench-dir BENCH/extracted/bench-release/swe-bench/verified \
  --static-checks RUNS/static-checks-linux.csv --repo-tests RUNS/repo-tests.csv \
  --dataset DATA/swe-bench-verified.jsonl --cases RUNS/judge/cases-pilot.jsonl --key RUNS/judge/key-pilot.csv \
  --sample 150 --seed 42 --models claude-opus-4.1 claude-sonnet-4 deepseek-r1-0528 deepseek-v3.1-terminus \
  deepseek-v3-0324 kimi-k2-0905 glm-4.6 qwen3-235b-a22b-2507
python scripts/judge_patches.py score --key RUNS/judge/key-pilot.csv --verdicts RUNS/judge/verdicts-session.csv
```

The session judge's verdicts cannot be regenerated exactly. A pinned judge can:
`judge_patches.py judge --base-url ... --model ...`.
