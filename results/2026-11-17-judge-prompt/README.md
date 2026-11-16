# A stricter judge prompt, 2026-11-17

**Paid run.** `claude-haiku-5-5` on 1,110 cases ($0.62) and `claude-sonnet-5-5` on
400 ($3.40), **$4.02** as reported by the API's usage figures, inside the team's $50
authorisation (total spent on judges so far: $47.73). No patch was executed.

## Question

Reading 60 of the judge's errors (`results/2026-11-16-judge-designs`) suggested that
the prompt, not the design, was the weak point. Most wrong patches the judge accepted
were partial fixes or behaved differently from what the issue asks. In 7 of 30 it
named the gap in its own reason and accepted anyway. Its reasons often cite "the
upstream fix" (`results/2026-11-16-judge-probes`). Does a prompt that addresses those
three things catch more wrong patches?

## Setup

- `STRICT_INSTRUCTIONS` in `tine/judge.py` (`--style strict`). It keeps the original
  task and reply format and adds: judge the patch as written, not by resemblance to a
  remembered fix; check every behaviour and edge case the issue names, whether the
  change is on the issue's code path, and whether it changes behaviour the issue did
  not ask about; a concrete gap means reject, and do not accept while naming one.
- The prompt was written after reading 60 of these cases. Results are shown for all
  cases and for the cases that were not read.
- Haiku on all 1,110 cases; Sonnet on the fixed 400-case sample used by the probes.
  Same models, effort, parser and grades (`run_resolved`) as before.
- Run-to-run noise reference: the probes' rerun of Sonnet with one neutral sentence
  added moved its catch rate from 78.3% to 78.9% on these 400 cases.

## Results

`strict-prompt.csv`.

| judge | cases | prompt | wrong rejected | right rejected | AUC |
| --- | --- | --- | --- | --- | --- |
| Haiku | 1,050 not read | original | 75.7% | 6.2% | 0.918 |
| Haiku | 1,050 not read | strict | 74.8% | 6.7% | 0.904 |
| Sonnet | 379 not read | original | 83.9% | 3.1% | 0.963 |
| Sonnet | 379 not read | original, rerun | 84.5% | 3.1% | 0.958 |
| Sonnet | 379 not read | strict | 81.9% | 4.5% | 0.961 |
| Haiku | all 1,110 | original / strict | 71.7% / 71.0% | 9.5% / 9.3% | 0.895 / 0.883 |
| Sonnet | all 400 | original / strict | 78.3% / 76.5% | 7.3% / 7.7% | 0.929 / 0.934 |

**The stricter prompt does not help either judge.** Both catch slightly fewer wrong
patches with it and reject slightly more right ones; every difference is within the
run-to-run noise. Telling the judge to reject named gaps and to ignore remembered
fixes did not change what it decides. It costs 4-9% more per check, from slightly
longer replies.

## What this means

Three changes to what the judge is shown or told have now failed to move it: a
hardened prompt, the surrounding code, and this stricter prompt. Only a stronger
model has (`claude-opus-5-5`, `results/2026-11-16-judge-probes`). The judge's
remaining errors look like limits of the model reading the diff, not of the
instructions. Keep the original prompt (`PROMPT_VERSION` 2026-11-11), which every
measured result uses. `--style strict` stays available but is not recommended.

## Limits

- One wording, one effort setting, two models. Opus was not run with it.
- 166 wrong patches in the Sonnet sample: a change smaller than about five points in
  catch rate cannot be told from noise.

## Reproduce

```bash
# paid: needs authorisation and ANTHROPIC_API_KEY in the environment
python scripts/judge_patches.py judge --provider anthropic --model claude-sonnet-5-5 --style strict \
  --cases RUNS/probes/sample400.jsonl --output RUNS/probes/verdicts-strict.csv --max-cost-usd 4.2
python scripts/judge_probes.py compare --key RUNS/judge/key.csv --baseline RUNS/judge/verdicts-sonnet.csv \
  --probe RUNS/probes/verdicts-strict.csv
```
