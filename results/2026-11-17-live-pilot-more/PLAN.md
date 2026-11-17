# The live pilot on four more kinds of traffic: plan, 2026-11-17

Written before any model was called for this run. It repeats
[`../2026-11-17-live-pilot/PLAN.md`](../2026-11-17-live-pilot/PLAN.md) (same customer,
models, settings, steps and pass marks) on four more benchmarks, one at a time, as four
separate customers. Results go in `README.md` next to this file; anything that changes
from this plan is listed there as a deviation.

## Why

The first pilot used one benchmark of 10-option multiple-choice questions. Routing by
agreement depends on how close the cheaper models are to the frontier and on whether two
right answers are written the same way. These four vary both.

| benchmark | stands in for | answer the proxy reads | calibrate / shadow / live |
| --- | --- | --- | --- |
| MedQA | clinical questions | `Answer: X`, letters A-D | 800 / 200 / 150 |
| ARC-Challenge | easy school science | `Answer: X`, letters (some 1-4) | 800 / 200 / 150 |
| FinQA | numbers read off financial tables | the last `\boxed{...}` | 800 / 200 / 133 |
| MATH-500 | competition maths | the last `\boxed{...}` | 330 / 100 / 70 |

## What is new

- **Agreement on free-form answers.** For FinQA and MATH-500 the proxy compares the two
  cheaper models' boxed text after collapsing whitespace only, so `14.5\%`, `14.5%`
  and `14.50` disagree and go to the frontier. That is safe but costs savings; this run
  measures how much.
- **Grading.** Letters must match the key. MATH-500 answers match after normalising the
  LaTeX (`\left`, `\dfrac`, `\text{}`, spaces and so on) or as equal numbers. FinQA
  answers match as the benchmark grades them: a number counts when it rounds to the key
  at the key's precision, and yes/no ignores case. On recorded answers this grader
  agrees with the benchmark's own grades on 95-100% of MedQA, ARC and MATH-500 answers
  and 84-95% of FinQA answers; the rest are replies with no `\boxed{}`, which the proxy
  could not read either and which count as wrong here.
- **Set sizes.** MATH-500 has 500 questions, so its certify part is 165 and 2% cannot
  certify (`scripts/plan_certification.py`); 4% can only if the loss is near zero.

## Steps, limits and caps

For each benchmark: a 20-question probe (part of the calibrate set), calibrate with all
three models, certify with `--fractions 0.3 0.5 0.2` at 0.04 and 0.02 with the
benchmark's reader written into the policy, then, only if a chain certified at 0.04,
shadow through the proxy and route live. No free text this time.

Hard cap for the whole run: **$90**, paid from the team's separate credits key. Per
benchmark: $25 (FinQA, MATH-500 replies are longer) or $18 (MedQA, ARC), split as the
probe projects. A benchmark whose probe projects over its cap is cut to fit.

## Pass marks (per benchmark, as before)

1. Shadow loss on new questions at most 4%.
2. Share answered by cheaper models within 7 points of the certified test-part estimate.
3. The reader matches at least 98% of replies for each model (for boxed answers this is
   also a measurement of how often models box their answer).
4. Live routing: accuracy against the key, cost and latency against the frontier.

Nothing here executes model output. Prompts and replies stay under
`artifacts/live-pilot/`; only summary tables go in this folder.
