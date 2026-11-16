# A live pilot with real models, 2026-11-17

**Paid run.** `claude-haiku-5-5`, `claude-sonnet-5-5` and `claude-opus-5-5`, called
through the Anthropic API and, in steps 3 and 4, through Tine's real proxy.
**$23.65** recorded from the API's own token counts, plus at most about $0.70 for
calls cut off when one run was stopped (see Deviations). The cap was $35. No model
output was executed.

The plan, with the questions and pass marks, was written before any model was
called: [`PLAN.md`](PLAN.md). Summary tables: `models.csv`, `checks.csv`,
`shadow-by-traffic.csv`. Prompts, replies and per-question rows stay local under
`artifacts/live-pilot/`.

## Bottom line

On real current models, end to end through the proxy, Tine did what it says on
multiple-choice traffic.

- **Certification.** A policy certified at both 4% and 2%: Haiku and Sonnet
  answer, the answer is used when they agree, and otherwise the request goes to
  Opus.
- **New questions.** On 300 new questions the loss against Opus was 2.0%. The
  limit was 4%.
- **Shadow mode.** Cheaper models would have answered 89% of these questions, at
  48% lower cost.
- **Live routing.** On 200 more questions, 91.5% were answered by Haiku. Accuracy
  was 91.5%, against Opus's 91.7-92.2%. Cost was about 52% lower, and the median
  response was faster.

Two things a customer would hit:

- **Free text on the same token costs more and is never saved.** It was never
  wrongly accepted, but it paid for two cheaper attempts every time and was 31%
  dearer than Opus alone.
- **The savings report shows nothing under live routing.** `/v1/savings` reported
  no saving for the routed run, because Tine cannot measure it without a random
  share of requests always sent to the frontier.

## The customer and the setup

An Anthropic-only team whose traffic all goes to `claude-opus-5-5`. Most of it
asks a question with fixed answer options and ends with `Answer: X`. Some of it
is free text. The exact-answer traffic was stood in for by MMLU-Pro questions, and
the free text by Arena-Hard creative-writing prompts. The four question sets do
not overlap. Every call used the same settings: the benchmark's own prompt,
`max_tokens` 4096, effort `medium`. The answer reader was
`Answer:\s*\**\s*\$?\(?([A-J])\b`, written into the policy with the new
`freeze-policy --answer-pattern`.

Prices, $ per million input / output tokens: Haiku 0.10 / 0.50, Sonnet 2 / 10,
Opus 4 / 20.

## Results

### Step 1: the team's graded history (1,000 questions, all three models)

`models.csv`

| model | accuracy | reader matched | mean cost per question | median / p95 latency |
| --- | --- | --- | --- | --- |
| Haiku 5.5 | 84.3% | 99.4% (6 cut off at 4,096 tokens) | $0.00037 | 3.0s / 8.0s |
| Sonnet 5.5 | 88.4% | 100% | $0.0038 | 2.8s / 6.1s |
| Opus 5.5 | 92.2% | 100% | $0.0112 | 4.6s / 10.0s |

Haiku and Sonnet gave the same answer on 86.9% of the questions.

### Step 2: certification

`scripts/agreement_pairs.py` built the candidates: the Haiku+Sonnet pair, and
each model alone. `freeze-policy` used a 0.3 / 0.5 / 0.2 split and delta 0.05. The
same policy came out at both limits.

| | value |
| --- | --- |
| chain | Haiku + Sonnet must agree, else Opus |
| certified at 4% and at 2% | yes and yes |
| upper bound on loss (95%) | 1.84% |
| loss on the 495 certify questions | 0.81% |
| held-out test part (197 questions) | 41.1% saving, 13.2% sent to Opus, 2.5% loss |
| all 1,000 questions | 86.9% answered by cheaper models, 1.3% loss, 0.6% gain |

The held-out loss of 2.5% is above 2% on a small sample. Over all 1,000 questions
the loss was 1.3%.

### Step 3: shadow mode through the real proxy (300 new questions and 20 free text)

`checks.csv`, `shadow-by-traffic.csv`. Code at commit `3331dd7`, with the
simulated pilots' fixes merged.

| pre-registered check | pass mark | result | |
| --- | --- | --- | --- |
| loss on new questions | at most 4% | 2.0% (6 of 300) | pass |
| share answered by cheaper models | within 7 points of 86.8% | 89.3% | pass |
| reader matched real replies | at least 98% for each model | Haiku 99.3%, Sonnet 100%, Opus 100% | pass |
| free text accepted | none | 0 of 20 | pass |

| traffic | requests | saving against Opus | median / p95 latency if routed (Opus alone) |
| --- | --- | --- | --- |
| multiple choice | 297 | 47.8% | 3.6s / 13.9s (5.0s / 10.4s) |
| free text | 20 | -31% | 49.8s / 87.1s (34.7s / 51.7s) |
| proxy's own report, both together | 317 | 28.4% | - |

What the proxy's shadow report said:

- **Coverage.** 317 of the 320 sampled requests were shadowed, which is 99.1%.
  Three were skipped as busy, and the new coverage counter showed them.
- **Drift check.** "Within the limit: 9 of 268 accepted answers (3.36%) differ
  from the frontier's". A review fix after this run (`4dd5ff7`) now counts
  differing answers over all shadowed requests: 9 of 317, 2.84%, still within 4%.
  The check counts answers that differ from the frontier's, which is more than the
  loss, because it includes questions the frontier got wrong. At a 2% policy this
  run would have warned at 2.84% while the true loss was 2.0%.
- **Policy checks.** It showed the answer reader the policy file names, and that
  the freeze digest matched.
- **Cost.** Shadow calls cost 35% of the frontier spend on the same requests,
  $1.58 against $4.50.
- **Waiting time.** The caller never waited for shadow calls. The caller's median
  latency was the frontier's alone.

Latencies "if routed" are worked out from the measured call times. They assume
the two cheaper models are called in parallel, and that Opus is called after them
when they disagree.

### Step 4: live routing (200 new questions)

Code at commit `1d0afb9`.

| | live routing | Opus alone |
| --- | --- | --- |
| answered by a cheaper model | 91.5% (183 of 200; all served by Haiku) | - |
| accuracy against the answer key | 91.5% | 91.7% on the shadow set, 92.2% on the history |
| cost per question, including every cheaper attempt | $0.0054 | $0.0112 (mean on the history) → about 52% lower |
| caller latency, median / p95 | 3.7s / 12.5s | 4.6s / 10.0s on the history |

Opus was not called for the questions Haiku and Sonnet agreed on, so "Opus alone"
comes from the other two sets of the same kind of question.

**What the product itself reported.** `/v1/savings` showed no saving for this run
(`savings_fraction: null`). Its two baselines were both empty:

- **Measured baseline.** It covers only requests that escalated. Those also paid
  for the cheaper attempts, and none had an outcome reported.
- **Estimated baseline.** It needs a caller-reported success.

Its `savings_scope` text now says this plainly. This is the simulated buyer's
finding #10, seen for real. A team cannot be billed on a share of savings until a
random frontier holdout exists.

## What this shows, and what it does not

- **Shown.**
  - The agreement check certified and held on new questions from the same source.
  - The answer reader written into the policy read 99-100% of real replies.
  - Shadow mode's projection matched what live routing then did.
  - Free text was kept safe.
  - The drift check, coverage counters and freeze digest all reported correctly.
- **Not shown.**
  - **Different traffic.** That it holds on traffic unlike MMLU-Pro, or on a
    customer's own labels. The models may also have seen MMLU-Pro before.
  - **Drift over time.** Everything ran in one afternoon.
  - **The deployed server.** It was not tested under real load. The proxy ran
    in-process, because this machine cannot open a port, and the container was
    not used.
- **Price ratios decide the saving.** The saving depends on Sonnet costing half
  of Opus and Haiku a fortieth. Haiku's long replies and Sonnet's short ones made
  the cheaper stage cost about 37% of Opus when accepted, not the 55% the list
  prices suggest.

## Deviations from the plan

1. **Shadow step, first attempt: no model reached.** All 360 requests got a 502
   without reaching the provider. The harness's own transport ignored this
   machine's network proxy setting. Nothing was spent. The fix was in the
   harness only; Tine's proxy uses a normal client that honours the setting.
2. **Shadow step, second attempt: stopped.** Long free-text replies took over 5
   minutes each and held every request slot, because in-process calls wait for
   their shadow work. They also cost about 4 times what the plan assumed. The run
   was stopped after 42 requests, with $1.14 recorded, plus at most about $0.70
   for calls in flight. The rerun used:
   - 20 free-text prompts instead of 60,
   - 12 requests at a time instead of 6,
   - a 600-second upstream timeout.

   The 300 multiple-choice questions were unchanged.
3. **One extra call.** A single-question check of the proxy path cost $0.007.
4. **One request sent alone first.** Each proxy run sent one request on its own
   first. Shadow mode admits one run at a time until a run has finished and its
   cost is known.
5. **Steps 3 and 4 ran on different code.** Step 4 ran on a later commit
   (`1d0afb9`) than step 3 (`3331dd7`). Between them only documentation changed,
   plus the wording of `/v1/savings`'s `savings_scope` text in `tine/ledger.py`.

## Spend

| step | spent |
| --- | --- |
| probe and history (3,000 answers) | $15.32 |
| shadow, stopped attempt | $1.14 recorded (up to about $0.70 more unrecorded) |
| proxy path check | $0.01 |
| shadow | $6.10 |
| live routing | $1.08 |
| **total** | **$23.65 recorded** |

## Reproduce

```bash
# paid: needs authorisation and ANTHROPIC_API_KEY in the environment
python scripts/live_pilot.py sample --bench-dir artifacts/bench/extracted/bench-release --output RUNS/sets
python scripts/live_pilot.py ask --questions RUNS/sets/calibrate.jsonl --output RUNS/asked-calibrate.jsonl \
  --models claude-haiku-5-5 claude-sonnet-5-5 claude-opus-5-5 --max-cost-usd 22
python scripts/live_pilot.py export --asked RUNS/asked-calibrate.jsonl --outcomes RUNS/outcomes.jsonl --answers RUNS/answers.csv
python scripts/agreement_pairs.py --input RUNS/outcomes.jsonl --answers RUNS/answers.csv --frontier claude-opus-5-5 \
  --output RUNS/pairs.jsonl --checks RUNS/checks.csv --include-singles --grade-conflicts wrong
python -m tine freeze-policy --input RUNS/pairs.jsonl --frontier claude-opus-5-5 --checks RUNS/checks.csv \
  --epsilon 0.04 --fractions 0.3 0.5 0.2 --answer-pattern 'Answer:\s*\**\s*\$?\(?([A-J])\b' --output RUNS/policy.json
python scripts/live_pilot.py proxy --policy RUNS/policy.json --questions RUNS/sets/shadow.jsonl RUNS/sets/freetext20.jsonl \
  --mode shadow --max-cost-usd 7.5 --concurrency 12 --ledger RUNS/shadow.db --output RUNS/shadow.json --summary RUNS/shadow-summary.json
python scripts/live_pilot.py proxy --policy RUNS/policy.json --questions RUNS/sets/live.jsonl \
  --mode route --max-cost-usd 3 --concurrency 12 --ledger RUNS/route.db --output RUNS/route.json --summary RUNS/route-summary.json
```
