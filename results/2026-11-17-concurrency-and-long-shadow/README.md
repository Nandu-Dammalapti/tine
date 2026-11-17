# Rate limits under real refusals, and a 1,500-question shadow run, 2026-11-17

**Paid run.** `claude-haiku-5-5`, `claude-sonnet-5-5` and `claude-opus-5-5`, called
through the Anthropic API and through Tine's proxy. The run used the team's separate
credits key.

- **Spend:** **$26.43**, from the API's own token counts. The cap was $32. Part 1
  spent $4.12 against its own $3 cap (see Deviations).
- **The plan:** [`PLAN.md`](PLAN.md). It was committed after Part 1 had run (see
  Deviations).
- **Data:** the summary tables are `ramp.csv`, `pause-test.csv` and
  `long-shadow-blocks.csv`. Prompts, replies and per-question rows stay local.

## Bottom line

- **This key allows 25 requests in flight.** That is the limit that binds. A
  refusal comes back in about half a second and names no wait.
- **The pause kept real traffic safe when the proxy's own calls filled the
  account.** The same 60 questions were sent 24 at a time through each code version.
  - Without the pause: the provider refused 9 of 69 real attempts (13%). The 60
    questions took 46 seconds.
  - With the pause: it refused none of 60. They took 30 seconds.
- **The price is shadow coverage.** With the pause, 4 of the 60 requests were
  shadowed (6.7%) and 55 were skipped. The report counted them and said why.
- **The first pause test showed nothing.** A background load left only a few slots
  free, but no real request was refused with either code version. A second test was
  added.
- **On 1,500 new questions the loss against Opus was 2.47%** (37 questions). The
  one-sided 95% upper bound is 3.2%. The pass mark was a bound of at most 4%, so it
  passes.
- **The run does not confirm the 2% limit.** The same policy also certified at 2%
  on its history. Here the bound is 3.2%, above 2%. The run does not show the true
  loss is above 2% either: the lower bound is 1.8%.
- **Five blocks of 300 lost 1.3% to 3.3%.** That is the spread of a larger sample
  within 79 minutes of one day. It is not drift over weeks.
- **The drift check warned in 2 of the 5 blocks** while the loss in each was under
  the limit.

## Results

### Part 1A: the limit on requests in flight

`ramp.csv`. `scripts/concurrency_probe.py ramp` sent N Haiku requests at once, each
taking about 1.5 seconds.

| sent at once | accepted | refused (429) | median seconds: accepted / refused |
| --- | --- | --- | --- |
| 8 | 8 | 0 | 1.60 / - |
| 16 | 16 | 0 | 1.38 / - |
| 24 | 24 | 0 | 1.50 / - |
| 32 | 25 | 7 | 1.49 / 0.53 |
| 48 | 25 | 23 | 1.61 / 0.59 |
| 64 | 25 | 39 | 1.65 / 0.68 |
| 96 | 25 | 71 | 1.48 / 0.57 |
| 128 | 42 | 86 | 1.68 / 0.56 |

- **The limit is 25.** From 32 to 96 at once, exactly 25 were accepted.
- **At 128, 42 were accepted.** The likely reason is that the first replies
  finished while later requests were still being sent. This was not checked.
- **The refusal names the limit:** "Number of concurrent requests across all models
  has exceeded your organization's limit". It is one limit for all three models.
- **No refusal carried a `retry-after` header.** So the proxy's pause uses its own
  30 seconds.
- **Refusals are fast:** a median of 0.52 to 0.68 seconds at each level that had
  any.
- **The per-minute limits were not the cause.** The plan records 4,000 requests a
  minute per model, from the response headers. The token limits read from the same
  headers were 2,000,000 input and 400,000 output tokens a minute per model. That
  reading was noted during the run and is not in a saved file.

### Part 1B: the pause under real refusals

`pause-test.csv`. Shadow mode, sample rate 1, the MedQA policy (Haiku and Sonnet
must agree, else Opus). "Before" is the proxy code at `bc627ae`, which has no
pause. "After" is the current code.

| test | code | real attempts | refused (429) | seconds | shadowed | skipped | shadow coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| background load, 30 questions, 3 in flight | before | 30 | 0 | 112 | 30 | 0 | 100% |
| | after | 30 | 0 | 105 | 30 | 0 | 100% |
| the proxy's own traffic, 60 questions, 24 in flight | before | 69 | 9 (13.0%) | 46 | 44 | 16, as busy | 73.3% |
| | after | 60 | 0 | 30 | 4 | 55, as rate limited | 6.7% |

**The first test, as planned.** A background load held 21 Haiku requests in flight
while the proxy sent 30 MedQA questions, 3 at a time. Each code version got 30
different questions, none of them in another MedQA set.

- **No real request was refused, with either version.** The test did not exercise
  the pause.
- **The account was rarely full.** One shadow call was refused with the old code and
  none with the new. The load itself was refused 1 time in 4,735 requests and 5
  times in 4,743.

**The second test, added.** No background load. The proxy sent the same 60 MedQA
questions 24 at a time, first on the old code, then on the new. The real requests
alone fit under the limit of 25. The shadow calls do not.

- **Old code.** The provider refused 9 real requests and 16 shadow calls. The
  client retried the 9, so 60 questions took 69 attempts. 16 requests were not
  shadowed, and 10 of the 44 that were had a cost that is not known.
- **New code.** One shadow call was refused. The proxy paused shadow mode for 30
  seconds and no real request was refused.
- **What the report said.** `skipped_rate_limited` 55, `rate_limited_runs` 1,
  coverage 6.7%, and `shadow_paused` with the reason: "a shadow call got HTTP 429
  from anthropic; no retry-after, so 30 seconds (pause 1 in a row)".
- **Through the proxy, a refusal came back in 0.30 to 0.48 seconds** over both tests.

The plan's pass marks, applied to the second test:

| pass mark | result | |
| --- | --- | --- |
| after has fewer refused real requests than before, and under 10% | 0 of 60 against 9 of 69 | pass |
| the report counts the skipped shadow runs | 55 skipped as rate limited | pass |
| coverage is below 100% and says why | 6.7%, with the pause and its reason | pass |

On the first test the marks say nothing: neither version had a refused real request
or a skipped run.

### Addition: the Message Batches API on the same account

Not in the plan. `scripts/live_pilot.py ask --batch` sent MedQA questions that
`../2026-11-17-live-pilot-more` had already asked directly.

- **49 of 60 requests were submitted.** The 60 are 20 questions for each of three
  models. The harness held back 11 because their worst-case cost did not fit under
  the cap. The 49 cover 17 questions: 17 Haiku, 17 Sonnet and 15 Opus.
- **All 49 succeeded.** They were collected 153 seconds after the first batch was
  submitted.
- **47 of 49 answers were the same as the direct call's.** In the other two the
  batch answer was right and the direct one wrong.
- **Cost: $0.14, against $0.29 for the same 49 requests asked directly.** The
  harness prices each batch reply at half the list price for its own tokens. The
  figure was not checked against an invoice.
- **Batches did not use the limit on requests in flight.** After the three batches
  were submitted and before they were collected, the ramp was repeated: 24 at once
  were all accepted, and 40 at once were capped at 25, as before.

### Part 2: a longer shadow run

`long-shadow-blocks.csv`. The 1,500 MMLU-Pro questions that the first pilot did not
use, through the proxy in shadow mode, sample rate 1, 3 requests in flight. The
policy is the first pilot's `policy-eps0.04.json`: Haiku and Sonnet must agree, else
Opus. The run took 79 minutes.

| | value |
| --- | --- |
| loss against Opus | 2.47% (37 of 1,500) |
| one-sided 95% upper bound (Clopper-Pearson) | 3.23% |
| pass mark: bound at most 4% | pass |
| against 2% | not confirmed: the bound is above 2% |
| gain (the policy right, Opus wrong) | 0.67% (10) |
| accuracy: policy / Opus | 91.1% / 92.9% |
| answered by cheaper models | 92.7% |
| projected saving | 50.8% |

**How to read the loss against 2%.** The policy file certified at both 4% and 2%,
with a bound of 1.84% on its 495 certify questions.

| sample | loss | one-sided 95% bounds: lower / upper |
| --- | --- | --- |
| history, all 1,000 (the policy was chosen on these) | 1.30% (13) | 0.77% / 2.06% |
| first shadow run, 300 new questions | 2.00% (6) | 0.87% / 3.91% |
| this run, 1,500 new questions | 2.47% (37) | 1.84% / 3.23% |
| both runs on new questions, 1,800 | 2.39% (43) | 1.83% / 3.07% |

The 4% claim is solid. The 2% claim on this policy was optimistic: the best estimate
on new questions is 2.4%, though 2% is not ruled out. A customer who needs 2% should
not rely on this policy for it.

**By block,** in the order sent:

| block | loss (limit 4%) | answered by cheaper models | projected saving | accepted answers that differ from Opus's | drift check |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.7% (8) | 91.0% | 49.9% | 9 (3.0%) | within limit |
| 2 | 3.3% (10) | 94.3% | 54.5% | 17 (5.7%) | warning |
| 3 | 2.0% (6) | 92.3% | 50.4% | 11 (3.7%) | within limit |
| 4 | 3.0% (9) | 93.7% | 50.7% | 14 (4.7%) | warning |
| 5 | 1.3% (4) | 92.0% | 48.8% | 7 (2.3%) | within limit |

- **This is spread, not drift.** The five blocks ran one after another in 79
  minutes, on questions shuffled from one benchmark. If every block had the pooled
  loss rate, a block of 300 would lose between 3 and 13 questions 95% of the time.
  These lost 4 to 10. A chi-square test finds no difference between the blocks
  (p = 0.52). Drift over weeks needs a real deployment.
- **Above the certified estimate.** The policy file estimated 86.8% answered by
  cheaper models and a 41.1% saving, on 197 held-out questions. Every block was above
  both.
- **The drift check gave two false warnings.** It counts accepted answers that
  differ from Opus's, which is more than the loss. Of the 58 that differed, 37 were
  losses, 10 were gains, and in 11 both answers were wrong. Blocks 2 and 4 warned at
  5.7% and 4.7% while their losses were 3.3% and 3.0%. Over all 1,500 requests the
  count is 3.9%, under the limit.

**Refusals and errors:**

- **No rate-limit refusals.** None of the 4,500 provider calls got a 429 or 529.
  No real request failed, the client never retried, no shadow run was skipped and
  shadow mode never paused. Coverage was 100% in every block.
- **One shadow call lost its connection** after 17 seconds (Sonnet, block 3). That
  request is left out of the projected saving. Its cost, if any, is not recorded.
- **One question was declined by Opus and by Haiku** (stop reason `refusal`,
  block 2). It counts as wrong for both and is not a loss.
- **Four Haiku replies were cut off at 4,096 tokens.** A reply with no readable
  answer never agrees, so the policy would send those requests to Opus.

## Deviations from the plan

1. **The plan was committed after Part 1 ran.** The commit (`c5fe825`) is at 09:29
   UTC. The data files show Part 1 from 09:13 to 09:27 and the long run's first
   request at 09:28. The plan's text names no result from either part. Git cannot
   show that it was written first.
2. **Part 1 went over its $3 cap: $4.12.** The background load did not stop when the
   30 questions finished. Both times it ran on to its own spending limit ($0.70),
   about three minutes after the proxy run had ended. The total stayed inside the
   $32 cap.
3. **A second pause test was added,** because the first one produced no refused
   real request. It has no background load and puts 24 real requests in flight, with
   the same 60 questions for both code versions. The pass marks were written for the
   first design and are applied to the second.
4. **Two more additions:** the batch validation, and the ramp repeated while
   batches were pending ($0.15 together).
5. **The long run was five runs of 300, not one run of 1,500.** Each block was its
   own proxy process with its own ledger. So each report covers 300 requests, and no
   report shows figures building up over 1,500. The totals here were added up
   afterwards.
6. **Code versions.**
   - *Before:* `bc627ae`, as the plan says. Its reports have none of the pause's
     fields. No file records the version.
   - *After, and the long run:* `6dcd48d`, from a version file written just before
     block 1. The proxy code under `tine/` is the same from the pause's merge
     (`2e56353`) to `6dcd48d`.
   - *Blocks 4 and 5* started after `2341a23`, which gave `scripts/live_pilot.py` a
     tickets entry and a `--max-tokens` flag that defaults to the old value. Nothing
     under `tine/` changed, so the run behaved the same.

## Limits

- **One key, one day.** The limit of 25 belongs to this account's tier. Other
  accounts differ. Everything here ran between about 09:12 and 10:47 UTC.
- **Each pause test ran once.** The two code versions ran one after the other on 60
  questions, so some of the difference could be the minute they ran in.
- **The pause was never seen to end.** The second test finished with 16 seconds of
  the pause left. Resuming, a longer backoff, a `retry-after` header and a 529 were
  not exercised.
- **The pause trades coverage for safety.** While real traffic alone nearly fills
  the account, shadow mode measures almost nothing. That is the intended behaviour,
  and it means a savings report needs spare capacity.
- **Public benchmark.** MMLU-Pro stands in for one customer's traffic. The models
  may have seen it.
- **The batch figures are small:** 49 requests on one benchmark.

## Spend (credits key)

| step | spent |
| --- | --- |
| ramp to 128 at once | $0.03 |
| background load, twice | $1.41 |
| first pause test, two proxy runs | $1.01 |
| second pause test, two proxy runs | $1.67 |
| **Part 1** | **$4.12** |
| ramp while batches were pending | $0.01 |
| batch validation | $0.14 |
| long shadow run, five blocks | $22.16 |
| **total** | **$26.43** |

## Reproduce

The commands as typed were not saved. These give the same steps; the caps are
examples.

```bash
# paid: needs authorisation and ANTHROPIC_API_KEY in the environment
python scripts/concurrency_probe.py ramp --max-cost-usd 0.1 --output RUNS/ramp.json
python scripts/concurrency_probe.py load --in-flight 21 --seconds 400 --max-cost-usd 0.7 --output RUNS/load.json &
python scripts/live_pilot.py proxy --policy RUNS/policy-medqa-eps0.04.json --questions RUNS/sets/medqa-pause-after.jsonl \
  --mode shadow --max-cost-usd 1 --concurrency 3 --ledger RUNS/pause-after.db --output RUNS/proxy-pause-after.json \
  --summary RUNS/summary-pause-after.json
# second test: both 30-question sets, --concurrency 24, and no background load
python scripts/live_pilot.py ask --batch --questions RUNS/sets/medqa-batch.jsonl --output RUNS/asked-medqa-batch.jsonl \
  --models claude-haiku-5-5 claude-sonnet-5-5 claude-opus-5-5 --max-cost-usd 1
# long run: once for each of the five blocks of 300
python scripts/live_pilot.py proxy --policy RUNS/policy-eps0.04.json --questions RUNS/sets/long-shadow-1.jsonl \
  --mode shadow --max-cost-usd 5.4 --concurrency 3 --ledger RUNS/long-1.db --output RUNS/proxy-long-1.json \
  --summary RUNS/summary-long-1.json
```

- For "before", run the same proxy command from a checkout of `bc627ae`.
- The long set is the 1,500 MMLU-Pro questions left over by `live_pilot.py sample`.
  No command writes it; it was cut by hand and split into five files in order.
- The loss bounds use `tine.calibration.clopper_pearson_upper` with alpha 0.05.
