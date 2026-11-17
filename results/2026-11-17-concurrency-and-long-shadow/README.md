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
- **The planned pause test did not meet the plan's pass marks.** A background load
  left only a few slots free, but no real request was refused and no shadow run was
  skipped with either code version, so the test did not exercise the pause. The
  marks asked for fewer refused real requests with the pause than without, and for
  shadow coverage below 100%. It was 0 of 30 against 0 of 30, and 100% with both.
- **A test added after the plan showed the pause working.** It is not a
  pre-registered result. The proxy's own calls filled the account: the same 60
  questions were sent 24 at a time through each code version.
  - Without the pause: the provider refused 9 of 69 real attempts (13%). The 60
    questions took 46 seconds.
  - With the pause: it refused none of 60. They took 30 seconds.
- **The price is shadow coverage.** With the pause, 4 of the 60 requests were
  shadowed (6.7%) and 55 were skipped. The report counted them and said why.
- **On 1,500 new questions the loss against Opus was 2.47%** (37 questions). The
  one-sided 95% upper bound is 3.2%. The pass mark was a bound of at most 4%, so it
  passes.
- **Do not quote 2% for this policy.** It certified at 4%. Its 2% certificate
  passed on the last allowed loss with 495 certify questions, and it did not hold
  up: the loss was 2.3% on the 2,305 questions outside the certify part (95% upper
  bound 2.8%) and 2.4% on the 1,800 new ones. This run does not show the true loss
  is above 2% either: its lower bound is 1.8%.
- **These questions are easier than the history's.** So the share answered by
  cheaper models and the saving here are not comparable with the history's. The
  loss stands on its own.
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
- **Share answered by cheaper models, over the requests that were shadowed:** 32 of
  44 (72.7%) on the old code and 3 of 4 on the new. The harness's summaries said
  53.3% and 5.0%, because they counted requests that were never shadowed as turned
  down. The harness now counts shadowed requests only.

**The plan's pass marks.** They were written for the first test, and on it they
were not met: neither version had a refused real request or a skipped run, so the
test did not exercise the pause. The plan states the last two rows as one mark.

| pass mark | first test, as planned | |
| --- | --- | --- |
| after has fewer refused real requests than before, and under 10% | 0 of 30 against 0 of 30 | not met |
| the report counts the skipped shadow runs | none were skipped | not met |
| coverage is below 100% and says why | 100% with both versions | not met |

**The same marks on the added test.** That test was designed after the first one
had been seen, so read this as a description of what happened, not as a
pre-registered pass.

| pass mark | second test, added | |
| --- | --- | --- |
| after has fewer refused real requests than before, and under 10% | 0 of 60 against 9 of 69 | met |
| the report counts the skipped shadow runs | 55 skipped as rate limited | met |
| coverage is below 100% and says why | 6.7%, with the pause and its reason | met |

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

**How to read the loss against 2%.** The policy certified at 4%. It also passed at
2%, with a bound of 1.84% on its 495 certify questions, on the last allowed loss:
it had 4 losses and the limit allows 4.

| sample | loss | one-sided 95% bounds: lower / upper |
| --- | --- | --- |
| history, all 1,000 (the policy was chosen on these) | 1.30% (13) | 0.77% / 2.06% |
| first shadow run, 300 new questions | 2.00% (6) | 0.87% / 3.91% |
| this run, 1,500 new questions | 2.47% (37) | 1.84% / 3.23% |
| both runs on new questions, 1,800 | 2.39% (43) | 1.83% / 3.07% |
| everything outside the certify part: 505 history questions and the 1,800 | 2.26% (52 of 2,305) | 1.77% / 2.83% |

The 4% claim is solid. The 2% certificate did not hold up, and 2% should not be
quoted for this policy.

- **Outside the certify part the loss is above 2%:** 2.3% on 2,305 questions (95%
  upper bound 2.8%), and 2.4% on the 1,800 new ones.
- **Neither is significantly above 2%.** A true loss of exactly 2% would give 43 or
  more losses in 1,800 about one time in seven. So 2% is not ruled out.
- **It was never established.** The whole history's bound is 2.06%. Freezing that
  history with 300 other seeds certifies at 2% in 17 of them
  (`../2026-11-17-live-pilot`, Step 2).

A customer who needs 2% should not rely on this policy for it.

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
- **Above the certified estimate, on easier questions.** The policy file estimated
  86.8% answered by cheaper models and a 41.1% saving, on 197 held-out questions.
  Every block was above both. The two sets are not like for like (below).
- **The drift check gave two false warnings.** It counts accepted answers that
  differ from Opus's, which is more than the loss. Of the 58 that differed, 37 were
  losses, 10 were gains, and in 11 both answers were wrong. Blocks 2 and 4 warned at
  5.7% and 4.7% while their losses were 3.3% and 3.0%. Over all 1,500 requests the
  count is 3.9%, under the limit.

**The long-run questions are easier than the history's.** Both sets are slices of
one shuffle of the benchmark's 3,000 questions, and the draw was uneven.

- **By the benchmark's own records.** The archive holds 15 models' graded answers
  to every question. On average they got 80.7% of the history's 1,000 questions
  right and 83.9% of these 1,500. The gap is 3.2 points, 2.6 standard errors.
- **By our own models.** Haiku and Sonnet agreed on 86.9% of the history and 92.7%
  here. Haiku was right on 84.3% of the history and 87.9% here.
- **What this changes.** The share answered by cheaper models (92.7%) and the
  projected saving (50.8%) are not comparable with the history's 86.9% and the
  policy's estimate of 41.1%. Traffic like the history should expect the history's
  figures.
- **What it does not change.** The loss moved the other way: 13 of 1,000 on the
  history and 37 of 1,500 here, on easier questions. The loss of 2.47% (bound
  3.23%) stands on its own.
- **Block 5 is the easiest.** The archive's models got 86.4% of its questions
  right. Opus was right on 290 of its 300, against 275 to 278 in the other blocks
  (Fisher p = 0.004).

`python scripts/live_pilot.py sample --compare HISTORY.jsonl SET.jsonl` now prints
these means, and warns about a draw like this one before any money is spent.

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
   first design. On the first test they were not met. What the second test shows
   is reported apart from them, as an addition made after the plan.
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

7. **The laptop slept for 106 seconds during block 4.** The system's power log has a
   lid-closed sleep on battery from 10:27:57 to 10:29:43 UTC. The block's ledger has
   no finished request between 10:27:57 and 10:30:01.
   - Four provider calls took 112 to 206 seconds by the wall clock. All returned 200.
   - For the three requests that waited longest (128 to 211 seconds for the
     client), the ledger's latency is 5 to 21 seconds. The proxy times a request
     with a clock that stops while the machine sleeps.
   - No request failed and no count changed. A longer sleep would have cost the
     block: a proxy run could not be resumed then. It can now (`--resume`), and the
     harness reports `suspended_seconds`.
   - Run paid runs on mains power with the lid open. `caffeinate -i` stops idle
     sleep but not a lid-closed sleep on battery.
8. **Other runs used the same account during blocks 4 and 5.** The plan says one
   run at 3 in flight. Block 4 ran from 10:14 to 10:31 UTC and block 5 from 10:32 to
   10:46. Beside them ran the ticket pilot's shadow run (10:22 to 10:30) and live
   run (10:30 to 10:33), each 3 in flight, and a 32-request server test through the
   proxy's OpenAI-format route (10:36 to 10:37). None of these runs got a 429. A
   container test followed at 10:48, after block 5 had ended.
9. **A figure in the plan is wrong.** Its first reason says the provider refused
   19-45% of real requests in `../2026-11-17-live-pilot-more`. The ledgers give
   19-38% in the shadow runs, 25-58% in the routed runs and 65% in the server test.
   The plan is left as it was written.

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
