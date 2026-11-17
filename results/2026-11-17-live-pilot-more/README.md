# The live pilot on four more kinds of traffic, 2026-11-17

**Paid run.** The same three models as the first live pilot (`claude-haiku-5-5`,
`claude-sonnet-5-5`, `claude-opus-5-5`) were called through the Anthropic API and then
through Tine's proxy. The run used the team's separate credits key.

- **Spend:** **$60.00**, from the API's own token counts, including about $5.32
  spent on runs that were stopped (see Deviations). The cap was $90.
- **The plan:** written before any call ([`PLAN.md`](PLAN.md)).
- **Data:** the summary tables are `models.csv`, `certification.csv`, `shadow.csv`
  and `live-routing.csv`. Prompts, replies and per-question rows stay local.

## Bottom line

- **Four kinds of traffic, all certified.** Each benchmark stood in for one
  customer: clinical questions (MedQA), easy science (ARC-Challenge), numbers read
  off financial tables (FinQA) and competition maths (MATH-500). A policy certified
  at the 4% limit on each one. None certified at 2%, because the certify sets were
  smaller here (about 400, or 165 for MATH-500).
- **The limit held on new questions.** In shadow mode the loss against Opus was
  0.5-2.5%, under 4% on every benchmark.
- **The saving under live routing was measured, not estimated.** This used the
  new random frontier holdout (`TINE_AUDIT_RATE`).
  - MedQA: **56%** (95% interval 49-64%).
  - ARC: **97%** (97-98%).
  - FinQA: **18%** (5-32%).
  - MATH-500: too few held-out requests for an estimate.
- **The saving follows how close the cheap models are to the frontier.** Easy
  science saves almost everything, because Haiku alone is close enough to Opus.
  Finance saves least: Haiku gets 64% of FinQA right against Opus's 79%, and the
  two cheaper models agree on only about half the questions.
- **Comparing answers by value helps when one answer can be written several
  ways.** Comparing `14.5\%` and `14.50` as numbers (`--answer-compare math`) made
  these changes in shadow mode on the same questions:

  | benchmark | share answered by cheaper models | projected saving | loss |
  | --- | --- | --- | --- |
  | MATH-500 | 85% → 97% | 44.5% → 60.5% | 1.0% → 1.0% |
  | FinQA | 51% → 59% | 10% → 17% | 1.0% → 2.0% |

- **Rate limits are a real risk for shadow mode.** When shadow calls and real
  calls shared one account at modest limits, the provider refused 19-38% of the
  customer's real requests in the shadow runs, 25-58% in the routed runs and 65%
  in the server test. That led to the rate-limit pause now in the proxy.
- **`tine serve` worked as a real server.** It ran as its own process and answered
  40 requests over HTTP with routing and the holdout on, with no errors.

## Results

### Calibration (all three models on each set)

`models.csv`

| benchmark | questions | accuracy: Haiku / Sonnet / Opus | reader matched (lowest model) | Opus cost per question |
| --- | --- | --- | --- | --- |
| MedQA | 800 | 92.8% / 96.6% / 97.5% | 99.9% | $0.0128 |
| ARC-Challenge | 800 | 94.5% / 95.5% / 96.6% | 100% | $0.0057 |
| FinQA | 800 | 63.5% / 73.9% / 78.9% | 100% | $0.0115 |
| MATH-500 | 330 | 94.8% / 93.9% / 96.1% | 99.4% | $0.0099 |

How graded:

- **MedQA and ARC:** the letter must match the key.
- **MATH-500:** the answer must be equal to the key after normalising the LaTeX,
  or as a number.
- **FinQA:** graded as the benchmark grades it. A number counts when it rounds to
  the key at the key's precision.

On recorded answers this grader agreed with the benchmarks' own grades on 95-100%
of answers, and 84-95% for FinQA. The difference is replies with no `\boxed{}`,
which the proxy could not read either, so they count as wrong here.

### Certification

`certification.csv`. Split 0.3 / 0.5 / 0.2, delta 0.05, with the benchmark's answer
reader written into each policy.

| benchmark | policy at 4% | bound | held-out saving | sent to Opus | held-out loss |
| --- | --- | --- | --- | --- | --- |
| MedQA | Haiku + Sonnet agree | 2.6% | 55.3% | 6.5% | 0.65% |
| ARC-Challenge | **Haiku alone, no check** | 3.6% | 97.3% | 0% | 3.25% |
| FinQA (exact) | Haiku + Sonnet agree | 2.6% | 10.8% | 47.4% | 1.3% |
| FinQA (by value) | Haiku + Sonnet agree | 2.9% | 23.5% | 33.1% | 3.9% |
| MATH-500 (exact) | Haiku + Sonnet agree | 2.8% | 54.7% | 14.5% | 0% |
| MATH-500 (by value) | Haiku + Sonnet agree | 3.8% | 68.4% | 1.6% | 0% |

Nothing certified at 2%. ARC is the first benchmark where a single cheap model
certified with no check at all. Its held-out loss is close to the limit.

### Shadow mode on new questions

`shadow.csv`. Each benchmark sent new questions through the proxy, sample rate 1:
200 each, or 100 for MATH-500.

| benchmark | answered by cheaper models (certified estimate) | loss vs Opus (limit 4%) | projected saving | drift check |
| --- | --- | --- | --- | --- |
| MedQA | 95.5% (93.5%) | 0.5% | 55.7% | within limit |
| ARC-Challenge | 100% (100%) | 2.5% | 97.4% | within limit |
| FinQA, exact | 51.0% (52.6%) | 1.0% | 10.0% | warning |
| FinQA, by value | 59.0% (66.9%) | 2.0% | 16.9% | warning |
| MATH-500, exact | 85.0% (85.5%) | 1.0% | 44.5% | warning |
| MATH-500, by value | 97.0% (98.4%) | 1.0% | 60.5% | within limit |

Every pre-registered check passed:

- the loss was at most 4% on every benchmark;
- the share answered by cheaper models was within 7 points of the estimate, apart
  from FinQA by value, which was not in the plan (7.9 points below);
- the reader matched at least 98% of every model's replies.

There were no provider errors and coverage was 100%.

**Two of the drift warnings were false alarms.** The drift check counts accepted
answers that differ from the frontier's, which runs above the loss.

- **MATH-500, exact.** The differences were formatting. Comparing by value cleared
  the warning: 2 of 97 accepted answers differed, and both did differ in
  correctness.
- **FinQA.** Even compared by value, 23 of 102 accepted answers differed from
  Opus's, but only 4 differed in being right or wrong. The rest were rounding
  (`14.5` against `14.53`, where the key is `14.5%`). The value comparison has no
  tolerance by design, so on rounded figures it warns early. Whether to allow a
  rounding tolerance there is open.

### Live routing with a random frontier holdout

`live-routing.csv`. `TINE_AUDIT_RATE=0.3`: 30% of requests, chosen at random, skipped
the policy and went to Opus. This was the first use of the holdout with real models.

| benchmark | routed / held out | accuracy: routed vs held out | mean cost per request: routed vs held out | measured saving (95% interval) |
| --- | --- | --- | --- | --- |
| MedQA | 105 / 45 | 97.1% vs 97.8% | $0.0056 vs $0.0128 | 56.3% (48.8-63.8%) |
| ARC-Challenge | 104 / 46 | 93.3% vs 100% | $0.00015 vs $0.0054 | 97.3% (97.0-97.6%) |
| FinQA (exact) | 96 / 37 | 76.0% vs 75.7% | $0.0093 vs $0.0114 | 18.3% (4.9-31.6%) |
| MATH-500 (exact) | 56 / 14 | 98.2% vs 92.9% | $0.0047 vs $0.0107 | not shown (fewer than 30 held out) |

How to read the accuracy columns:

- **They come from different questions.** The routed and held-out questions are
  not the same ones. With 14-46 held-out questions, the accuracy gaps are within
  noise.
- **The paired comparison is the shadow loss above.** There, both answers come from
  the same questions.
- **ARC's routed accuracy, 93.3%,** is in line with Haiku's 94.5% on the history.

### `tine serve` as a real process

Code at `2e56353`. Settings came from environment variables: route policy MedQA
(exact), holdout 0.3. 40 MedQA questions not in any other set were sent over HTTP
from a separate client.

- **All 40 requests answered.** Routing was on, and `/health` reported `policy`.
- **27 routed,** all answered by Haiku + Sonnet agreeing: 96.3% right, $0.0044
  each, 4.2s mean latency.
- **13 held out:** 100% right, $0.0145 each, 6.9s mean latency.
- **No holdout estimate shown.** Each group needs 30 requests, and the proxy said
  so.

This was the first time the proxy ran as a server rather than in-process.

## Deviations from the plan

1. **The first shadow and live runs were stopped for rate limits.** The four
   benchmarks and a first server test ran in parallel on one account. The
   provider refused part of the real requests with 429, so the runs were stopped.
   The share was 19-38% in the shadow runs (MATH-500 19 of 100, FinQA 53 of 200,
   MedQA 45 of 124, ARC 75 of 200), 25-58% in the routed runs (FinQA 14 of 57, ARC
   31 of 88, MATH-500 40 of 69) and 65% in the server test (26 of 40). An earlier
   version of this page said 19-45%; no run gives 45%, which was MedQA's count.
   - Spend: $5.32 recorded, including $0.13 for the stopped server test.
   - Rerun: one benchmark at a time, 3 requests in flight, with client retries.
     No retries were needed.
   - What it showed: shadow calls compete with real traffic for the same limits.
     The proxy now pauses shadow mode on a 429 or 529 (`docs/PROXY.md`, Rate
     limits). That pause was merged after these runs, so it was not exercised.
2. **Additions after the plan:**
   - the by-value comparison (certification and the shadow reruns);
   - the random frontier holdout in live routing;
   - the `tine serve` test.

   All three features were built during the run. Their results are reported apart
   and were not part of the pre-registered checks.
3. **Code versions:**
   - shadow and live reruns at `bc627ae`;
   - by-value shadow runs at `2b205e6`;
   - server test at `2e56353`.

   The changes between them do not affect exact-comparison policies.
4. **Gap-fill calls.** 24 of about 6,900 calibration calls hit a rate limit and
   were asked again ($0.13).

## Limits

- **Public benchmarks.** The models may have seen them. Each benchmark stands in
  for one customer's traffic, not for any real customer.
- **One afternoon.** Nothing here shows drift over time.
- **Small samples.** The holdout groups are small. The MATH-500 sets especially
  are small.
- **Answer format.** FinQA and MATH-500 depend on the models putting the answer in
  `\boxed{}`. They did in over 99% of replies.

## Spend (credits key)

| step | spent |
| --- | --- |
| probe and calibration (2,730 questions × 3 models, with gap-fill) | $37.65 |
| stopped runs (rate-limited) | $5.32 |
| shadow and live reruns | $12.25 |
| by-value shadow (FinQA, MATH-500) | $4.47 |
| `tine serve` test | $0.31 |
| **total** | **$60.00** |

## Reproduce

See `../2026-11-17-live-pilot/README.md`. Add:

- `--dataset {medqa,arcc,finqa,math500}` to `scripts/live_pilot.py sample`;
- `--answer-compare math` to `scripts/agreement_pairs.py` and `freeze-policy` for
  the by-value runs;
- `--mode route --audit-rate 0.3` for live routing with the holdout.

For the server test, start `python -m tine serve` with `TINE_ROUTE_POLICIES`,
`TINE_AUDIT_RATE`, `TINE_MODEL_PRICES` and `TINE_TENANT_KEYS` set, then send
requests to it over HTTP.
