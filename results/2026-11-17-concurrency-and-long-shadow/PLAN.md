# Rate limits under real refusals, and a longer shadow run: plan, 2026-11-17

Written before any model was called for this run. Results go in `README.md` next to
this file; anything that changes from this plan is listed there as a deviation.
Paid from the team's credits key. Hard cap for everything here: **$32**.

## Why

1. In `../2026-11-17-live-pilot-more` the provider refused 19-45% of real requests when
   shadow calls shared the account with them. The stored refusals say which limit it
   was: "Number of concurrent requests across all models has exceeded your organization's
   limit". The per-minute limits on this key are high (4,000 requests a minute per
   model, from the response headers), so the limit that binds is how many requests are
   in flight at once. The team cannot raise the tier. The proxy's new pause on a 429
   (`docs/PROXY.md`, Rate limits) has only been tested with a stand-in provider.
2. Every shadow result so far rests on 100-300 new questions. A longer run gives a
   tighter figure for the loss and shows how the report's numbers move as requests
   accumulate.

## Part 1: the concurrency limit and the pause (cap $3)

**A. Find the limit.** Send N `claude-haiku-5-5` requests at once that each take a few
seconds (`max_tokens` 300), for N = 8, 16, 24, 32, 48, 64, 96, 128. Record how many are
refused at each N, how long a refusal takes to come back, and whether it carries a
`retry-after` header. The limit is read from the largest count accepted at once.

**B. The pause under real refusals.** A background load holds (limit - 4) Haiku requests
in flight. Through the proxy in shadow mode (sample rate 1, MedQA policy), 3 requests in
flight, send 30 new MedQA questions twice, each time with 30 different questions:

- *before*: the proxy code at `bc627ae`, which has no pause;
- *after*: the current code, with the pause.

The real requests (3) fit under the limit; the shadow calls (up to 6 more) do not.
Measured for each: the share of real requests the provider refused, how long the 30
questions took, how many shadow runs completed, and what `/v1/shadow` reports
(`skipped_rate_limited`, `rate_limited_runs`, `shadow_paused`).

Pass marks, fixed now:
- *after* has fewer refused real requests than *before*, and under 10% of them;
- the report counts the skipped shadow runs, and coverage is below 100% and says why.

No pass mark is set for *before*: it may show few refusals at this load, and that is a
result too.

## Part 2: a longer shadow run (cap $27)

The 1,500 MMLU-Pro questions not used by the first pilot, through the proxy in shadow
mode with the policy certified there (`policy-eps0.04.json`: Haiku and Sonnet agree, else
Opus; certified at 4% and 2%), 3 requests in flight, on the current code. One run, a few
hours.

Fixed now:
- **Loss** over all 1,500, with a one-sided 95% upper bound (Clopper-Pearson). Pass: the
  bound is at most 4%. Also reported against 2%.
- **By block.** Loss, share answered by cheaper models, projected saving and the drift
  check's verdict in five consecutive blocks of 300, in the order sent. Reported, with
  the range across blocks; no pass mark. This is spread across a larger sample on one
  day. It is not drift over weeks, which only a real deployment can show.
- **Refusals.** Real requests refused, shadow runs skipped, and pauses, from the report.

Nothing here executes model output. Prompts and replies stay under
`artifacts/live-pilot/`; only summary tables go in this folder.
