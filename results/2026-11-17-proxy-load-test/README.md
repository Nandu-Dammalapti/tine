# What the proxy adds under load, 2026-11-17

**No model was called and nothing was paid for.** `scripts/load_test.py matrix` sent
the same load to a stand-in provider directly and through Tine's proxy, on one Mac,
over loopback.

- **Spend:** $0.
- **Code:** the commit `2d1cca5`.
- **Scenarios:** 24. `matrix.csv` has 25 rows, because the ledger-growth scenario
  writes one row for each ledger size.
- **Data:** `matrix.csv` is the whole record. Every number below is read from it.
- **An earlier run the same day is not used.** A time limit cut it off after 10
  scenarios.
- **Three rows are flagged `busy_machine`.** Another job was running during the first
  three scenarios (`a-passthrough-instant-1`, `-10` and `-50`). The first of them
  agrees with the earlier run's clean figure of 0.67 ms at one request in flight.

## Bottom line

- **About 0.7 ms added at one request in flight** (median 0.695 ms, 95th percentile
  1.04 ms).
- **About 6 ms added at 10 in flight with 2 s replies**, and about 9 ms at 50 in
  flight.
- **About 1,100 requests a second is the ceiling for one process** when the provider
  answers at once (1,087 at one in flight, 1,182 at 10). It falls with more requests in
  flight and with larger prompts.
- **Over 100 requests in flight queue on the upstream pool.** The proxy holds 100
  connections to the provider. At 250 in flight each request waited about 3 s more. At
  400 in flight with a 3 s timeout, 1,992 of 2,532 requests got a 502.
- **Shadow mode and routing add about the same 6 to 7 ms** as passing a request
  through.
- **Shadowing every request does not keep up at 20 in flight.** At a sample rate of
  1.0 the shadow coverage was 0.34: 1,928 runs were skipped as busy.
- **The reports slow down as the ledger grows.** At 100,000 rows the savings report
  took 254 ms and the shadow report 420 ms, and a request sent while a report ran took
  268 ms.
- **No reply was changed.** `relayed_bytes_different` is 0 in every row that records
  it (the two routing rows do not).

## Results

"Added" is the latency through the proxy minus the latency of the same load sent
straight to the stand-in. `added_basis` in the table says how each row was worked out.

### One request in flight

The stand-in answers at once, so this is the cost of one request.

| scenario | prompt characters | added p50 ms | added p95 ms | requests a second through the proxy |
| --- | --- | --- | --- | --- |
| `a-passthrough-instant-1` (busy machine) | 2,000 | 0.695 | 1.043 | 1,087 |
| `a-bigprompt-50k-instant-1` | 50,000 | 1.183 | 1.328 | 654 |
| `a-bigprompt-200k-instant-1` | 200,000 | 3.231 | 6.730 | 236 |

- The median over the three rounds of the first row ran from 0.652 to 0.789 ms.
- The proxy used 0.64 ms of CPU per request in the first row.

### Replies of about 2 s

| scenario | in flight | added p50 ms | added p95 ms | added p99 ms |
| --- | --- | --- | --- | --- |
| `b-passthrough-realistic-10` | 10 | 6.097 | 9.280 | 12.667 |
| `b-passthrough-realistic-50` | 50 | 8.667 | 18.500 | 25.623 |

- Throughput through the proxy equalled the direct figure in both rows (ratios 1.004
  and 0.998).
- `b-bigprompt-200k-10` sent 200,000-character prompts 10 at a time, with replies of
  about 0.45 s. It added 13.5 ms at the median and 25.4 ms at the 95th percentile. The
  stand-in was itself queueing in this row (`direct_saturated`), so the table uses the
  `over_planned` basis for it.

### Streams

Added time to the first byte, and added time for the whole reply.

| scenario | in flight | added first byte p50 ms | added first byte p95 ms | added whole reply p50 ms | added whole reply p95 ms |
| --- | --- | --- | --- | --- | --- |
| `c-streams-instant` | 10 | 4.342 | 5.933 | 4.855 | 6.659 |
| `c-streams-realistic` | 20 | 6.542 | 10.432 | 4.442 | 14.464 |
| `c-streams-long-reply` | 20 | 2.004 | 5.600 | -54.611 | -25.357 |

- **The negative figures in the last row are noise of the paired estimate**, not a
  proxy that is faster than no proxy. The direct side of that row ran 106 ms over its
  planned time at the median, so the pairing had a noisy partner. Read the first-byte
  figures for that row and ignore the whole-reply ones.
- `c-streams-instant` was kept saturated, so its figures include queueing in the proxy.

### Shadow mode

20 in flight, replies of about 0.45 s.

| scenario | sample rate | added p50 ms | added p95 ms | shadow coverage | skipped as busy | skipped as rate limited |
| --- | --- | --- | --- | --- | --- | --- |
| `d-shadow-0.0` | 0 | 5.490 | 10.742 | - | - | - |
| `d-shadow-0.1` | 0.1 | 5.782 | 12.579 | 0.98 | 5 | 0 |
| `d-shadow-1.0` | 1.0 | 7.052 | 13.734 | 0.34 | 1,928 | 0 |
| `d-shadow-rate-limited` | (provider refuses some) | 5.864 | 11.902 | 0.0007 | 9 | 2,880 |

- **Shadow runs do not slow the caller.** The added latency at a rate of 1.0 is 1.6 ms
  above the rate of 0.
- **They are dropped when the proxy is busy.** At 1.0 about a third of the sampled
  requests were shadowed. The report counts the rest as skipped.
- **Under provider refusals shadow mode stands back.** In the last row the stand-in
  answered some requests with 429 and 529 (the proxy relayed 73 and 75 of them among
  2,756), and nearly every shadow run was skipped as rate limited.
- Peak memory was 87 to 88 MB at rates 0.1 and 1.0, against 62 MB at 0.

### Routing through a policy

20 in flight. A routed request calls other models, so its latency is not comparable
with the direct load. The comparison is on the random share the proxy held out and
sent to the frontier (`holdout_added_p50_ms`).

| scenario | held out | added p50 ms on held-out requests | added p95 ms | routes |
| --- | --- | --- | --- | --- |
| `e-route-agree-0.9` | 202 | 6.005 | 12.877 | stage-0 1,777, frontier 194, holdout 202 |
| `e-route-agree-0.3` | 152 | 6.188 | 10.707 | stage-0 402, frontier 951, holdout 152 |

- A request that fell through to the frontier took about 980 ms at the median in both
  rows, against about 530 ms for one answered at the first stage and 444 to 474 ms for a
  held-out one. That is the price of asking the cheap models first, and it comes from
  the stand-in's reply times, not from a real model.

### Provider errors and slow replies

`f-errors-and-slow`, 20 in flight.

- Added 5.967 ms at the median and 10.499 ms at the 95th percentile on paired requests.
- Statuses through the proxy: 1,522 of 200, 50 of 429, 67 of 500, 192 of 502 and 60 of
  529. The table does not say which 502s the proxy made itself.
- The 99th percentile was 606 ms through the proxy and 6,280 ms direct.

### Soak

`h-soak`: 20 requests a second, 6,000 requests through the proxy.

- All 6,000 returned 200.
- Added 6.405 ms at the median, 10.523 ms at the 95th percentile, 12.979 ms at the 99th.
- Peak memory of the proxy process was 61.0 MB.

### The ceiling

The stand-in answers at once and the load is kept saturated, so latency in these rows
is time spent queueing, not the cost of a request.

| scenario | in flight | requests a second through the proxy | added p50 ms | busy machine |
| --- | --- | --- | --- | --- |
| `a-passthrough-instant-1` | 1 | 1,087 | 0.695 | yes |
| `a-passthrough-instant-10` | 10 | 1,182 | 7.640 | yes |
| `a-passthrough-instant-50` | 50 | 796 | 44.761 | yes |
| `a-passthrough-instant-200` | 200 | 430 | 455.722 | no |
| `c-streams-instant` | 10 | 845 | 4.855 | no |

- **About 1,100 requests a second for one process**, with 2,000-character prompts.
- It is lower with 200 in flight, where the proxy opened 10,401 upstream connections
  for 18,455 requests.

### Ledger growth

The ledger was filled to a set number of rows, then the reports were timed, and a
request was timed while a report ran.

| scenario | ledger rows | ledger bytes | savings report ms | shadow report ms | request ms while a report runs | peak memory MB |
| --- | --- | --- | --- | --- | --- | --- |
| `g-ledger-growth-10000` | 10,063 | 56,791,872 | 29.30 | 42.42 | 19.12 | 102.8 |
| `g-ledger-growth-100000` | 100,063 | 541,163,872 | 253.74 | 419.56 | 267.67 | 238.4 |
| `g-ledger-large-rows-2000` | 2,063 | 139,472,808 | 25.31 | 8.92 | 14.03 | 91.1 |

- **Ten times the rows cost about ten times the time and the disk.** About 5.4 KB a
  row with 2,000-character prompts, and about 68 KB a row with 50,000-character
  prompts and 2,000-token replies.
- **A request waits while a report runs.** A request with nothing else going on took
  about 1.6 ms at the median in these rows.

### More requests in flight than the upstream pool holds

The proxy keeps 100 connections to the provider. Replies of about 2 s.

| scenario | in flight | proxy timeout | statuses | added p50 s | added p95 s |
| --- | --- | --- | --- | --- | --- |
| `i-pool-queue` | 250 | default | 200: 1,502 | 3.06 | 3.30 |
| `i-pool-timeout` | 400 | 3 s | 200: 540, 502: 1,992 | 4.21 | 4.74 |

- **At 250 in flight every request succeeded and waited about 3 s** for a connection.
  Throughput was 50 requests a second against 125 direct.
- **At 400 in flight with a 3 s timeout, 79% of requests got a 502** (1,992 of 2,532).
  Only 740 requests reached the stand-in.

## What it means

- **The proxy's own cost is small beside a model's reply.** 6 to 9 ms on a 2 s reply
  is under half a percent.
- **One process is enough for a pilot's request rate.** The ceiling is far above the
  25 requests in flight the team's own provider key allows
  (`../2026-11-17-concurrency-and-long-shadow`).
- **The first limit a busy tenant meets is the 100-connection upstream pool**, not the
  CPU. Past it requests queue, and with a short timeout they fail.
- **Shadowing every request needs spare capacity.** Plan on a sample rate below 1.0,
  and read the coverage in the shadow report.
- **Keep the ledger small or read reports off a copy.** At 100,000 rows a report
  holds a request up for a quarter of a second.

## Limits

- **Not a capacity promise.** One laptop, one run, one day.
- **Same machine.** The load generator, the stand-in provider and the proxy shared
  the Mac's cores. The proxy's ceiling here is partly the load generator's.
- **Loopback.** No network, no TLS, no gateway in front.
- **A stand-in provider.** Its reply times are set by the scenario. No real provider's
  latency, streaming or refusals were measured.
- **The first three rows ran on a busy machine.** Only the one-in-flight figure has a
  clean run to agree with. The ceilings at 10 and 50 in flight do not.
- **The policy is synthetic.** The shadow and routing scenarios use a made-up
  agreement policy. They measure the proxy's work, not a saving or a loss.
- **SQLite on a laptop's disk.** The ledger figures depend on that disk.
- **The soak is five minutes.** It does not show a leak that takes hours.
- **Not run in the container.** The proxy ran as a local process.

## Reproduce

```sh
python scripts/load_test.py matrix --output results/NEW-FOLDER
```

It writes one JSON file per scenario and the combined `matrix.csv`. It calls no model.
Close other work first: a busy machine sets `busy_machine`, and such a row is not a
measurement.
