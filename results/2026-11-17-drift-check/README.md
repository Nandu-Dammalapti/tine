# The drift check on recorded runs: comparing with epsilon against comparing with what the certification allows for, 2026-11-17

Shadow mode's drift check reads accepted cheaper answers that differ from the frontier's
(`docs/PROXY.md`, The drift check). The first version warned when the differing share of
shadowed requests was above the policy's epsilon. Differing is not the loss, so it warned
on healthy traffic. This folder measures the first rule and the one that replaces it on
every recorded shadow run, and on the same runs with drift injected.

No model was called and no paid call was made. Everything was replayed from recorded
answers and grades that are local files (`artifacts/`, not in git). The folder holds
counts and rates only: no prompt, no answer and no per-request row.

## Question

1. Does the new rule stay quiet on the recorded runs whose graded loss is under the limit?
2. Does it still catch cheaper models that agree on wrong answers?
3. How large must a loss be before it shows, and after how many requests?

## The two rules

- **Old** (commit `2341a23`): warn when differing answers / shadowed requests is above
  epsilon, once 30 accepted answers were compared.
- **New** (this branch). Two shares are checked: differing answers over the requests the
  policy was run on, and over the accepted answers alone. Each is compared with what the
  certification allows for: epsilon plus the 95% upper bound on the share of graded tasks
  whose accepted answer differed from the frontier's without being a loss, recorded in the
  policy file at freezing (`drift_baseline`).
  - **Warning**: a share is clearly above its allowance. "Clearly" is a test that stays
    valid however often the report is read (evidence of 20 needed; at most a 5% chance of
    a warning at any read when both shares keep to their allowance). The evidence is
    weighed for all requests and again from every 250th request on.
  - **Notice**: no warning, and a share is clearly above what the graded tasks themselves
    showed (their differing answers, losses included, at the 95% upper bound). A notice
    is not an alarm. It says the traffic is unlike the graded tasks.
  - **Default**: a policy file without a baseline (every file frozen before this change)
    gets epsilon plus 10 points of requests, and a third of accepted answers. No notice.

## Setup

- **Runs.** 13 live shadow runs through the real proxy (`results/2026-11-17-live-pilot`,
  `results/2026-11-17-live-pilot-more`, `results/2026-11-17-ticket-pilot`): MMLU-Pro (317
  requests, then five blocks of 300), MedQA, ARC-Challenge, FinQA and MATH-500 with exact
  comparison and by value, and real IT support tickets. Beside them the xRouteBench
  replay of `results/2026-11-16-shadow-replay` (1,414 requests, the policy's test part)
  and the same policy's choose part (1,922).
- **Graded loss.** Each request was graded with the benchmark's answer key
  (`scripts/live_pilot.py`), so the true loss of every run is known. The proxy does not
  see it.
- **Baselines.** The recorded policy files have no baseline. Each was frozen again from
  its recorded graded history (`scripts/agreement_pairs.py`, then `tine freeze-policy`);
  every re-frozen file has the same stages, guarantee and estimate as the original.
  "Baseline" below is the re-frozen file, "default" the original file.
- **Checks on the replay.** `ShadowStore.report` on copies of the 13 recorded ledgers and
  `GET /v1/shadow` of the proxy app in the xRouteBench replay give the same verdicts as
  the function called directly.
- **Injected drift.** An accepted answer is replaced by a wrong answer that the cheaper
  models agree on. The share replaced is set so the expected loss hits a multiple of the
  limit. "40% agree wrong" replaces the answer of every request with chance 0.4, accepted
  or not: the case of `results/2026-11-16-simulated-pilots`.
- **Lengths.** "Recorded length" keeps the recorded requests in order. 1,000 and more are
  drawn with replacement from the run. 200 runs per cell (400 for repeated reads), so 0%
  means under about 1.5%.
- Cells read **old rule / new with baseline: warning / new with baseline: warning or
  notice / new with default: warning**, in percent of runs.

## Results

### 1. Recorded runs: the false warnings are gone

`recorded-runs.csv`. "quiet" is no warning and no notice.

| run | limit | requests | accepted | differ | differ share | lost / graded | graded loss | old rule | new, baseline (allowance; evidence for a warning; for a notice; 20 needed) | new, default (allowance; evidence) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MMLU-Pro first pilot | 4% | 317 | 268 | 9 | 2.84% | 6 / 297 | 2.02% | quiet | quiet (5.89%; 0.0962; 0.213) | quiet (14.00%; 0.0402) |
| MMLU-Pro long-1 | 4% | 300 | 273 | 9 | 3.00% | 8 / 300 | 2.67% | quiet | quiet (5.89%; 0.102; 0.192) | quiet (14.00%; 0.048) |
| MMLU-Pro long-2 | 4% | 300 | 283 | 17 | 5.67% | 10 / 300 | 3.33% | WARNS | quiet (5.89%; 0.104; 0.612) | quiet (14.00%; 0.0447) |
| MMLU-Pro long-3 | 4% | 300 | 277 | 11 | 3.67% | 6 / 300 | 2.00% | quiet | quiet (5.89%; 0.0537; 0.0871) | quiet (14.00%; 0.038) |
| MMLU-Pro long-4 | 4% | 300 | 281 | 14 | 4.67% | 9 / 300 | 3.00% | WARNS | quiet (5.89%; 0.0717; 0.188) | quiet (14.00%; 0.041) |
| MMLU-Pro long-5 | 4% | 300 | 276 | 7 | 2.33% | 4 / 300 | 1.33% | quiet | quiet (5.89%; 0.055; 0.0705) | quiet (14.00%; 0.039) |
| MedQA | 4% | 200 | 191 | 1 | 0.50% | 1 / 200 | 0.50% | quiet | quiet (6.35%; 0.0167; 0.0181) | quiet (14.00%; 0.0137) |
| ARC-Challenge | 4% | 200 | 200 | 7 | 3.50% | 5 / 200 | 2.50% | quiet | quiet (5.13%; 0.0382; 0.0505) | quiet (14.00%; 0.0161) |
| FinQA exact | 4% | 200 | 102 | 25 | 12.50% | 2 / 200 | 1.00% | WARNS | quiet (19.20%; 0.0363; 0.0499) | quiet (14.00%; 0.073) |
| FinQA by value | 4% | 200 | 118 | 29 | 14.50% | 4 / 200 | 2.00% | WARNS | quiet (21.70%; 0.0404; 0.049) | quiet (14.00%; 0.0989) |
| MATH-500 exact | 4% | 100 | 85 | 5 | 5.00% | 1 / 100 | 1.00% | WARNS | quiet (11.36%; 0.0499; 0.0668) | quiet (14.00%; 0.0379) |
| MATH-500 by value | 4% | 100 | 97 | 2 | 2.00% | 1 / 100 | 1.00% | quiet | quiet (5.31%; 0.0464; 0.0656) | quiet (14.00%; 0.0294) |
| IT tickets | 4% | 300 | 300 | 32 | 10.67% | 12 / 300 | 4.00% | WARNS | quiet (8.00%; 1.1; 2.6) | quiet (14.00%; 0.109) |
| MMLU-Pro long-1..5 as one history | 4% | 1,500 | 1,390 | 58 | 3.87% | 37 / 1,500 | 2.47% | quiet | quiet (5.89%; 0.0126; 0.0872) | quiet (14.00%; 0.0054) |
| MMLU-Pro all six as one history | 4% | 1,817 | 1,658 | 67 | 3.69% | 43 / 1,797 | 2.39% | quiet | quiet (5.89%; 0.0112; 0.0719) | quiet (14.00%; 0.00512) |
| MMLU-Pro first pilot | 2% | 317 | 268 | 9 | 2.84% | 6 / 297 | 2.02% | WARNS | quiet (3.89%; 0.235; 0.275) | quiet (12.00%; 0.0753) |
| MMLU-Pro long-1..5 as one history | 2% | 1,500 | 1,390 | 58 | 3.87% | 37 / 1,500 | 2.47% | WARNS | quiet (3.89%; 0.0811; 0.167) | quiet (12.00%; 0.0108) |
| MMLU-Pro all six as one history | 2% | 1,817 | 1,658 | 67 | 3.69% | 43 / 1,797 | 2.39% | WARNS | quiet (3.89%; 0.0676; 0.138) | quiet (12.00%; 0.0101) |
| xRouteBench replay, test part | 2% | 1,414 | 1,099 | 58 | 4.10% | 14 / 1,414 | 0.99% | WARNS | quiet (5.23%; 0.0437; 0.0844) | quiet (12.00%; 0.0138) |
| xRouteBench replay, test part | 4% | 1,414 | 1,099 | 58 | 4.10% | 14 / 1,414 | 0.99% | WARNS | quiet (7.23%; 0.0127; 0.0442) | quiet (14.00%; 0.0069) |
| xRouteBench replay, choose part | 2% | 1,922 | 1,499 | 74 | 3.85% | 26 / 1,922 | 1.35% | WARNS | quiet (5.23%; 0.0291; 0.0555) | quiet (12.00%; 0.00977) |
| xRouteBench replay, choose part | 4% | 1,922 | 1,499 | 74 | 3.85% | 26 / 1,922 | 1.35% | quiet | quiet (7.23%; 0.00873; 0.0291) | quiet (14.00%; 0.00489) |

- The old rule warns on 12 of these 22 rows: on 6 of the 13 live runs at the limit they
  were run with, on the three MMLU-Pro rows at 2% and on three of the four xRouteBench
  rows, the replay at its own 2% limit among them.
- The new rule gives no warning and no notice on any row, with the baseline or with the
  default. The largest evidence for a warning is 1.1 and for a notice 2.6 (both IT
  tickets), against the 20 needed.
- Read after every single request in recorded order, the new rule never warns and never
  gives a notice on any run (table in section 5).
- The MMLU-Pro rows at 2% put the same requests against the 2% file, which has the same
  stages. Their loss (2.0% to 2.5%) is at or just over 2%, so the old rule's warning
  there is not clearly false.
- The xRouteBench test part is half of that policy's own baseline, so that row is not an
  independent test.
- IT tickets is not a healthy run that the rule leaves alone. Its loss is at the limit
  (12 of 300), and its 10.67% is above its own allowance of 8.00%. It is quiet only
  because 300 requests are too few to say so clearly (section 5).

`baselines.csv`:

| policy | graded tasks (certify + test) | accepted and compared | differed | differed without a loss | 95% upper bound | allowance at 4%, of requests | of accepted answers | graded tasks' own share, at most | in the shadow run: differed without a loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MMLU-Pro | 692 | 612 | 16 | 7 (1.01%) | 1.89% | 5.89% | 6.66% | 3.49% | 3 of 297 (1.01%) |
| MedQA | 556 | 512 | 13 | 7 (1.26%) | 2.35% | 6.35% | 6.90% | 3.69% | 0 of 200 (0.00%) |
| ARC-Challenge | 556 | 556 | 15 | 2 (0.36%) | 1.13% | 5.13% | 5.13% | 4.12% | 2 of 200 (1.00%) |
| FinQA exact | 554 | 294 | 77 | 70 (12.64%) | 15.20% | 19.20% | 35.79% | 16.55% | 23 of 200 (11.50%) |
| FinQA by value | 554 | 357 | 95 | 83 (14.98%) | 17.70% | 21.70% | 33.43% | 20.00% | 25 of 200 (12.50%) |
| MATH-500 exact | 227 | 197 | 11 | 10 (4.41%) | 7.36% | 11.36% | 13.07% | 7.89% | 4 of 100 (4.00%) |
| MATH-500 by value | 227 | 223 | 2 | 0 (0.00%) | 1.31% | 5.31% | 5.41% | 2.75% | 1 of 100 (1.00%) |
| IT tickets | 692 | 692 | 38 | 19 (2.75%) | 4.00% | 8.00% | 8.00% | 7.14% | 20 of 300 (6.67%) |
| xRouteBench triples | 2,840 | 2,190 | 105 | 76 (2.68%) | 3.23% | 7.23% | 9.37% | 4.33% | 44 of 1,414 (3.11%) |

### 2. Cheaper models agreeing on wrong answers is still caught

`injected-drift.csv`.

From the start, 40% of requests: every rule warns in 200 of 200 runs for every policy at
every length, down to the 100 requests of the MATH-500 runs.

Starting half way through the run and climbing to 40% at the last request:

| run | limit | recorded length | 1,000 | 3,000 |
| --- | --- | --- | --- | --- |
| MMLU-Pro first pilot | 4% | 100 / 100 / 100 / 63 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| MMLU-Pro long-1..5 | 4% | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| MedQA | 4% | 100 / 13 / 80 / 0 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| ARC-Challenge | 4% | 100 / 92 / 100 / 0 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| FinQA exact | 4% | 100 / 0 / 0 / 12 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| FinQA by value | 4% | 100 / 0 / 0 / 46 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| MATH-500 exact | 4% | 100 / 0 / 15 / 0 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| MATH-500 by value | 4% | 100 / 32 / 82 / 0 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| IT tickets | 4% | 100 / 100 / 100 / 91 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| xRouteBench test part | 4% | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| xRouteBench choose part | 4% | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| MMLU-Pro first pilot | 2% | 100 / 100 / 100 / 82 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| MMLU-Pro long-1..5 | 2% | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| xRouteBench test part | 2% | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| xRouteBench choose part | 2% | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |

At 1,000 requests and more every policy warns in every run, with the baseline and with
the default. Before the evidence was weighed from every 250th request the same climb at
1,000 requests gave 8% and 1% on FinQA, 39% on MATH-500 exact and 0% to 4% with the
default. At the size of a pilot (100 to 300 requests, the climb covers 50 to 150 of them)
it is still mostly missed on FinQA, MATH-500 and MedQA.

- A sudden change to 40% after 0 to 20,500 healthy requests is caught within 500
  requests in every run (`late-change.csv`).
- Mixed traffic, where only part of the requests is of the certified kind and the rest
  is tried and never accepted: with 20,000 requests, 15% to 35% of the certified kind and
  44% to 100% of the accepted answers differing, the rule warns in each of five cases
  through `ShadowStore`, with a baseline and without, where the share of requests alone
  stayed under the allowance. Healthy mixed traffic (3% of accepted answers differing on
  the MMLU-Pro policy, 24% on FinQA) gives nothing.

### 3. A loss of twice the limit: seldom a warning at 1,000 requests, mostly a notice

| run | limit | recorded length | 1,000 | 3,000 | 10,000 | 30,000 |
| --- | --- | --- | --- | --- | --- | --- |
| MMLU-Pro first pilot | 4% | 100 / 22 / 94 / 0 | 100 / 76 / 100 / 0 | 100 / 100 / 100 / 0 | 100 / 100 / 100 / 0 | 100 / 100 / 100 / 0 |
| MMLU-Pro long-1..5 | 4% | 100 / 99 / 100 / 0 | 100 / 90 / 100 / 0 | 100 / 100 / 100 / 0 | 100 / 100 / 100 / 0 | 100 / 100 / 100 / 0 |
| MedQA | 4% | 99 / 0 / 36 / 0 | 100 / 11 / 98 / 0 | 100 / 55 / 100 / 0 | 100 / 99 / 100 / 0 | 100 / 100 / 100 / 0 |
| ARC-Challenge | 4% | 100 / 20 / 48 / 0 | 100 / 95 / 100 / 0 | 100 / 100 / 100 / 0 | 100 / 100 / 100 / 0 | 100 / 100 / 100 / 0 |
| FinQA exact | 4% | 100 / 0 / 1 / 1 | 100 / 1 / 36 / 78 | 100 / 0 / 94 / 100 | 100 / 2 / 100 / 100 | 100 / 9 / 100 / 100 |
| FinQA by value | 4% | 100 / 0 / 0 / 4 | 100 / 0 / 6 / 94 | 100 / 0 / 30 / 100 | 100 / 0 / 98 / 100 | 100 / 2 / 100 / 100 |
| MATH-500 exact | 4% | 100 / 0 / 0 / 0 | 100 / 1 / 94 / 0 | 100 / 6 / 100 / 0 | 100 / 32 / 100 / 0 | 100 / 93 / 100 / 0 |
| MATH-500 by value | 4% | 98 / 10 / 54 / 0 | 100 / 96 / 100 / 0 | 100 / 100 / 100 / 0 | 100 / 100 / 100 / 0 | 100 / 100 / 100 / 0 |
| IT tickets | 4% | 100 / 96 / 100 / 0 | 100 / 100 / 100 / 2 | 100 / 100 / 100 / 10 | 100 / 100 / 100 / 66 | 100 / 100 / 100 / 100 |
| xRouteBench test part | 4% | 100 / 100 / 100 / 0 | 100 / 92 / 100 / 0 | 100 / 100 / 100 / 0 | 100 / 100 / 100 / 0 | 100 / 100 / 100 / 0 |
| xRouteBench choose part | 4% | 100 / 100 / 100 / 0 | 100 / 78 / 100 / 0 | 100 / 100 / 100 / 0 | 100 / 100 / 100 / 0 | 100 / 100 / 100 / 0 |
| MMLU-Pro first pilot | 2% | 100 / 0 / 2 / 0 | 100 / 8 / 32 / 0 | 100 / 46 / 84 / 0 | 100 / 98 / 100 / 0 | 100 / 100 / 100 / 0 |
| MMLU-Pro long-1..5 | 2% | 100 / 36 / 88 / 0 | 100 / 18 / 48 / 0 | 100 / 81 / 98 / 0 | 100 / 100 / 100 / 0 | 100 / 100 / 100 / 0 |
| xRouteBench test part | 2% | 100 / 58 / 100 / 0 | 100 / 30 / 77 / 0 | 100 / 90 / 100 / 0 | 100 / 100 / 100 / 0 | 100 / 100 / 100 / 0 |
| xRouteBench choose part | 2% | 100 / 25 / 99 / 0 | 100 / 10 / 47 / 0 | 100 / 45 / 96 / 0 | 100 / 98 / 100 / 0 | 100 / 100 / 100 / 0 |

Of the 15 rows, a steady loss of twice the limit from the start gives:

| requests | a warning in at least 80% of runs | a warning or a notice in at least 80% | default: a warning in at least 80% |
| --- | --- | --- | --- |
| 1,000 | 5 | 9 | 1 |
| 3,000 | 9 | 14 | 2 |
| 10,000 | 12 | 15 | 2 |
| 30,000 | 13 | 15 | 3 |

**The new rule misses a loss of twice the limit at 1,000 requests on most policies.** A
warning shows in at least 80% of runs on 5 of 15 rows. With the notice counted it is 9 of
15. The six rows where neither shows in 80% of runs at 1,000 requests are FinQA exact
(36%), FinQA by value (6%) and the four rows at a 2% limit (32% to 77%). On FinQA a
warning comes in at most 9% of runs at any length up to 30,000 requests. With the default
allowance a loss of twice the limit shows only where harmless differences happen to fill
the default 10 points (FinQA) and, late, on IT tickets.

The old rule warns in 100% of these runs. It also warns on the same runs with nothing
injected (section 5), so its warning does not tell the two apart.

The smallest steady loss each rule catches in at least 80% of runs
(`smallest-loss-caught.csv`):

| run | limit | 1,000 requests | 3,000 requests |
| --- | --- | --- | --- |
| MMLU-Pro first pilot | 4% | 1x / 2.5x / 1.5x / 4.5x | 1x / 2x / 1.5x / 4x |
| MMLU-Pro long-1..5 | 4% | 1x / 2x / 1.5x / 4.5x | 1x / 2x / 1x / 4x |
| MedQA | 4% | 1.5x / 2.5x / 2x / 5x | 1.5x / 2.5x / 1.5x / 4.5x |
| ARC-Challenge | 4% | 1x / 2x / 1.5x / 4.5x | 1x / 1.5x / 1.5x / 4x |
| FinQA exact | 4% | 1x / 3.5x / 2.5x / 2x | 1x / 3x / 2x / 1.5x |
| FinQA by value | 4% | 1x / 3.5x / 3x / 2x | 1x / 3x / 2.5x / 1.5x |
| MATH-500 exact | 4% | 1x / 3x / 2x / 4x | 1x / 2.5x / 1.5x / 3.5x |
| MATH-500 by value | 4% | 1x / 2x / 1x / 4.5x | 1x / 1.5x / 1x / 4x |
| IT tickets | 4% | 1x / 1.5x / 1x / 3x | 1x / 1x / 1x / 2.5x |
| xRouteBench test part | 4% | 1x / 2x / 1.5x / 4x | 1x / 2x / 1x / 3.5x |
| xRouteBench choose part | 4% | 1x / 2.5x / 1.5x / 4.5x | 1x / 2x / 1x / 4x |
| MMLU-Pro first pilot | 2% | 1x / 3x / 3x / 8x | 1x / 2.5x / 2x / 7x |
| MMLU-Pro long-1..5 | 2% | 1x / 3x / 2.5x / 7.5x | 1x / 2.5x / 2x / 6.5x |
| xRouteBench test part | 2% | 1x / 3x / 2.5x / 7x | 1x / 2x / 1.5x / 6x |
| xRouteBench choose part | 2% | 1x / 3x / 2.5x / 7x | 1x / 2.5x / 2x / 6x |

Why a warning cannot do better, whatever the length: the differing share is the loss plus
the harmless differences. The allowance is epsilon plus the upper bound on the harmless
share at freezing. A loss below "allowance minus the harmless share of the traffic" keeps
the expected share under the allowance for ever. On these runs that loss is 1.0 to 1.6
times the limit for MMLU-Pro, MedQA, ARC, MATH-500 by value and xRouteBench, and 1.8 to
2.3 times for FinQA and MATH-500 exact, where 4% to 15% of requests differ harmlessly and
the bound on that share is 3 points above it.

### 4. A loss at the limit, which the guarantee allows

`injected-drift.csv`, level "1x eps". A warning here is a false one.

- With the baseline: no warning in any run at any length, except IT tickets (42% at
  1,000 requests, 98% at 3,000), whose harmless differences more than doubled against
  its graded history.
- A notice comes often: 28% to 90% of MMLU-Pro runs, 44% to 100% for xRouteBench and 80%
  to 100% for MATH-500 by value, at 1,000 to 3,000 requests. A loss at the limit is
  three to four times the loss on those policies' graded tasks, and the notice says that
  the traffic differs from them. It does not say the limit is broken.
- With the default: 13% of FinQA by value runs warn at 1,000 requests and 62% at 3,000
  (FinQA exact 0% and 3%). Its healthy differing share, 14.5%, is already above the
  default 14%.
- The old rule warns in 40% to 100% of these runs.

### 5. Healthy runs read again and again

`healthy-repeated-reads.csv`. Nothing injected.

| run | limit | requests | recorded order, read after every request: first at request | shuffled, every 50 | resampled to 1,000, every 50 | resampled to 3,000, every 50 | resampled to 10,000, every 500 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MMLU-Pro first pilot | 4% | 297 | 40 / never / never / never | 28 / 0 / 0 / 0 | 43 / 0 / 0 / 0 | 40 / 0 / 0 / 0 | 10 / 0 / 0 / 0 |
| MMLU-Pro long-1..5 | 4% | 1,500 | 472 / never / never / never | 82 / 0 / 0 / 0 | 76 / 0 / 2 / 0 | 81 / 0 / 4 / 0 | 72 / 0 / 7 / 0 |
| MedQA | 4% | 200 | never / never / never / never | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| ARC-Challenge | 4% | 200 | 30 / never / never / never | 38 / 0 / 0 / 0 | 58 / 0 / 0 / 0 | 62 / 0 / 0 / 0 | 34 / 0 / 0 / 0 |
| FinQA exact | 4% | 200 | 56 / never / never / never | 100 / 0 / 0 / 0 | 100 / 0 / 0 / 0 | 100 / 0 / 0 / 0 | 100 / 0 / 0 / 0 |
| FinQA by value | 4% | 200 | 40 / never / never / never | 100 / 0 / 0 / 0 | 100 / 0 / 0 / 1 | 100 / 0 / 0 / 3 | 100 / 0 / 0 / 4 |
| MATH-500 exact | 4% | 100 | 34 / never / never / never | 100 / 0 / 0 / 0 | 97 / 0 / 0 / 0 | 100 / 0 / 0 / 0 | 100 / 0 / 0 / 0 |
| MATH-500 by value | 4% | 100 | never / never / never / never | 0 / 0 / 0 / 0 | 11 / 0 / 0 / 0 | 9 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| IT tickets | 4% | 300 | 30 / never / never / never | 100 / 0 / 6 / 0 | 100 / 55 / 85 / 0 | 100 / 98 / 100 / 0 | 100 / 100 / 100 / 0 |
| xRouteBench test part | 4% | 1,414 | 64 / never / never / never | 100 / 0 / 0 / 0 | 84 / 0 / 0 / 0 | 90 / 0 / 0 / 0 | 92 / 0 / 0 / 0 |
| xRouteBench choose part | 4% | 1,922 | 683 / never / never / never | 84 / 0 / 0 / 0 | 74 / 0 / 0 / 0 | 83 / 0 / 0 / 0 | 70 / 0 / 0 / 0 |
| MMLU-Pro first pilot | 2% | 297 | 36 / never / never / never | 100 / 0 / 0 / 0 | 100 / 0 / 0 / 0 | 100 / 0 / 0 / 0 | 100 / 0 / 0 / 0 |
| MMLU-Pro long-1..5 | 2% | 1,500 | 32 / never / never / never | 100 / 0 / 0 / 0 | 100 / 1 / 3 / 0 | 100 / 1 / 4 / 0 | 100 / 0 / 13 / 0 |
| xRouteBench test part | 2% | 1,414 | 33 / never / never / never | 100 / 0 / 0 / 0 | 100 / 0 / 0 / 0 | 100 / 0 / 0 / 0 | 100 / 0 / 0 / 0 |
| xRouteBench choose part | 2% | 1,922 | 374 / never / never / never | 100 / 0 / 0 / 0 | 100 / 0 / 0 / 0 | 100 / 0 / 0 / 0 | 100 / 0 / 0 / 0 |

- New rule with the baseline: no warning at any read on any run except IT tickets (55%
  of runs by 1,000 requests, 98% by 3,000, 100% by 10,000) and 1% on MMLU-Pro at 2%.
- Notices on healthy traffic: MMLU-Pro 2% to 7% of runs at 4% and 3% to 13% at 2%, where
  the live loss (2.47%) is about twice the loss on the graded tasks (1.3%); IT tickets
  85% by 1,000.
- New rule with the default: no warning except FinQA by value (1%, 3% and 4%).
- Old rule: a warning at some read on 13 of 15 rows in recorded order.

**IT tickets.** On the 692 graded tickets, Sonnet's accepted answer differed from Opus's
without a loss on 19 (2.75%); on the 300 shadowed tickets on 20 (6.67%). The graded loss
went from 2.9% to 4.0%, which is at the limit and not over it. So the traffic did differ
from the graded history, and the rule cannot tell that from a loss over the limit. A
longer ticket run at these rates would get a notice and then a warning. Whether that is
wanted is a product decision (open questions).

False alarms at the edge of what the rule allows, simulated (`false-alarms.csv`): 300
streams of 20,000 requests per row, read every 25 requests. Each accepted answer differs
at a fixed rate: exactly the allowed one, or the graded tasks' own.

| policy | limit | each accepted answer differs with chance | streams with a warning at any read | with a warning or a notice |
| --- | --- | --- | --- | --- |
| MMLU-Pro | 4% | 6.66% (at the allowance) | 1.7% |  |
| MMLU-Pro | 4% | 2.61% (like the graded tasks) | 0.0% | 0.3% |
| MMLU-Pro at 2% | 2% | 4.40% (at the allowance) | 1.0% |  |
| MMLU-Pro at 2% | 2% | 2.61% (like the graded tasks) | 0.0% | 0.0% |
| FinQA exact | 4% | 35.79% (at the allowance) | 4.7% |  |
| FinQA exact | 4% | 26.19% (like the graded tasks) | 0.0% | 0.0% |
| IT tickets | 4% | 8.00% (at the allowance) | 1.3% |  |
| IT tickets | 4% | 5.49% (like the graded tasks) | 0.0% | 0.0% |
| no baseline, 88% accepted | 4% | 15.91% (at the allowance) | 1.3% |  |

The bound for the rows "at the allowance" is 5%.

### 6. Other drift

- **Noise with no change in right or wrong** (`injected-drift.csv`, "noise"): 10% of the
  matching accepted answers made to read as different warns in 82% to 100% of the
  multiple-choice runs at 1,000 requests and in none of the FinQA runs; 25% warns
  everywhere at 3,000. Limited to rewritings of a number that the policy's comparison
  really reads as different, 10% gives no warning at any length (a notice on MATH-500)
  and 25% warns in 50% to 100% of runs at 1,000 requests and in 98% to 100% at 3,000. The rule sees only "same" or "different". The
  old rule warns in every one of these cases.
- **A sudden change to three times the limit after a healthy history**
  (`late-change.csv`): with the baseline, a warning within 1,000 requests in every run at
  a 4% limit, after up to 20,500 healthy requests (median 200 to 500 requests); at a 2%
  limit 15% to 91% within 1,000 and 98% to 100% within 3,000. With the default it is
  never seen within 3,000.
- **A chain with a named-check stage before an agreement stage** (synthetic graded
  history of 6,000 tasks, no real policy has this shape): traffic drawn like the history
  warned in 70% of streams by 3,000 requests before this change, because the baseline
  left the named check's tasks out while epsilon is a share of all tasks. Now 0 of 200
  streams warn within 10,000 requests, and 1 of 200 gets a notice.

## Findings from review, and what was done

Three reviews of the first version of the new rule (commit `ff49699`) raised 22
findings. Each was reproduced on that commit before any change.

| # | Finding | State |
| --- | --- | --- |
| 1, 3, 8 | With a baseline, a loss of twice the limit seldom warns at 1,000 requests, never on FinQA, and MedQA is worse than under the old rule | Real. A warning cannot do better (section 3). Added the notice, which shows a loss of twice the limit in at least 80% of runs on 9 of 15 rows at 1,000 requests and on all 15 at 10,000, MedQA included (98% at 1,000). Still missed at 1,000 requests: FinQA, and every policy at a 2% limit |
| 2 | With the default allowance nothing under about four times the limit shows | Real, not fixed. A policy file without a baseline says nothing about its harmless differences. Measured and documented; freezing again records a baseline and now keeps the requests already shadowed |
| 4 | Drift that climbs in the second half of a run of 1,000 requests is missed | Fixed: the evidence is weighed from every 250th request on, not every 1,000th. The climb to 40% is caught in every run at 1,000 requests. Still missed at pilot size on FinQA, MATH-500 exact and MedQA |
| 5, 11, 22 | The ticket run warns when extended, with the loss at the limit | Real, not changed. The test and the docs no longer list it as a healthy run that stays quiet; they say it is quiet at 300 requests only |
| 6 | With the default, healthy FinQA by value can warn | Real, not fixed (1% to 4% of healthy runs; 13% to 62% with the loss at the limit). Its harmless differences are above the default 10 points |
| 7 | Noise that changes nothing right or wrong warns once it passes the slack in the baseline | Real, inherent: the proxy sees only same or different. Documented |
| 9, 15, 18 | Freezing a policy again restarted the check, dropped a standing warning, and the page mixed counts of two files | Fixed. Shadow rows keep a second digest of how their policy runs (frontier, stages, answer pattern, comparison); a file that runs the same way counts them. Rows stored before the second digest count once the proxy has started with their file. The differing count beside the matching count covers every request again; the check's own counts have their own fields, and the page shows how many requests the check left out |
| 10 | Mixed traffic: total failure on the certified part read "compatible" | Fixed: the share among accepted answers is checked too, and requests sent to the frontier untried (tools) are left out of the share of requests |
| 12 | A chain with a named check before an agreement stage warned on traffic like its own history | Fixed: tasks and requests accepted by a named check stay in both denominators |
| 13 | Traffic in batches by topic breaks the 5% bound | Documented. Not changed |
| 14 | A warning leaves no trace when it goes away | Not fixed. Documented with measured durations (a 200-request episode shows as a warning for a median of 775 requests after it ends, 60 simulated streams) |
| 16 | "compatible with the certification" said too much | Fixed: the result reads "no clear sign of drift", adds that only a large change shows where harmless differences were common, and the counts the finding lists now get a notice |
| 17, 20 | The page described a default 10-point check for policies with only named checks and beside reports saved before the change | Fixed |
| 19 | A baseline count of 10**30 passed validation and then failed every report | Fixed: counts above 1,000,000,000 are refused |
| 21 | `AGENTS.md` section 12 and `docs/EVIDENCE.md` still describe the first rule | Not changed here: this task was told not to edit those files. They need a line each |

## What this means

- The false warnings that led to this change are gone on every recorded run, and the one
  failure the check is for, cheaper models agreeing on wrong answers, is caught in every
  run, also under mixed traffic and when it starts late in a run of 1,000 requests.
- The check is not a measurement of the loss, and it is weak near the limit. A warning
  needs a loss of about twice the limit and several thousand requests on most policies,
  and three times the limit where harmless differences are common. The notice shows
  smaller changes, at the price of also showing a loss that rose within the limit.
- A policy file without a baseline gets a rough check that shows gross failure only.
  Freeze such files again before relying on the check.
- To know the loss on live traffic, grade a sample. Nothing here replaces that.

## Limits

- The lengths beyond the recorded ones are drawn with replacement from 100 to 1,922
  recorded requests. They have no order effects and say nothing about a customer's
  traffic. Only MMLU-Pro (1,500 and 1,817) and xRouteBench have real histories over
  1,000 requests.
- All runs but one are public benchmark questions. The ticket run is one company's
  tickets answered as multiple choice.
- The constants (95% for the baseline bound, 10 points and a third for the default, 20
  for the evidence, 250 for the block) were chosen with these runs in view, so "quiet on
  the recorded runs" is in-sample.
- xRouteBench has no answer key in the local files, only grades per model. Its injected
  wrong answers differ from a right frontier answer always and from a wrong one 2 times
  in 3 (four options).
- The share of the ticket history's `--max-cost-share` is not recorded; 0.9 was used and
  reproduces the stored policy.
- The mixed-chain and mixed-traffic cases use synthetic counts. No real policy or traffic
  of those shapes exists yet.
- The notice was not part of the reviewed version. It has had no review of its own.
- Shadow mode has never run on a customer's traffic.

## Reproduce

The inputs are local files under `artifacts/live-pilot` and `artifacts/xroutebench`
(recorded answers and grades; not in git). The replay script is about 800 lines tied to
those files, so it is not in `scripts/`. It does the following, with `tine` imported from
this branch and both provider keys unset:

1. Freeze each policy again from its recorded history: `scripts/agreement_pairs.py` on
   the recorded answers (with `--answer-compare math` for the by-value sets), then
   `tine freeze-policy` with the original seed and epsilon. Check that stages, guarantee
   and estimate equal the stored file.
2. Per recorded run, read the proxy's stored rows (accepted, matches the frontier) and
   grade each request with `scripts/live_pilot.py` (`score_proxy`, `grade`). For
   xRouteBench, replay the recorded answers through the proxy app as
   `scripts/shadow_replay.py` does.
3. Call the old rule (`drift` of `git show 2341a23:tine/shadow.py`) and the new one
   (`tine.shadow.drift` with `tine.shadow.drift_scopes`, as `ShadowStore.report` calls
   them) on the rows in stored order, and on the rows with drift injected as described
   under Setup, 200 seeds per cell.

The unit tests carry the counts of every recorded run and assert the verdicts above:

```
python -m pytest tests/test_drift.py -q
```
