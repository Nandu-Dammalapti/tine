# Judge designs for code: tiered, selector, calibration, failures, 2026-11-16

Offline results on recorded data. **No model was called, no paid call was made and no
patch was executed in this folder.** The judge verdicts are the recorded ones from
`results/2026-11-14-api-judge` (`claude-haiku-5-5` and `claude-sonnet-5-5` on the
1,110 patches that passed the free checks, paid then, not now). The grades are from the
Colab test run (`results/2026-11-10-swe-static-checks`). Small summary tables only:
no prompts, issue text, patches or per-task rows.

**Short answers.**

- **Tiered judge (Haiku first, Sonnet only when Haiku is unsure):** on the 1,110 cases
  it keeps Sonnet's catch rate at 43% to 82% of Sonnet's cost. In chains this barely
  matters. The judge is 2.5% of always-frontier spend with Sonnet, so halving it moves
  savings by about a point. Tiered chains save 31% to 35%, against 34% with Sonnet and
  38% to 39% with Haiku alone. Certified, one of the four settings matches Sonnet when half
  the tasks certify and certifies 20% on the default split, where Sonnet certifies nothing;
  with four settings tried, that is a hint, not a finding.
- **Judge as selector (call 2 or 3 cheap models, let the judge pick):** it does not beat
  trying models one by one. With Haiku, selector chains save 3 to 5 points more
  than the plain 3-stage chain, uncertified. But a plain chain allowed 5 stages saves
  about as much (checked after seeing the selector results). The gain comes from
  reaching more models, not from selection. With the same members, selection adds
  0.2 to 0.5 points of right patches for 24% to 47% more spend per stage. Certified, it
  is worse with Haiku. With Sonnet it is mixed: better on one split, worse on the other.
- **Calibration:** both judges rank patches well (AUC 0.90 and 0.93) and are
  overconfident between 0.8 and 0.95. Isotonic recalibration fixes the calibration
  error but does not change what the judge catches at a fixed false-reject rate.
  Weakest slices: patches written by some models (catch 53% to 68%), sympy, and large
  patches (false rejects 13% to 18%).
- **Failure reading:** 21 of 30 sampled Sonnet false rejects look like gaps in the
  hidden tests rather than judge errors: the judge named a real flaw that the tests do
  not check. Most false accepts are partial fixes or behaviour the tests check that
  the issue does not state. In 7 of 30 false accepts the judge named the gap and
  accepted anyway.

## Question

The judged chain (free checks, then a judge, one cheap model at a time) is about a
third cheaper than `claude-opus-4.1` and certifies 22% at the 4% limit when half the
tasks certify. Can the judge be used better? Four questions:

1. Can a cheap judge that defers to Sonnet when unsure keep Sonnet's quality at a
   fraction of its cost?
2. Does calling several cheap models at once and letting the judge pick the best patch
   beat trying them one by one, on cost and accuracy?
3. Is the judge's probability calibrated, does recalibration help, and where is the
   judge weakest?
4. Why is Sonnet wrong when it is wrong, and what change would fix it?

## Setup

- Data: the 274 valid SWE-bench Verified tasks (django and sympy), eight models,
  frontier `claude-opus-4.1`, grades from our test run (`run_resolved`). Free checks
  from the Colab run. Recorded verdicts of Haiku and Sonnet on all 1,110 cases (644
  right, 466 wrong).
- `scripts/judge_designs.py`, one subcommand per analysis. Chains: `python -m tine
  cascade`, epsilon 0.02, 0.03, 0.04, delta 0.05, 10 seeds, the default split
  (40 / 30 / 30) and 30 / 50 / 20, the same as the earlier folders.
- **Baseline first.** `baseline` rebuilt the 24 published Haiku and Sonnet rows of
  `results/2026-11-14-api-judge/cascade-judge-variants.csv` with `judge_patches.py
  checks` and `tine cascade`. All 24 match exactly (`baseline-reproduction.csv`).
- **The comparison rows in this folder are built one way for every design:** the
  frontier's own answers are never judged or charged. The published join added the
  judge's cost to the frontier's answers too. This moves the Sonnet rows by up to 0.4
  points (33.6% here against 34.0% there, default split, 4%), as the
  swe-agreement folder found. Haiku rows do not move.
- **Cost.** Savings are net of every model call and every judge call. A judge call is
  charged on every patch the judge reads. Free checks and latency are not charged, as
  in the earlier folders.
- **Paying once.** `tine cascade` gives a candidate one cost per task. A chain such as
  `a+b > a+c` would then pay for `a` twice, and for a second reading of `a`'s patch.
  The `saved` column pays each model once per task and chain, and judges each distinct
  patch once per task and chain. It reuses the cascade's search: same parts per seed,
  same candidate pool, same uncertified pick, same fixed-sequence certified pick.
  The same search charged the cascade's way must reproduce the cascade's summary on
  every variant and split, or the script stops; it did on all of them.
  `saved_cascade_charge` is the cascade's own figure. The two charges can pick
  different chains, so a certified figure can move either way between them.
- **Fixed before any chain was run:** the tiered grid HI in {0.9, 0.95}, LO in {0.1,
  0.3}. The selector's TAU: the judge's own verdict, and probability >= 0.9. The
  selector sets: pairs, triples, and mixed (singles, pairs and triples offered
  together). The oracle's definition. The false-reject targets for recalibration (5%
  and 10%) and the probability buckets. The failure sample's seed (20261116) and the
  category list.
- **Chosen after seeing results:** the longer sequential chains in Result 3 (`length`).
  Two failure labels were harmonised by me after reading (Result 5). The check of
  miscounted diffs in Result 5 was prompted by the readers' notes.

## Results

Limit 4% unless stated. Cells read: saved / accuracy / worst held-out run's loss /
runs with nothing certified (certified), or saved / accuracy / worst run
(uncertified). Frontier accuracy on the test parts: 50.3% (default split), 49.2%
(30 / 50 / 20). No row in this folder has a held-out run over the 4% limit unless
stated.

### Result 1: the tiered judge

`tiered-judge-quality.csv` (all 1,110 cases), `cascade-tiered.csv` (chains).

Haiku reads every patch that passed the free checks. Probability >= HI accepts,
<= LO rejects, and anything between, or a reply with no probability, goes to Sonnet,
whose verdict decides. Haiku is paid on every read, Sonnet only where consulted.

| judge | wrong rejected | right rejected | accepted that are right | Sonnet consulted | cost per check | vs Sonnet |
| --- | --- | --- | --- | --- | --- | --- |
| Haiku | 71.7% | 9.5% | 81.5% | - | $0.00053 | 7% |
| Sonnet | 79.4% | 9.0% | 85.9% | - | $0.00798 | 100% |
| Haiku, then Sonnet (published: both must accept) | 82.4% | 13.2% | 87.2% | 64.4% | $0.00546 | 68% |
| tiered HI 0.9, LO 0.1 | 78.3% | 8.2% | 85.4% | 41.9% | $0.00441 | 55% |
| tiered HI 0.9, LO 0.3 | 79.4% | 10.2% | 85.8% | 31.9% | $0.00345 | 43% |
| tiered HI 0.95, LO 0.1 | 80.3% | 9.9% | 86.3% | 75.2% | $0.00656 | 82% |
| tiered HI 0.95, LO 0.3 | 81.3% | 12.0% | 86.7% | 65.2% | $0.00560 | 70% |

- **At the judge level, yes.** HI 0.9 / LO 0.3 catches exactly as many wrong patches
  as Sonnet at 43% of its cost, and rejects 1.2 points more right ones. HI 0.9 / LO
  0.1 catches 1.1 points fewer, rejects 0.8 points fewer right ones, at 55% of the cost.
- Haiku's probability never exceeds 0.95 on these cases (23 at exactly 0.95). So with
  HI 0.95 Haiku accepts almost nothing on its own and Sonnet is consulted on 65% to 75%.

Chains:

| judge | certified, 40/30/30 | certified, 30/50/20 | uncertified, 40/30/30 | uncertified, 30/50/20 | judge cost, % of frontier spend |
| --- | --- | --- | --- | --- | --- |
| Haiku | 2.1% / 50.5% / 1.3% / 9 | 8.5% / 50.5% / 2.0% / 7 | **39.3%** / 54.8% / 2.6% | **37.9%** / 54.6% / 2.9% | 0.2% |
| Sonnet | 0.0% / 50.3% / - / 10 | **21.9%** / 52.7% / 1.9% / 3 | 33.6% / 56.1% / 2.6% | 34.3% / 55.1% / 1.9% | 2.5% |
| Haiku, then Sonnet | 3.9% / 50.6% / 0.0% / 9 | **22.5%** / 53.8% / 1.7% / 2 | 29.7% / 56.1% / 2.6% | 29.0% / 55.6% / 2.0% | 1.5% |
| tiered 0.9 / 0.1 | 0.0% / 50.3% / - / 10 | 9.1% / 51.4% / 3.8% / 7 | 35.0% / 55.5% / 3.8% | 33.7% / 54.6% / 3.8% | 1.4% |
| tiered 0.9 / 0.3 | 0.0% / 50.3% / - / 10 | 7.2% / 50.5% / 0.0% / 8 | 31.9% / 55.7% / 2.6% | 32.2% / 55.1% / 1.9% | 1.0% |
| tiered 0.95 / 0.1 | 0.0% / 50.3% / - / 10 | 15.5% / 51.9% / 1.9% / 5 | 31.8% / 55.8% / 2.4% | 32.2% / 54.9% / 1.9% | 2.0% |
| tiered 0.95 / 0.3 | **20.0%** / 54.1% / 1.2% / 4 | **21.1%** / 52.8% / 1.7% / 3 | 31.0% / 56.3% / 1.3% | 31.3% / 55.3% / 1.9% | 1.5% |

- **Nothing certifies at 2%.** At 3% on 30 / 50 / 20: tiered 0.95 / 0.3 6.5% (8 of 10
  runs empty), tiered 0.95 / 0.1 2.8% (9 empty), Sonnet 3.1% (9 empty), the others 0%
  or close.
- **In the chain, the cheaper judge does not buy much.** Sonnet's calls cost 2.5% of
  always-frontier spend in the chosen chains. The tiered settings cut that to 1.0% to
  2.0%, a saving of 0.5 to 1.5 points, and they let through a different mix of
  patches, which moves the chain more than the judge's cost does. Uncertified, every
  tiered setting saves 3.0 points less to 1.4 points more than Sonnet, and 4 to 8
  points less than Haiku alone.
- Tiered 0.95 / 0.3 is the only judge here that certifies a sizeable amount on the
  default split (20.0%, in 6 of 10 runs; Haiku and Haiku-then-Sonnet certify 2% to 4%
  there, Sonnet nothing). Its chains lost the fewest frontier-solved tasks (worst
  run 1.3%). That is one of four grid points, so treat it as a hint, not a finding. On
  30 / 50 / 20 it certifies the same as Sonnet (21.1% against 21.9%).
- Tiered 0.9 / 0.1 has the highest worst-run loss of any judge here (3.8%), within the
  4% limit but close.

### Result 2: the judge as selector (best of k)

`cascade-selector.csv` (chains), `selector-stages.csv` (stages outside certification).

A selector stage calls k cheap models (k = 2 or 3, from the seven non-frontier
models; `claude-opus-4.1` is never a member), runs the free checks, and the judge
reads every distinct patch that passed. It returns the patch with the highest judge
probability among those accepted at TAU (a tie goes to the cheaper model). Cost: all k
model calls plus every judge read. Success: the returned patch's grade. The plain
sequential chain is the same code with k = 1.

Stages, over every group of k models and all 274 tasks (shares of group-tasks; the
judge at its own verdict):

| judge, k | a right patch exists (oracle) | selector: right / wrong returned | selector cost, % of a frontier answer | one by one: right / wrong | one by one cost |
| --- | --- | --- | --- | --- | --- |
| Haiku, 1 | 29.7% | 26.7% / 5.9% | 5.0% | same | same |
| Haiku, 2 | 41.2% | 37.1% / 9.2% | 9.9% | 36.8% / 9.4% | 7.9% |
| Haiku, 3 | 47.8% | 42.8% / 11.4% | 14.8% | 42.3% / 11.9% | 10.1% |
| Sonnet, 1 | 29.7% | 27.0% / 4.3% | 6.1% | same | same |
| Sonnet, 2 | 41.2% | 37.7% / 6.7% | 12.2% | 37.5% / 6.9% | 9.8% |
| Sonnet, 3 | 47.8% | 43.8% / 8.4% | 18.3% | 43.5% / 8.7% | 12.8% |

- **The judge already closes most of the oracle gap: 88% to 92% of the right patches
  that exist are returned**, whether the stage selects or tries members in order. At
  probability >= 0.9 it is 60% to 61% with Haiku and 76% to 77% with Sonnet.
- **Selecting instead of trying in order adds 0.2 to 0.5 points of right patches and
  removes 0.2 to 0.5 points of wrong ones**, for 24% to 47% more spend on the stage.
  At probability >= 0.9 the right and wrong shares are the same either way.

Chains, paid once:

| judge, TAU | candidates | certified, 40/30/30 | certified, 30/50/20 | uncertified, 40/30/30 | uncertified, 30/50/20 |
| --- | --- | --- | --- | --- | --- |
| Haiku, verdict | one by one, 3 stages | 2.1% / 50.5% / 1.3% / 9 | 8.5% / 50.5% / 2.0% / 7 | 39.3% / 54.8% / 2.6% | 37.9% / 54.6% / 2.9% |
| | one by one, up to 5 stages (post hoc) | 2.1% / 50.5% / 1.3% / 9 | 6.2% / 49.7% / 2.0% / 8 | 45.1% / 57.4% / 3.0% | 42.6% / 56.1% / 2.9% |
| | pairs | 0.0% / 10 | 0.0% / 10 | 44.4% / 57.0% / 2.6% | 41.3% / 56.0% / 2.0% |
| | triples | 0.0% / 10 | 0.0% / 10 | 44.6% / 57.3% / 2.6% | 42.9% / 56.5% / 2.0% |
| | mixed | 0.0% / 10 | 0.0% / 10 | 44.5% / 57.1% / 2.6% | 41.8% / 56.2% / 2.0% |
| Haiku, >= 0.9 | one by one, 3 stages | 2.3% / 50.7% / 1.3% / 8 | 11.8% / 50.3% / 2.0% / 3 | 15.4% / 53.4% / 1.3% | 13.7% / 50.8% / 1.9% |
| | pairs | 8.5% / 52.5% / 1.3% / 4 | 13.4% / 51.0% / 2.0% / 1 | 15.4% / 53.3% / 1.3% | 14.6% / 51.4% / 2.0% |
| | triples | 6.0% / 51.4% / 1.3% / 6 | 11.2% / 51.1% / 2.0% / 2 | 15.0% / 53.4% / 1.3% | 14.3% / 51.6% / 2.0% |
| | mixed | 7.7% / 52.0% / 1.3% / 5 | 13.4% / 51.0% / 2.0% / 1 | 15.2% / 53.3% / 1.3% | 14.5% / 51.4% / 2.0% |
| Sonnet, verdict | one by one, 3 stages | 0.0% / 10 | 21.9% / 52.7% / 1.9% / 3 | 33.6% / 56.1% / 2.6% | 34.3% / 55.1% / 1.9% |
| | one by one, up to 5 stages (post hoc) | 0.0% / 10 | 3.6% / 49.4% / 1.4% / 9 | 36.7% / 58.3% / 1.3% | 35.3% / 56.6% / 1.9% |
| | pairs | 10.1% / 53.0% / 1.3% / 7 | 13.5% / 51.5% / 2.9% / 6 | 35.2% / 57.8% / 2.0% | 33.7% / 56.4% / 2.9% |
| | triples | 2.9% / 51.2% / 1.3% / 9 | 19.5% / 53.9% / 2.9% / 4 | 34.7% / 58.4% / 1.3% | 32.5% / 57.3% / 2.9% |
| | mixed | 10.0% / 53.0% / 1.3% / 7 | 18.1% / 52.9% / 2.9% / 4 | 34.8% / 57.8% / 1.3% | 33.3% / 56.1% / 2.9% |
| Sonnet, >= 0.9 | one by one, 3 stages | 7.6% / 51.6% / 1.1% / 7 | 19.4% / 52.6% / 1.5% / 1 | 21.4% / 54.5% / 1.3% | 20.1% / 53.2% / 0.0% |
| | pairs | 20.0% / 54.3% / 0.0% / 1 | 16.1% / 52.6% / 1.7% / 2 | 21.3% / 55.0% / 0.0% | 20.0% / 53.3% / 1.7% |
| | triples | 21.4% / 55.0% / 0.0% / 0 | 16.5% / 52.7% / 1.7% / 2 | 21.1% / 55.0% / 0.0% | 20.6% / 53.6% / 1.7% |
| | mixed | 22.0% / 54.9% / 0.0% / 0 | 17.0% / 52.9% / 0.0% / 2 | 21.9% / 54.9% / 0.0% | 20.8% / 53.6% / 1.5% |
| none: oracle (cannot run) | one by one, 3 stages | 21.8% / 55.1% / 1.2% / 3 | 30.5% / 57.2% / 0.0% / 0 | 30.6% / 58.1% / 1.2% | 30.5% / 57.2% / 0.0% |
| | pairs | 32.5% / 60.1% / 0.0% / 0 | 31.3% / 58.4% / 0.0% / 0 | 32.5% / 60.1% / 0.0% | 31.3% / 58.4% / 0.0% |
| | triples | 32.4% / 60.4% / 0.0% / 0 | 31.2% / 58.7% / 0.0% / 0 | 32.4% / 60.4% / 0.0% | 31.2% / 58.7% / 0.0% |

- **With Haiku, selector chains save 3 to 5 points more than the 3-stage chain
  (uncertified), and are 1 to 2.5 points more accurate.** But `tine cascade` tries
  chains of at most three stages, so a chain of three pairs reaches six models and a
  plain chain three. **A plain chain allowed up to 5 stages saves about as much**:
  45.1% and 42.6% against 44.4% and 41.3% for pairs and 44.6% and 42.9% for triples,
  at the same accuracy within half a point (`cascade-sequential-length.csv`, run after seeing the selector rows). With
  Sonnet, 5 stages give 36.7% and 35.3% against 35.2% and 33.7% for pairs. So the
  gain is the reach, and trying models one by one gets it for less.
- **Certified, the selector does not help with Haiku** (0% in every run at the
  verdict, against 2.1% and 8.5%). With Sonnet at the verdict it certifies more on
  the default split (10.1% against 0%) and less on 30 / 50 / 20 (13.5% to 19.5%
  against 21.9%). With Sonnet at >= 0.9, triples and mixed certify 21.4% and 22.0% in
  every run on the default split, against 7.6% for the plain chain, and less on 30 /
  50 / 20 (16.5% and 17.0% against 19.4%). Twelve selector rows were run against four
  baselines; one design winning on one split and losing on the other is not a result.
- Longer plain chains certify less, not more: more chains enter the fixed-sequence
  test and the picks lose more tasks. With Haiku and 6 or 7 stages one held-out run
  went over the 4% limit on 30 / 50 / 20 (4.3%).
- At 2% nothing selector-based certifies more than 2.4% (Sonnet, >= 0.9, 9 of 10 runs
  empty). At 3% on 30 / 50 / 20 the best is Sonnet, >= 0.9, mixed: 17.0% with 2 runs
  empty, against 11.3% for the plain chain with 6 empty.
- **The oracle is an accuracy ceiling, not a savings ceiling.** It never accepts a
  wrong patch, so it escalates to the frontier on tasks where the frontier also fails
  and the judge would have accepted a wrong patch at no counted loss (a degradation
  is a task the frontier solves and the chain misses). So oracle chains save less
  (31% to 33%) than judged chains, at 57% to 60% accuracy against 54% to 58%. With a
  perfect check, three-model selection would add about 2 points of accuracy over one
  model at a time.
- Paid once against charged by the cascade: up to 3.2 points more for selector chains
  at 4% (Haiku, mixed, 30 / 50 / 20: 41.8% against 38.6%; 3.6 points at 2%). At the
  verdict, 8 to 10 of the 10 uncertified selector picks reuse a model across stages.
- Not measured: latency. A selector stage can call its members in parallel, so it is
  faster than trying them in order. That is its one clear advantage.

### Result 3: calibration

`calibration-summary.csv`, `calibration-reliability.csv`, `calibration-target.csv`.

16 Haiku and 18 Sonnet replies have no probability. They are left out of Brier, AUC
and the reliability table and count as rejections everywhere else.

| judge | scores | Brier | AUC | calibration error (bucketed) |
| --- | --- | --- | --- | --- |
| Haiku | raw | 0.128 | 0.898 | 0.066 |
| Haiku | isotonic, held out | 0.126 | 0.885 | 0.026 |
| Sonnet | raw | 0.109 | 0.928 | 0.061 |
| Sonnet | isotonic, held out | 0.106 | 0.919 | 0.030 |

- **Both judges are overconfident between 0.8 and 0.95.** Haiku's 0.80 to 0.90 bucket
  (228 cases, mean 0.84) is right 69% of the time; Sonnet's (121 cases, mean 0.84) also
  69%, and its 0.90 to 0.95 bucket (185 cases, mean 0.93) 83%. The ends are good:
  Sonnet at 0.95 or more is right 96% of the time (353 cases), and below 0.1 right
  7% (305 cases).
- **Recalibration does not change what the judge catches.** Isotonic regression fitted
  on four folds of tasks and applied to the fifth halves the calibration error. With
  the threshold chosen on the training folds for a false-reject target, and the
  mean over 10 random fold assignments:

| judge | false-reject target | raw: wrong rejected / right rejected | recalibrated | own verdict |
| --- | --- | --- | --- | --- |
| Haiku | 5% | 57.4% / 4.6% | 54.9% / 4.4% | 71.7% / 9.5% |
| Haiku | 10% | 72.2% / 9.4% | 70.6% / 8.5% | |
| Sonnet | 5% | 66.8% / 5.1% | 65.3% / 4.7% | 79.4% / 9.0% |
| Sonnet | 10% | 80.6% / 9.5% | 79.7% / 9.4% | |

  Recalibration is monotone, so it cannot reorder patches; it only coarsens the
  thresholds available (ties), which costs 1 to 2.5 points of catch rate. The
  probability is useful as a ranking. Its absolute value is not a probability, but
  nothing in the chain needs it to be.
- Held-out thresholds land close to their targets (4.4% to 5.1% at 5%, 8.5% to 9.5% at
  10%). The judge's own verdict sits at about the 10% target, and a threshold chosen
  for 10% matches it or slightly beats it (Sonnet 80.6% against 79.4%).

### Result 4: where the judge is weakest

`judge-slices.csv`. Judge's own verdict. 95% Wilson intervals are in the file; most
slices have 50 to 150 wrong patches, so differences under about 10 points are within
noise.

| slice | cases (wrong / right) | Sonnet: wrong rejected / right rejected | Haiku |
| --- | --- | --- | --- |
| all | 466 / 644 | 79.4% / 9.0% | 71.7% / 9.5% |
| django | 325 / 531 | 81.5% / 8.9% | 72.9% / 8.9% |
| sympy | 141 / 113 | **74.5%** / 9.7% | 68.8% / 12.4% |
| written by `claude-opus-4.1` | 49 / 130 | **65.3%** / 7.7% | **53.1%** / 5.4% |
| written by `deepseek-r1-0528` | 47 / 98 | **68.1%** / 9.2% | **53.2%** / 11.2% |
| written by `glm-4.6` | 50 / 71 | **68.0%** / 8.5% | 64.0% / **16.9%** |
| written by `claude-sonnet-4` | 52 / 103 | 78.8% / 11.7% | 69.2% / 7.8% |
| written by `deepseek-v3.1-terminus` | 66 / 77 | 80.3% / 9.1% | 77.3% / 13.0% |
| written by `kimi-k2-0905` | 53 / 77 | 83.0% / 3.9% | 73.6% / 5.2% |
| written by `qwen3-235b-a22b-2507` | 70 / 68 | 82.9% / 8.8% | 77.1% / 8.8% |
| written by `deepseek-v3-0324` | 84 / 75 | 91.7% / 12.0% | 85.7% / 8.0% |
| patch of 1 to 2 changed lines | 120 / 250 | 80.8% / 6.0% | 68.3% / 5.6% |
| 3 to 6 lines | 172 / 253 | 80.2% / 9.5% | 70.9% / 8.3% |
| more than 6 lines | 174 / 141 | 77.6% / **13.5%** | 74.7% / **18.4%** |
| identical patch from 2 or more models | 4 / 50 | 1 of 4 / 8.0% | 1 of 4 / 6.0% |

- **Wrong patches from some writers are much harder to catch.** Sonnet catches 65% to
  68% of the wrong patches from `claude-opus-4.1`, `deepseek-r1-0528` and `glm-4.6`,
  against 83% to 92% from `kimi-k2-0905`, `qwen3-235b-a22b-2507` and
  `deepseek-v3-0324`. Haiku drops to 53% on the first two. A case written by several
  models counts in each of their rows. The frontier's patches are never judged in a
  chain; they are in the table as a reference.
- **sympy is harder** (74.5% against 81.5% with Sonnet), as the api-judge folder found.
- **Large patches are falsely rejected more**: 13.5% with Sonnet and 18.4% with Haiku
  above 6 changed lines, against 6% for 1 to 2 lines. Catch rate does not depend much on
  size. Tercile edges: 2 and 6 changed lines, over all 1,110 cases.
- Identical patches from several models are almost always right (50 of 54), so there
  is too little to say about the judge on them.

### Result 5: why Sonnet was wrong (30 false accepts, 30 false rejects)

`failure-categories.csv`. Drawn with seed 20261116 from Sonnet's 96 false accepts
and the 52 of its 58 false rejects that came with a reason (the 6 replies with no
verdict were not drawn). Each
case was read against the issue, the reference fix and the hidden tests, by four
readers (Claude sub-agents, 15 cases each, with one fixed category list), and
reviewed by me. The 30 false accepts come from 23 tasks and the 30 false rejects from
21, so several cases share a task. One category per case.

| false accepts (30) | cases | false rejects (30) | cases |
| --- | --- | --- | --- |
| partial fix | 10 | grade questionable (judge arguably right) | **21** |
| behaviour differs from what the tests check | 9 | judge misread the code | 5 |
| grade questionable | 8 | judge too strict | 2 |
| judge misread the code | 3 | issue ambiguous | 1 |
| | | right fix by a different route | 1 |

Paraphrased examples (no code, no issue text):

- *Partial fix.* The patch fixed the main code path but left the same fault in a
  sibling function or a second display path that the tests also check. The patch
  fixed one of two symptoms the issue describes. It changed one backend's code but not
  the shared caller that the tests exercise directly.
- *Behaviour differs from the tests.* The patch read a value strictly where the tests
  expect a missing value to be tolerated. It stopped a crash with an early guard,
  while the tests expect the input to be rejected earlier and the result fully
  simplified. Two patches copied a fix proposed in the issue that is wrong for inputs
  beyond the issue's example, and the judge checked only that example.
- *Grade questionable (false accepts).* The tests check a capability or an internal
  cache that the issue never mentions. Four patches look right in content, and
  readers suspected their diffs did not apply. A hunk-count check confirms a
  miscounted diff in two of the four, but 25 of the 48 miscounted patches among all
  1,110 still passed our test run, so failure to apply is not confirmed.
- *Judge misread the code.* It described what an unseen parent method or printer would
  do, and was wrong. On false rejects, it rejected the reference fix in effect,
  believing surrounding code still had the bug, or compared the patch with an upstream
  fix it remembered wrongly.
- *Grade questionable (false rejects).* The patch adds a condition that can never be
  true and silently removes an existing optimisation. It changes serialised output for
  every case rather than the one the issue is about. It crashes on single-line input
  that the issue's own example contains. In each case the judge named the flaw and
  the hidden tests do not check it. In 3 of the 21 the benchmark's own grade also
  says the patch fails, so our run's pass looks wrong there; 5 of the 30 sampled false
  rejects have that disagreement.
- *Judge too strict.* It rejected over an edge case outside the issue that the
  reference fix shares, or because the patch changes an old expected output that the
  reference also changes.

What this suggests:

- **The measured false-reject rate overstates the judge's real mistakes.** Only 9 of
  the 30 sampled false rejects look like judge errors (95% interval 17% to 48% of all
  false rejects, Wilson). The rest are flawed patches that the selected hidden tests do
  not catch. A chain that escalates them loses nothing a stricter test would credit.
- **In 7 of 30 false accepts the judge's own reason named the gap** (a missing case, an
  untouched sibling path, a backend the issue names) and it accepted anyway, guessing
  the tests would not cover it. A prompt rule, "if you name a gap the issue covers,
  reject", would target these. Readers' suggested changes: prompt in 9 false accepts
  and 6 false rejects, more context (callers, sibling functions, parent classes) in 7
  and 2, a second look in 2 and 1, nothing reasonable in 12 and 21.
- On false rejects the prompt change is the opposite: judge against the issue and the
  code shown, not a remembered upstream fix.
- 8 of 30 false accepts are hard to blame on the judge: the tests ask for more than the
  issue says, or the diff may not have applied. A strict apply check that matches the
  test harness would catch the second kind.

## What this means

- **Keep the plain chain: one cheap model at a time, free checks, then the judge.**
  Neither design beats it. The tiered judge is a real saving on the judge's bill, but
  the judge's bill is 2.5% of spend at most. The selector's apparent gain is reach,
  and a longer plain chain gets the same reach for less. A selector is worth it only
  if latency matters, which was not measured.
- **Which judge:** Haiku for the most uncertified saving (38% to 39%); Sonnet, or
  Haiku-then-Sonnet, where the goal is certification (22% at 4% with half the tasks
  certifying). If Sonnet's cost matters, tiered HI 0.9 / LO 0.1 has Sonnet's catch and
  false-reject rates within about a point, at 55% of its cost. Its chains are not
  distinguishable from Sonnet's on this data.
- **The cheapest gain found here is longer chains.** Allowing 4 or 5 stages raised the
  uncertified saving by 2.6 to 5.8 points with Haiku and 1.0 to 3.1 with Sonnet. That was
  seen after the selector results and certifies less, so it needs a test on fresh tasks.
- **The judge's errors point at the prompt and the grades, not the design.** Two
  prompt changes and more surrounding code are the next judge experiment (paid; about
  $9 for Sonnet on the 1,110 cases). Many false rejects, and some false accepts, are
  grading problems: the hidden tests check less than the issue, or more.
- Certified savings are still limited by the number of graded tasks.

## Limits

- 274 tasks from two repositories, single-shot patches, one benchmark. The certify part
  has about 82 or 137 tasks, so one failure moves a certified figure a lot.
- Many comparisons: 7 judges in Result 1, 16 judged candidate sets in Result 2 and
  five chain lengths. The best certified row of any family is optimistic.
- The selector's candidates are drawn as `tine cascade` draws them: the 8 candidates
  that save most alone on the choose part, so most of the 21 pairs and 35 triples are
  never tried in a chain.
- Paying once still assumes a model's answer and a judge's verdict are kept for the
  whole task. The free checks are not charged.
- Probability thresholds read the judges' stated probabilities, which are coarse
  (Haiku never above 0.95; most values are multiples of 0.05).
- The failure reading is a judgement by readers who saw the reference fix and the
  tests; the judge did not. "Grade questionable" means the readers found a flaw the
  tests do not check, without running any further test. Some labels rest on the
  readers' knowledge of code the packets do not show, marked medium confidence. The
  sample is clustered: several cases share a task.
- The judges' verdicts come from public issues a strong model may have seen.
- Latency not counted.

## Reproduce

```bash
# no model is called and no patch is executed; inputs default to artifacts/judge and artifacts/swe
python scripts/judge_designs.py baseline       # stops unless all 24 published rows match
python scripts/judge_designs.py tiered         # about 1 minute
python scripts/judge_designs.py selector       # about 6 minutes
python scripts/judge_designs.py length         # post hoc: plain chains of up to 7 stages, about 3 minutes
python scripts/judge_designs.py calibration
python scripts/judge_designs.py failures sample        # reading packets to artifacts/judge-designs/failures
# label the packets (packet, kind, category, secondary, confidence, note, suggested_change), then
python scripts/judge_designs.py failures summary --labels artifacts/judge-designs/failures/labels-reviewed.csv
```

Scratch inputs and the cascade outputs go to `artifacts/judge-designs/`; summary tables
to this folder. `--key`, `--cases`, `--haiku`, `--sonnet`, `--free-checks`,
`--outcomes`, `--work` and `--results` override the defaults.

## Read with the judge probes

This folder's recommendation was written before `results/2026-11-16-judge-probes`
finished. That folder found that the judges partly remember these public fixes, that
on tasks the judge model cannot recall the Haiku and Sonnet chains lose about 2.3% of
the frontier's solved tasks, and that `claude-opus-5-5`, with Haiku screening in
front of it, brings that loss to 0.46%. For code a customer owns, that two-step judge
is the recommendation in `docs/JUDGE.md`. The tiered and selector designs here were
not run with Opus.
