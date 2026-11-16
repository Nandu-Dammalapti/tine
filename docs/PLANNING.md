# Planning: how many graded tasks certification needs

Written 2026-11-16. Everything here is arithmetic on the certification test. No model
was called and no new data was used. The numbers are exact given their assumptions;
the assumptions are the weak part, and they are listed.

## Question

A chain certifies when the one-sided Clopper-Pearson bound on its degradation rate,
measured on the certify part, is at most epsilon (`tine.cascade.certify_chain`). The
code chain on SWE-bench certifies at 4% only when half the 274 tasks go to the certify
part, and never at 2% (`results/2026-11-14-api-judge`). How many graded tasks would
it take to certify at 2%, 3% and 4%?

## Setup

`tine/planning.py` answers this from two facts:

- For a certify part of `n` tasks there is a largest number of degradations that
  still passes. `max_failures(n, epsilon, delta)` returns it, using the same bound
  function as the certifier. `tasks_needed(epsilon, delta, failures)` is the reverse:
  the smallest `n` at which that many failures pass.
- If tasks are independent and the chain's true degradation rate is `r`, the number
  of degradations in `n` tasks is binomial. `pass_probability(n, epsilon, delta, r)`
  is the chance it is at most `max_failures`.

`tasks_for_power` finds the certify size that gives a chosen chance of passing.
`plan` and `total_tasks_for_power` convert between certify tasks and graded tasks in
total for a split. `rate_estimate` gives a rate from a count, with confidence limits.
`sequence_pass_bounds` covers the fixed-sequence test (see Limits).

A degradation is a task the frontier solves and the chain misses. The rate is per
task, not per frontier-solved task, because that is what the certifier bounds.
Delta is 0.05 throughout.

Two details matter more than they look:

- **More tasks can lower the chance of passing.** One more allowed failure arrives
  in steps (at 4%: 74, 117, 156, 192 certify tasks). Between steps, extra tasks only
  add chances to fail. So each "tasks needed" figure comes in two forms: the first
  size that reaches the target, and the size from which it never drops below again.
  The tables below use the second.
- **The code splits tasks at random.** `run_cascade_experiment` and
  `tine freeze-policy` assign each task to a part independently, so a "30%" certify
  part of 274 tasks is 82 on average, not exactly. "Fixed split" below means the
  certify part has exactly its share. "Random split" is the exact average over the
  sizes the current code would produce.

## Results

### The test itself

Smallest certify part that passes with k degradations (delta 0.05):

| epsilon | k = 0 | k = 1 | k = 2 | k = 3 | k = 4 |
| --- | --- | --- | --- | --- | --- |
| 0.02 | 149 | 236 | 313 | 386 | 456 |
| 0.03 | 99 | 157 | 208 | 257 | 303 |
| 0.04 | 74 | 117 | 156 | 192 | 227 |

This reproduces the earlier write-up exactly:

- 82 certify tasks (40/30/30 of 274): zero degradations give a bound of 3.59%. One
  gives 5.65% and fails the 4% limit.
- 137 certify tasks (30/50/20 of 274): one degradation gives 3.42% and passes at 4%.
  Zero gives 2.16%, so nothing can pass at 2%.
- 2% needs 149 certify tasks with zero degradations, 236 with one.

### The rate we are planning for

On all 274 tasks at once the judge chain lost 2 tasks the frontier solves: 0.73% per
task. The one-sided 95% limits on that rate are 0.13% and 2.28%.

This estimate is weak in two ways. Two events say little about a rate. And the chain
was looked at on the same 274 tasks, so the count is optimistic and no interval
corrects for that. On held-out test parts the chosen chains lost 0.3% to 0.7% of
tasks on average over ten seeds (0.7% on 40/30/30, 0.3% on 30/50/20), with a worst
run of 2.6%; those seeds reuse the same
274 tasks and are not independent. So the tables give 0.73% as the main case and
0.5%, 1% and 1.5% beside it.

### What 274 tasks can do

Chance that one chain with a true rate of 0.73% certifies:

| split of 274 | certify tasks | epsilon | degradations allowed | fixed split | random split |
| --- | --- | --- | --- | --- | --- |
| 40 / 30 / 30 | 82 | 0.02 | none | 0% | 0.0% |
| | | 0.03 | none | 0% | 0.8% |
| | | 0.04 | 0 | 54.8% | 47.3% |
| 30 / 50 / 20 | 137 | 0.02 | none | 0% | 2.7% |
| | | 0.03 | 0 | 36.7% | 37.1% |
| | | 0.04 | 1 | 73.6% | 73.6% |

These figures are for one chain tested alone. That is not what the measured
experiment counted. It certified something at 4% in 7 of 10 seeds on 30/50/20 and in
0 of 10 on 40/30/30, but "something" is whichever chains came first in the order,
and that was rarely the chain with 2 of 274. So the 7 of 10 does not confirm the
74% here, though the two are close.

### Check against the measured experiment

Added on review. The certifier's ordering was re-run on the recorded outcomes for
the same ten seeds, following the chain with 2 of 274 (4% limit; 212 to 259 chains
entered the sequence in a seed):

| split of 274 | passes its own test | this model, random split | chains ordered before it | reached and certified |
| --- | --- | --- | --- | --- |
| 40 / 30 / 30 | 4 of 10 | 47.3% | 0 to 209, median 60.5 | 0 of 10 |
| 30 / 50 / 20 | 6 of 10 | 73.6% | 1 to 242, median 51.5 | 1 of 10 |

The arithmetic describes the chain's own test about as well as ten seeds can show.
It does not describe whether the chain is certified. The sequence stopped at the
first chain in every 40/30/30 seed, and after at most three chains in 8 of the 10
30/50/20 seeds. The chains at the front had no degradation on the choose part, which
is why they were at the front, and 1 to 4 on the whole pool. Ten seeds over the same
274 tasks are not independent, so these counts are an illustration, not an estimate.
No script in the repository prints this table; it took a few lines around
`tine.cascade.candidate_chains` and `chain_metrics`.

### Graded tasks needed for an 80% chance

Certify tasks needed, then graded tasks in total under each split. "Cannot" means the
rate is at or over the limit, where the chance of passing never exceeds 5%.

| epsilon | true rate | certify tasks | total, 40/30/30 fixed | total, 40/30/30 random | total, 30/50/20 fixed | total, 30/50/20 random |
| --- | --- | --- | --- | --- | --- | --- |
| 0.02 | 0.50% | 386 | 1,287 | 1,301 | 772 | 779 |
| 0.02 | **0.73%** | **655** | **2,184** | **1,969** | **1,310** | **1,181** |
| 0.02 | 1.00% | 1,091 | 3,637 | 3,397 | 2,182 | 2,042 |
| 0.02 | 1.50% | 4,646 | 15,487 | 14,974 | 9,292 | 8,984 |
| 0.02 | 2.28% | cannot | cannot | cannot | cannot | cannot |
| 0.03 | 0.50% | 208 | 694 | 677 | 416 | 408 |
| 0.03 | **0.73%** | **257** | **857** | **856** | **514** | **514** |
| 0.03 | 1.00% | 392 | 1,307 | 1,160 | 784 | 696 |
| 0.03 | 1.50% | 726 | 2,420 | 2,250 | 1,452 | 1,351 |
| 0.03 | 2.28% | 3,349 | 11,164 | 10,771 | 6,698 | 6,462 |
| 0.04 | 0.50% | 117 | 390 | 412 | 234 | 245 |
| 0.04 | **0.73%** | **156** | **520** | **522** | **312** | **313** |
| 0.04 | 1.00% | 192 | 640 | 647 | 384 | 388 |
| 0.04 | 1.50% | 326 | 1,087 | 1,008 | 652 | 603 |
| 0.04 | 2.28% | 752 | 2,507 | 2,327 | 1,504 | 1,396 |

Reading it for the code chain, if its true rate is the observed 0.73% and it is
tested alone (the two figures are the random and the fixed split):

- **4%:** 312 to 313 graded tasks with 30/50/20, 520 to 522 with 40/30/30. We have 274.
- **3%:** 514 with 30/50/20, 856 to 857 with 40/30/30.
- **2%:** 1,181 to 1,310 with 30/50/20, 1,969 to 2,184 with 40/30/30.
  SWE-bench Verified has 500 tasks, so 2% is out of reach on that set alone at this
  rate. With all 500 graded and 30/50/20 the chance at 2% is 42% to 45%.
- The answer is very sensitive to the true rate. At 1% the 2% limit needs 2,042 to
  2,182 tasks with 30/50/20; at 1.5%, 8,984 to 9,292. If the true rate is at the upper
  confidence limit of 2.28%, the 2% limit cannot be certified at any size.
- These are lower limits for the procedure as it runs today, and the check above
  shows the gap is large. See the first two points under Limits.

With all 500 SWE-bench Verified tasks graded, one chain at 0.73% would pass with:

| split | epsilon 0.02 | epsilon 0.03 | epsilon 0.04 |
| --- | --- | --- | --- |
| 40 / 30 / 30, fixed (random) | 33% (18%) | 33% (43%) | 70% (76%) |
| 30 / 50 / 20, fixed (random) | 45% (42%) | 72% (77%) | 96% (97%) |

The first two cells of the top row are equal because 150 certify tasks allow zero
degradations at both limits.

### When earlier chains must pass too

The certifier tests chains in a fixed order and stops at the first failure. The
worst case for m earlier chains with the same true rate is that their failures never
overlap; then each of the m + 1 chains needs a pass chance of 1 - 0.2 / (m + 1) for
80% overall. Certify tasks, then totals for a fixed 40/30/30 and 30/50/20 split, at a
true rate of 0.73%:

| earlier chains | epsilon 0.02 | epsilon 0.03 | epsilon 0.04 |
| --- | --- | --- | --- |
| 0 | 655: 2,184 / 1,310 | 257: 857 / 514 | 156: 520 / 312 |
| 1 | 782: 2,607 / 1,564 | 348: 1,160 / 696 | 192: 640 / 384 |
| 3 | 969: 3,230 / 1,938 | 392: 1,307 / 784 | 227: 757 / 454 |
| 9 | 1,151: 3,837 / 2,302 | 478: 1,594 / 956 | 294: 980 / 588 |
| 49 | 1,508: 5,027 / 3,016 | 604: 2,014 / 1,208 | 358: 1,194 / 716 |

In the ten-seed experiment the chain had 50 to 60 chains ahead of it in a typical
seed, so the last row is the scale of the current procedure. It is still not a
forecast: it is a worst case over how failures overlap, and it assumes the earlier
chains are as good as this one, which the chains at the front were not shown to be.

## Limits

- **Tasks are assumed independent and alike.** SWE-bench tasks come in groups from
  the same repository, and new tasks may be harder or easier than the 274 graded so
  far. Neither is modelled.
- **The true rate is not known.** It rests on 2 events, on tasks the chain was also
  examined on. Treat the 0.73% rows as the hopeful case and read the 1% and 1.5%
  rows before committing to a run.
- **The fixed-sequence test makes every figure above optimistic, and by a lot.** A
  chain is only tested if every chain ordered before it passed. Chains are ordered
  by their degradation on the choose part, lowest first, so a chain that looked
  clean there by luck and is worse in truth can come early, fail, and block the
  rest. With over 200 chains in the sequence, the chain planned for here had 50 to
  60 ahead of it in a typical seed. It was certified in 1 of 20 runs, though it
  passed its own test in 10 of 20 (see the check above). The single-chain probability is therefore
  an upper limit on the chance of certifying, and the task counts are lower limits.
  `sequence_pass_bounds` gives the range for a known prefix: between
  `1 - sum(1 - p)` and `min(p)`. Chains share models and tasks, so their failures
  overlap, but that overlap is not measured here. The "earlier chains" table is the
  worst case for chains of equal quality; it does not cover an earlier chain that
  is worse.
- **The figures apply as they stand only if one chain is named before the new tasks
  are graded.** Then the sequence has one entry, the new tasks need no choose part,
  and the "certify tasks" column is the number of new graded tasks: 156 at 4%, 257
  at 3%, 655 at 2%, if the true rate is 0.73%. `tine cascade` and
  `tine freeze-policy` do not work this way today. Both build the candidate list
  from a choose part of the same tasks.
- **The choose step is not modelled.** The plan assumes the chain we want is a
  candidate and is picked. A smaller choose part (30/50/20 gives it 30%) makes that
  less sure and the savings estimate behind the choice noisier.
- **A random split's "stays above" total is exact to a tolerance of 1e-9 in
  probability**, not to the last digit, because the search stops once smaller certify
  parts are that unlikely.
- **80% is a planning target chosen here**, not a project rule. Pass `--power` for
  another.
- This says nothing about savings. A chain that certifies may be a cautious one.

## Reproduce

```bash
python scripts/plan_certification.py --epsilon 0.02 0.03 0.04 --observed 2 274 \
  --rate 0.005 0.01 0.015 --total 274 --split 40/30/30 30/50/20 --earlier-chains 3
python scripts/plan_certification.py --epsilon 0.02 0.03 0.04 --observed 2 274 --total 500
pytest tests/test_planning.py
```

The first command takes about half a minute; rates close to the limit need long
searches. The other rows of the "earlier chains" table come from the same command
with `--earlier-chains 1`, `9` and `49`. `--certify 82 137` describes certify sizes
directly. In Python:

```python
from tine.planning import plan, tasks_for_power, total_tasks_for_power
plan(274, 0.5, 0.04, true_rate=2 / 274)["pass_probability"]             # 0.736
tasks_for_power(0.02, 0.05, 2 / 274, stable=True)                       # 655
total_tasks_for_power(0.5, 0.02, 0.05, 2 / 274, split="random")         # 1181
```
