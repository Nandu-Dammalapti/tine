# Agreement between cheap models' patches, next to the judge, 2026-11-16

Offline results on recorded SWE-bench Verified answers. **No model was called and no
patch was executed in this folder's runs.** The judge verdicts are the recorded ones
from `results/2026-11-14-api-judge` (paid then, not now); the grades are from the
Colab test run (`results/2026-11-10-swe-static-checks`). Patches were compared as
text only. Small summary tables only.

**Short answer: no reliable gain from agreement on code.** It does not raise what
certifies. On cost it is a wash. Charged as `tine cascade` charges them, chains of
pairs save the same as the plain chain or less. That charge counts a model twice when
it sits in two groups of one chain; with it paid once (Result 5, a check added by the
verifier), pairs that agree at 0.8 land within about a point and a half of the plain
chain, above it with one judge and at or below it with the other. Details and the one
small exception are below.

## Question

On exact-answer work, two cheap models giving the same answer is a strong and nearly
free check. On code the check is "free checks, then a judge". Can agreement between
cheap models' patches cut what the judge costs, or certify more?

## Setup

- Data: the 274 valid tasks, seven cheap models, frontier `claude-opus-4.1`, this
  run's grades. Free checks from the Colab run. Recorded verdicts of
  `claude-haiku-5-5` and `claude-sonnet-5-5` on the 1,110 patches that passed the
  free checks.
- `scripts/swe_agreement_checks.py` builds inputs for `python -m tine cascade
  --checks`. Every variant: epsilon 0.02, 0.03, 0.04, delta 0.05, 10 seeds, the
  default split (40 / 30 / 30) and 30 / 50 / 20, the same two splits as the api-judge
  folder.
- A candidate is one cheap model (`singles`, the existing chain) or a **group** of
  two or three cheap models that are **all called on every task the stage sees**.
  This is how the cost of agreement is charged: a group costs the sum of its members'
  recorded costs, whether or not they agree, plus the judge's recorded cost where the
  judge was consulted. Members are ordered by mean recorded cost.
  - `agree-or-judge`: if two members wrote the same patch and both passed the free
    checks, that patch is accepted with no judge call. Otherwise the judge reads the
    cheapest member's patch that passed the free checks.
  - `judge-only`: the same groups with no agreement rule. The control: it shows what
    grouping does without agreement.
  - `agree-only`: accept on agreement, never call a judge.
  - `mixed`: singles, pairs and triples offered together; the choose part picks.
- "The same patch": identical, or **soft**: the Jaccard overlap of the two patches'
  changed lines is at or above a threshold. **The thresholds, 0.8 (main) and 0.5
  (looser), were fixed in the code before any chain with soft agreement was run.**
  The table of Result 1, which shows how often an agreed patch is right at both
  thresholds, was printed before the chains were run. The thresholds were not
  changed after it, and no other was tried.
- `claude-opus-4.1` is never an agreement partner and its answers are never judged.
- Which pair or triple is used is chosen on the choose part of each seed by the
  cascade. With default settings it searches chains of up to three stages over the
  eight candidates that would save most alone on the choose part if every wrong
  answer were caught.
- `tine cascade` gives a candidate one cost per task. So a chain that has the same
  model in two groups is charged for that model in both. Results 2 to 4 are charged
  that way. Result 5 repeats the search with a model paid once.
- The certify part has 76 to 94 tasks on the default split and 127 to 149 on
  30 / 50 / 20, depending on the seed. With no task lost the bound needs 74 tasks at
  4%, 99 at 3% and 149 at 2%. So on the default split nothing can certify at 3% or
  2% whatever the chain, and on 30 / 50 / 20 one seed of ten has enough tasks for 2%.
- One variant, `agree-and-judge`, **was added after seeing the first results** (see
  Result 4). Read it as exploratory.

### Baseline check

`judge_patches.py checks` plus `tine cascade` reproduced all 24 Haiku and Sonnet rows
of `results/2026-11-14-api-judge/cascade-judge-variants.csv` exactly. A group of one
in the new script gives the same check for every cheap answer (1,918 of 1,918). One
difference: the earlier join also added the judge's cost to the frontier's own
answers, which a chain never judges. The new script does not. That moves the `singles`
rows below by up to 0.4 points from the published ones (Sonnet, 4%, default split:
33.6% here, 34.0% there).

## Result 1: how often cheap models agree

`agreement-rates.csv` (every group on every task), `case-precision.csv` (per patch).

| agreement between | group has a passing patch | two members agree | agreed patch is right |
| --- | --- | --- | --- |
| pairs, identical patch | 70.6% | 0.5% | 83.3% |
| pairs, overlap >= 0.8 | 70.6% | 10.7% | 84.1% |
| pairs, overlap >= 0.5 | 70.6% | 13.8% | 82.4% |
| triples, identical patch | 80.2% | 1.4% | 83.3% |
| triples, overlap >= 0.8 | 80.2% | 17.3% | 84.9% |
| triples, overlap >= 0.5 | 80.2% | 23.0% | 81.4% |

- **Identical patches are too rare to matter.** 21 of the 966 cheap-model patches
  (2%) were written by two or more cheap models; 54 of all 1,110 (5%) if the frontier
  is counted, which a chain cannot use. A pair of cheap models writes the same patch
  on 0.5% of tasks.
- Similar patches are common enough to matter: a pair agrees at 0.8 on about one task
  in nine.
- **An agreed patch is right about 84% of the time.** That is the same as a patch the
  judge accepts (Haiku 81.8%, Sonnet 86.1% on the same 966 patches), and far above
  the 56.7% of all patches that passed the free checks.
- Agreement and the judge are partly independent. Patches that agree at 0.8 **and**
  that the judge accepts are right 90.3% (Haiku) and 92.4% (Sonnet) of the time.
  Patches the judge accepts when no other cheap model's patch overlaps at 0.5 or
  more are right 69.0% and 77.0%.

## Result 2: "agree or judge" chains

`cascade-agreement-variants.csv` has every row. Limit 4%. Cells: saved / accuracy,
then for certified rows the runs over the limit and the runs where nothing certified
(of 10), and for uncertified rows the worst run's loss and the runs over the limit.
The last column is per 100 tasks, the mean over the ten chosen uncertified chains
(default split) run on all 274 tasks.

Haiku as the judge (frontier accuracy 50.3% and 49.2% on the two splits):

| candidates | certified, 40/30/30 | certified, 30/50/20 | uncertified, 40/30/30 | uncertified, 30/50/20 | judge calls / agreement accepts / frontier calls |
| --- | --- | --- | --- | --- | --- |
| `singles` (existing chain) | 2.1% / 50.5% / 0 / 9 | 8.5% / 50.5% / 0 / 7 | 39.3% / 54.8% / 2.6% / 0 | 37.9% / 54.6% / 2.9% / 0 | 106 / 0 / 44 |
| pairs, `judge-only` | 4.6% / 50.4% / 0 / 9 | 1.8% / 49.7% / 0 / 9 | 38.4% / 57.4% / 2.6% / 0 | 34.6% / 55.5% / 2.9% / 0 | 134 / 0 / 41 |
| pairs, agree-or-judge, identical | 4.6% / 50.4% / 0 / 9 | 1.8% / 49.7% / 0 / 9 | 38.5% / 57.4% / 2.6% / 0 | 34.6% / 55.5% / 2.9% / 0 | 134 / 0.3 / 41 |
| pairs, agree-or-judge, 0.8 | 4.6% / 50.4% / 0 / 9 | 1.8% / 49.7% / 0 / 9 | 39.3% / 57.2% / 2.6% / 0 | 35.2% / 55.5% / 2.9% / 0 | 121 / 10.3 / 40 |
| pairs, agree-or-judge, 0.5 | 0.0% / 50.3% / 0 / 10 | 0.0% / 49.2% / 0 / 10 | 38.9% / 56.6% / 3.9% / 0 | 37.6% / 55.3% / 2.9% / 0 | 116 / 14.5 / 39 |
| triples, `judge-only` | 3.0% / 51.1% / 0 / 9 | 9.7% / 50.4% / 0 / 7 | 33.0% / 57.1% / 1.3% / 0 | 31.5% / 54.9% / 4.3% / 1 | 134 / 0 / 44 |
| triples, agree-or-judge, identical | 3.0% / 51.1% / 0 / 9 | 4.8% / 50.0% / 0 / 8 | 33.3% / 57.1% / 1.3% / 0 | 31.6% / 54.9% / 4.3% / 1 | 133 / 0.9 / 44 |
| triples, agree-or-judge, 0.8 | 0.0% / 50.3% / 0 / 10 | 5.1% / 49.5% / 0 / 9 | 34.3% / 56.2% / 3.7% / 0 | 32.6% / 54.1% / 4.3% / 1 | 120 / 17.2 / 42 |
| triples, agree-or-judge, 0.5 | 0.0% / 50.3% / 0 / 10 | 0.0% / 49.2% / 0 / 10 | 35.2% / 55.8% / 5.9% / 1 | 32.9% / 54.5% / 4.3% / 1 | 116 / 23.4 / 40 |
| mixed, agree-or-judge, identical | 5.1% / 51.5% / 0 / 8 | 16.9% / 50.8% / **1** / 5 | 35.0% / 56.1% / 3.0% / 0 | 32.6% / 54.7% / 2.9% / 0 | 133 / 0.4 / 44 |
| mixed, agree-or-judge, 0.8 | 3.2% / 51.2% / 0 / 9 | 8.2% / 49.9% / 0 / 7 | 35.4% / 56.0% / 3.0% / 0 | 33.4% / 54.5% / 3.9% / 0 | 125 / 11.8 / 42 |
| mixed, agree-or-judge, 0.5 | 3.2% / 51.2% / 0 / 9 | 7.1% / 50.1% / 0 / 8 | 36.5% / 55.8% / 4.0% / 0 | 34.1% / 54.1% / 4.3% / 1 | 121 / 19.3 / 40 |

Sonnet as the judge:

| candidates | certified, 40/30/30 | certified, 30/50/20 | uncertified, 40/30/30 | uncertified, 30/50/20 | judge calls / agreement accepts / frontier calls |
| --- | --- | --- | --- | --- | --- |
| `singles` (existing chain) | 0.0% / 50.3% / 0 / 10 | 21.9% / 52.7% / 0 / 3 | 33.6% / 56.1% / 2.6% / 0 | 34.3% / 55.1% / 1.9% / 0 | 103 / 0 / 46 |
| pairs, `judge-only` | 11.3% / 52.9% / 0 / 6 | 23.2% / 54.4% / 0 / 3 | 32.3% / 57.6% / 1.3% / 0 | 32.2% / 56.8% / 2.9% / 0 | 129 / 0 / 43 |
| pairs, agree-or-judge, identical | 11.4% / 52.8% / 0 / 6 | 8.3% / 51.2% / 0 / 7 | 31.9% / 57.5% / 2.2% / 0 | 31.9% / 56.6% / 3.8% / 0 | 132 / 0.4 / 43 |
| pairs, agree-or-judge, 0.8 | 11.3% / 53.0% / 0 / 6 | 2.3% / 49.9% / 0 / 9 | 33.6% / 57.6% / 2.2% / 0 | 32.6% / 56.6% / 3.8% / 0 | 121 / 10.3 / 41 |
| pairs, agree-or-judge, 0.5 | 11.2% / 52.9% / 0 / 6 | 2.3% / 50.1% / 0 / 9 | 32.2% / 56.7% / 2.6% / 0 | 31.8% / 55.4% / 4.3% / 1 | 116 / 15.1 / 41 |
| triples, `judge-only` | 15.9% / 53.6% / 0 / 4 | 11.3% / 51.6% / 0 / 6 | 28.0% / 57.1% / 2.5% / 0 | 25.6% / 55.2% / 2.9% / 0 | 126 / 0 / 46 |
| triples, agree-or-judge, identical | 15.9% / 53.6% / 0 / 4 | 10.8% / 51.6% / 0 / 6 | 28.1% / 57.1% / 2.5% / 0 | 25.8% / 55.2% / 2.9% / 0 | 125 / 1.4 / 46 |
| triples, agree-or-judge, 0.8 | 8.1% / 52.0% / 0 / 7 | 6.9% / 50.1% / 0 / 7 | 28.9% / 56.7% / 2.5% / 0 | 27.8% / 54.6% / 2.9% / 0 | 108 / 17.3 / 44 |
| triples, agree-or-judge, 0.5 | 8.3% / 51.3% / 0 / 7 | 3.8% / 49.9% / 0 / 8 | 31.3% / 56.4% / 5.9% / 1 | 27.9% / 53.6% / 4.3% / 1 | 114 / 24.1 / 41 |
| mixed, agree-or-judge, identical | 20.8% / 53.5% / 0 / 3 | 13.2% / 52.3% / 0 / 5 | 31.0% / 56.3% / 2.0% / 0 | 27.4% / 54.5% / 1.9% / 0 | 124 / 0.4 / 46 |
| mixed, agree-or-judge, 0.8 | 11.7% / 52.5% / 0 / 6 | 4.6% / 50.5% / 0 / 8 | 31.0% / 56.4% / 2.2% / 0 | 29.6% / 55.2% / 2.9% / 0 | 113 / 11.5 / 44 |
| mixed, agree-or-judge, 0.5 | 10.5% / 52.3% / 0 / 6 | 3.8% / 49.9% / 0 / 8 | 31.1% / 56.5% / 2.6% / 0 | 30.3% / 54.6% / 3.8% / 0 | 112 / 20.1 / 42 |

At the lower limits (certified saved, with the runs where nothing certified):

| limit | best `singles` row | best agree-or-judge row |
| --- | --- | --- |
| 3% | 3.1% (9 of 10 empty), Sonnet, 30/50/20 | 3.5% (8 of 10 empty), Sonnet, mixed at 0.5, 30/50/20 |
| 2% | 0% everywhere | 0% everywhere |

Reading:

- **Certified: no reliable gain.** With Sonnet on the default split the groups
  certify 8-21% where `singles` certifies nothing, but the `judge-only` control
  certifies the same 11-16% there, so that is grouping, not agreement. On
  30 / 50 / 20 it reverses: `singles` 21.9%, agree-or-judge rows 2-13%. One
  agree-or-judge row beats `singles` on both splits (mixed, identical, Haiku: 5.1%
  and 16.9%). It is also the row with a certified run over the limit, and identical
  agreement fires on fewer than 1 task in 100, so its difference from `singles` is
  the candidate set, not agreement. Two `judge-only` control rows also beat `singles`
  on both splits (triples with Haiku, pairs with Sonnet), with no agreement rule at
  all. Nothing certifies at 2%, and at 3% only in one or two runs of ten.
- **These certified numbers are fragile.** Pairs with identical agreement differ
  from the `judge-only` control on fewer than 1 task in 100, yet on 30 / 50 / 20 with
  Sonnet one certifies 8.3% and the other 23.2%. One failure in the certify part
  decides a run. The differences between rows above are inside that noise.
- **Certified chains went over the limit in three cells**, one run of ten each:
  mixed identical with Haiku at 4% (30 / 50 / 20, worst run 4.3%), pairs identical
  with Sonnet at 3% (3.8%), and pairs agree-only at 0.5 at 4% (4.3%). At delta 0.05
  one run in twenty may do that, so it is within the statement, but `singles` had
  none.
- **Uncertified, as charged here: chains of groups do not cost less than the plain
  chain.** Pairs at 0.8 match `singles` on the default split (39.3% with Haiku, 33.6%
  with Sonnet) and are 2.7 and 1.7 points lower on the other. Over all pairs rows the
  gap on 30 / 50 / 20 is 0.3 to 3.3 points. Triples are 2.3 to 8.5 points lower.
  **These rows charge a model twice when it sits in two groups of a chain, and 5 to
  10 of the 10 chosen pair chains do that.** Result 5 removes the double charge.
- **Against the same pairs without it, agreement at 0.8 saves 0.4 to 1.3 points**
  (Haiku 38.4% to 39.3% and 34.6% to 35.2%; Sonnet 32.3% to 33.6% and 32.2% to
  32.6%) at the same accuracy within 0.2 points. So the rule does save judge cost
  once two models are called anyway. With Sonnet the worst run's loss is higher
  (2.2% and 3.8% against 1.3% and 2.9%).
- **Pairs are a little more accurate**, by 0.9 to 2.6 points at identical and 0.8 and
  by 0.3 to 1.8 at 0.5 (triples by less, and not in every row), because the judge can
  be given a second model's patch when the first failed the free checks. The
  `judge-only` control shows the same gain, so it comes from calling two models, not
  from agreement.
- **The looser threshold is worse.** At 0.5, four of the six rows have a held-out
  run over the 4% limit. `singles` and the pairs at 0.8 have none.

### Judge calls saved

- Identical agreement saves 0.3 to 1.4 judge calls per 100 tasks. Nothing.
- Soft agreement at 0.8 accepts 10 (pairs) to 17 (triples) patches per 100 tasks with
  no judge call, and cuts judge calls by 6 to 14% against the same groups without it
  (pairs: 134 to 121 with Haiku, 129 to 121 with Sonnet; triples: 134 to 120 and 126
  to 108). The chosen chains differ between the two rows, so this is not the same
  chain with and without the rule.
- **But groups call the judge more than the plain chain does**: 121 calls per 100
  tasks against 103 to 106. A group more often has some patch that passed the free
  checks, and each stage that does not agree still pays for a judge call.
- **There was little to save.** In the `singles` chain the judge costs 0.2% of
  always-frontier cost with Haiku and 2.7% with Sonnet. A second cheap model costs
  0.7% to 5.8% of a frontier answer each time its stage is reached (`claude-sonnet-4`
  20%). Agreement spends a model call to save part of a judge call, and a Haiku judge
  call costs a fifth of the cheapest model's answer.

## Result 3: agreement with no judge at all

| candidates | certified, 40/30/30 | certified, 30/50/20 | uncertified, 40/30/30 | uncertified, 30/50/20 | agreement accepts / frontier calls per 100 |
| --- | --- | --- | --- | --- | --- |
| pairs, identical | 0.0% / 50.3% / 0 / 10 | -0.3% / 49.2% / 0 / 9 | 0.0% / 50.3% / 0.0% / 0 | -0.3% / 49.2% / 0.0% / 0 | 0 / 100 |
| pairs, 0.8 | 2.4% / 50.3% / 0 / 3 | 2.8% / 49.2% / 0 / 1 | 3.7% / 50.3% / 1.3% / 0 | 3.2% / 49.2% / 0.0% / 0 | 13.7 / 86 |
| pairs, 0.5 | 2.8% / 50.4% / 0 / 3 | 2.2% / 48.6% / **1** / 3 | 6.4% / 49.6% / 3.7% / 0 | 4.8% / 48.7% / 2.9% / 0 | 20.7 / 79 |
| triples, identical | -0.9% / 50.3% / 0 / 8 | -1.5% / 49.2% / 0 / 7 | -0.9% / 50.3% / 0.0% / 0 | -1.5% / 49.2% / 0.0% / 0 | 0.7 / 99 |
| triples, 0.8 | 2.0% / 50.2% / 0 / 6 | 3.4% / 49.0% / 0 / 2 | 6.4% / 50.2% / 1.3% / 0 | 5.8% / 49.4% / 1.4% / 0 | 16.8 / 83 |
| triples, 0.5 | 2.1% / 51.0% / 0 / 7 | 2.9% / 49.5% / 0 / 6 | 8.5% / 49.6% / 5.0% / 2 | 6.6% / 48.7% / 5.7% / 1 | 28.6 / 71 |

- **Agreement cannot replace the judge.** With identical patches it accepts almost
  nothing and the chain costs more than the frontier alone. With similar patches it
  saves 3 to 9% uncertified against 34 to 39% for the judged chain, at the frontier's
  accuracy, not above it.
- What it does do: pairs at 0.8 certify 2.4 to 2.8% at the 4% limit in 7 and 9 of 10
  runs with no judge and no paid check. It is small but steady.

## Result 4 (added after seeing Results 1 to 3): agreement and the judge together

Result 1 showed that a patch that agrees and is accepted by the judge is right more
often than either alone. `agree-and-judge` accepts only those: two members agree at
0.8 and the judge accepts the agreed patch. **This variant was not planned; it was
chosen after looking at the tables above, on the same 274 tasks.**

| candidates, judge | certified at 4%, 40/30/30 | certified at 4%, 30/50/20 | certified at 3%, 30/50/20 | certified at 2%, 30/50/20 | worst uncertified run |
| --- | --- | --- | --- | --- | --- |
| pairs, Haiku | 2.6% / 50.4% / 0 / 2 | 2.7% / 49.2% / 0 / 0 | 1.7% / 0 / 5 | 0% / 0 / 10 | 0.0% |
| pairs, Sonnet | 3.7% / 50.4% / 0 / 0 | 3.0% / 49.2% / 0 / 0 | 3.0% / 0 / 0 | 0.3% / 0 / 9 | 0.0% |
| triples, Haiku | 4.1% / 50.4% / 0 / 2 | 4.7% / 49.6% / 0 / 0 | 2.6% / 0 / 7 | 0% / 0 / 10 | 1.4% |
| triples, Sonnet | 5.2% / 50.5% / 0 / 0 | 4.7% / 49.5% / 0 / 0 | 4.7% / 0 / 0 | 0.1% / 0 / 9 | 0.0% |

- It certifies in every run at 4% with Sonnet on both splits, and in every run at 3%
  when half the tasks certify. On the default split it certifies nothing at 3%, like
  every other variant: the certify part is too small there (see Setup). Nothing else
  here or in the api-judge folder certifies in all ten runs at 3% (Sonnet at
  probability >= 0.9 did in 4 of 10, for 11.5% on average). With pairs it lost no
  frontier-solved task in any held-out run.
- **The saving is small: 3 to 5%.** It accepts 11 to 14 patches per 100 tasks and
  pays for two or three models on every task. It certifies reliably because it
  accepts little, not because it saves much. The judged chain's 22% at 4% is a larger
  certified number where it certifies.
- It is post hoc and should be re-tested on tasks not used to pick it before it is
  quoted.

## Result 5 (a check added by the verifier, after the results above were seen): a model paid once

`paid-once-check.csv`. `tine cascade` charges a chain such as `a+b > a+c` for model
`a` twice. A real chain would call `a` once and reuse its patch. Most chosen chains
of groups are of that kind (3 to 10 of the 10 uncertified picks at the 4% limit in
Result 2; 5 to 10 for pairs), so Results 2 to 4 understate what groups save.
`swe_agreement_checks.py paid-once` repeats the cascade's search (the same parts per
seed, the same pool of eight candidates, the same uncertified and certified picks)
and pays each model once per task and chain.
The judge is still paid every time a stage consults it. Charged the cascade's way,
the same code reproduces all 408 rows of `cascade-agreement-variants.csv`, and it
stops if it does not. **This check was not planned. It changes how cost is counted
and nothing else; no threshold or candidate list was changed for it.**

Limit 4%. Cells as in Result 2. The last column repeats Result 2's uncertified
savings on the two splits.

Haiku as the judge:

| candidates | certified, 40/30/30 | certified, 30/50/20 | uncertified, 40/30/30 | uncertified, 30/50/20 | uncertified as charged in Result 2 |
| --- | --- | --- | --- | --- | --- |
| `singles` (existing chain) | 2.1% / 50.5% / 0 / 9 | 8.5% / 50.5% / 0 / 7 | 39.3% / 54.8% / 2.6% / 0 | 37.9% / 54.6% / 2.9% / 0 | 39.3% / 37.9% |
| pairs, `judge-only` | 4.7% / 50.4% / 0 / 9 | 7.2% / 50.1% / 0 / 8 | 39.1% / 56.9% / 2.6% / 0 | 38.0% / 55.7% / 2.9% / 0 | 38.4% / 34.6% |
| pairs, agree-or-judge, identical | 4.7% / 50.4% / 0 / 9 | 7.2% / 50.1% / 0 / 8 | 39.1% / 56.9% / 2.6% / 0 | 38.2% / 55.6% / 2.9% / 0 | 38.5% / 34.6% |
| pairs, agree-or-judge, 0.8 | 4.8% / 50.4% / 0 / 9 | 7.4% / 50.1% / 0 / 8 | 40.2% / 56.3% / 2.6% / 0 | 38.7% / 55.4% / 2.9% / 0 | 39.3% / 35.2% |
| pairs, agree-or-judge, 0.5 | 0.0% / 50.3% / 0 / 10 | 0.0% / 49.2% / 0 / 10 | 40.4% / 55.6% / 3.9% / 0 | 38.9% / 54.8% / 2.9% / 0 | 38.9% / 37.6% |
| triples, `judge-only` | 3.0% / 51.1% / 0 / 9 | 7.2% / 50.1% / 0 / 8 | 35.3% / 56.9% / 3.0% / 0 | 34.3% / 54.5% / 2.9% / 0 | 33.0% / 31.5% |
| triples, agree-or-judge, identical | 3.1% / 51.1% / 0 / 9 | 2.1% / 49.7% / 0 / 9 | 35.5% / 57.0% / 3.0% / 0 | 34.9% / 54.4% / 4.3% / 1 | 33.3% / 31.6% |
| triples, agree-or-judge, 0.8 | 0.0% / 50.3% / 0 / 10 | 0.0% / 49.2% / 0 / 10 | 37.2% / 55.9% / 3.7% / 0 | 36.0% / 54.2% / 4.3% / 1 | 34.3% / 32.6% |
| triples, agree-or-judge, 0.5 | 0.0% / 50.3% / 0 / 10 | 0.0% / 49.2% / 0 / 10 | 38.3% / 55.2% / 5.9% / 1 | 36.4% / 54.3% / 4.3% / 1 | 35.2% / 32.9% |
| mixed, agree-or-judge, identical | 5.6% / 51.6% / 0 / 8 | 14.6% / 50.0% / **1** / 6 | 37.1% / 56.6% / 3.0% / 0 | 35.6% / 55.4% / 2.9% / 0 | 35.0% / 32.6% |
| mixed, agree-or-judge, 0.8 | 3.6% / 51.2% / 0 / 9 | 9.5% / 50.1% / 0 / 7 | 37.8% / 56.2% / 3.0% / 0 | 35.9% / 54.9% / 2.9% / 0 | 35.4% / 33.4% |
| mixed, agree-or-judge, 0.5 | 3.6% / 51.2% / 0 / 9 | 8.0% / 50.7% / 0 / 8 | 39.1% / 55.8% / 4.0% / 0 | 37.4% / 54.4% / 4.3% / 1 | 36.5% / 34.1% |

Sonnet as the judge:

| candidates | certified, 40/30/30 | certified, 30/50/20 | uncertified, 40/30/30 | uncertified, 30/50/20 | uncertified as charged in Result 2 |
| --- | --- | --- | --- | --- | --- |
| `singles` (existing chain) | 0.0% / 50.3% / 0 / 10 | 21.9% / 52.7% / 0 / 3 | 33.6% / 56.1% / 2.6% / 0 | 34.3% / 55.1% / 1.9% / 0 | 33.6% / 34.3% |
| pairs, `judge-only` | 12.0% / 53.2% / 0 / 6 | 21.2% / 53.6% / 0 / 4 | 32.2% / 57.6% / 1.3% / 0 | 32.5% / 56.8% / 1.9% / 0 | 32.3% / 32.2% |
| pairs, agree-or-judge, identical | 12.0% / 53.2% / 0 / 6 | 12.9% / 52.7% / 0 / 6 | 32.5% / 57.5% / 1.3% / 0 | 32.4% / 56.8% / 1.9% / 0 | 31.9% / 31.9% |
| pairs, agree-or-judge, 0.8 | 12.2% / 53.1% / 0 / 6 | 4.2% / 50.3% / 0 / 8 | 33.6% / 57.1% / 2.2% / 0 | 32.9% / 56.4% / 1.9% / 0 | 33.6% / 32.6% |
| pairs, agree-or-judge, 0.5 | 12.3% / 53.0% / 0 / 6 | 2.5% / 49.9% / 0 / 9 | 34.5% / 56.1% / 2.6% / 0 | 33.6% / 55.5% / 4.3% / 1 | 32.2% / 31.8% |
| triples, `judge-only` | 16.5% / 53.6% / 0 / 4 | 11.8% / 51.8% / 0 / 6 | 30.0% / 57.1% / 2.5% / 0 | 28.4% / 55.2% / 2.9% / 0 | 28.0% / 25.6% |
| triples, agree-or-judge, identical | 16.6% / 53.6% / 0 / 4 | 7.9% / 50.8% / 0 / 7 | 30.1% / 57.0% / 2.5% / 0 | 28.2% / 54.6% / 2.9% / 0 | 28.1% / 25.8% |
| triples, agree-or-judge, 0.8 | 5.8% / 51.7% / 0 / 8 | 7.5% / 49.9% / 0 / 7 | 31.1% / 56.7% / 3.7% / 0 | 29.5% / 55.0% / 3.8% / 0 | 28.9% / 27.8% |
| triples, agree-or-judge, 0.5 | 8.8% / 51.2% / 0 / 7 | 3.9% / 49.9% / 0 / 8 | 33.9% / 56.7% / 5.9% / 1 | 31.9% / 53.6% / 4.3% / 1 | 31.3% / 27.9% |
| mixed, agree-or-judge, identical | 21.4% / 53.7% / 0 / 3 | 12.1% / 52.3% / 0 / 5 | 32.3% / 57.0% / 2.0% / 0 | 28.0% / 54.7% / 1.9% / 0 | 31.0% / 27.4% |
| mixed, agree-or-judge, 0.8 | 11.8% / 52.6% / 0 / 6 | 10.7% / 52.2% / 0 / 6 | 33.3% / 57.0% / 2.2% / 0 | 30.4% / 55.2% / 2.9% / 0 | 31.0% / 29.6% |
| mixed, agree-or-judge, 0.5 | 11.4% / 52.2% / 0 / 6 | 5.1% / 51.1% / 0 / 8 | 33.7% / 56.6% / 3.0% / 0 | 31.7% / 55.4% / 2.9% / 0 | 31.1% / 30.3% |

Reading:

- **The double charge was not small.** Paying a model once raises the uncertified
  savings of pairs by up to 3.6 points (one row is 0.1 lower), of triples by 1.7 to
  4.0 and of mixed sets by 0.6 to 3.3, at the 4% limit. `singles` rows do not change.
- **Pairs at 0.8 against the plain chain: no reliable difference.** With Haiku they
  save 0.9 and 0.8 points more (40.2% against 39.3%, 38.7% against 37.9%). With
  Sonnet they save the same and 1.4 points less (33.6% against 33.6%, 32.9% against
  34.3%). Seed by seed the pairs chain saved more in 7, 6, 4 and 3 of 10 runs (Haiku
  on the two splits, then Sonnet), and single runs differ by up to 11 points either
  way. Accuracy is 0.8 to 1.5 points higher with pairs. That is not a saving one can
  claim. It is also not a loss.
- **Agreement at 0.8 against the same pairs without it: about a point, in most
  runs.** 40.2% against 39.1% and 38.7% against 38.0% with Haiku, 33.6% against
  32.2% and 32.9% against 32.5% with Sonnet; more in 9, 9, 10 and 7 of 10 runs.
  The seed counts in this bullet and the one above come from the per-seed rows of
  the same search and are not in the CSV.
- The 0.5 threshold saves a little more again (34.5% with Sonnet on the default
  split is the only pairs row above `singles` with that judge) and still has the
  runs over the limit and the higher worst-run losses seen in Result 2.
- Triples stay below `singles` by 1.0 to 6.1 points in every row but one, and mixed
  sets in every row but one. Both exceptions are Sonnet at 0.5 on the default split
  (triples 33.9%, with a run over the limit, and mixed 33.7%, against 33.6%).
- **Certified: still no reliable gain.** Pairs at 0.8 certify 4.8% and 7.4% with
  Haiku (`singles` 2.1% and 8.5%) and 12.2% and 4.2% with Sonnet (`singles` 0% and
  21.9%). The `judge-only` control certifies about the same with Haiku (4.7% and
  7.2%) and 12.0% and 21.2% with Sonnet.
- Agreement with no judge (Result 3) at 0.8 saves 6.7% and 5.9% uncertified with
  pairs and 7.5% and 6.7% with triples, against 3.7% / 3.2% and 6.4% / 5.8% as
  charged. Agreement and the judge together (Result 4) with Sonnet certify 5.2% and
  3.6% (pairs) and 5.8% and 6.3% (triples) at 4% in 10 of 10 runs, and 3.6% and 6.3%
  at 3% on 30 / 50 / 20 in 10 of 10. With Haiku and triples on the default split it
  certifies less than before (2.0%, 6 of 10 runs empty, against 4.1% and 2).
- Paid once, the search prefers chains that share a model (8 to 10 of 10 chosen
  pair chains). Such a chain adds one new model per stage and compares its patch
  with the ones already written.

## What this means

- For code there is no measured reason to replace the chain as it is: one cheap
  model at a time, free checks, then the judge. Chains of pairs with agreement do
  not save reliably more, even when no model is paid twice, and they do not certify
  more. A second model's answer costs several times a Haiku judge call. Identical
  patches almost never happen.
- Similarity between cheap models' patches is real evidence (84% right when two
  agree at 0.8, 92% with the Sonnet judge as well). Where two cheap models are
  called anyway, accepting on agreement at 0.8 and judging the rest saved about a
  point over judging everything, at the same accuracy. Its other use is as a strict
  extra condition for a small saving that certifies at a tight limit.
- Pairs were about a point more accurate than the plain chain at about the same
  cost once a model is paid once. That is a reason to test them again on more tasks,
  not a result.
- The limit on certified savings is still the number of graded tasks. Nothing here
  changes that: no variant certifies a useful amount at 2%.

## Limits

- 274 tasks from django and sympy, single-shot patches, one benchmark. The certify
  part has 82 or 137 tasks, so certified numbers move a lot with one failure.
- Result 4 was chosen after seeing results on these same tasks. The two similarity
  thresholds were fixed beforehand, but only two were tried.
- A group pays for every member on every task it sees. A design that calls the second
  model only sometimes was not tried; it could not agree before both have answered.
- In Results 2 to 4 a chain that uses the same model in two groups pays for it
  twice, and most chosen chains do. Result 5 pays once. It still pays the judge again
  when a later group sends it a patch an earlier stage's judge call already
  rejected, so it still overstates cost a little.
- Result 5 was added by the verifier after Results 1 to 4 were seen. Its search runs
  in this folder's script, not in `tine cascade`; it reproduces the cascade's rows
  when charged the same way. It was run once, with nothing tuned.
- Members are ordered by mean recorded cost over all 274 tasks. Only costs are used
  for that, not outcomes.
- The search keeps the eight candidates that save most alone on the choose part, so
  with 21 pairs or 35 triples most groups are never tried in a chain.
- Similarity is measured on changed lines as a set. Two patches that make the same
  change with different formatting count as different.
  A changed line that itself starts with `--` or `++` (an SQL comment, say) is read
  as a file header and ignored, as in `scripts/train_verifier.py`.
- The judge verdicts come from public issues a strong model may have seen (see the
  api-judge folder). Agreement has no such problem, but the judged rows do.
- Judge calls and group members add latency; not counted. A group can call its
  members in parallel.
- "Per 100 tasks" columns run each seed's chosen chain over all 274 tasks, including
  the ones it was chosen on. They describe usage, not held-out performance.

## Reproduce

```bash
# no model is called; needs the recorded verdicts and the judge cases and key
python scripts/swe_agreement_checks.py rates --key RUNS/judge/key-277.csv --cases RUNS/judge/cases-277.jsonl \
  --free-checks RUNS/repo-test-checks.csv --outcomes DATA/swe-274-own.jsonl --frontier claude-opus-4.1 \
  --output RUNS/agreement-rates.csv
python scripts/swe_agreement_checks.py cases --key RUNS/judge/key-277.csv --cases RUNS/judge/cases-277.jsonl \
  --free-checks RUNS/repo-test-checks.csv --outcomes DATA/swe-274-own.jsonl --frontier claude-opus-4.1 \
  --judge haiku=RUNS/judge/verdicts-haiku55-277.csv sonnet=RUNS/judge/verdicts-sonnet55-277.csv \
  --output RUNS/case-precision.csv
# every variant, both judges, both splits, through `python -m tine cascade` (about 10 minutes)
python scripts/swe_agreement_checks.py run --key RUNS/judge/key-277.csv --cases RUNS/judge/cases-277.jsonl \
  --free-checks RUNS/repo-test-checks.csv --outcomes DATA/swe-274-own.jsonl --frontier claude-opus-4.1 \
  --judge haiku=RUNS/judge/verdicts-haiku55-277.csv sonnet=RUNS/judge/verdicts-sonnet55-277.csv \
  --work RUNS/swe-agreement --summary RUNS/cascade-agreement-variants.csv
```

The same search with a model paid once per task and chain (about 3 minutes; needs the
`--work` folder of the `run` above, and stops if it cannot reproduce the cascade):

```bash
python scripts/swe_agreement_checks.py paid-once --key RUNS/judge/key-277.csv --cases RUNS/judge/cases-277.jsonl \
  --free-checks RUNS/repo-test-checks.csv --outcomes DATA/swe-274-own.jsonl --frontier claude-opus-4.1 \
  --judge haiku=RUNS/judge/verdicts-haiku55-277.csv sonnet=RUNS/judge/verdicts-sonnet55-277.csv \
  --work RUNS/swe-agreement --summary RUNS/paid-once-check.csv
```

One variant by hand:

```bash
python scripts/swe_agreement_checks.py build --key RUNS/judge/key-277.csv --cases RUNS/judge/cases-277.jsonl \
  --free-checks RUNS/repo-test-checks.csv --outcomes DATA/swe-274-own.jsonl --frontier claude-opus-4.1 \
  --verdicts RUNS/judge/verdicts-haiku55-277.csv --mode agree-or-judge --group-size 2 --similarity 0.8 \
  --output RUNS/pairs-soft80
python -m tine cascade --input RUNS/pairs-soft80/outcomes.jsonl --frontier claude-opus-4.1 \
  --checks RUNS/pairs-soft80/checks.csv --epsilon 0.02 0.03 0.04 --fractions 0.3 0.5 0.2 --output RUNS/pairs-soft80/cascade
```

Columns of `cascade-agreement-variants.csv`: `split` is `d` (40 / 30 / 30) or `c50`
(30 / 50 / 20); `over` is runs over the limit and `none` runs where nothing certified,
of 10; the three `per_100` columns and `judge_cost_pct_of_frontier` are the mean over
the ten chosen chains, each run on all 274 tasks.

Columns of `paid-once-check.csv`: the same first eleven, with a model paid once;
`saved_charged_every_stage` is the cascade's figure for the same row;
`runs_reusing_a_model_every_stage` and `runs_reusing_a_model` count, of 10, the chosen
chains in which a model sits in two groups, as the cascade charges and paid once.
