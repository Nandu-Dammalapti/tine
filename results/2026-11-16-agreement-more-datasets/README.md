# Agreement check on the other exact-answer datasets, 2026-11-16

Offline results on recorded LLMRouterBench outcomes and the answers recorded with
them. No model was called and no recorded answer was executed. Small summary tables
only.

**Short answer: no, not as it stands.** On 16 further datasets and a second split of
MMLU-Pro the agreement check certifies almost nothing at the 2% limit and little at
the 4% limit. The 45-64% results on MMLU-Pro and xRouteBench multiple choice do not
repeat here. Three things explain it, and each can be measured:

- The cheap models in the archive's small-model pool are too weak next to its
  frontier. This is the main reason at the 4% limit.
- Four datasets are too small to certify anything, and about 1,000 tasks is too few
  for the 2% limit.
- Free-form answers rarely match as strings.

## Question

The agreement check (two or three cheaper models answer; accept only if they give
the same answer) certified 45-64% savings on MMLU-Pro
(`results/2026-11-11-agreement-check`) and on xRouteBench multiple choice
(`results/2026-11-14-xroutebench`). Does it hold on the other exact-answer datasets
in the LLMRouterBench archive?

## Setup

`scripts/agreement_datasets.py` runs one fixed recipe on every dataset. Each step is
the existing command: `tine import-llmrouterbench`, `tine combine-outcomes`
(dropping `openrouter`, which is a router, and models with no recorded cost),
`scripts/agreement_pairs.py` for pairs and triples, and `tine cascade --checks
--epsilon 0.02 0.04` with 10 seeds, tasks split 40 / 30 / 30 into choose / certify /
test, chains of up to three groups ending in the frontier, delta 0.05. Savings are
against always using the frontier and include every call made.

Rules fixed before any run:

- **Frontier:** the most accurate model with a recorded cost, measured on all tasks
  of the dataset. Ties go to the cheaper model (GPQA: `gpt-5` and `gemini-2.5-pro`
  tie at 84.8%).
- **Cheaper models (rule `cheapest`):** up to eight cheapest models whose mean cost
  is at most a quarter of the frontier's. Accuracy is not used. Groups costing more
  than half of the frontier are dropped, as before. This is the rule the xRouteBench
  run used.

**The driver gives the published numbers back.** Run on the MMLU-Pro `test_3000`
split it returns the numbers of `results/2026-11-11-agreement-check` exactly: pairs
11.1% at 2% and 63.8% at 4%, triples 32.3% and 42.6% (`reference-certified.csv`).
The model list is not the same: rule `cheapest` offers 6 models (15 pairs, 20
triples), the first run offered 13 (25 pairs) and a hand-picked 7 (29 triples). The
certified chains come out the same.

In a review pass, every pairs and triples cell below (both rules, the pools, the
sample-size rows and the every-group rows) was recomputed from the raw
archive files by a second script that shares no code with `tine`: its own import,
split, chain search and Clopper-Pearson test. All cells matched. That script is
not in the repository.

### Datasets

The archive holds three model pools. They behave differently, so they are kept apart.

| pool | datasets | models with a recorded cost | frontier |
| --- | --- | --- | --- |
| small open models | arcc, medqa, winogrande, mathbench, emorynlp, meld, bbh, korbench, finqa, math500, kandk | 11 of 20, all 7B to 9B | an 8B reasoning model (`DeepSeek-R1-0528-Qwen3-8B`, `Qwen3-8B` or `Intern-S1-mini`); `gemma-2-9b-it` on emorynlp |
| wide | gpqa, mmlupro (`test_1000` split), aime, livemathbench | 25 to 28, from 8B open models to frontier APIs | `gpt-5` |
| API models only | arc-agi, hle | 16 and 14 | `gpt-5` |

Answer format: one option letter for arcc (4 options), medqa (4), winogrande (2),
mathbench (4), emorynlp (7), meld (7), gpqa (4) and mmlupro (10). Free-form for the
rest: short strings and option letters mixed (bbh, korbench, hle), numbers (finqa,
aime), LaTeX (math500, livemathbench), a sentence list (kandk) and a grid (arc-agi).

`mmlupro` here is the `test_1000` split with 26 priced models. Its 998 distinct
questions are all among the 3,000 of the first result. It is a third of the same
questions with a wider model pool, not a new dataset and not a new sample.

`hle` was added in a review pass, after the other results were seen, under the same
rule. The shared importer refuses the split as recorded, so six of its 2,158 tasks
are left out for every model: five where the `gpt-5` record has a negative token
count (cost 0, graded wrong) and one where the prompt text differs between models
by whitespace. Leaving these out raises the frontier's accuracy from 26.3% to 26.4%.

Left out, with the reason:

- `simpleqa`: the recorded prediction is the grader's verdict letter, not the answer.
- `tau2`: no recorded prediction. `humaneval`, `mbpp`, `livecodebench`,
  `swe-bench`: code. `arenahard*`: judged free text.

`cheapest-datasets.csv` lists every dataset with its frontier and the models paired.
`cheapest-models.csv` has accuracy and cost per model.

## Results

### Certified savings, rule `cheapest`

Certified picks, mean over 10 seeds on held-out tasks. Each cell is **saved /
accuracy / runs over the limit / runs that certified a chain**. A run that certifies
nothing uses the frontier and saves 0%; "0%" alone means all 10 runs did that.
`cheapest-certified.csv`, rows `pairs`, `triples` and `singles`.

| dataset | tasks | frontier (accuracy on test parts) | pairs 2% | pairs 4% | triples 2% | triples 4% | one cheaper model alone, 2% / 4% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arcc | 1,170 | `DeepSeek-R1-0528-Qwen3-8B` (94.7%) | 0% | 6.8% / 94.4% / 0 / 1 | 0% | **38.9% / 93.9% / 0 / 7** | 0% / 0% |
| medqa | 1,273 | `DeepSeek-R1-0528-Qwen3-8B` (81.8%) | 0% | 0% | 0% | 3.2% / 81.7% / 0 / 1 | 0% / 0% |
| winogrande | 1,267 | `Qwen3-8B` (83.2%) | 0% | 0% | 0% | 0% | 0% / 0% |
| mathbench | 150 | `Qwen3-8B` (94.6%) | 0% | 0% | 0% | 0% | 0% / 0% |
| emorynlp | 697 | `gemma-2-9b-it` (39.7% on all tasks) | no group | no group | no group | no group | none |
| meld | 1,231 | `Qwen3-8B` (54.3%) | 0% | 0% | 0% | 4.2% / 54.1% / 0 / 8 | 0% / 0% |
| bbh | 1,080 | `DeepSeek-R1-0528-Qwen3-8B` (89.1%) | 0% | -0.1% / 88.6% / 1 of 10 / 1 | 0% | 0.9% / 88.7% / 0 / 2 | 0% / 0% |
| korbench | 1,210 | `Intern-S1-mini` (57.7%) | 0% | 0% | 0% | 0% | 0% / 0% |
| finqa | 1,138 | `Qwen3-8B` (74.1%) | 0.1% / 73.9% / 0 / 3 | 0.6% / 73.7% / 0 / 7 | 0% | 0% | 0% / 0% |
| math500 | 500 | `DeepSeek-R1-0528-Qwen3-8B` (93.9%) | 6.3% / 94.1% / 0 / 5 | **12.1% / 94.2% / 0 / 10** | 0% | 0% | 0% / 0% |
| kandk | 700 | `Intern-S1-mini` (79.6%) | 0% | 0% | 0% | 0% | 0% / 0% |
| gpqa | 198 | `gpt-5` (86.4%) | 0% | 0% | 0% | 0% | 0% / 0% |
| mmlupro (`test_1000`) | 998 | `gpt-5` (85.5%) | 0% | 5.7% / 84.9% / 1 of 10 / 2 | 0% | 7.8% / 84.3% / 0 / 8 | 0% / 0% |
| aime | 60 | `gpt-5` (86.0%) | 0% | 0% | 0% | 0% | 0% / 0% |
| livemathbench | 121 | `gpt-5` (80.5%) | 0% | 0% | 0% | 0% | 0% / 0% |
| arc-agi | 400 | `gpt-5` (51.1%) | 0% | 0% | 0% | -0.4% / 51.1% / 0 / 1 | 0% / 0% |
| hle | 2,152 | `gpt-5` (26.6%) | 0% | 0% | 0% | -0.2% / 25.1% / 0 / 6 | 0% / 0% |

"Tasks" is what the cascade used: it keeps one task per identical prompt (korbench
loses 40 of 1,250 that way).

- **At the 2% limit nothing certifies**, apart from math500 pairs (6.3%, in 5 of 10
  runs) and 0.1% on finqa.
- **At the 4% limit two datasets certify more than 10%:** arcc with triples (38.9%,
  in 7 of 10 runs, 0.8 points below the frontier) and math500 with pairs (12.1%, in
  all 10 runs, accuracy 94.2% against 93.9%). Every other cell is between -0.4% and
  7.8%.
- **A certified chain can cost more than the frontier.** On hle a triple certifies
  in 6 of 10 runs, saves nothing (-0.2%) and is 1.5 points less accurate. The
  certificate bounds the loss. It says nothing about savings on new tasks.
- **Runs over the limit:** bbh pairs at 4% (1 of 10, worst 5.5%) and mmlupro pairs
  at 4% (1 of 10, worst 4.8%). With delta 0.05 about 1 run in 20 is expected.
- **Just switching to one cheaper model certifies 0% everywhere**, at both limits.
- **Offering single models next to the pairs** (`pairs+singles` rows) changes five
  cells. Four get slightly worse (finqa 0.1% to -0.1% and 0.6% to 0.2%, math500 6.3%
  to 5.2% and 12.1% to 10.0%). One gets better: mmlupro at 4%, 5.7% to 7.3%.
- **emorynlp has nothing to route.** Its most accurate priced model, `gemma-2-9b-it`
  (39.7%), is also the cheapest, so no group costs less than half of it.
- **Agreeing answers graded differently:** none on 16 datasets. On hle there are 21
  such cases across the 28 pairs and 24 across the 56 triples (at most 0.5% and 1.2%
  of a group's agreed tasks). The group is counted as wrong on those tasks, which
  is what `--grade-conflicts wrong` does.

### Why: model strength, size, answer format

`cheapest-datasets.csv`, `cheapest-agreement-rates.csv`. "Loss" is a task the
frontier gets right and the group agrees on a wrong answer. "Can pass" is the highest
loss rate on the certify tasks whose Clopper-Pearson bound is still inside the
limit; "none" means even zero losses cannot certify. The lowest loss rate is taken
over all groups and all tasks, in hindsight, among groups that agree on at least a
tenth of tasks. No run could pick it. It is a floor.

| dataset | certify tasks (about) | loss rate that can pass, 2% / 4% | lowest loss rate of any pair / triple | highest agreement rate, pair / triple | all members right and same string (median pair) |
| --- | --- | --- | --- | --- | --- |
| MMLU-Pro `test_3000`, for comparison | 900 | 1.11% / 2.89% | 1.33% / 0.67% | 87.6% / 80.1% | 100% |
| arcc | 351 | 0.57% / 1.99% | 2.56% / 1.37% | 87.0% / 76.4% | 100% |
| medqa | 381 | 0.52% / 2.10% | 5.50% / 2.59% | 68.0% / 52.1% | 100% |
| winogrande | 380 | 0.53% / 2.11% | 8.37% / 4.97% | 68.4% / 50.3% | 100% |
| mathbench | 45 | none / none | 4.00% / 1.33% | 40.7% / 20.0% | 100% |
| meld | 369 | 0.54% / 2.17% | 3.08% / 1.38% | 63.5% / 46.3% | 100% |
| bbh | 324 | 0.62% / 1.85% | 3.33% / 2.04% | 54.6% / 37.0% | 99.5% |
| korbench | 363 | 0.55% / 2.20% | 1.60% / 0.64% | 36.2% / 25.6% | 98.7% |
| finqa | 341 | 0.59% / 2.05% | 0.09% / no triple agrees on a tenth | 17.3% / 7.5% | 16.5% |
| math500 | 150 | 0.00% / 0.67% | 0.00% / 0.00% | 68.2% / 37.0% | 82.5% |
| kandk | 210 | 0.00% / 1.43% | no group agrees on a tenth | 4.0% / 0.1% | 8.0% |
| gpqa | 59 | none / none | 2.53% / 1.52% | 51.0% / 22.7% | 100% |
| mmlupro (`test_1000`) | 299 | 0.33% / 2.01% | 2.10% / 0.60% | 63.7% / 43.3% | 100% |
| aime | 18 | none / none | 0.00% / 0.00% | 33.3% / 25.0% | 100% |
| livemathbench | 36 | none / none | 0.00% / 0.00% | 58.7% / 45.5% | 80.0% |
| arc-agi | 120 | none / 0.83% | 1.00% / no triple agrees on a tenth | 38.0% / 9.8% | 100% |
| hle | 645 | 0.93% / 2.64% | 2.04% / 1.39% | 30.7% / 16.3% | 65.8% |

**1. Model strength.** In the small-model pool the best pair in hindsight loses
2.6% (arcc), 3.1% (meld), 3.3% (bbh), 4.0% (mathbench), 5.5% (medqa) and 8.4%
(winogrande) of tasks. Only arcc's is below the 2.89% that MMLU-Pro's 900 certify
tasks could pass at 4%. Triples lose less (1.3% to 5.0%), and all but winogrande's
best triple are below 2.89%, which is why triples certify more than pairs on these
sets.
The models that qualify by cost are much weaker than the frontier: on arcc they are
76-88% accurate against 94.3%, on medqa 41-68% against 81.6%. When two of them agree
they are still less accurate than the frontier is on those same questions (arcc
91.8% against 95.1%, medqa 77.8% against 86.1%, median pair). On MMLU-Pro the
agreeing pair was more accurate than the frontier overall. On arcc, medqa,
winogrande, mathbench, bbh, korbench and kandk every priced model within five
points of the frontier costs more than half of what it costs (on arcc 66%, 75% and
253%), because all of these are 7B to 9B models and the price differences are
mostly reasoning tokens. So they cannot be paired under the half-cost limit. meld,
finqa and math500 each have one cheap model within five points. math500's is in
the pair that certifies. Nine of the 20 models in this pool have no recorded cost
and were dropped, including some of the most accurate (`MiniCPM4.1-8B`,
`GLM-Z1-9B-0414`).

hle shows the same thing with API models. The eight cheapest are right on 4-9% of
tasks and `gpt-5` on 26%. When two of them agree they are right 5.5% of the time
(median pair), and `gpt-5` is right on 24% of those same tasks. Agreement between
models that are far below the frontier is not a sign of a right answer.

**2. Size.** A dataset needs about 250 tasks before anything can certify at 4% and
about 500 at 2%, even with zero losses (74 and 149 certify tasks). mathbench, gpqa,
aime and livemathbench are below that and cannot certify whatever the check does.
The datasets with 1,000 to 1,300 tasks can pass only a loss rate near 0.5% at the
2% limit and 2% at the 4% limit. MMLU-Pro with 3,000 could pass 1.1% and 2.9%.

How much does size alone cost? MMLU-Pro `test_3000` was cut to random samples of
1,000 and 400 questions, six samples per size, same recipe
(`reference-1000-certified.csv`, `reference-400-certified.csv`). Mean certified
savings over the samples, with the lowest and highest sample in brackets and the
number of seed-level runs that certified a chain:

| MMLU-Pro `test_3000` | pairs 2% | pairs 4% | triples 2% | triples 4% |
| --- | --- | --- | --- | --- |
| all 3,000 | 11.1%, 2 of 10 runs (1 run over the limit) | 63.8%, 10 of 10 | 32.3%, 9 of 10 | 42.6%, 10 of 10 |
| 1,000 at random, six samples | 3.0% (0% to 11.3%), 3 of 60 (1 run over) | 40.0% (17.1% to 57.4%), 40 of 60 (2 runs over) | 10.6% (3.0% to 25.8%), 17 of 60 | 42.0% (32.0% to 63.5%), 57 of 60 |
| 400 at random, six samples | 0% | 9.2% (0% to 13.5%), 10 of 60 | 0% | 13.2% (1.3% to 27.0%), 24 of 60 (1 run over) |

- With 1,000 tasks the triples result at 4% holds and the pairs result keeps about
  two thirds. The 2% results fall to about a third. With 400 tasks most of it is
  gone.
- One sample of 1,000 (seed 2) has `gemini-2.5-pro` as its most accurate model, so
  the frontier rule picks it and offers eight cheaper models. Without that sample
  the means are 2.3%, 39.2%, 11.5% and 37.7%.
- The samples differ a lot. The first version of this page showed only the seed-0
  sample of each size (17.1% for pairs at 4% with 1,000 tasks, the lowest of the
  six) and read too much into it.

So size explains the four small datasets and most of the 2% column. It does not
explain 0% at the 4% limit on datasets with 1,000 to 1,300 tasks: MMLU-Pro keeps
about 40% with 1,000. Model strength does.

**3. Answer format.**

- *Two options (winogrande).* Two weak models agree on a wrong answer often: the
  best pair loses 8.4% of tasks. The first agreement README predicted this.
- *Free-form numbers and LaTeX (finqa, math500, livemathbench).* Right answers are
  spelled differently. When both members of a pair are right they give the same
  string only 16.5% of the time on finqa (one writes a currency sign or a unit, the
  other does not), 82.5% on math500 and 80% on livemathbench. On finqa the pair that agrees most agrees on 17% of tasks and
  costs 11% of the frontier, so almost nothing is saved.
- *Long free-form answers (kandk, arc-agi).* kandk's recorded prediction is a
  sentence list or the whole response; pairs agree on at most 4% of tasks. On
  arc-agi the cheap models are rarely right (6-25% against 52%), so they rarely
  agree (38% at most, 9% for the median pair).
- *Low ceiling (meld, korbench, hle).* The frontier itself is right on 26-58%.
  Losses are capped by that, and so are savings: on meld the median pair loses 4.6%
  and gains 4.4%.

A group saves only when its agreement rate is higher than its cost as a share of
the frontier, since the frontier is still called on every disagreement. math500
passes that test (68% agreement for 30% of the cost, no losses on 500 tasks), which
is why it certifies despite only 150 certify tasks.

### Pooled datasets

Added after the per-dataset results, to give the check enough tasks. Each pool is
one task mix certified as a whole, so the limit holds for the mix, not for each
dataset in it. Same rule `cheapest`; rows are in the same CSVs.

| pool | tasks | frontier (accuracy on test parts) | pairs 2% | pairs 4% | triples 2% | triples 4% |
| --- | --- | --- | --- | --- | --- | --- |
| `pool-8b`: all 11 small-model datasets | 10,415 | `Qwen3-8B` (73.2%) | 0% | 4.5% / 72.6% / 0 / 4 | 0% | 0% |
| `pool-8b-letters`: arcc, medqa, winogrande, mathbench, emorynlp, meld | 5,787 | `Qwen3-8B` (72.8%) | 0% | 0% | 0% | **20.5% / 72.1% / 0 / 10** |
| `pool-wide`: gpqa, mmlupro `test_1000`, aime, livemathbench | 1,377 | `gpt-5` (84.6%) | 0% | **10.4% / 83.6% / 0 / 5** | 1.2% / 84.3% / 0 / 2 | 5.9% / 83.1% / 0 / 10 |
| `pool-wide-small`: gpqa, aime, livemathbench | 379 | `gpt-5` (83.1%) | 0% | 0% | 0% | 0.4% / 82.1% / 0 / 7 |

- With 5,787 option-letter tasks, three small models that agree certify **20.5% at
  the 4% limit in all 10 runs**, at 0.7 points below the frontier. At 2% nothing
  certifies even with 1,736 certify tasks: the best triple loses 2.3% of tasks. So
  in this pool the check is real but weak: about a fifth at 4%, nothing at 2%. More
  than four times the tasks of its largest dataset did not rescue it, which again
  points at model strength.
- Adding the free-form sets (`pool-8b`) removes the result: triples then agree too
  rarely to pay for three calls (the one that agrees most agrees on 35% of tasks and
  costs 35% of the frontier).
- `pool-wide` is 73% mmlupro. arc-agi and hle are not in a pool: they share too few
  models with the others.

### A second model rule, added after seeing the results above

Rule `cheapest` picks only 8B models in the wide pool, because they are the eight
cheapest, and they are weak next to `gpt-5`. Rule `all-cheaper` offers every model
cheaper than the frontier, as the first MMLU-Pro run did (triples from the 14
cheapest only, to keep files small). **This rule was chosen after the first results
were seen.** `all-cheaper-*.csv`.

- Small-model pool and hle: the same results on every dataset. The extra models
  cost more than half of the frontier, or the cascade does not pick their groups.
  arc-agi stays at or below 0%.
- Wide pool: mmlupro `test_1000` triples at 4% go from 7.8% to 15.0% (8 of 10 runs
  certify, accuracy 84.6% against 85.5%); pairs 4.8% (1 run). gpqa, aime and
  livemathbench stay at 0%. `pool-wide` is unchanged.

### What the cascade's shortlist costs, added after seeing the results above

`tine cascade` builds its chains from eight candidates per seed. It ranks candidates
by what each would save on the choosing tasks if its check caught every wrong
answer. For an agreement group that depends on its cost and on the recorded grade
of its first-named member. How often the group agrees is not used. With 10 groups
that drops two. With 77 to 286 pairs (rule `all-cheaper`, wide pool) it drops almost
all of them.

`--every-group` runs the same certification with every group offered, each as a
one-group chain (`tine.cascade.run_cascade_experiment` with `top_k` set to the
number of groups and `max_length` 1; rows `pairs-every-group` and
`triples-every-group`). **This check was chosen after the results were seen.** It
is not the method of the earlier agreement results.

| | as run (eight candidates, chains of up to three) | every group, one-group chains |
| --- | --- | --- |
| rule `cheapest`, 17 datasets, 4% | largest cell 38.9% (arcc triples) | no cell moves by more than 5.0 points: meld pairs 0% to 5.0% (2 runs, 1 over the limit), math500 pairs 12.1% to 14.2%, mmlupro triples 7.8% to 9.8%, mmlupro pairs 5.7% to 3.5%, arcc triples 38.9% to 38.5% |
| rule `cheapest`, 17 datasets, 2% | math500 pairs 6.3% | math500 pairs 8.4%; mmlupro triples 0% to 1.4% (2 runs) |
| rule `all-cheaper`, mmlupro `test_1000`, pairs 4% | 4.8% / 85.3% / 0 / 1 | **16.1% / 84.7% / 0 / 9** |
| rule `all-cheaper`, mmlupro `test_1000`, triples 4% | 15.0% / 84.6% / 0 / 8 | **24.2% / 85.1% / 0 / 10** |
| rule `all-cheaper`, mmlupro `test_1000`, triples 2% | 0% | 1.6% / 85.4% / 0 / 2 |
| rule `all-cheaper`, `pool-wide`, pairs 4% | 10.4% / 83.6% / 0 / 5 | **21.7% / 84.7% / 0 / 9** |
| rule `all-cheaper`, `pool-wide`, triples 4% | 5.9% / 83.1% / 0 / 10 | **16.5% / 84.2% / 0 / 10** |
| rule `all-cheaper`, `pool-wide`, triples 2% | 1.2% / 84.3% / 0 / 2 | 4.7% / 84.3% / 0 / 8 |
| MMLU-Pro `test_3000`, rule `cheapest` | pairs 11.1% and 63.8%, triples 32.3% and 42.6% | pairs 5.7% and 61.5%, triples 31.1% and 42.7% |

- **Under the rule fixed in advance the shortlist is not the reason for the weak
  result.** Offering every group changes little on any of the 17 datasets.
- **In the wide pool with mid-priced API models offered, the shortlist hides most of
  what the check can certify.** On mmlupro `test_1000` it is 24% instead of 15% for
  triples at 4%, and 16% instead of 5% for pairs. In hindsight the pair `intern-s1`
  + `kimi-k2-0905` agrees on 81% of those tasks and loses 1.4%, for 23% of
  `gpt-5`'s cost, close to the best pair on `test_3000` (85%, 1.3%, 19%). The
  shortlist never offers it.
- gpqa, aime, livemathbench, arc-agi and hle stay at or below 0% either way.
- At the 2% limit the largest cell either way is math500 pairs (6.3% as run, 8.4%
  with every group). In the wide pool it is 4.7%.

### Pooled picture

A single pooled savings number would not be a fair summary: the datasets differ too
much in size, format and model pool, and most cells are zero. In counts, under the
rule fixed in advance: of 17 datasets, none certifies more than 7% at the 2% limit
and two certify more than 10% at the 4% limit (arcc 38.9%, math500 12.1%).

## What this means for Gacrux

- The agreement check is not a general result for exact-answer work. It needs three
  things at once. Cheap models that are close to the frontier in accuracy, so that
  an agreed answer is as likely to be right as the frontier's. Enough graded tasks
  for the limit: about 300 certify tasks carried most of MMLU-Pro's 4% result, the
  2% result needed the full 900. And answers that compare as strings. MMLU-Pro and
  xRouteBench multiple choice had all three. Nothing else in the archive does.
- A tenant with a few hundred graded tasks per task type should expect little
  certified saving at first, whatever the check. Shadow mode has to collect graded
  tasks before a policy can be frozen.
- Free-form numeric answers need a normaliser before agreement is worth trying
  (finqa). The xRouteBench math result warns that a normaliser can match answers the
  grader treats differently, so it has to be measured, not assumed.
- With many candidate groups, how `tine cascade` shortlists them matters as much as
  the check. Ranking by a measured check's own results is the first thing to try.

## Limits

- Benchmark runs, not customer traffic. One recorded run per model.
- Three model pools only. The small-model pool has no strong cheap model by
  construction, so its result says "8B models cannot stand in for an 8B reasoning
  model", not "agreement fails on ARC or MedQA". A pool with mid-priced API models
  on those datasets was not available.
- Models with no recorded cost were dropped (9 of 20 in the small-model pool), as
  `combine-outcomes` does by default. Some are among the most accurate.
- The frontier is picked on all tasks of a dataset, including the ones later held
  out. That is the stated rule, and the same as in the earlier results.
- Model choice by cost alone pairs related models: `Llama-3.1-8B-Instruct` and
  `Llama-3.1-8B-UltraMedical` share a base model, and the certified arcc triple
  contains both. Related models make the same mistakes more often.
- A chain of two groups that share a member is charged for that member twice, so
  savings for multi-group chains are slightly understated.
- The certify split is random, so "certify tasks" is 30% of the tasks on average,
  not exactly.
- Exact string comparison after trimming whitespace, as in the earlier results. No
  answer normaliser was tried here.
- hle leaves out six tasks (see Datasets). Some identical recorded answers are
  graded differently there, so its grader does not compare the recorded string
  alone. The other 16 datasets have no such case.
- Added after the per-dataset results were seen: the pools, the `all-cheaper` rule,
  the sample-size rows, hle, and the `--every-group` check. The per-dataset table
  under rule `cheapest`, without hle, is the result fixed in advance.
- Delta applies per seed and limit, not across the whole table. In the pairs and
  triples rows under rule `cheapest` (17 datasets and 4 pools), 100 seed-level runs
  certified a chain and 2 of them went over their limit. About 1 in 20 is expected.

## Next

1. In `tine cascade`, shortlist candidates with their measured check results, not
   as if the check were perfect, and rerun the wide pool.
2. An answer normaliser for numeric answers, measured against the grader on finqa.
3. More graded tasks per task type is the lever for the 2% limit. For the wide pool
   that means new runs (paid), not more analysis.

## Reproduce

After `scripts/fetch_benchmark.py` has extracted the archive to `BENCH`:

```bash
OUT=results/2026-11-16-agreement-more-datasets
for RULE in cheapest all-cheaper; do
  python scripts/agreement_datasets.py --bench BENCH/extracted/bench-release --rule $RULE --every-group \
    --work WORK/$RULE --results $OUT --label $RULE \
    --datasets arcc medqa winogrande mathbench emorynlp meld bbh korbench finqa math500 kandk \
      gpqa mmlupro aime livemathbench arc-agi hle pool-8b pool-8b-letters pool-wide pool-wide-small
done
python scripts/agreement_datasets.py --bench BENCH/extracted/bench-release --brief --every-group \
  --work WORK/reference --results $OUT --label reference --datasets mmlupro-3000
for SIZE in 1000 400; do
  python scripts/agreement_datasets.py --bench BENCH/extracted/bench-release --brief \
    --max-tasks $SIZE --sample-seeds 0 1 2 3 4 5 \
    --work WORK/reference-$SIZE --results $OUT --label reference-$SIZE --datasets mmlupro-3000
done
```

The first two take about a quarter of an hour each on a laptop and write several
gigabytes of per-task files to `WORK` (keep it outside the repository or under
`artifacts/`). Tests: `pytest tests/test_agreement_datasets.py`.
