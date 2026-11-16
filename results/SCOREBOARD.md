# Scoreboard: Gacrux against the alternatives

One page for "where do we stand". **Gacrux** here means our routing model as it
currently works offline: start with cheaper models, verify the answer, climb to the
frontier model when the check fails, and only use a policy that passed certification
at the quality limit. Limit: 4% (the team's working limit), delta 0.05. Baseline:
always using `gpt-5`, except the SWE-bench section, where it is `claude-opus-4.1`, and
the section on other exact-answer datasets, where each dataset has its own frontier.
All numbers are offline, on recorded benchmark answers. Details
and caveats are in the linked folders; update this page when a result changes.

## MMLU-Pro (3,000 multiple-choice tasks)

| Router | Cost vs always gpt-5 | Accuracy (gpt-5: 87.4%) | Holds a quality limit? |
| --- | --- | --- | --- |
| **Gacrux** (two cheaper models must agree) | **64% cheaper** | 86.9% | Yes, certified at 4%; 0 of 10 runs over |
| **Gacrux** at a 2% limit (three must agree) | **32% cheaper** | 87.6% | Yes, certified at 2%; 0 of 10 runs over |
| OpenRouter auto-router | 65% more expensive | 85.8% | No limit offered |
| Always gpt-5 | baseline | 87.4% | - |

Source: [agreement-check](2026-11-11-agreement-check/README.md).

## Agreement on other exact-answer datasets

Source: [agreement-more-datasets](2026-11-16-agreement-more-datasets/README.md). The
MMLU-Pro result above does not repeat on 16 more datasets and a second split of
MMLU-Pro from the same archive. Each dataset is compared with its own most accurate
priced model, and the cheaper models are picked by cost only.

| Dataset | Gacrux, certified at 4% | Gacrux, certified at 2% |
| --- | --- | --- |
| arcc (1,170 tasks), three must agree | 38.9% cheaper, in 7 of 10 runs; 0 runs over | 0% |
| math500 (500 tasks), two must agree | 12.1% cheaper, in 10 of 10 runs; 0 runs over | 6.3% cheaper, in 5 of 10 runs |
| Six option-letter sets of the small-model pool, pooled (5,787 tasks), three must agree | 20.5% cheaper, in 10 of 10 runs; 0 runs over | 0% |
| Every other dataset | between -0.4% and 7.8% | 0%, apart from 0.1% on finqa |

One cheaper model alone certifies 0% everywhere. The pooled row was added after the
per-dataset results were seen. Main cause at 4%: the cheap 8B models are too weak next
to the frontier. Size explains the four small sets and the 2% column. Free-form
answers rarely match as strings.

## LiveCodeBench (1,055 coding tasks)

| Router | Cost vs always gpt-5 | Accuracy (gpt-5: 86.4%) | Holds a quality limit? |
| --- | --- | --- | --- |
| **Gacrux** (response check only, everything measured) | **9% cheaper** | 1.0 point below gpt-5 | Yes, certified at 4%; 0 of 10 runs over |
| **Gacrux** (response check + sample tests; sample tests simulated at measured rates) | **17% cheaper** | 0.4 points above gpt-5 | Yes, certified at 4%; 0 of 10 runs over |
| **Gacrux** prompt-only router (the original design) | 2% cheaper | 0.4 points below | Yes, but routes almost nothing |
| Avengers-Pro, official code (weight picked at a 5% limit) | 28% cheaper | 3.0 points below gpt-5 | No: over the limit in 4 of 10 runs |
| OpenRouter auto-router | 75% cheaper | 61.1%, 25 points below | No limit offered; on 27% of all tasks gpt-5 is right and it is wrong (31% of the tasks gpt-5 solves) |
| Always gpt-5 | baseline | 86.4% | - |

Sources: [response-check](2026-11-10-response-check/README.md) (its committed tables
cover epsilon 0.01, 0.02 and 0.05; the 4% rows here come from a later run of the same
command with `--epsilon 0.04`, whose summary is not yet in that folder),
[router-rules](2026-11-11-router-rules/README.md),
[avengers-pro](2026-11-10-avengers-pro/README.md).

## SWE-bench Verified (500 repository tasks)

First numbers with a paid judge (2026-11-14, [api-judge](2026-11-14-api-judge/README.md)): on 274 django and sympy tasks, Gacrux with free checks and a `claude-sonnet-5-5` judge is about a third cheaper than `claude-opus-4.1` net of the judge and 5 points more accurate on held-out tasks; it certifies 22% at a 4% limit when half the tasks are used to certify, and 0% with the default split. Before the judge: What is measured: free checks catch about 69% of wrong
patches on django, and a frontier-class model reading the patch caught 95% of the
rest in a 150-case pilot, while open local models of 14B and 32B could not tell right from wrong
([judge-pilot](2026-11-11-judge-pilot/README.md)). The
OpenRouter auto-router is 41% cheaper than gpt-5 there at 10.6% accuracy against 15.8%.

## OpenRouter on the other datasets

`openrouter-vs-gpt5.csv`. Recorded in the benchmark archive by its authors in late
2025; a snapshot of that router then, choosing from its own catalogue.

| Dataset | Tasks | OpenRouter accuracy | gpt-5 accuracy | OpenRouter cost vs gpt-5 | Share of all tasks where gpt-5 is right and OpenRouter is wrong |
| --- | --- | --- | --- | --- | --- |
| LiveCodeBench | 1,055 | 61.1% | 86.4% | 75% cheaper | 26.8% |
| SimpleQA | 4,326 | 42.4% | 47.9% | 82% cheaper | 28.4% |
| SWE-bench Verified | 500 | 10.6% | 15.8% | 41% cheaper | 11.0% |
| HLE | 2,158 | 13.8% | 26.3% | 7% cheaper | 17.1% |
| ARC-AGI v1 | 400 | 48.7% | 52.0% | 15% cheaper | 9.8% |
| MMLU-Pro | 3,000 | 85.8% | 87.4% | 65% more expensive | 4.1% |
| GPQA | 198 | 76.8% | 84.8% | 8% more expensive | 10.6% |
| AIME | 60 | 43.3% | 88.3% | 32% more expensive | 46.7% |
| LiveMathBench | 121 | 57.9% | 79.3% | 138% more expensive | 23.1% |

The last column is a share of all tasks in the dataset, not of the tasks gpt-5 solves.

## How to read this

- Other routers trade accuracy for cost and make no promise. Gacrux saves less on
  coding today, but what it saves is inside a stated limit.
- Gacrux is strong where answers can be checked cheaply (exact-answer tasks) and the
  cheap models are close to the frontier in accuracy. For code a strong judge now
  works and pays for itself, but certified savings are limited by the number of graded
  tasks: 22% at a 4% limit, nothing at 2% yet with the judged chain.
- None of this is live. Nothing here has run on customer traffic, and the part of
  the design that depends on one company's history has not been measured at all
  (`docs/ARCHITECTURE_STATUS.md`).
- Different routers choose from different sets of models, so these are comparisons
  of outcomes, not of algorithms on equal terms.
