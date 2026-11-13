# Scoreboard: Gacrux against the alternatives

One page for "where do we stand". **Gacrux** here means our routing model as it
currently works offline: start with cheaper models, verify the answer, climb to the
frontier model when the check fails, and only use a policy that passed certification
at the quality limit. Limit: 4% (the team's working limit), delta 0.05. Baseline:
always using `gpt-5`. All numbers are offline, on recorded benchmark answers. Details
and caveats are in the linked folders; update this page when a result changes.

## MMLU-Pro (3,000 multiple-choice tasks)

| Router | Cost vs always gpt-5 | Accuracy (gpt-5: 87.4%) | Holds a quality limit? |
| --- | --- | --- | --- |
| **Gacrux** (two cheaper models must agree) | **64% cheaper** | 86.9% | Yes, certified at 4%; 0 of 10 runs over |
| **Gacrux** at a 2% limit (three must agree) | **32% cheaper** | 87.6% | Yes, certified at 2%; 0 of 10 runs over |
| OpenRouter auto-router | 65% more expensive | 85.8% | No limit offered |
| Always gpt-5 | baseline | 87.4% | - |

Source: [agreement-check](2026-11-11-agreement-check/README.md).

## LiveCodeBench (1,055 coding tasks)

| Router | Cost vs always gpt-5 | Accuracy (gpt-5: 86.4%) | Holds a quality limit? |
| --- | --- | --- | --- |
| **Gacrux** (response check only, everything measured) | **9% cheaper** | 1.0 point below gpt-5 | Yes, certified at 4%; 0 of 10 runs over |
| **Gacrux** (response check + sample tests; sample tests simulated at measured rates) | **17% cheaper** | 0.4 points above gpt-5 | Yes, certified at 4%; 0 of 10 runs over |
| **Gacrux** prompt-only router (the original design) | 2% cheaper | 0.4 points below | Yes, but routes almost nothing |
| Avengers-Pro, official code (weight picked at a 5% limit) | 28% cheaper | 3.0 points below gpt-5 | No: over the limit in 4 of 10 runs |
| OpenRouter auto-router | 75% cheaper | 61.1%, 25 points below | No limit offered; loses 27% of the tasks gpt-5 solves |
| Always gpt-5 | baseline | 86.4% | - |

Sources: [response-check](2026-11-10-response-check/README.md),
[router-rules](2026-11-11-router-rules/README.md),
[avengers-pro](2026-11-10-avengers-pro/README.md).

## SWE-bench Verified (500 repository tasks)

No certified Gacrux number yet. What is measured: free checks catch about 69% of wrong
patches on django, and a frontier-class model reading the patch caught 95% of the
rest in a 150-case pilot, while open local models of 14B and 32B could not tell right from wrong
([judge-pilot](2026-11-11-judge-pilot/README.md)). The
OpenRouter auto-router is 41% cheaper than gpt-5 there at 10.6% accuracy against 15.8%.

## OpenRouter on the other datasets

`openrouter-vs-gpt5.csv`. Recorded in the benchmark archive by its authors in late
2025; a snapshot of that router then, choosing from its own catalogue.

| Dataset | Tasks | OpenRouter accuracy | gpt-5 accuracy | OpenRouter cost vs gpt-5 | Tasks lost that gpt-5 solved |
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

## How to read this

- Other routers trade accuracy for cost and make no promise. Gacrux saves less on
  coding today, but what it saves is inside a stated limit.
- Gacrux is strong where answers can be checked cheaply (exact-answer tasks) and weak
  where they cannot yet (code). The verifier for code is the open work.
- None of this is live. Nothing here has run on customer traffic, and the part of
  the design that depends on one company's history has not been measured at all
  (`docs/ARCHITECTURE_STATUS.md`).
- Different routers choose from different sets of models, so these are comparisons
  of outcomes, not of algorithms on equal terms.
