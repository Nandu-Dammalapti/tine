# Shortlisting chain candidates by their measured checks, 2026-11-17

Offline results on recorded outcomes and recorded check results. No model was
called. Small summary tables only.

**Short answer: mixed.** The new rule raises certified savings a lot where there are
hundreds of agreement groups to choose from (MMLU-Pro `test_1000` wide pool, triples
at 4%: 15.0% to 30.5%; `pool-wide` triples at 4%: 5.9% to 23.0%) and on xRouteBench
pairs at 2% (45.2% to 54.3%). It lowers 7 of 44 certified cells, by 0.2 to 2.2
points, and adds two runs over the limit in cells that had none. It changes nothing
on the code chains, which have only seven candidates. The default stays as it is.

## Question

`tine cascade` and `tine freeze-policy` build chains from a shortlist of eight
candidates. The default rule ranks each cheaper candidate by what it would save if
its check caught every wrong answer. With measured checks (agreement groups, judged
patches) that ignores how often a candidate's check passes and how often a passed
answer is wrong. With hundreds of agreement groups it keeps the cheapest groups, not
the reliable ones (`results/2026-11-16-agreement-more-datasets`, section "What the
cascade's shortlist costs"). Does a shortlist ranked by the measured check results
certify more?

## Setup

New option `--shortlist measured` (default `--shortlist savings`, which is the old
behaviour) in `tine cascade` and `tine freeze-policy`, and `shortlist=` in
`tine.cascade.candidate_chains`, `run_cascade_experiment` and
`tine.policy.freeze_policy`.

The rule, on the choose part only. For each cheaper candidate, tried alone in front of
the frontier behind its measured check, take its savings and its degradation rate
(`chain_metrics([j], ..., passes=choose passes)`). Fill the eight slots in this order:

1. candidates with savings above 0 and degradation rate at most epsilon, most
   savings first;
2. other candidates with savings above 0, lowest degradation rate first;
3. candidates that save nothing alone.

Because the first group depends on epsilon, the shortlist is built once per seed and
epsilon. Certify and test rows are never passed to it (a test checks this by
flipping every check result outside the choose part and getting the same
shortlists). Chains of up to three, `top_k` 8, the fixed-sequence Clopper-Pearson
test, delta 0.05 and the splits are unchanged.

**The rule was written, with its tests, before any run with it.** It follows the
rule suggested when this work was set, plus step 3, which was added before any run so that a group
that almost never agrees (no losses, but no savings either) does not take a slot.
The motivating finding (one strong pair on MMLU-Pro `test_1000` that the old
shortlist never offers) was known in advance.

Experiments rerun, each with the command or driver from its folder's Reproduce
section and the local data under `artifacts/`. The xRouteBench folder has no
Reproduce section; its runs used `tine cascade --checks` with the settings its README
states (10 seeds, 40 / 30 / 30, epsilon 0.01 0.02 0.04), which is what was rerun:

- (a) `results/2026-11-11-agreement-check`: MMLU-Pro `test_3000`, pairs (25 groups)
  and triples (29), frontier `gpt-5`.
- (b) `results/2026-11-14-xroutebench`: multiple choice, pairs (28) and triples (56),
  frontier `cogito-v2-1-671b`.
- (c) `results/2026-11-16-agreement-more-datasets`: mmlupro `test_1000`, arcc and
  `pool-wide`, under both model rules (`cheapest`, `all-cheaper`), rebuilt with
  `scripts/agreement_datasets.py`.
- (d) `results/2026-11-14-api-judge`: SWE-bench chains (274 tasks) with the Haiku
  and Sonnet judges, frontier `claude-opus-4.1`, splits 40 / 30 / 30 and
  30 / 50 / 20.

**The default rule gives every published number back exactly.** All 40 + 30 + 30 +
30 + 120 + 360 compared cells match, at full precision for (a) and (b), at the
published 3 and 4 decimals for (d) and (c) (`reproduction-check.csv`).

## Results

Certified picks, mean over 10 seeds on held-out tasks. Each cell is **saved /
accuracy / runs over the limit / runs that certified a chain**. "0%" alone means no
run certified anything (frontier only). `certified-by-rule.csv` has every cell,
including 1% and 5% where the original runs had them, and the worst run.

### (a) MMLU-Pro `test_3000` (frontier 87.3%)

| | 2%, savings | 2%, measured | 4%, savings | 4%, measured |
| --- | --- | --- | --- | --- |
| pairs | 11.1% / 87.2% / 1 / 2 | 10.9% / 87.2% / 1 / 2 | 63.8% / 86.9% / 0 / 10 | 65.4% / 86.9% / 0 / 10 |
| triples | 32.3% / 87.6% / 0 / 9 | 33.0% / 87.5% / 0 / 8 | 42.6% / 87.5% / 0 / 10 | 42.6% / 87.4% / 0 / 10 |

### (b) xRouteBench multiple choice (frontier 88.9%)

| | 2%, savings | 2%, measured | 4%, savings | 4%, measured |
| --- | --- | --- | --- | --- |
| pairs | 45.2% / 89.8% / 0 / 9 | **54.3% / 90.0% / 0 / 10** | 73.8% / 89.4% / 2 / 10 | 73.1% / 89.7% / 1 / 10 |
| triples | 53.5% / 90.1% / 0 / 10 | 53.5% / 90.1% / 0 / 10 | 53.3% / 90.1% / 0 / 10 | 53.8% / 89.9% / 0 / 10 |

### (c) Other LLMRouterBench datasets

| data, model rule | groups | 2%, savings | 2%, measured | 4%, savings | 4%, measured |
| --- | --- | --- | --- | --- | --- |
| mmlupro `test_1000` pairs, `cheapest` (frontier 85.5%) | 28 | 0% | 0% | 5.7% / 84.9% / 1 / 2 | 3.5% / 85.1% / 1 / 1 |
| mmlupro `test_1000` triples, `cheapest` | 56 | 0% | 1.0% / 85.5% / 0 / 1 | 7.8% / 84.3% / 0 / 8 | 10.9% / 84.1% / **1** / 9 |
| mmlupro `test_1000` pairs, `all-cheaper` | 207 | 0% | 0% | 4.8% / 85.3% / 0 / 1 | 9.7% / 85.3% / 0 / 2 |
| mmlupro `test_1000` triples, `all-cheaper` | 364 | 0% | 3.5% / 85.4% / 0 / 1 | 15.0% / 84.6% / 0 / 8 | **30.5% / 84.8% / 0 / 10** |
| arcc pairs, either rule (frontier 94.7%) | 10 | 0% | 0% | 6.8% / 94.4% / 0 / 1 | 6.8% / 94.4% / 0 / 1 |
| arcc triples, either rule | 10 | 0% | 0% | 38.9% / 93.9% / 0 / 7 | 38.5% / 93.9% / 0 / 7 |
| `pool-wide` pairs, `cheapest` (frontier 84.6%) | 28 | 0% | 0% | 10.4% / 83.6% / 0 / 5 | 10.4% / 83.6% / 0 / 5 |
| `pool-wide` triples, `cheapest` | 56 | 1.2% / 84.3% / 0 / 2 | 2.4% / 84.3% / 0 / 4 | 5.9% / 83.1% / 0 / 10 | 5.3% / 83.2% / 0 / 7 |
| `pool-wide` pairs, `all-cheaper` | 180 | 0% | 0% | 10.4% / 83.6% / 0 / 5 | 8.6% / 84.4% / **1** / 2 |
| `pool-wide` triples, `all-cheaper` | 364 | 1.2% / 84.3% / 0 / 2 | 6.3% / 84.5% / 0 / 3 | 5.9% / 83.1% / 0 / 10 | **23.0% / 84.2% / 0 / 10** |

Group counts are from the earlier folder's `*-datasets.csv`. On arcc both model
rules leave the same five models and ten groups of each size, so the rows are the
same.

For reference, the earlier folder's post hoc check that offers every group as a
one-group chain (`every group` rows in `certified-by-rule.csv`) gives 16.1% and
24.2% on mmlupro `test_1000` `all-cheaper` pairs and triples at 4%, and 21.7% and
16.5% on `pool-wide`. The measured shortlist beats it on triples and stays well
below it on pairs.

### (d) Code chains with a judge (frontier 49-50% on test parts)

| judge, split | 2% | 3% | 4% |
| --- | --- | --- | --- |
| Haiku, 40 / 30 / 30 | 0% either rule | 0% either rule | 2.1% / 50.5% / 0 / 1, both rules |
| Haiku, 30 / 50 / 20 | 0% either rule | 0% either rule | 8.5% / 50.5% / 0 / 3, both rules |
| Sonnet, 40 / 30 / 30 | 0% either rule | 0% either rule | 0% either rule |
| Sonnet, 30 / 50 / 20 | 0% either rule | 3.1% / 49.5% / 0 / 1, both rules | 22.2% / 52.7% / 0 / 7, both rules |

Identical by construction: the code pool has seven cheaper models, so either rule
shortlists all of them and builds the same chains.

### What changed, in counts

Over the 44 certified cells at 2%, 3% and 4% above (pairs and triples of (a),
(b) and (c), with arcc counted once per model rule, and the 12 code cells of (d)):

- **Higher with the measured rule: 13 cells.** The large gains are where the
  candidate list is long: mmlupro `test_1000` `all-cheaper` triples at 4% (+15.5
  points, 10 of 10 runs certify), `pool-wide` `all-cheaper` triples at 4% (+17.1),
  and xRouteBench pairs at 2% (+9.1, 10 of 10 runs instead of 9). Four 2% cells
  rise from 0-1.2% to 1.0-6.3%, each certifying in 1 to 4 runs. Three of the 13 rise by
  less than a point.
- **Lower: 7 cells**, by 0.2 to 2.2 points: MMLU-Pro `test_3000` pairs at 2% (11.1%
  to 10.9%), xRouteBench pairs at 4% (73.8% to 73.1%), mmlupro `test_1000`
  `cheapest` pairs at 4% (5.7% to 3.5%), arcc triples at 4% under both model rules
  (38.9% to 38.5%), `pool-wide` `cheapest` triples at 4% (5.9% to 5.3%, 7 of 10 runs
  certify instead of 10), `pool-wide` `all-cheaper` pairs at 4% (10.4% to 8.6%).
- **Unchanged: 24 cells**, including all 12 code cells.
- **Runs over the limit: 4 with the default rule, 5 with the measured rule**, out
  of 151 and 153 runs that certified a chain. The measured rule removes one on
  xRouteBench pairs at 4% and adds one each on mmlupro `test_1000` `cheapest`
  triples at 4% (worst run 4.84%) and `pool-wide` `all-cheaper` pairs at 4% (worst
  4.52%). With delta 0.05 about 1 run in 20 is expected; both are within that.
- **Accuracy** moves by at most 1.0 point between the rules (up, on `pool-wide`
  `all-cheaper` triples at 4%) and stays within 1.5 points of the frontier in every
  cell under either rule.

### Why the pairs in the wide pool barely move

The strong pair named in the earlier folder (`intern-s1` + `kimi-k2-0905` on mmlupro
`test_1000`) is still not shortlisted at 4% in any seed: at a 4% limit many pairs
pass step 1 and the rule then ranks by savings, where other pairs are ahead. The
bigger reason is the fixed-sequence test, not the shortlist. Chains are tested from
lowest degradation on the choose part upwards and the test stops at the first
failure. On mmlupro `test_1000` `all-cheaper` pairs at 4%, the first chain tested
fails on the certify part in 9 of 10 seeds with the default rule and 8 of 10 with
the measured one (for example 9 losses in 305 certify tasks, bound 5.1%), so nothing
behind it is reached. Choosing the lowest of many noisy loss rates picks one that is
low partly by luck. (Diagnosis made after the results were seen.)

## Limits

- One recorded run per model, benchmark data, not customer traffic.
- The rule was fixed before results with it were seen, but the problem it targets
  was found on some of these same datasets. The wide-pool gains are on the data that
  prompted the change.
- Under a gate or response check, `run_cascade_experiment` ranks with the setting's
  check alone; that path was not rerun here.
- A chain of two groups that share a member is still charged for that member twice
  (see `results/2026-11-16-swe-agreement-and-judge`). Not addressed here.
- `tine freeze-policy` does not yet record which shortlist rule built a policy in the
  policy file.
- Delta applies per seed and limit, not across this table.

## What this means

The measured rule is not better or equal everywhere certified, so this page does not
recommend changing the default. Where the candidate list is long (hundreds of
agreement groups) it is clearly better for triples; elsewhere it is within about two
points of the default either way. For pairs in the wide pool, the order of the
fixed-sequence test is now the limiting step, not the shortlist.

## Reproduce

Per-task inputs as in each folder's README; here they were the files under
`artifacts/`. For (a), (b) and (d), each folder's `tine cascade` command, run twice,
with `--shortlist savings` and `--shortlist measured`:

```bash
for RULE in savings measured; do
  python -m tine cascade --input DATA/mmlupro-pairs.jsonl --frontier gpt-5 \
    --checks RUNS/mmlupro-agreement.csv --epsilon 0.02 0.01 0.04 0.05 --shortlist $RULE --output RUNS/mmlupro-pairs-$RULE
  python -m tine cascade --input DATA/mmlupro-triples.jsonl --frontier gpt-5 \
    --checks RUNS/mmlupro-agreement3.csv --epsilon 0.02 0.01 0.04 --shortlist $RULE --output RUNS/mmlupro-triples-$RULE
  python -m tine cascade --input DATA/multiple-choice-pairs.jsonl --frontier cogito-v2-1-671b \
    --checks DATA/multiple-choice-agreement.csv --epsilon 0.01 0.02 0.04 --shortlist $RULE --output RUNS/xrb-pairs-$RULE
  python -m tine cascade --input DATA/multiple-choice-triples.jsonl --frontier cogito-v2-1-671b \
    --checks DATA/multiple-choice-agreement3.csv --epsilon 0.01 0.02 0.04 --shortlist $RULE --output RUNS/xrb-triples-$RULE
  for J in haiku sonnet; do for SPLIT in "0.4 0.3 0.3" "0.3 0.5 0.2"; do
    python -m tine cascade --input RUNS/judge/v-$J.jsonl --frontier claude-opus-4.1 --checks RUNS/judge/v-$J-checks.csv \
      --epsilon 0.02 0.03 0.04 --fractions $SPLIT --shortlist $RULE --output "RUNS/code-$J-${SPLIT// /-}-$RULE"
  done; done
done
```

For (c), the earlier folder's driver rebuilds the per-task files and runs the default
rule; the measured rule is then run on the files it wrote:

```bash
for RULE in cheapest all-cheaper; do
  python scripts/agreement_datasets.py --bench BENCH/extracted/bench-release --rule $RULE --every-group \
    --work WORK/$RULE --results WORK/results-$RULE --label $RULE --datasets mmlupro arcc pool-wide
  for DS in mmlupro arcc pool-wide; do for KIND in pairs triples; do
    python -m tine cascade --input WORK/$RULE/$DS/$DS-$KIND.jsonl --checks WORK/$RULE/$DS/$DS-$KIND-checks.csv \
      --frontier FRONTIER --epsilon 0.02 0.04 --seeds 10 --shortlist measured --output WORK/$RULE/$DS/run-$KIND-measured
  done; done
done
```

`FRONTIER` is the `frontier` column of `WORK/results-$RULE/$RULE-datasets.csv`
(`gpt-5` for mmlupro and `pool-wide`, `DeepSeek-R1-0528-Qwen3-8B` for arcc). The
tables here were collected from the `cascade-summary.csv` of each run. Tests:
`pytest tests/test_cascade.py tests/test_policy.py`.
