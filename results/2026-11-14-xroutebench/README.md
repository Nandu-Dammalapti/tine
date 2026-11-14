# xRouteBench: agreement checks and the LLMRouter library's routers, 2026-11-14

Offline results on recorded outcomes. No model was called. Small summary tables only.

Two questions, both prompted by the LLMRouter library
(<https://github.com/ulab-uiuc/LLMRouter>, see `docs/RESEARCH_NOTES.md`):

1. Does the agreement check, which certified large savings on MMLU-Pro
   (`results/2026-11-11-agreement-check`), hold on a different benchmark with different
   models?
2. How do the library's own routers do on our pools once a quality limit is applied?

## Data

xRouteBench, `ulab-ai/xRouteBench` on Hugging Face at revision `ea4b6e1b`, config
`llmrouter_generic` (train and test files together): every query run on 18 open
models, with each model's response, a score and token counts. Imported with
`scripts/import_xroutebench.py`. Cost per row is tokens times the benchmark's own
price table. The answer a model gave is extracted from its response: the last
parenthesised letter (multiple choice), the last number (GSM8K-style) or the last
`\boxed{}` (MATH). For multiple choice, "extracted letter equals the reference"
matches the benchmark's score on 100% of 85,716 rows.

| group | tasks in it | queries | frontier (most accurate) | its accuracy |
| --- | --- | --- | --- | --- |
| multiple choice | ARC-Challenge, BoolQ, CommonsenseQA, HellaSwag, MMLU, MMLU-Pro, OpenbookQA | 4,762 | `cogito-v2-1-671b` | 88.6% |
| math | GSM8K, MATH, AIME | 2,020 | `cogito-v2-1-671b` | 72.3% |

`models-multiple-choice.csv`, `models-math.csv`. Left out: SQuAD (scored by F1, not
right or wrong) and the two code tasks (no comparable answer; the cheapest model is
also the most accurate there, so there is nothing to route).

This pool differs from ours in a way that matters. All 18 models are open models at
$0.05 to $1.25 per million input tokens. On multiple choice the second most accurate
model costs 6% of the most accurate one and is 3 points behind it. There is no
frontier API model in the pool.

## Result 1: agreement

Setup as on MMLU-Pro: `scripts/agreement_pairs.py` builds one candidate per pair (or
triple) of cheaper models, costing the sum of its members because all are always
called; its check passes when all members gave the same answer. Eight cheaper models
costing at most a quarter of the frontier (one of two copies of the same model on
different providers was dropped, since a model agreeing with itself is not a second
opinion): 28 pairs, 56 triples (multiple choice), 53 triples (math). `tine cascade
--checks`, 10 seeds, tasks split 40 / 30 / 30, chains of up to three groups ending in
the frontier, delta 0.05. Savings are against always using the frontier and include
every call made.

Certified picks, mean over 10 seeds on held-out tasks:

| group | epsilon | pairs: saved / accuracy / runs over | triples: saved / accuracy / runs over |
| --- | --- | --- | --- |
| multiple choice (frontier 88.9%) | 0.01 | 0% / 88.9% / 0 | 15.2% / 89.4% / 0 |
| | 0.02 | **45.2% / 89.8% / 0** | **53.5% / 90.1% / 0** |
| | 0.04 | 73.8% / 89.4% / 2 of 10 | 53.3% / 90.1% / 0 |
| math (frontier 71.9%) | 0.01 | 0% / 71.9% / 0 | 1.1% / 71.7% / 2 of 10 |
| | 0.02 | 14.8% / 72.3% / 0 | 3.6% / 71.6% / 0 |
| | 0.04 | 19.5% / 71.8% / 0 | 4.8% / 71.6% / 0 |

`agreement-pairs-*-summary.csv`, `agreement-triples-*-summary.csv`.

- **The MMLU-Pro result repeats on a second benchmark with different models.** On
  multiple choice, two cheap models that agree save 45% at the 2% limit and 74% at 4%,
  and the routed answers are slightly more accurate than the frontier's.
- **Why it works:** the pair `gpt-oss-120b` + `gpt-oss-20b` agrees on 82% of questions
  and is right on 94.7% of those; the frontier is right on 93.4% of the same
  questions. Where they disagree the frontier is right on 67%, so those go to it.
- **At 4% the pairs went over the limit in 2 of 10 runs** (worst 4.9%). With delta
  0.05 that should happen in about 1 run in 20, so 2 in 10 is more than expected and
  is reported as it is; triples did not go over.
- **Just switching to a cheaper model does not certify** (`agreement-singles-*`): 0%
  at every limit. The best cheap model is 3.1 points less accurate than the frontier
  overall, and it gets wrong 7.2% of the questions the frontier gets right, which is
  what the limit counts.
- Offering single models alongside the pairs (`agreement-pairs-singles-*`) gives
  less at 2% (24%), not more. More candidates means more chains tried on the same
  choosing data, and the picked chain fails certification more often (5 of 10 runs
  fall back to the frontier).
- **Math is weaker: 15-20%.** Models agree on about half the questions, and free-form
  answers do not always match when they mean the same thing (`0.5` and `1/2`), which
  counts as disagreement here. The extraction also misses 11-27% of answers per
  model. A normaliser for math answers would raise this; not built.

## Limits

- Open models only, with a narrow price range. Savings here are a share of a small
  bill; what carries over is that the check certifies, not the dollar amount.
- The choice of which eight models to pair used their recorded cost (at most a
  quarter of the frontier), not their accuracy.
- The benchmark's train and test files were pooled and re-split by seed. Its own
  split was not used.
- Exact-match comparison of extracted answers; see the math note above.
