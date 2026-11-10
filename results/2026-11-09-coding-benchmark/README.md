# Coding benchmark results, 2026-11-09

Offline results on already-graded LLMRouterBench outcomes. No model was called.
These are small summary tables only: per-task outputs, prompts and the benchmark
files stay out of Git (AGENTS.md: never commit generated benchmark outputs).
The commands at the bottom regenerate everything.

## Setup

- Data: LLMRouterBench `NPULH/LLMRouterBench` at revision
  `b95870f9e60c589e82f7972c91ffa8de9ca89f85`, archive SHA-256 in `data-manifest.json`.
- Two pools, each relabelled as one `coding` task type:
  - `lcb`: LiveCodeBench, 1,055 tasks, 28 models (9 models with no recorded cost
    and `openrouter`, itself a router, removed).
  - `lcb-swe`: LiveCodeBench + SWE-bench Verified, 1,555 tasks, the 13 models both have.
  - HumanEval and MBPP were left out: their results only cover small open models,
    so pooling them would drop gpt-5.
- Frontier (comparison) model: `gpt-5`, the most accurate model in both pools
  (86.4% on `lcb`, 63.7% on `lcb-swe`). Per-model accuracy and cost: `data-manifest.json`.
- epsilon = 0.02 (max quality loss vs gpt-5), delta = 0.05.

## 1. Current router (predict the model before calling it)

`router-*-metrics.csv`, `router-*-calibration.json` (`tine run`, seed 42).

- Certified savings vs gpt-5: **0%** on both pools. Only the `cluster` method
  certified a threshold, and on the test split it still sent every task to gpt-5.
  In a Colab run with different library versions the same method saved 1.8%
  (1.3 points less accurate).
- The hindsight oracle saves 63% (`lcb`) and 50% (`lcb-swe`), so the savings exist;
  the predictors cannot tell from the prompt which tasks a cheaper model solves.
- Even choosing the best threshold on the test set (optimistic), the best method
  saved about 6% at <= 2% loss.
- Sentence-transformer embeddings (all-MiniLM-L6-v2, 256-token truncation) did worse
  than TF-IDF: all three methods failed calibration at the first threshold.
- Stronger encoders do not change this. Qwen3-Embedding-0.6B and -8B (the 8B is the
  encoder the Avengers-Pro paper names), 2,048-token context, computed on Colab and
  run through `tine run --embeddings` on `lcb` and `lcb-swe` (seed 42, main 9f384d9):
  0% savings in all four runs. Only `cluster` certified a threshold (3 of 4 runs) and
  it routed no test task. Reported by the Colab session on 2026-11-10; the embedding
  files are on Drive, not in Git, and were not re-run here.
- Prompt length matters for any text features: LiveCodeBench prompts average 1,408
  characters, SWE-bench prompts 69,562 (median 46,616): a short `<issue>` block, then
  repository files and an example patch. Truncated encoders mostly see the start of
  the issue; issue-only features (extract `<issue>...</issue>`) are untested.
- Calibration uses fixed-sequence testing over thresholds (PR #5): one group with
  three methods needs >= 203 risk-calibration tasks at zero degradations; `lcb`
  has 211, `lcb-swe` 311.

## 2. Try cheaper models first, check the answer, escalate to gpt-5

`cascade-*-summary.csv` (`tine cascade`). For each of 10 seeds, tasks split
40/30/30 into choose / certify / test parts. A chain such as
`qwen3-235b-a22b-2507 > qwen3-235b-a22b-thinking > intern-s1-new > gpt-5` tries each
model in turn and escalates when its answer fails the check.

- `catch` = share of wrong answers the check catches. Benchmark labels come from
  hidden tests, so `catch = 1.0` is an optimistic upper bound; 0.95 and 0.90 are
  simulated per wrong answer. Section 3 measures real public-sample checks.
- `uncertified` = cheapest chain with degradation <= epsilon on the choose part.
  `certified` = additionally passes a one-sided Clopper-Pearson bound <= epsilon on
  the certify part (chains tested in a fixed order, stop at first failure).
- Degradation = gpt-5 solved the task but the chain's final answer did not.

Held-out results, `lcb` (gpt-5 accuracy on the test parts: 87.1%):

| catch | epsilon | uncertified: saved / runs over epsilon | certified: saved / runs over epsilon | certified accuracy |
| --- | --- | --- | --- | --- |
| 1.00 | 0.02 | 42% / 0 | 42% / 0 | 91.0% |
| 1.00 | 0.01 | 42% / 0 | 38% / 0 | 90.6% |
| 1.00 | 0.005 | 42% / 0 | 0% / 0 (needs ~600 certify tasks) | 87.1% |
| 0.95 | 0.02 | 47% / 2 of 10 | 14% / 0 | 87.9% |
| 0.95 | 0.01 | 45% / 6 of 10 | 4% / 0 | 87.0% |
| 0.90 | 0.02 | 50% / 9 of 10 | 0% / 0 | 87.1% |

`lcb-swe` (gpt-5 accuracy on test parts 63.9%): 29% saved and 75.5% accurate with
catch 1.0 at epsilon 0.01-0.02 (certified); 6% at catch 0.95 / epsilon 0.02; ~0% below that.

Takeaways:
- Without the certification step, chains broke the epsilon limit in up to 9 of 10
  runs once checks were imperfect. With it, no certified run went over.
- Check quality now decides the result: 38-42% savings with perfect checks,
  4-14% at 95%, ~0% at 90%.
- Average calls per task: 1.7 (`lcb`) to 2.2 (`lcb-swe`), so failed attempts add latency.
- Costs are recorded per-task benchmark costs; running the checks is not costed.

## 3. Measured public-sample checks (LiveCodeBench)

`sample-check-rates-lcb.csv`, `cascade-lcb-measured-summary.csv`.

`scripts/lcb_sample_checks.py` ran every recorded answer of the 28 `lcb` models
(29,540 answers) against the problem's public samples: stdin problems as real
programs, LeetCode-style problems through the method. Each answer ran as the
`nobody` user with an empty environment, no network (`unshare -n`), CPU, memory
and file-size limits and a 2-second limit per sample.

- Coverage: samples parsed for 99.8% of answers (2 of 1,055 problems have none;
  those count as "passed", nothing caught).
- False rejects: 0.2% of right answers fail their samples (1 of 911 for gpt-5).
- Catch rate (share of wrong answers that fail the samples) is **low for the strong
  cheap models**: qwen3-235b-a22b-2507 38%, qwen3-235b-a22b-thinking 52%,
  intern-s1-new 68%, deepseek-v3.1-terminus 70%, qwen3-235b-a22b-thinking-2507 74%,
  gpt-5 31%. Weak models' wrong answers are easy to catch (88-100%), but those
  models are rarely right.
- Overall catch rate 79% across all models, dominated by weak models.
- Moving from a 4-second to a 2-second per-sample limit changed 4 of 1,055 gpt-5 verdicts.

Cascade with these measured checks (`tine cascade --checks`, 10 seeds, gpt-5 frontier):

| epsilon | uncertified: saved / runs over epsilon | certified: saved |
| --- | --- | --- |
| 0.02 | 0% (no chain fits) | 0% |
| 0.01 | 0% (no chain fits) | 0% |
| 0.05 | 51% / 4 of 10 (accuracy 85.0% vs 87.1%) | 0% |

**Conclusion:** public samples are far too weak a check. The 38-42% certified
savings in section 2 need checks that catch about 95% or more of wrong answers;
real visible tests catch 38-74% for the models worth trying first.

## Next

1. Stronger checks: combine the sample check with the router's predictor (try a cheap
   model only when it is predicted likely right *and* its answer passes the samples),
   differential testing between models on generated inputs, or a judge model.
   Measure each the same way (catch rate, false-reject rate, certified savings).
2. Customer test suites and CI are usually stronger than benchmark public samples;
   measure their catch rate in shadow mode before relying on escalation.
3. Run the official Avengers-Pro code on the same pools for a head-to-head
   (our `cluster` method is only an Avengers-style heuristic).

## Reproduce

From the repository root, after `bash scripts/setup.sh` (plus `huggingface-hub==0.36.0`):

```bash
python scripts/fetch_benchmark.py --output-dir BENCH --revision b95870f9e60c589e82f7972c91ffa8de9ca89f85
python -m tine import-llmrouterbench --input BENCH/extracted/bench-release/livecodebench --output DATA/outcomes-livecodebench.jsonl --dataset livecodebench --run-selection latest
python -m tine import-llmrouterbench --input BENCH/extracted/bench-release/swe-bench --output DATA/outcomes-swe-bench.jsonl --dataset swe-bench --run-selection latest
python -m tine combine-outcomes --input DATA/outcomes-livecodebench.jsonl --output DATA/lcb.jsonl --task-type coding --exclude-model openrouter
python -m tine combine-outcomes --input DATA/outcomes-livecodebench.jsonl DATA/outcomes-swe-bench.jsonl --output DATA/lcb-swe.jsonl --task-type coding --exclude-model openrouter
python -m tine run --input DATA/lcb.jsonl --frontier gpt-5 --output RUNS/router-lcb
python -m tine cascade --input DATA/lcb.jsonl --frontier gpt-5 --output RUNS/cascade-lcb
# Executes model-written code: Linux, run as root so it can drop to `nobody`, isolated machine only.
python scripts/lcb_sample_checks.py --bench-dir BENCH/extracted/bench-release/livecodebench --output RUNS/lcb-checks.csv --models <the 28 lcb models> --workers 4
python -m tine cascade --input DATA/lcb.jsonl --frontier gpt-5 --checks RUNS/lcb-checks.csv --epsilon 0.02 0.01 0.05 --output RUNS/cascade-lcb-measured
```
