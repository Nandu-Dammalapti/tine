# A live pilot with real models: plan, 2026-11-17

Written before any model was called. Results go in `README.md` next to this file;
anything that changes from this plan is listed there as a deviation.

## Why

Five simulated pilots (`results/2026-11-16-simulated-pilots`) tested onboarding,
wiring, docs and security with stand-in providers. They could not measure what a
real pilot has to measure: what real replies look like, whether a certification
made on one set of questions holds on new ones, and what shadow mode really costs
and adds in latency. This run does those things with real models, as one
customer would, end to end through the real proxy.

## The customer

An Anthropic-only team. All their traffic goes to `claude-opus-5-5` today. Most of
it asks a question with a fixed set of answers and ends with a line `Answer: X`,
like ticket triage or document classification. Some of it is free text on the
same token. The team has graded history (questions with known right answers) and
wants to pay less without losing more than 4% of right answers (2% reported too).

Stand-in traffic: MMLU-Pro questions (`artifacts/bench/.../mmlupro/test_3000`,
3,000 questions, 10 options, gold letters) for the exact-answer part, and
Arena-Hard creative-writing prompts for the free-text part. Both are public
benchmarks; the models may have seen them. That affects all three models alike
and does not change what is being tested here (routing, not model ability).

## Models and prices

| role | model | $ per M input / output |
| --- | --- | --- |
| frontier | `claude-opus-5-5` | 4 / 20 |
| cheaper | `claude-sonnet-5-5` | 2 / 10 |
| cheaper | `claude-haiku-5-5` | 0.10 / 0.50 |

Same settings for every call: no system prompt, the benchmark's own prompt (which
asks for `Answer: $LETTER` on the last line), `max_tokens` 4096, effort `medium`.

Note on prices: Sonnet costs half of Opus, so a chain where Haiku and Sonnet must
agree costs about 55% of Opus when it is accepted and about 155% when it
escalates. Its saving is roughly (share accepted) minus 0.55. Sonnet alone,
unchecked, would save about 50% if it certifies. Whatever the search picks is
the result.

## Answer reader

`Answer:\s*\**\s*\$?\(?([A-J])\b` - the last match's letter. The offline grading uses
the proxy's own `extract_answer`, so the certification and the proxy read
answers the same way (pilot finding #2). A reply with no match is wrong.

## Steps and spending caps

Hard cap for the whole run: **$35**, from the API's own token counts at the
prices above. Each step has its own cap and stops cleanly at it.

| step | what | questions | cap |
| --- | --- | --- | --- |
| 0. probe | all three models, to measure real cost per question | 40 (part of step 1's set) | $1.50 |
| 1. calibrate | all three models: the team's graded history | 1,000 | $22 |
| 2. certify | `agreement_pairs.py` (Haiku+Sonnet pair, singles) then `freeze-policy --answer-pattern`, split 0.3 / 0.5 / 0.2, epsilon 0.04 and 0.02, delta 0.05 | - | $0 |
| 3. shadow | the real proxy in shadow mode, sample rate 1, policy at 0.04 | 300 new MC + 60 free text | $9 |
| 4. live | the real proxy routing through the policy | 200 new MC | $3 |

If the probe projects step 1 over its cap, step 1 is cut to the largest size that
fits, and this is reported. The four question sets do not overlap (seed 0).

## Questions and what counts as an answer

1. **Does a chain certify on real current models?** Report the chain, its
   certified bound, the test-part saving and loss, at 0.04 and 0.02. No pass
   mark: "nothing certifies" is a result.
2. **Does the certification hold on new questions?** In shadow mode, on the
   300 new questions: loss = share of questions where Opus is right and the
   policy's answer is wrong. Questions the policy gets right and Opus gets wrong
   do not offset it; this is the definition the certification uses
   (`degradation_rate`, `tine/cascade.py`). Pass: loss at or under epsilon, and the
   share answered by cheaper models within 7 points of the test-part estimate.
3. **Can the reader read real replies?** Share of replies with a match, per
   model. Pass: at least 98% for every model.
4. **What does shadow mode cost?** Shadow spend as a share of frontier spend on
   the same requests; added latency for the caller (should be none); the
   proxy's coverage counters; API errors.
5. **What happens to free text on the same token?** Pass: no free-text request
   gets a cheaper answer accepted. Report the money spent on cheaper stages that
   could never pass.
6. **Live routing.** Accuracy of the served answers against gold, against Opus's
   accuracy on the shadow set; cost per request against Opus; latency p50 and
   p95 against Opus in the shadow set; what `/v1/savings` reports.

## Handling

- The key is read from the environment by the person running the steps, never
  by an agent and never written to the repository.
- Prompts, replies and per-question rows stay under `artifacts/live-pilot/`
  (git-ignored). The results folder gets only summary tables.
- No model-written code is run. Nothing here executes model output.
