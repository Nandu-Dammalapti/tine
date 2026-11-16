# Shadow mode and live routing replayed on recorded answers, 2026-11-16

A wiring check of the proxy's shadow mode and of routing through a policy
(`docs/PROXY.md`), with no model called.
`scripts/shadow_replay.py` stands in for the provider with recorded xRouteBench
responses, serves the requests through the real proxy app with a frozen policy in
shadow mode, and reads the report back from `GET /v1/shadow`.

## Setup

- Policy: `results/2026-11-14-xroutebench/policy-multiple-choice-triples-eps02.json`
  (three cheap models must agree, then a second trio, then the frontier; certified at
  the 2% limit).
- Requests: the 1,414 multiple-choice questions of that policy's own held-out test
  part, each answered by the frontier model `cogito-v2-1-671b` first, as a customer's
  request would be. Sample rate 1. Prices from the benchmark's table.
- The stand-in provider returns, for each model, the answer that model recorded and
  its recorded token counts.

## Result

`shadow-replay-multiple-choice-triples.json`.

| | shadow report, from the proxy | live routing, from the ledger | the policy file's stored estimate |
| --- | --- | --- | --- |
| requests answered by cheaper models | 77.7% (would be) | 77.7% | 77.7% |
| cost against always using the frontier | 53.5% lower | 53.5% lower | 52.5% lower |
| accuracy (from the recorded grades) | 88.9% | 88.9% | 88.9% |
| frontier accuracy | 87.0% | 87.0% | 87.0% |
| tasks lost that the frontier solves | 0.99% | 0.99% | 0.99% |

The live-routing column (`route-replay-multiple-choice-triples.json`, `--mode route`)
serves the same 1,414 requests from the policy and adds up each request's answer cost
and routing cost as the ledger recorded them.

- **Both live paths reproduce the offline estimate, and each other.** Acceptance, accuracy and
  degradation match exactly. Savings are one point higher in the replay for a known
  reason: `gpt-oss-20b` sits in both stages; the stored estimate charges it twice and a
  live run calls it once per request.
- Of the answers the policy would have returned, 94.7% equal the frontier's answer.
  That is the only quality signal a live shadow report has, and it understates the
  policy here: where the two differ, the policy is right more often than the frontier
  (its accuracy is 1.9 points higher).
- Running shadow mode on all 1,414 requests cost 18% of the frontier's bill for them
  ($0.127 against $0.698). Shadow mode is not free; that is what the sample rate and
  the spending cap are for.

## What the replay caught

The first run reported 74% savings. A cheaper model's reply with no extractable answer
was being treated as a failed call and its cost dropped, which removed 198 of the most
expensive requests from the cost comparison. A reply that was paid for now keeps its
cost and simply fails the agreement check. Fixed before this result, with a test.

## Limits

- Recorded answers from one benchmark of open models; the dollar amounts are tiny.
- This checks that the proxy does what the policy says. It is not new evidence about
  the policy's quality, which rests on its certification.

- Note added 2026-11-16 after a code review: the proxy now treats a reply with no
  answer in it as a provider failure. The stand-in provider sends an empty reply where
  no answer was extracted, so a re-run of the replay reports 317 failed shadow calls
  where the recorded JSON has 0. The reviewer's re-run reproduced every money,
  acceptance and accuracy figure above in both modes.

## Reproduce

```bash
python scripts/shadow_replay.py --outcomes DATA/xroutebench-multiple-choice.jsonl \
  --answers DATA/xroutebench-multiple-choice-answers.csv --policy RUNS/policy.json \
  --prices DATA/xroutebench-prices.json --group-outcomes DATA/multiple-choice-triples.jsonl
# the same with --mode route for live routing
```
