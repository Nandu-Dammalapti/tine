# The ticket pilot again, without tickets that repeat a prompt example, 2026-11-17

**Paid run.** `claude-haiku-5-5`, `claude-sonnet-5-5` and `claude-opus-5-5` through the
Anthropic API and Tine's proxy, on the team's credits key.

- **Spend:** **$6.45** from the API's token counts, against a $9 cap.
- **The plan:** written before any call ([`PLAN.md`](PLAN.md)). No deviations.
- **Data:** Zenodo record 7648117, "Classification of IT Support Tickets" (Leonardo
  Santiago Benitez Pereira, CC BY 4.0). Summary table: `results.csv`. Ticket text,
  prompts and replies stay local.

## Bottom line

The first ticket pilot (`../2026-11-17-ticket-pilot`) was flattered by tickets whose text
also appeared as an example in the prompt. With those removed:

- **Sonnet alone does not certify.** On the first pilot's own shadow tickets, without
  the 31 repeats, it lost 12 of 269 (4.5%), over the 4% limit. The 49% saving reported
  for it should not be quoted.
- **The pair certifies at 4% and holds on fresh tickets.** Haiku and Sonnet must agree,
  else Opus. On 280 tickets never sent to any model before, the loss against Opus was
  **0.7%** (2 of 280; one-sided 95% upper bound 2.2%).
- **The measured saving under live routing is 35%** (95% interval 29% to 40%), from a
  random 30% frontier holdout on 250 more fresh tickets.

## What was removed

A ticket counts as repeating an example when its text, with spaces collapsed and cut to
the 300 characters the prompt shows, equals the text of one of the prompt's examples
(30 distinct texts among the 35 examples). That removed 107 of the 1,000 history
tickets, and it applies to 31 of the first pilot's 300 shadow tickets and 34 of its 300
routed ones.

## Results

`results.csv`.

### Certification on the 893 clean history tickets (no model called)

Same procedure as the first pilot. With and without single-model candidates, the search
picks the pair at 4%. Nothing certifies at 2%.

| | value |
| --- | --- |
| certify part | 9 of 450 lost (2.0%), bound 3.46% |
| held-out test part | 1.2% lost, 89.6% answered by the cheaper models, 33.9% estimated saving |

### Shadow mode on 280 fresh tickets

| pre-registered check | pass mark | result | |
| --- | --- | --- | --- |
| loss against Opus | at most 4% | 0.7% (2 of 280), upper bound 2.2% | pass |
| share answered by cheaper models | within 7 points of 89.6% | 90.4% | pass |
| reader matched replies | at least 98% for each model | 100% for all three | pass |

- Policy accuracy against the dataset's labels 79.6%, Opus 80.0%; projected saving 35.3%.
- Of the 280, 76 repeat the text of a history ticket (1 lost, 1.3%) and 204 have new
  text (1 lost, 0.5%). The result does not rest on repeats.
- The drift check (the version in this checkout) reported within the limit: 5 of 253
  accepted answers differed from Opus's. No provider errors, no retries.

### Live routing on 250 fresh tickets, 30% held out at random

| | routed (175) | held out (75) |
| --- | --- | --- |
| answered by the cheaper models | 90.9% | - |
| accuracy against the labels | 81.1% | 80.0% |
| mean cost per ticket, routing included | $0.0066 | $0.0101 |

Measured saving 34.7% (95% interval 29.0% to 40.5%). The two groups are different
tickets, so the accuracy columns are not a paired comparison; the shadow loss above is.

### The first pilot's shadow tickets, regraded

Haiku was asked the first pilot's 300 shadow tickets ($0.08) so the pair could be scored
on them with the recorded Sonnet and Opus replies. Without the 31 repeats (269 tickets):
Sonnet alone lost 12 (4.5%); the pair lost 3 (1.1%) and answered 85.9% without Opus.

## What this means

On this workload the supported claim is: the pair policy, certified at a 4% limit, lost
about 1% against Opus on new tickets and saved about a third of the cost, measured. The
saving is smaller than on the benchmarks because the prompt is long and the answer is
one line, so Sonnet costs half of Opus on every ticket it reads.

## Limits

- One public dataset from 2020, one task, one day. The models may have seen it. Not a
  customer.
- Labels are noisy: Opus matches them about 80% of the time, so "loss" includes cases
  where the label, not the policy, is at fault.
- 280 shadow tickets and 75 held-out requests: the bounds and the interval are wide.
- The certification's history was re-cut after the fact (the repeats were found by a
  review of the first pilot). The fresh shadow and live sets were fixed before any call.

## Reproduce

Build the sets with `scripts/ticket_pilot_prep.py`, drop tickets whose text equals an
example's, then follow `../2026-11-17-ticket-pilot/README.md` with
`agreement_pairs.py --max-cost-share 0.6` and the pair policy.
