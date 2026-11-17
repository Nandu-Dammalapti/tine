# The ticket pilot again, without tickets that repeat a prompt example: plan, 2026-11-17

Written before any model was called for this run. Results go in `README.md` next to
this file; anything that changes from this plan is listed there as a deviation. Paid
from the team's credits key. Hard cap: **$9**.

## Why

In `../2026-11-17-ticket-pilot` the prompt showed 35 past tickets with their category.
Once names are replaced by tags, many tickets share their exact text, and some of the
evaluated tickets had the same text as one of those examples: 107 of the 1,000 history
tickets, 31 of the 300 shadow tickets and 34 of the 300 routed ones (text compared after
collapsing spaces, first 300 characters, as the prompt shows it). For those the prompt
held the answer. Every model was right on them, so they made the policy look safer than
it is on other tickets. That pilot also chose Sonnet alone, which sat exactly at the 4%
limit. This run removes those tickets and tests the policy that the clean data certifies.

## What is fixed before any call

Done offline, from the recorded answers, at no cost:

- **Clean history.** 893 tickets (the 1,000 minus the 107).
- **Certification on it**, same procedure as before (`agreement_pairs.py` with
  `--max-cost-share 0.6`, with and without single models; `freeze-policy`, split
  0.3 / 0.5 / 0.2, delta 0.05). At 4% the search now picks the pair in both cases:
  Haiku and Sonnet must agree, else Opus. Bound 3.46%, certify part 2.0% lost (450
  tickets), held-out test part 1.2%, estimated saving 33.9%, 10.4% sent to Opus.
  Sonnet alone no longer certifies. Nothing certifies at 2%.
- **Fresh tickets.** 530 tickets that were never sent to any model and do not repeat an
  example: 280 for shadow mode, 250 for live routing. 142 of the 530 repeat the text of
  a history ticket; results are also given without those.

## Paid steps

Same prompt, examples, models, prices and settings as the first ticket pilot
(`max_tokens` 512).

| step | tickets | cap |
| --- | --- | --- |
| Haiku on the first pilot's 300 shadow tickets (Sonnet and Opus are recorded), to score the pair on them | 300 | $0.25 |
| shadow mode through the proxy, the pair policy, sample rate 1 | 280 fresh | $5.50 |
| live routing with a random 30% frontier holdout | 250 fresh | $3.00 |

## Pass marks

1. Shadow loss on the 280 fresh tickets (Opus right, the policy wrong) at most 4%. Its
   one-sided 95% upper bound is reported too; with 280 tickets it will be above 4%
   unless the loss is under about 2%.
2. Share answered by the cheaper models within 7 points of the estimate (89.6%).
3. The reader matches at least 98% of each model's replies.
4. Live: measured saving with its interval, and accuracy of routed against held-out
   requests. Reported, no pass mark.

Also reported: the first pilot's shadow run regraded without its 31 repeated tickets
(Sonnet alone), and the pair scored on those 269 tickets.

## Limits known in advance

One public dataset from 2020; the models may have seen it; labels are noisy (Opus matches
them about 80% of the time). Tickets that repeat a history ticket's text remain in the
fresh sets because repeats are real traffic; they are reported apart.
