# A live pilot on one company's real IT support tickets: plan, 2026-11-17

Written before any model was called for this run. Results go in `README.md` next to
this file; anything that changes from this plan is listed there as a deviation. Paid
from the team's credits key. Hard cap: **$15**.

## Why

Every live result so far is on benchmark questions. This is the first run on real
traffic from one company: tickets typed by its users and labelled by its own staff.

## Data

Zenodo record 7648117, "Classification of IT Support Tickets" (Leonardo Santiago
Benitez Pereira, CC BY 4.0). 2,229 support tickets from one IT support company in
Brazil, each given one of 7 categories by three of its IT staff in 2020. Tickets are in
several languages, mainly English, German, Portuguese and Spanish. Personal details
were replaced by tags by the author. The author's own trained classifier reaches 79%
on these labels, so the labels are noisy and no model should be expected near 100%.

What the data looks like, measured before any model call:
- median ticket 72 characters; 830 of the 2,229 share their exact text with another
  ticket once names are tags (templated requests such as file-share access);
- categories are uneven: "Support general" 762, "Fileservice" 684, the other five
  103-222 each.

## The customer and the setup

The company's triage step sends every new ticket to `claude-opus-5-5` to pick the
category. The prompt lists the 7 categories as options A-G, shows 35 past tickets with
their category (5 per category, set aside for the prompt and used nowhere else), then
the new ticket with its id, and asks for one line, `Answer: X`
(`scripts/ticket_pilot_prep.py`). Same models, prices, settings and steps as the first
pilot (`../2026-11-17-live-pilot/PLAN.md`): `claude-haiku-5-5` and `claude-sonnet-5-5`
as the cheaper models, the reader `Answer:\s*\**\s*\$?\(?([A-J])\b`, answers compared
exactly.

| step | tickets | how | projected cost |
| --- | --- | --- | --- |
| probe | 20 (part of the history) | direct calls, to check the prompt and reader | $0.25 |
| graded history | 1,200, all three models | Message Batches API, half price | $7 |
| certify | - | `agreement_pairs.py` with singles, `freeze-policy`, split 0.3 / 0.5 / 0.2, limits 0.04 and 0.02 | $0 |
| shadow | 300 new | through the proxy, sample rate 1, 3 in flight | $3.50 |
| live | 400 new | routed, with a random 30% frontier holdout | $2.50 |

Shadow and live run only if a chain certifies at 0.04. "Right" means the category the
company's staff gave.

## Pass marks, fixed now

1. Shadow loss on new tickets (Opus right, the policy wrong) at most 4%.
2. Share answered by cheaper models within 7 points of the certified test-part estimate.
3. The reader matches at least 98% of each model's replies.
4. Live: accuracy of routed against held-out requests, and the measured saving with its
   95% interval, reported. No pass mark; with about 120 held out the interval is wide.

## Reported as well

- Each model's accuracy against the staff labels, overall and per category.
- Shadow results split by whether the ticket's exact text also appears in the graded
  history or the prompt's examples (97 of the 300 shadow tickets do). Repeats are real
  traffic, but a result that held only on repeats would not carry to new kinds of ticket.
- How often Opus itself disagrees with the staff label, since that bounds what "loss
  against the frontier" can mean here.

## Limits known in advance

- One company, one task, labels from 2020, tags in place of names. Not a customer of ours.
- The data is public, so the models may have seen it.
- The 35 examples were drawn at random, not chosen by the company.

Nothing here executes model output. The tickets, prompts and replies stay under
`artifacts/` (git-ignored); only summary tables go in this folder.
