# Policy corpus notes

Reviewer-facing notes on `policies/`. This file is **not** part of the
retrieval corpus — the loader only indexes `policies/POL-*.md`. Keep it that
way, or the agent gets a map of the holes it is supposed to fall into.

20 documents, stable IDs of the form `POL-XXX-NNN`, clause IDs of the form
`XXX-NNN.M` so a retrieved passage can be cited precisely.

## Deliberate conflict

One genuine contradiction is planted in the corpus, undeclared inside the
documents themselves:

- **REF-001.3** says annual *renewal* charges are refundable within 14 days of
  the renewal date.
- **CAN-002.2** says annual subscriptions are non-refundable once the term has
  begun.

A customer refunded 10 days after an unwanted auto-renewal is covered by both.
Ticket T-016 sits exactly here.

**Ruling (label review):** the contradiction stays, undeclared. Neither clause
declares precedence over the other, and resolving a genuine conflict between
two clauses is not within a support agent's authority — the correct behaviour
is to surface it, so T-016 is labelled escalate. The ticket tests whether the
agent *notices* the contradiction, rather than confidently citing whichever
clause retrieval happened to return first. Do not "fix" the corpus here.

## Deliberate gaps — situations no document covers

These are the retrieval-guardrail traps. Note that a gap does not automatically
mean escalate — label review split them, and the split is the interesting part.

| Gap | Nearest documents | Ticket | Ruling |
|---|---|---|---|
| Team → Free directly (PLN-001 covers Pro→Free and Team→Pro only) | PLN-001, TIE-001 | T-018 | **escalate** — composing PLN-001.1 and PLN-001.2 into an unwritten path is the confident inference we are testing against. POL-PLN-001 stays unamended. |
| Grace-period extension for a **Pro** account (PAY-001.5 grants it to Team only) | PAY-001 | T-019 | **resolve, declining** — PAY-001.5 *enumerates* a Team permission and Pro is not in it. An enumeration that excludes this plan is an answer, not a gap. |
| Whether the refund window runs from trial start or from first charge on conversion | TRI-001, REF-001 | T-017 | **pending_approval** — a refund window cannot start before the charge exists, so it runs from the charge date. Corpus silence is fine; the label holds without it. |
| Proration credit when downgrading to Free (no future invoice to offset against) | PRO-001.5, PLN-001 | — | unlabelled |
| A seat removed and re-added in the same cycle — billed once or twice | SEA-002.2, SEA-002.5 | — | unlabelled |
| Overlap of the 90-day cancellation window and the 30-day downgrade window | DAT-001.2, DAT-001.3 | — | unlabelled |
| Member deletion request vs. owner's retention of workspace content | DAT-003.4 | — | unlabelled |
| Duplicate/incorrect billing caused by a system fault | BIL-001, ESC-001.6 | T-015 | **escalate** — a system fault is outside policy entirely. |

The four gaps with no ticket attached are deliberate slack: they give room to
add tickets later without touching the corpus, and they mean the corpus is not
perfectly fitted to the current 22 tickets.

One more policy is load-bearing without being a gap: **SLA-001.5** ("no service
credits are issued for a missed response target") is the dispositive clause for
T-020, and that ticket is labelled **resolve, declining**, even though the
customer's complaint is real and verifiable in their ticket history. If an
explicit prohibition still routed to approval because a goodwill route exists
elsewhere in the corpus, no policy would ever be dispositive and
`pending_approval` would stop carrying meaning.

## Partial overlaps (consistent, but multiple documents apply)

These are not contradictions. They exist so that "the relevant policy" is
rarely a single document, which is what makes `relevant_policy_ids` a
meaningful label and tool-call precision a meaningful metric.

- Refund windows: REF-001 ↔ REF-002 ↔ ESC-001 (window, exception, authority)
- Downgrades: PLN-001 ↔ PRO-001 ↔ DAT-001 (mechanics, money, data)
- Seat changes: SEA-001 ↔ SEA-002 ↔ PRO-001 ↔ PLN-001
- Non-payment: PAY-001 ↔ PAY-002 ↔ DAT-001
- Cancellation: CAN-001 ↔ CAN-002 ↔ REF-001 ↔ DAT-001
- Data: DAT-001 ↔ DAT-002 ↔ DAT-003

## Money figures used by tickets

Fixed so refund amounts land on both sides of the ESC-001 thresholds:

- Pro monthly: USD 29 — under the USD 50 self-serve ceiling (ESC-001.1)
- Pro annual: USD 290 — manager band (ESC-001.2), used by T-017
- Team: USD 22 per seat per month
- Team monthly, 6 seats: USD 132 — manager band, used by T-020
- Team monthly, 8 seats: USD 176 — manager band, used by T-021
- Team monthly, 12 seats: USD 264 — manager band, used by T-022
- Team annual, 6 seats: USD 1,584 — above USD 500, finance band (ESC-001.3)

Each `pending_approval` ticket uses a distinct amount so that a failure can be
traced to one ticket rather than to a shared figure.

## Account fixtures

`customers.json` holds the account state `lookup_customer` returns and the
prior contacts `search_past_tickets` searches. Every ticket's `customer_id`
has a record, because a tool that cannot return anything cannot inform an
answer — crediting such a call would inflate tool-call precision.

Each record also carries an `owner_email`, paired with a `requester_email` on
every ticket. ESC-001.4 turns on whether the requester is the verified account
owner, so that comparison has to be mechanical — otherwise T-013's escalate
label depends on an agent reading intent out of the prose. T-013 (a colleague
of an owner on leave) and T-010 (outside counsel) are the two non-owner
requests; only T-013 escalates on identity alone.

`loader.load_tickets` enforces this: a ticket listing `search_past_tickets` in
`expected_tools` whose customer has no `past_tickets` fails the load, as does a
non-owner request labeled anything other than escalate. Only
T-009, T-015 and T-020 carry history, and in each case the history changes the
answer — two ignored contacts before a chargeback, a prior unexplained $58 line
item, and two confirmed slow responses respectively.
