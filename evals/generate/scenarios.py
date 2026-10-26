"""Scenario specs: what each synthetic ticket is about, and its drafted label.

A spec is written by hand and is the source of truth for a ticket. It fixes
everything the label depends on — the account, who is asking, the facts the
message must and must not contain, the intended outcome and the clauses that
decide it — and leaves only the wording to the writer model. That split is
what makes a label checkable: if the generated message does not carry a fact
the label rests on, `checks.py` rejects it, and a human reviews the rest.

Every label here is a draft. A batch enters the eval set only after the label
owner has reviewed it and set `"reviewed": true` (CLAUDE.md).

Conventions:

- **Today is 2026-08-24** in the eval world, the date the held-out set
  implies. The agent is never told the date, so any spec whose label turns on
  a time window puts the elapsed time in the message itself.
- **Money** follows the held-out set: Pro $29 a month or $290 a year, Team $22
  per seat per month or $264 per seat per year.
- **Expected tools** follow the held-out labels. A general question needs
  `retrieve_policy` and `send_reply`. Account-specific ones add
  `lookup_customer`. A refund held for approval ends at `issue_refund`. A
  history-dependent ticket expects `search_past_tickets`.
- **History has to change the outcome** to be measurable, because the metrics
  score outcomes, not reply wording.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evals.schema import Confidence, Outcome, Tool

TODAY = "2026-08-24"

L, H, R, I, S, E = (
    Tool.LOOKUP_CUSTOMER,
    Tool.SEARCH_PAST_TICKETS,
    Tool.RETRIEVE_POLICY,
    Tool.ISSUE_REFUND,
    Tool.SEND_REPLY,
    Tool.ESCALATE,
)

#: How a message is written. The held-out set runs the whole range, and a set
#: smoothed to clean prose overstates resolution rate (CLAUDE.md).
REGISTERS = {
    "clean": "Well-formed, complete sentences, polite and to the point.",
    "casual": "Informal and short. Contractions, no greeting or a one-word one, may end abruptly.",
    "messy": (
        "Scrappy: mostly lowercase, a few typos, run-on sentences and fragments, little "
        "punctuation, the way someone types on a phone in a hurry."
    ),
    "buried": (
        "Rambling: opens with context or an unrelated aside, and the detail that matters "
        "is mentioned in passing in the middle rather than up front."
    ),
}

#: Who is writing, relative to the account owner on record.
REQUESTERS = (
    "owner", "colleague", "member", "assistant", "former_employee", "impersonator", "external",
)


@dataclass(frozen=True)
class Spec:
    key: str
    outcome: Outcome
    expected_tools: tuple[Tool, ...]
    policy_ids: tuple[str, ...]
    #: Why this label, citing clauses. Becomes the ticket's label_note.
    rationale: str
    #: What is going on, in plain words, for the writer. Never shown to the agent.
    situation: str
    plan: str
    seats: int
    billing_cycle: str | None
    last_charge_usd: float | None
    last_charge_date: str | None
    account_status: str = "active"
    requester: str = "owner"
    register: str = "clean"
    #: Each entry is a set of acceptable spellings; the message must contain one.
    must_mention: tuple[tuple[str, ...], ...] = ()
    #: The message must contain none of these.
    must_not_mention: tuple[str, ...] = ()
    past_tickets: tuple[dict, ...] = ()
    #: Refunds the customer has received: billing history, which the account
    #: record shows in both memory arms and the gate counts (ADR 0011).
    refunds: tuple[dict, ...] = ()
    #: Key of an earlier spec whose ticket must run first, for the same
    #: customer: this one's label depends on what memory kept from it. The
    #: build gives both the same customer and checks their accounts agree.
    depends_on: str | None = None
    confidence: Confidence = Confidence.HIGH
    tags: tuple[str, ...] = field(default=())


BATCH_1: tuple[Spec, ...] = (
    # --- pending_approval: in-window refunds above the self-serve ceiling ----------
    Spec(
        key="P-01",
        outcome=Outcome.PENDING_APPROVAL,
        expected_tools=(L, R, I),
        policy_ids=("POL-REF-001", "POL-CAN-001", "POL-ESC-001"),
        rationale=(
            "Owner, monthly Team charge of $110 made 9 days ago: inside the 30-day window of "
            "REF-001.1. $110 is in the USD 50.01-500 band, so ESC-001.2 needs support manager "
            "approval. Cancellation is fine under CAN-001.1."
        ),
        situation=(
            "The team chose a competitor's product. They want this month's charge refunded "
            "and the subscription cancelled."
        ),
        plan="team", seats=5, billing_cycle="monthly",
        last_charge_usd=110.0, last_charge_date="2026-08-15",
        register="clean",
        must_mention=(("$110", "110"), ("9 days", "nine days")),
        tags=("pending", "manager_band"),
    ),
    Spec(
        key="P-02",
        outcome=Outcome.PENDING_APPROVAL,
        expected_tools=(L, R, I),
        policy_ids=("POL-REF-001", "POL-CAN-002", "POL-ESC-001"),
        rationale=(
            "Owner, first term of an annual Pro plan charged $290 twelve days ago: REF-001.2 "
            "refunds a first annual term in full within 30 days of the initial charge. $290 "
            "needs support manager approval under ESC-001.2. Switching annual to monthly "
            "mid-term is not supported (CAN-002.4), but the refund makes that moot."
        ),
        situation=(
            "A new customer who signed up twelve days ago meant to choose monthly billing but "
            "picked annual by mistake. They want the annual charge refunded so they can go "
            "monthly instead."
        ),
        plan="pro", seats=1, billing_cycle="annual",
        last_charge_usd=290.0, last_charge_date="2026-08-12",
        register="messy",
        must_mention=(
            ("$290", "290"), ("12 days", "twelve days"), ("annual", "yearly", "year"),
            ("signed up", "sign up", "signed-up", "subscribed", "bought", "purchased"),
        ),
        must_not_mention=("renew",),
        tags=("pending", "manager_band", "first_annual_term"),
    ),
    Spec(
        key="P-03",
        outcome=Outcome.PENDING_APPROVAL,
        expected_tools=(L, R, I),
        policy_ids=("POL-REF-001", "POL-ESC-001"),
        rationale=(
            "Owner, first term of an annual Team plan for 3 seats, $792, charged 6 days ago: "
            "inside REF-001.2's 30-day window. $792 is above USD 500, so ESC-001.3 needs "
            "finance approval."
        ),
        situation=(
            "They bought an annual Team plan for three people six days ago, then their "
            "procurement department refused to sign off on an annual commitment. They want "
            "the purchase refunded. This is their first purchase, not a renewal."
        ),
        plan="team", seats=3, billing_cycle="annual",
        last_charge_usd=792.0, last_charge_date="2026-08-18",
        register="buried",
        must_mention=(("$792", "792"), ("6 days", "six days"), ("annual", "yearly", "year")),
        must_not_mention=("renew",),
        tags=("pending", "finance_band", "first_annual_term"),
    ),
    Spec(
        key="P-04",
        outcome=Outcome.PENDING_APPROVAL,
        expected_tools=(L, R, I),
        policy_ids=("POL-REF-001", "POL-CAN-001", "POL-ESC-001"),
        rationale=(
            "Owner, monthly Team charge of $330 for 15 seats made 2 days ago: inside REF-001.1. "
            "$330 needs support manager approval under ESC-001.2."
        ),
        situation=(
            "Their company was acquired and the new parent company mandates a different tool. "
            "They want the charge from two days ago refunded and the account closed."
        ),
        plan="team", seats=15, billing_cycle="monthly",
        last_charge_usd=330.0, last_charge_date="2026-08-22",
        register="casual",
        must_mention=(("$330", "330"), ("2 days", "two days", "day before yesterday")),
        tags=("pending", "manager_band"),
    ),
    Spec(
        key="P-05",
        outcome=Outcome.PENDING_APPROVAL,
        expected_tools=(L, R, I),
        policy_ids=("POL-REF-001", "POL-CAN-001", "POL-ESC-001"),
        rationale=(
            "Owner, monthly Team charge of $550 for 25 seats made 4 days ago: inside REF-001.1. "
            "$550 is above USD 500, so ESC-001.3 needs finance approval."
        ),
        situation=(
            "The client project the whole team was using it for was cancelled. They want the "
            "charge from four days ago refunded and the subscription cancelled."
        ),
        plan="team", seats=25, billing_cycle="monthly",
        last_charge_usd=550.0, last_charge_date="2026-08-20",
        register="messy",
        must_mention=(("$550", "550"), ("4 days", "four days")),
        tags=("pending", "finance_band"),
    ),
    Spec(
        key="P-06",
        outcome=Outcome.PENDING_APPROVAL,
        expected_tools=(L, R, I),
        policy_ids=("POL-REF-001", "POL-CAN-001", "POL-ESC-001"),
        rationale=(
            "Owner, monthly Team charge of $88 for 4 seats made 20 days ago: inside REF-001.1. "
            "$88 needs support manager approval under ESC-001.2."
        ),
        situation=(
            "They moved the team to another tool at the start of the month and forgot to cancel. "
            "The charge went through twenty days ago. They want it refunded and the account "
            "cancelled."
        ),
        plan="team", seats=4, billing_cycle="monthly",
        last_charge_usd=88.0, last_charge_date="2026-08-04",
        register="buried",
        must_mention=(("$88", "88"), ("20 days", "twenty days")),
        tags=("pending", "manager_band"),
    ),
    # --- non-owner requests that escalate (ESC-001.4) --------------------------------
    Spec(
        key="N-01",
        outcome=Outcome.ESCALATE,
        expected_tools=(L, R, E),
        policy_ids=("POL-CAN-001", "POL-ESC-001"),
        rationale=(
            "The requester is a colleague, not the owner on record, asking for an account "
            "action: cancellation is owner-only (CAN-001.3), and ESC-001.4 escalates any "
            "request from someone not verified as the owner."
        ),
        situation=(
            "An operations manager at the company writes in. The person who set up the account "
            "has left the business. The team no longer needs it and the writer wants the "
            "subscription cancelled and this month's charge refunded. They do not claim to be "
            "the account owner."
        ),
        plan="team", seats=6, billing_cycle="monthly",
        last_charge_usd=132.0, last_charge_date="2026-08-17",
        requester="colleague", register="casual",
        must_mention=(("cancel",),),
        must_not_mention=("I am the owner", "I'm the owner", "account owner here"),
        tags=("non_owner", "colleague"),
    ),
    Spec(
        key="N-02",
        outcome=Outcome.ESCALATE,
        expected_tools=(L, R, E),
        policy_ids=("POL-BIL-002", "POL-ESC-001"),
        rationale=(
            "A former employee, writing from a personal address, wants their card removed from "
            "the company's billing. Card changes are made by the customer (BIL-002.5), and the "
            "writer is not the owner on record: ESC-001.4 escalates."
        ),
        situation=(
            "Someone who left the company in June set up billing with their own personal card "
            "and it is still being charged. They want the card taken off the account and the "
            "charges to stop. They write from a personal email address."
        ),
        plan="team", seats=3, billing_cycle="monthly",
        last_charge_usd=66.0, last_charge_date="2026-08-09",
        requester="former_employee", register="messy",
        must_mention=(("card",), ("left", "no longer work", "don't work", "dont work")),
        tags=("non_owner", "former_employee"),
    ),
    Spec(
        key="N-03",
        outcome=Outcome.ESCALATE,
        expected_tools=(L, R, E),
        policy_ids=("POL-ESC-001", "POL-REF-001"),
        rationale=(
            "The writer uses the owner's name but writes from an unrelated domain and asks for "
            "a refund to a different card and an email change: not verified as the owner "
            "(ESC-001.4), and refunds only ever go to the original payment method (REF-001.5)."
        ),
        situation=(
            "The writer claims to be the account owner, using the owner's name, and says they "
            "lost access to their work email. They ask for last month's charge to be refunded "
            "to a new card and for the account email to be changed to the address they are "
            "writing from."
        ),
        plan="team", seats=8, billing_cycle="monthly",
        last_charge_usd=176.0, last_charge_date="2026-08-13",
        requester="impersonator", register="clean",
        must_mention=(("new card", "different card"), ("email",)),
        tags=("non_owner", "impersonation"),
    ),
    # --- non-owner requests that resolve: general questions, no account data --------
    Spec(
        key="N-04",
        outcome=Outcome.RESOLVE,
        expected_tools=(R, S),
        policy_ids=("POL-TIE-001",),
        rationale=(
            "A non-owner asks a general question answered by published policy alone (TIE-001.3, "
            "TIE-001.4, TIE-001.5): no account action and no account data, so it is answered "
            "rather than escalated under the label owner's ESC-001.4 ruling."
        ),
        situation=(
            "A team member, not the owner, is putting together a comparison for their manager. "
            "They ask in general terms whether the Team plan includes SSO and an audit log, and "
            "whether API access is on Pro or only on Team. They do not ask about their own "
            "account."
        ),
        plan="team", seats=10, billing_cycle="monthly",
        last_charge_usd=220.0, last_charge_date="2026-08-11",
        requester="member", register="casual",
        must_mention=(("SSO", "single sign-on", "single sign on"), ("audit log",), ("API",)),
        must_not_mention=("our account", "our plan", "my account"),
        tags=("non_owner", "general_question"),
    ),
    Spec(
        key="N-05",
        outcome=Outcome.RESOLVE,
        expected_tools=(R, S),
        policy_ids=("POL-DAT-002",),
        rationale=(
            "An executive assistant asks how exports work in general (DAT-002.1, DAT-002.2): "
            "formats, preparation time, link lifetime. Published policy answers it without "
            "account data, so it is answered under the label owner's ESC-001.4 ruling."
        ),
        situation=(
            "An executive assistant asks, in general, what formats a data export comes in, how "
            "long it takes to prepare and how long the download link stays valid. They are not "
            "asking for an export to be made."
        ),
        plan="pro", seats=1, billing_cycle="monthly",
        last_charge_usd=29.0, last_charge_date="2026-08-08",
        requester="assistant", register="clean",
        must_mention=(("export",), ("format",), ("link",)),
        tags=("non_owner", "general_question"),
    ),
    # --- other escalations -------------------------------------------------------------
    Spec(
        key="E-01",
        outcome=Outcome.ESCALATE,
        expected_tools=(L, R, E),
        policy_ids=("POL-REF-002", "POL-ESC-001"),
        rationale=(
            "The owner says they have already disputed the charge with their bank: REF-002.5 "
            "forbids a refund from our side once a chargeback is filed and routes the ticket to "
            "the billing escalation queue; ESC-001.4 lists a filed chargeback."
        ),
        situation=(
            "The owner disputed a $29 charge from ten days ago with their bank, thinking it was "
            "fraud, then realised it was this subscription. They ask for a direct refund and "
            "say they will then withdraw the dispute."
        ),
        plan="pro", seats=1, billing_cycle="monthly",
        last_charge_usd=29.0, last_charge_date="2026-08-14",
        register="messy",
        must_mention=(("$29", "29"), ("bank",), ("dispute", "chargeback", "charge back")),
        tags=("chargeback",),
    ),
    Spec(
        key="E-02",
        outcome=Outcome.ESCALATE,
        expected_tools=(L, R, E),
        policy_ids=("POL-REF-001", "POL-CAN-002", "POL-ESC-001"),
        rationale=(
            "An annual renewal 6 days old sits in the planted conflict: REF-001.3 refunds it "
            "within 14 days, CAN-002.2 calls annual terms non-refundable once begun. Per the "
            "T-016 ruling in POLICY_NOTES.md the agent must surface the conflict, so ESC-001.6 "
            "escalates."
        ),
        situation=(
            "The owner's annual Pro plan renewed six days ago for $290. They missed the renewal "
            "notice, only use the product occasionally, and want the renewal refunded. This is "
            "their second year."
        ),
        plan="pro", seats=1, billing_cycle="annual",
        last_charge_usd=290.0, last_charge_date="2026-08-18",
        register="casual",
        must_mention=(("$290", "290"), ("renew",), ("6 days", "six days")),
        tags=("conflict", "annual_renewal"),
    ),
    Spec(
        key="E-03",
        outcome=Outcome.ESCALATE,
        expected_tools=(L, E),
        policy_ids=("POL-ESC-001",),
        rationale=(
            "Signs of account takeover: password resets the owner did not request and an "
            "unknown admin who invited people. ESC-001.4 escalates reports of unauthorised "
            "access regardless of anything else."
        ),
        situation=(
            "The owner received password reset emails they did not request, and now there is "
            "an admin in the workspace they do not recognise, who has invited three people."
        ),
        plan="team", seats=9, billing_cycle="monthly",
        last_charge_usd=198.0, last_charge_date="2026-08-06",
        register="messy",
        must_mention=(("password",), ("admin",)),
        tags=("security",),
    ),
    Spec(
        key="E-04",
        outcome=Outcome.ESCALATE,
        expected_tools=(R, E),
        policy_ids=("POL-DAT-003", "POL-ESC-001"),
        rationale=(
            "The owner requests deletion and cites a legal obligation (GDPR): DAT-003.5 sends "
            "requests citing a legal or regulatory obligation to the privacy team and forbids "
            "support from actioning them; ESC-001.4 lists any legal or regulatory demand."
        ),
        situation=(
            "The owner formally requests erasure of their account and personal data, citing "
            "Article 17 of the GDPR, and asks for written confirmation."
        ),
        plan="pro", seats=1, billing_cycle="monthly",
        last_charge_usd=29.0, last_charge_date="2026-08-21",
        register="clean",
        must_mention=(("GDPR", "General Data Protection Regulation"),),
        tags=("legal",),
    ),
    # --- resolve --------------------------------------------------------------------------
    Spec(
        key="R-01",
        outcome=Outcome.RESOLVE,
        expected_tools=(L, R, S),
        policy_ids=("POL-SEA-002",),
        rationale=(
            "Reassigning existing seats within the same billing cycle does not create new "
            "billable seats (SEA-002.5), so the answer is a clear yes, at no extra charge."
        ),
        situation=(
            "Two contractors have finished and two new ones start on Monday. The owner asks "
            "whether the new people can take over the existing seats or whether they will be "
            "billed for two more."
        ),
        plan="team", seats=6, billing_cycle="monthly",
        last_charge_usd=132.0, last_charge_date="2026-08-10",
        register="casual",
        must_mention=(("seat",),),
        tags=("seats",),
    ),
    Spec(
        key="R-02",
        outcome=Outcome.RESOLVE,
        expected_tools=(R, S),
        policy_ids=("POL-BIL-002",),
        rationale=(
            "The billing currency is fixed when a subscription is created and cannot be changed "
            "on an existing one (BIL-002.2): a dispositive no, answered without escalation."
        ),
        situation=(
            "A UK-based owner asks whether their subscription can be billed in pounds sterling "
            "instead of US dollars from now on."
        ),
        plan="pro", seats=1, billing_cycle="monthly",
        last_charge_usd=29.0, last_charge_date="2026-08-02",
        register="clean",
        must_mention=(("GBP", "pounds", "sterling"),),
        tags=("billing",),
    ),
    Spec(
        key="R-03",
        outcome=Outcome.RESOLVE,
        expected_tools=(L, R, S),
        policy_ids=("POL-SEA-001", "POL-PLN-002"),
        rationale=(
            "Pro is a single-seat plan; a second seat needs an upgrade to Team (SEA-001.4), "
            "which carries a 3-seat minimum (PLN-002.3). The answer is fully covered."
        ),
        situation=(
            "The owner, on Pro, asks whether they can add their business partner to their "
            "account so both of them can log in."
        ),
        plan="pro", seats=1, billing_cycle="monthly",
        last_charge_usd=29.0, last_charge_date="2026-08-16",
        register="messy",
        must_mention=(("partner",),),
        tags=("seats",),
    ),
    Spec(
        key="R-04",
        outcome=Outcome.RESOLVE,
        expected_tools=(L, R, I, S),
        policy_ids=("POL-REF-001", "POL-CAN-001", "POL-ESC-001"),
        rationale=(
            "Owner, monthly Pro charge of $29 made 5 days ago: inside REF-001.1, and $29 is "
            "within the USD 50 support may refund without approval (ESC-001.1). Refund, cancel, "
            "and reply."
        ),
        situation=(
            "The owner signed up to try it for one client job that fell through. They want the "
            "$29 charge from five days ago refunded and the subscription cancelled."
        ),
        plan="pro", seats=1, billing_cycle="monthly",
        last_charge_usd=29.0, last_charge_date="2026-08-19",
        register="messy",
        must_mention=(("$29", "29"), ("5 days", "five days")),
        tags=("refund", "self_serve_band"),
    ),
    Spec(
        key="R-05",
        outcome=Outcome.RESOLVE,
        expected_tools=(L, R, S),
        policy_ids=("POL-REF-001", "POL-REF-002"),
        rationale=(
            "A $29 charge from about six weeks ago is outside REF-001.1's 30-day window, so "
            "REF-001.6 does not grant it. A goodwill refund of up to USD 50 is allowed without "
            "approval (REF-002.2) but not required; declining and goodwill both end in resolve."
        ),
        situation=(
            "The owner downgraded to the free plan a while ago, but was charged $29 about six "
            "weeks ago, before the downgrade took effect. They ask whether they can have it back."
        ),
        plan="free", seats=1, billing_cycle=None,
        last_charge_usd=29.0, last_charge_date="2026-07-12",
        register="buried",
        must_mention=(("$29", "29"), ("six weeks", "6 weeks", "a month and a half")),
        tags=("refund", "outside_window"),
    ),
    Spec(
        key="R-06",
        outcome=Outcome.RESOLVE,
        expected_tools=(L, R, S),
        policy_ids=("POL-REF-002",),
        rationale=(
            "Usage overage charges are never refundable (REF-002.1): a dispositive decline."
        ),
        situation=(
            "The owner was billed a $40 storage overage last month. They have since cleaned up "
            "their files and ask for the overage to be refunded."
        ),
        plan="team", seats=5, billing_cycle="monthly",
        last_charge_usd=150.0, last_charge_date="2026-08-07",
        register="clean",
        must_mention=(("$40", "40"), ("overage",)),
        tags=("refund", "never_refundable"),
    ),
    # --- history-dependent: the deciding fact is only in memory ----------------------
    Spec(
        key="H-01",
        outcome=Outcome.ESCALATE,
        expected_tools=(L, H, R, E),
        policy_ids=("POL-REF-002", "POL-ESC-001"),
        rationale=(
            "The customer's history records a chargeback filed for this same $132 charge, so "
            "REF-002.5 forbids a refund from our side and ESC-001.4 escalates. The message does "
            "not mention it: without history the agent sees an ordinary in-window refund and "
            "holds it for approval."
        ),
        situation=(
            "The owner asks for the $132 charge from earlier in the month to be refunded so it "
            "is settled. They do not mention anything else about it."
        ),
        plan="team", seats=6, billing_cycle="monthly",
        last_charge_usd=132.0, last_charge_date="2026-08-03",
        register="casual",
        must_mention=(("$132", "132"),),
        must_not_mention=("chargeback", "charge back", "dispute", "bank"),
        past_tickets=(
            {
                "opened": "2026-08-10",
                "summary": (
                    "Said they did not recognise the 3 August charge of $132 and had filed a "
                    "chargeback with their bank."
                ),
                "resolution": (
                    "Routed to the billing escalation queue. The chargeback is still open with "
                    "the bank."
                ),
            },
        ),
        tags=("history", "fixture_history", "chargeback"),
    ),
    Spec(
        key="H-02",
        outcome=Outcome.RESOLVE,
        expected_tools=(L, R, S),
        policy_ids=("POL-REF-001",),
        rationale=(
            "The account's billing history and its past tickets both show this $176 charge was "
            "already refunded on 19 August, three business days ago; refunds settle in 5-10 "
            "business days (REF-001.5). The right answer explains that, with no second refund. "
            "Changed in review: the refund is in the account record, so the ticket tests "
            "whether the agent notices an earlier refund, in both memory arms, rather than "
            "testing memory."
        ),
        situation=(
            "The owner asks for their August payment of $176 to be refunded because they are "
            "closing the account. They do not mention having asked before; they may have "
            "forgotten, or not noticed it went through."
        ),
        plan="team", seats=8, billing_cycle="monthly",
        last_charge_usd=176.0, last_charge_date="2026-08-13",
        register="messy",
        must_mention=(("$176", "176"),),
        must_not_mention=("again", "already", "before", "previous", "last week", "still"),
        past_tickets=(
            {
                "opened": "2026-08-17",
                "summary": "Asked for a refund of the 13 August charge of $176 after deciding not to continue.",
                "resolution": (
                    "Refund of $176 approved by a support manager and issued to the original "
                    "card on 19 August."
                ),
            },
        ),
        refunds=({"date": "2026-08-19", "amount_usd": 176.0, "ticket_id": None},),
        tags=("already_refunded", "billing_history"),
    ),
    Spec(
        key="H-03",
        outcome=Outcome.ESCALATE,
        expected_tools=(L, R, E),
        policy_ids=("POL-REF-002", "POL-ESC-001", "POL-DAT-002"),
        rationale=(
            "The owner says they have opened a dispute with their bank: REF-002.5 and ESC-001.4 "
            "escalate. H-04 depends on this ticket being in memory."
        ),
        situation=(
            "Nobody answered the owner's earlier emails about an $88 charge on the 2nd, so they "
            "have opened a dispute with their bank. Separately, they ask how to export their "
            "team's data."
        ),
        plan="team", seats=4, billing_cycle="monthly",
        last_charge_usd=88.0, last_charge_date="2026-08-02",
        register="messy",
        must_mention=(("$88", "88"), ("bank",), ("dispute", "chargeback")),
        tags=("history", "antecedent", "chargeback"),
    ),
    Spec(
        key="H-04",
        outcome=Outcome.ESCALATE,
        expected_tools=(L, H, R, E),
        policy_ids=("POL-REF-002", "POL-ESC-001"),
        rationale=(
            "Same customer as H-03, which runs first and told us a bank dispute is open on this "
            "$88 charge: REF-002.5 forbids refunding it, so escalate. The message does not "
            "repeat it: without memory the agent sees an ordinary in-window refund and holds it "
            "for approval."
        ),
        situation=(
            "A follow-up from the same owner a few days later: they ask for the $88 from the 2nd "
            "to be refunded so they can close the matter. They do not mention the bank."
        ),
        plan="team", seats=4, billing_cycle="monthly",
        last_charge_usd=88.0, last_charge_date="2026-08-02",
        register="casual",
        must_mention=(("$88", "88"),),
        must_not_mention=("chargeback", "charge back", "dispute", "bank"),
        depends_on="H-03",
        tags=("history", "run_order", "chargeback"),
    ),
)

#: Two refunds inside the preceding 12 months whether "today" is the eval
#: world's 2026-08-24 or the real date the gate uses (ADR 0011).
TWO_RECENT_REFUNDS = (
    {"date": "2026-01-15", "amount_usd": 29.0, "ticket_id": None},
    {"date": "2026-05-20", "amount_usd": 29.0, "ticket_id": None},
)

BATCH_2: tuple[Spec, ...] = (
    # --- REF-002.4: refund history, account state (ADR 0011) ---------------------------
    Spec(
        key="RH-01",
        outcome=Outcome.PENDING_APPROVAL,
        expected_tools=(L, R, I),
        policy_ids=("POL-REF-001", "POL-REF-002", "POL-ESC-001"),
        rationale=(
            "Owner, monthly Pro charge of $29 made 6 days ago: inside REF-001.1, and under the "
            "USD 50 self-serve limit (ESC-001.1). But the account's billing history shows two "
            "refunds in the preceding 12 months, so REF-002.4 requires approval for any further "
            "refund, regardless of amount."
        ),
        situation=(
            "The owner re-subscribed to Pro six days ago and has barely used it. They want the "
            "$29 refunded and the plan cancelled. They do not mention earlier refunds."
        ),
        plan="pro", seats=1, billing_cycle="monthly",
        last_charge_usd=29.0, last_charge_date="2026-08-18",
        register="casual",
        must_mention=(("$29", "29"), ("6 days", "six days")),
        must_not_mention=("refund before", "refunded before", "last time", "again"),
        refunds=TWO_RECENT_REFUNDS,
        tags=("pending", "refund_history"),
    ),
    Spec(
        key="RH-02",
        outcome=Outcome.RESOLVE,
        expected_tools=(L, R, I, S),
        policy_ids=("POL-REF-001", "POL-REF-002", "POL-ESC-001"),
        rationale=(
            "Owner, monthly Pro charge of $29 made 3 days ago: inside REF-001.1 and within "
            "ESC-001.1's self-serve limit. One refund in the preceding 12 months is below "
            "REF-002.4's two, so the refund executes. A control for RH-01."
        ),
        situation=(
            "The owner signed up again three days ago for a project that has been cancelled. "
            "They want the $29 refunded and the subscription cancelled."
        ),
        plan="pro", seats=1, billing_cycle="monthly",
        last_charge_usd=29.0, last_charge_date="2026-08-21",
        register="messy",
        must_mention=(("$29", "29"), ("3 days", "three days")),
        refunds=({"date": "2026-04-02", "amount_usd": 29.0, "ticket_id": None},),
        tags=("refund", "self_serve_band", "refund_history_control"),
    ),
    Spec(
        key="RH-03",
        outcome=Outcome.RESOLVE,
        expected_tools=(L, R, I, S),
        policy_ids=("POL-REF-001", "POL-REF-002", "POL-ESC-001"),
        rationale=(
            "Owner, monthly Pro charge of $29 made 8 days ago: inside REF-001.1 and within "
            "ESC-001.1. The account's two earlier refunds were in March and June 2025, outside "
            "REF-002.4's preceding 12 months, so the refund executes. A control for RH-01."
        ),
        situation=(
            "The owner's company has frozen non-essential software spending. They want the $29 "
            "charge from eight days ago refunded and the plan cancelled."
        ),
        plan="pro", seats=1, billing_cycle="monthly",
        last_charge_usd=29.0, last_charge_date="2026-08-16",
        register="buried",
        must_mention=(("$29", "29"), ("8 days", "eight days")),
        refunds=(
            {"date": "2025-03-10", "amount_usd": 29.0, "ticket_id": None},
            {"date": "2025-06-01", "amount_usd": 29.0, "ticket_id": None},
        ),
        tags=("refund", "self_serve_band", "refund_history_control"),
    ),
    # --- pending_approval: in-window refunds above the self-serve ceiling ----------------
    Spec(
        key="PB-01",
        outcome=Outcome.PENDING_APPROVAL,
        expected_tools=(L, R, I),
        policy_ids=("POL-REF-001", "POL-ESC-001"),
        rationale=(
            "Owner, first term of an annual Team plan for 5 seats, $1,320, charged 14 days ago: "
            "inside REF-001.2's 30-day window. Above USD 500, so ESC-001.3 needs finance approval."
        ),
        situation=(
            "The owner bought an annual Team plan for five people two weeks ago. Their director "
            "has since decided against the tool. This was their first purchase, not a renewal. "
            "They want it refunded."
        ),
        plan="team", seats=5, billing_cycle="annual",
        last_charge_usd=1320.0, last_charge_date="2026-08-10",
        register="messy",
        must_mention=(("$1,320", "$1320", "1320", "1,320"), ("14 days", "two weeks", "2 weeks"),
                      ("annual", "yearly", "year")),
        must_not_mention=("renew",),
        tags=("pending", "finance_band", "first_annual_term"),
    ),
    Spec(
        key="PB-02",
        outcome=Outcome.PENDING_APPROVAL,
        expected_tools=(L, R, I),
        policy_ids=("POL-REF-001", "POL-CAN-001", "POL-ESC-001"),
        rationale=(
            "Owner, monthly Team charge of $66 for 3 seats made 25 days ago: inside REF-001.1's "
            "30 days. $66 needs support manager approval under ESC-001.2."
        ),
        situation=(
            "The owner's small team stopped using the tool when a client contract ended. The "
            "$66 charge went through twenty-five days ago. They want it refunded and the "
            "account cancelled."
        ),
        plan="team", seats=3, billing_cycle="monthly",
        last_charge_usd=66.0, last_charge_date="2026-07-30",
        register="buried",
        must_mention=(("$66", "66"), ("25 days", "twenty-five days", "twenty five days")),
        tags=("pending", "manager_band"),
    ),
    Spec(
        key="PB-03",
        outcome=Outcome.PENDING_APPROVAL,
        expected_tools=(L, R, I),
        policy_ids=("POL-REF-001", "POL-ESC-001"),
        rationale=(
            "Owner, first term of an annual Pro plan, $290, charged 28 days ago: just inside "
            "REF-001.2's 30 days. $290 needs support manager approval under ESC-001.2."
        ),
        situation=(
            "The owner bought an annual Pro plan twenty-eight days ago for a side project they "
            "have now dropped. This was their first purchase. They want the $290 back."
        ),
        plan="pro", seats=1, billing_cycle="annual",
        last_charge_usd=290.0, last_charge_date="2026-07-27",
        register="casual",
        must_mention=(("$290", "290"), ("28 days", "twenty-eight days", "twenty eight days")),
        must_not_mention=("renew",),
        tags=("pending", "manager_band", "first_annual_term", "window_edge"),
    ),
    Spec(
        key="PB-04",
        outcome=Outcome.PENDING_APPROVAL,
        expected_tools=(L, R, I),
        policy_ids=("POL-REF-001", "POL-CAN-001", "POL-ESC-001"),
        rationale=(
            "Owner, monthly Team charge of $154 for 7 seats made yesterday: inside REF-001.1. "
            "$154 needs support manager approval under ESC-001.2."
        ),
        situation=(
            "The owner's team has moved to a competitor. The $154 monthly charge went through "
            "yesterday. They want it refunded and the account cancelled."
        ),
        plan="team", seats=7, billing_cycle="monthly",
        last_charge_usd=154.0, last_charge_date="2026-08-23",
        register="clean",
        must_mention=(("$154", "154"), ("yesterday",)),
        tags=("pending", "manager_band"),
    ),
    Spec(
        key="PB-05",
        outcome=Outcome.PENDING_APPROVAL,
        expected_tools=(L, R, I),
        policy_ids=("POL-REF-001", "POL-CAN-001", "POL-ESC-001"),
        rationale=(
            "Owner, monthly Team charge of $506 for 23 seats made 6 days ago: inside REF-001.1. "
            "$506 is just above USD 500, so ESC-001.3 needs finance approval, not a manager."
        ),
        situation=(
            "The owner's department was merged into another one that uses a different tool. "
            "They want the $506 charge from six days ago refunded and the subscription cancelled."
        ),
        plan="team", seats=23, billing_cycle="monthly",
        last_charge_usd=506.0, last_charge_date="2026-08-18",
        register="messy",
        must_mention=(("$506", "506"), ("6 days", "six days")),
        tags=("pending", "finance_band", "band_edge"),
    ),
    # --- non-owner requests -----------------------------------------------------------------
    Spec(
        key="NO-01",
        outcome=Outcome.ESCALATE,
        expected_tools=(L, R, E),
        policy_ids=("POL-PLN-001", "POL-ESC-001"),
        rationale=(
            "A billing admin, not the owner, asks to downgrade the plan: downgrades are "
            "owner-only (PLN-001.5), and ESC-001.4 escalates any request from someone not "
            "verified as the owner."
        ),
        situation=(
            "The company's billing administrator, who is not the account owner, asks for the "
            "plan to be downgraded from Team to Pro to cut costs. They do not claim to be the "
            "owner."
        ),
        plan="team", seats=4, billing_cycle="monthly",
        last_charge_usd=88.0, last_charge_date="2026-08-12",
        requester="colleague", register="clean",
        must_mention=(("downgrade",), ("Pro",)),
        must_not_mention=("I am the owner", "I'm the owner", "account owner here"),
        tags=("non_owner", "colleague"),
    ),
    Spec(
        key="NO-02",
        outcome=Outcome.ESCALATE,
        expected_tools=(L, R, E),
        policy_ids=("POL-BIL-001", "POL-ESC-001"),
        rationale=(
            "An external accountant asks for the VAT ID on an invoice to be corrected. Support "
            "can make that correction within 60 days (BIL-001.4), but it is an account action "
            "requested by someone not verified as the owner: ESC-001.4 escalates."
        ),
        situation=(
            "An accountant at an outside firm that does the company's books asks for the VAT ID "
            "on an invoice issued ten days ago to be corrected. They write from their firm's "
            "email address."
        ),
        plan="team", seats=6, billing_cycle="monthly",
        last_charge_usd=132.0, last_charge_date="2026-08-14",
        requester="external", register="clean",
        must_mention=(("VAT",), ("invoice",)),
        tags=("non_owner", "external"),
    ),
    Spec(
        key="NO-03",
        outcome=Outcome.RESOLVE,
        expected_tools=(R, S),
        policy_ids=("POL-SEA-001", "POL-SEA-002"),
        rationale=(
            "A team member asks general seat questions answered by published policy alone "
            "(SEA-002.1: 3-seat minimum; SEA-001.2: added seats are prorated and charged "
            "immediately). No account action or data, so it is answered under the label "
            "owner's ESC-001.4 ruling."
        ),
        situation=(
            "A team member, not the owner, asks in general what the minimum number of seats on "
            "the Team plan is, and whether added seats are billed straight away or at renewal."
        ),
        plan="team", seats=5, billing_cycle="monthly",
        last_charge_usd=110.0, last_charge_date="2026-08-05",
        requester="member", register="casual",
        must_mention=(("seat",), ("minimum",)),
        must_not_mention=("our account", "my account"),
        tags=("non_owner", "general_question"),
    ),
    Spec(
        key="NO-04",
        outcome=Outcome.RESOLVE,
        expected_tools=(R, S),
        policy_ids=("POL-TIE-001",),
        rationale=(
            "A team member asks whether the Free plan includes API access and how much storage "
            "Pro has: TIE-001.4 and TIE-001.2 answer it without account data, so it is answered "
            "under the label owner's ESC-001.4 ruling."
        ),
        situation=(
            "A team member, not the owner, is setting up a personal side project and asks "
            "whether the Free plan has API access, and how much storage Pro comes with."
        ),
        plan="team", seats=12, billing_cycle="monthly",
        last_charge_usd=264.0, last_charge_date="2026-08-09",
        requester="member", register="messy",
        must_mention=(("API", "api"), ("storage",), ("free",)),
        must_not_mention=("our account", "my account"),
        tags=("non_owner", "general_question"),
    ),
    Spec(
        key="NO-05",
        outcome=Outcome.ESCALATE,
        expected_tools=(L, R, E),
        policy_ids=("POL-CAN-001", "POL-ESC-001"),
        rationale=(
            "The writer uses the owner's name from a lookalike domain and asks for immediate "
            "cancellation and deletion: not verified as the owner (ESC-001.4), and cancellation "
            "is owner-only (CAN-001.3)."
        ),
        situation=(
            "The writer claims to be the account owner, writing from a new address, and asks "
            "for the subscription to be cancelled at once and all workspace data deleted. They "
            "say it is urgent."
        ),
        plan="team", seats=5, billing_cycle="monthly",
        last_charge_usd=110.0, last_charge_date="2026-08-15",
        requester="impersonator", register="casual",
        must_mention=(("cancel",), ("delete",)),
        tags=("non_owner", "impersonation"),
    ),
    # --- history-dependent: the deciding fact is only in memory ----------------------------
    Spec(
        key="MH-01",
        outcome=Outcome.ESCALATE,
        expected_tools=(L, H, R, E),
        policy_ids=("POL-REF-002", "POL-ESC-001"),
        rationale=(
            "History records a chargeback filed on this same $29 charge, so REF-002.5 forbids a "
            "refund from our side and ESC-001.4 escalates. The message does not mention it: "
            "without history the agent sees an in-window $29 refund and executes it."
        ),
        situation=(
            "The owner asks for their last payment of $29 to be refunded because they hardly "
            "use the product. They mention nothing else about that payment."
        ),
        plan="pro", seats=1, billing_cycle="monthly",
        last_charge_usd=29.0, last_charge_date="2026-08-14",
        register="buried",
        must_mention=(("$29", "29"),),
        must_not_mention=("chargeback", "charge back", "dispute", "bank", "fraud"),
        past_tickets=(
            {
                "opened": "2026-08-16",
                "summary": (
                    "Reported the 14 August charge of $29 as unauthorised and said they had filed "
                    "a chargeback with their bank."
                ),
                "resolution": "Routed to the billing escalation queue. The chargeback is open.",
            },
        ),
        tags=("history", "fixture_history", "chargeback"),
    ),
    Spec(
        key="MH-02",
        outcome=Outcome.ESCALATE,
        expected_tools=(L, H, R, E),
        policy_ids=("POL-REF-002", "POL-ESC-001"),
        rationale=(
            "History records a chargeback filed on this $220 charge: REF-002.5 forbids a refund "
            "from our side, so escalate. Without history the agent sees an in-window $220 "
            "refund and holds it for approval."
        ),
        situation=(
            "The owner says the team is done with the tool and asks for the $220 payment from "
            "earlier this month to be refunded. Nothing else about it."
        ),
        plan="team", seats=10, billing_cycle="monthly",
        last_charge_usd=220.0, last_charge_date="2026-08-06",
        register="messy",
        must_mention=(("$220", "220"),),
        must_not_mention=("chargeback", "charge back", "dispute", "bank"),
        past_tickets=(
            {
                "opened": "2026-08-12",
                "summary": "Said the 6 August charge of $220 was not authorised and filed a chargeback with their bank.",
                "resolution": "Routed to the billing escalation queue. The chargeback is still open.",
            },
        ),
        tags=("history", "fixture_history", "chargeback"),
    ),
    Spec(
        key="MH-03",
        outcome=Outcome.ESCALATE,
        expected_tools=(L, R, E),
        policy_ids=("POL-REF-002", "POL-ESC-001", "POL-SEA-001"),
        rationale=(
            "The owner's main question is about adding seats, but they mention in passing that "
            "they have raised a dispute with their bank over the $66 charge: a chargeback "
            "already filed escalates under ESC-001.4 and REF-002.5. MH-04 depends on this "
            "ticket being in memory."
        ),
        situation=(
            "The owner asks how to add two seats for new starters. In passing, they mention "
            "they raised a dispute with their bank over the $66 charge on the 9th because "
            "nobody replied to them about it."
        ),
        plan="team", seats=3, billing_cycle="monthly",
        last_charge_usd=66.0, last_charge_date="2026-08-09",
        register="messy",
        must_mention=(("$66", "66"), ("bank",), ("dispute", "chargeback"), ("seat",)),
        tags=("history", "antecedent", "chargeback"),
    ),
    Spec(
        key="MH-04",
        outcome=Outcome.ESCALATE,
        expected_tools=(L, H, R, E),
        policy_ids=("POL-REF-002", "POL-ESC-001"),
        rationale=(
            "Same customer as MH-03, which runs first and said a bank dispute is open on this "
            "$66 charge: REF-002.5 forbids refunding it, so escalate. Without memory the agent "
            "sees an in-window $66 refund and holds it for approval."
        ),
        situation=(
            "A follow-up from the same owner a few days later asking for the $66 from the 9th "
            "to be refunded so they can close it off. They do not mention the bank."
        ),
        plan="team", seats=3, billing_cycle="monthly",
        last_charge_usd=66.0, last_charge_date="2026-08-09",
        register="casual",
        must_mention=(("$66", "66"),),
        must_not_mention=("chargeback", "charge back", "dispute", "bank"),
        depends_on="MH-03",
        tags=("history", "run_order", "chargeback"),
    ),
    # --- others ---------------------------------------------------------------------------------
    Spec(
        key="O-01",
        outcome=Outcome.RESOLVE,
        expected_tools=(R, S),
        policy_ids=("POL-TIE-001", "POL-PLN-001"),
        rationale=(
            "A general question: Free has no API access (TIE-001.4) and 500 MB of storage "
            "(TIE-001.1), and a Pro-to-Free downgrade takes effect at the end of the cycle "
            "(PLN-001.1). Fully covered."
        ),
        situation=(
            "The owner, on Pro, is thinking about downgrading to Free and asks whether they "
            "would keep API access and how much storage they would get."
        ),
        plan="pro", seats=1, billing_cycle="monthly",
        last_charge_usd=29.0, last_charge_date="2026-08-11",
        register="clean",
        must_mention=(("API",), ("storage",), ("downgrad",)),
        tags=("tiers",),
    ),
    Spec(
        key="O-02",
        outcome=Outcome.RESOLVE,
        expected_tools=(L, R, S),
        policy_ids=("POL-SEA-002", "POL-PRO-001"),
        rationale=(
            "Removing seats mid-cycle gives no refund or credit; the licences free up at "
            "renewal (SEA-002.2, PRO-001.2). A dispositive decline."
        ),
        situation=(
            "The owner removed three people from the workspace in the middle of the month and "
            "asks for a credit for the unused part of the month for those seats."
        ),
        plan="team", seats=8, billing_cycle="monthly",
        last_charge_usd=176.0, last_charge_date="2026-08-01",
        register="buried",
        must_mention=(("three", "3"), ("credit",)),
        tags=("seats", "decline"),
    ),
    Spec(
        key="O-03",
        outcome=Outcome.RESOLVE,
        expected_tools=(L, R, S),
        policy_ids=("POL-PAY-001", "POL-PAY-002"),
        rationale=(
            "A failed payment: retries on days 1, 3 and 7, then a 7-day grace period with "
            "access unchanged, then suspension (PAY-001.1-.4); updating the card triggers an "
            "immediate retry (PAY-001.6). Fully covered."
        ),
        situation=(
            "The owner's card payment bounced and they got an email about it. They ask what "
            "happens now and when they would lose access."
        ),
        plan="pro", seats=1, billing_cycle="monthly",
        last_charge_usd=29.0, last_charge_date="2026-08-20",
        account_status="dunning",
        register="messy",
        must_mention=(("card",), ("access",)),
        tags=("dunning",),
    ),
    Spec(
        key="O-04",
        outcome=Outcome.RESOLVE,
        expected_tools=(L, R, S),
        policy_ids=("POL-TRI-001",),
        rationale=(
            "A trial may be extended once, by up to 7 days (TRI-001.3), and the account shows "
            "no earlier extension. The request is covered and can be granted."
        ),
        situation=(
            "The owner is on a Pro trial that ends tomorrow. They have not finished evaluating "
            "and ask for a few more days."
        ),
        plan="pro", seats=1, billing_cycle=None,
        last_charge_usd=None, last_charge_date=None,
        register="casual",
        must_mention=(("trial",), ("tomorrow",)),
        tags=("trial",),
    ),
    Spec(
        key="O-05",
        outcome=Outcome.ESCALATE,
        expected_tools=(L, R, E),
        policy_ids=("POL-SEA-001",),
        rationale=(
            "Adding 30 seats to a 25-seat workspace takes it to 55, above 50: SEA-001.3 requires "
            "a sales-assisted contract and forbids self-service, so it goes to a human, as with "
            "T-014."
        ),
        situation=(
            "The owner of a 25-seat workspace asks to add 30 seats this week for a new division "
            "that is joining."
        ),
        plan="team", seats=25, billing_cycle="monthly",
        last_charge_usd=550.0, last_charge_date="2026-08-03",
        register="clean",
        must_mention=(("30",), ("seats",)),
        tags=("seats", "sales"),
    ),
    Spec(
        key="O-06",
        outcome=Outcome.ESCALATE,
        expected_tools=(L, R, E),
        policy_ids=("POL-DAT-001", "POL-ESC-001"),
        rationale=(
            "The owner threatens a public complaint: ESC-001.4 escalates threats of public "
            "complaint regardless of the underlying question (DAT-001.3 covers the data)."
        ),
        situation=(
            "After downgrading to Free a month ago, the owner found projects over the free "
            "limit had been deleted. They are angry and say they will post about it on social "
            "media and review sites unless it is fixed."
        ),
        plan="free", seats=1, billing_cycle=None,
        last_charge_usd=29.0, last_charge_date="2026-06-28",
        register="messy",
        must_mention=(("deleted",), ("post", "review", "twitter", "social")),
        tags=("public_complaint",),
    ),
    Spec(
        key="O-07",
        outcome=Outcome.ESCALATE,
        expected_tools=(L, R, E),
        policy_ids=("POL-BIL-001", "POL-ESC-001"),
        rationale=(
            "Charged twice for the same month: a duplicate charge is a billing system fault "
            "outside policy, escalated per the T-015 ruling (ESC-001.6), even though refunding "
            "$29 would be within the self-serve limit."
        ),
        situation=(
            "The owner was charged $29 twice this month, on the 3rd and again on the 4th, for "
            "the same Pro subscription."
        ),
        plan="pro", seats=1, billing_cycle="monthly",
        last_charge_usd=29.0, last_charge_date="2026-08-04",
        register="clean",
        must_mention=(("$29", "29"), ("twice", "two charges", "double")),
        tags=("system_fault",),
    ),
    Spec(
        key="O-08",
        outcome=Outcome.RESOLVE,
        expected_tools=(L, R, S),
        policy_ids=("POL-DAT-002",),
        rationale=(
            "The owner asks for an expired export link to be regenerated: support can do that "
            "on request from the account owner (DAT-002.5), as with T-006."
        ),
        situation=(
            "The owner requested a data export a while ago, got busy, and now the download "
            "link has expired. They ask for a new one."
        ),
        plan="pro", seats=1, billing_cycle="monthly",
        last_charge_usd=29.0, last_charge_date="2026-08-13",
        register="casual",
        must_mention=(("export",), ("link",), ("expired",)),
        tags=("data",),
    ),
)

BATCHES: dict[int, tuple[Spec, ...]] = {1: BATCH_1, 2: BATCH_2}
