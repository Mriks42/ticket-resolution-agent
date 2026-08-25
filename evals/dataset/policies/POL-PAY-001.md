---
policy_id: POL-PAY-001
title: Failed payments and dunning
tiers: [Pro, Team]
last_updated: 2026-06-18
---

# Failed payments and dunning

**PAY-001.1** A failed charge is retried automatically on days 1, 3, and 7
after the initial failure.

**PAY-001.2** The billing contact is emailed after each failed attempt.

**PAY-001.3** If the day-7 retry fails, the account enters a 7-day grace
period during which access is unchanged.

**PAY-001.4** At the end of the grace period the account is suspended under
POL-PAY-002.

**PAY-001.5** Support may extend the grace period once, by up to 7 days, for
Team accounts.

**PAY-001.6** Updating the payment method during dunning triggers an immediate
retry of the outstanding charge.
