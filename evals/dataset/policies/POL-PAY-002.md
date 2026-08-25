---
policy_id: POL-PAY-002
title: Suspension and reactivation
tiers: [Pro, Team]
last_updated: 2026-06-18
---

# Suspension and reactivation

**PAY-002.1** A suspended account is read-only. API access is disabled.
Workspace data is retained.

**PAY-002.2** Reactivation is automatic once the outstanding balance is paid
in full, and access is restored immediately.

**PAY-002.3** An account suspended for more than 60 consecutive days is
downgraded to Free, and its data is handled under POL-DAT-001.

**PAY-002.4** An outstanding balance is not cancelled by a downgrade or by
cancellation of the subscription.

**PAY-002.5** Waiving an outstanding balance follows the approval thresholds
in POL-ESC-001.
