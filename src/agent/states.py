"""The loop's states and the only transitions allowed between them.

CLAUDE.md asks for an explicit state machine rather than a while-loop over an
LLM call. The table below is that machine: the loop asks `check()` before
every hop, so a transition that is not listed here is a bug that raises, not a
path that quietly happens.

    PLAN ──> ACT ──> OBSERVE ──> PLAN ...
      │       │
      │       ├──> RESOLVE           a reply was proposed
      │       ├──> ESCALATE          an escalation was proposed, or the loop
      │       │                      stopped it (step budget, loop detected,
      │       │                      repeated invalid calls)
      │       └──> PENDING_APPROVAL  reserved for the HITL gate; unreachable
      │                              until that session lands
      ├──> ESCALATE                  token or wall-clock budget exhausted
      └──> FAILED                    no usable completion from the provider
"""

from __future__ import annotations

from enum import Enum


class State(str, Enum):
    PLAN = "PLAN"
    ACT = "ACT"
    OBSERVE = "OBSERVE"
    RESOLVE = "RESOLVE"
    ESCALATE = "ESCALATE"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    FAILED = "FAILED"


TERMINAL = frozenset({State.RESOLVE, State.ESCALATE, State.PENDING_APPROVAL, State.FAILED})

ALLOWED: dict[State, frozenset[State]] = {
    State.PLAN: frozenset({State.ACT, State.ESCALATE, State.FAILED}),
    State.ACT: frozenset({State.OBSERVE, State.RESOLVE, State.ESCALATE, State.PENDING_APPROVAL}),
    State.OBSERVE: frozenset({State.PLAN}),
    **{s: frozenset() for s in TERMINAL},
}


class IllegalTransition(RuntimeError):
    pass


def check(current: State, target: State) -> None:
    if target not in ALLOWED[current]:
        raise IllegalTransition(f"{current.value} -> {target.value} is not a legal transition")
