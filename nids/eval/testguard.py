"""Access control for the frozen test set.

`BRIEF.md` rule 3: "Test evaluations are budgeted and logged. Each run against
the test set is recorded in DECISIONS.md with the cycle number and what was
being tested. Ask before spending one. If the count starts climbing, say so -
the count itself is a measure of how much we have contaminated our own
result."

A rule that lives only in prose gets broken during a long session. This module
makes the test partition unreachable except through `spend`, which writes an
immutable ledger entry first. The count is therefore auditable after the fact,
and a reviewer asking "how many times did you look at the test set?" has an
answer backed by a file rather than by memory.

Phase A/B spends are pre-authorised by the supervisor (see SESSION.md).
Phase C spends require `authorised=True` passed explicitly, which the caller
may only set after checkpoint approval.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from nids import config

LEDGER = config.RESULTS / "test_set_ledger.jsonl"


class TestSetLocked(RuntimeError):
    """Raised when the test set is touched without an authorised spend."""


@dataclass
class Spend:
    cycle: int
    phase: str
    purpose: str
    model: str
    authorised: bool


def spend(
    *,
    cycle: int,
    phase: str,
    purpose: str,
    model: str,
    authorised: bool = False,
) -> Spend:
    """Record one test-set evaluation. Call BEFORE touching test data.

    Phases A and B are pre-authorised. Anything else must pass
    `authorised=True`, which the agent may only do after explicit supervisor
    approval at a checkpoint.
    """
    phase = phase.upper()
    if phase not in {"A", "B"} and not authorised:
        raise TestSetLocked(
            f"Phase {phase} test evaluation requires explicit supervisor "
            "approval (SESSION.md test-budget policy). Every Phase C spend "
            "waits for a checkpoint. Refusing."
        )
    rec = {
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cycle": cycle,
        "phase": phase,
        "purpose": purpose,
        "model": model,
        "authorised": authorised or phase in {"A", "B"},
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return Spend(cycle, phase, purpose, model, rec["authorised"])


def count() -> int:
    """How many test evaluations have been spent in total."""
    if not LEDGER.exists():
        return 0
    return sum(1 for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip())


def summary() -> str:
    """Human-readable ledger, for pasting into a checkpoint report."""
    if not LEDGER.exists():
        return "No test-set evaluations spent."
    rows = [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = [f"Total test-set evaluations: {len(rows)}", ""]
    out += [
        f"  [{r['ts_utc']}] cycle {r['cycle']} phase {r['phase']} "
        f"- {r['model']}: {r['purpose']}"
        for r in rows
    ]
    return "\n".join(out)
