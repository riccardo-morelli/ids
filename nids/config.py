"""Central configuration: seeds, split fractions, paths.

Everything that could silently change a result lives here, so a diff of this
file is a complete account of what changed between two runs.

The protocol constants at the top are frozen by `BRIEF.md` and must not be
edited to chase a number. `SPLIT_SEED` in particular defines the test set for
the whole revision: changing it re-splits the test set, which the brief
forbids outright.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Frozen experimental protocol (BRIEF.md, non-negotiable)
# --------------------------------------------------------------------------

#: Seed defining the ONE split of the data into train/val/test.
#: Recorded here, never changed. See data/SPLIT.md.
SPLIT_SEED = 20260803

#: Seeds for repeated model training. Every comparative claim reports mean and
#: dispersion across these. A margin inside this spread is not a margin.
MODEL_SEEDS = (0, 1, 2, 3, 4)

#: Fraction of the *malicious* pool held out as test. Matches both papers (30%).
TEST_FRACTION = 0.30

#: Significance level for all reported tests.
ALPHA = 0.05

#: Bootstrap resamples for confidence intervals on test metrics.
N_BOOTSTRAP = 10_000

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"
NOTES = ROOT / "notes"
MODELS = ROOT / "models"

for _p in (DATA_RAW, DATA_INTERIM, RESULTS, NOTES, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Attack taxonomy
# --------------------------------------------------------------------------

#: Attack classes treated as zero-day: withheld from all training, present in
#: test only. Both papers use Infiltration + Heartbleed on CIC-IDS2017.
#: Reviewer 2 point 2 attacks this choice (47 instances); the leave-one-class-
#: out protocol in nids/eval/zeroday.py is the answer, and this constant is
#: what it varies.
DEFAULT_ZERO_DAY = ("Infiltration", "Heartbleed")

#: Canonical label set after grouping. Every dataset adapter maps into this.
CANONICAL_CLASSES = (
    "Benign",
    "DoS",
    "Port Scan",
    "Brute Force",
    "Web Attack",
    "Botnet",
)

#: Final decision labels the multi-stage system can emit.
DECISION_LABELS = CANONICAL_CLASSES + ("Zero Day",)
