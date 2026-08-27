"""Stage 2, redesigned: three families of fusion rule, compared on equal terms.

**Why this module exists.** Cycle 4 measured that the detector contributes
nothing to the assembled system at *any* threshold — deltas between −0.0007 and
+0.0003 against a seed spread of ±0.0035. The cause is structural, not a
calibration accident: in the manuscript's rule table a confident classifier
label always wins, so with τ_M selected at 1.0 the detector only ever arbitrates
the 2.2% of flows the classifier declines to name, and 99.8% of those are
benign.

So the fusion rule, not the detector, is where the margin is. The supervisor
asked for all three families to be tried rather than picking one in advance,
with explicit freedom to add, remove and complicate rules.

**What stays fixed.** All three keep the three stages present and recognisable
— a benign-only detector, a malicious-only classifier, and a second stage that
combines them. That is what `BRIEF.md` calls the three-stage spirit; only the
combination changes, which the brief declares open.

**What each family gives up.**

* `RuleFusion` keeps the manuscript's selling point — behaviour tunable
  post-training without retraining anything — and its interpretability, at the
  cost of a coarse decision surface.
* `ScoreFusion` lets the detector influence *every* flow instead of a residual
  2.2%, still without any training, but the decision is a formula rather than a
  table a reader can check line by line.
* `LearnedFusion` is the most flexible and the least defensible: it needs
  fitting, so the "adjust without retraining" claim weakens, and a reviewer will
  reasonably ask why this is not simply one classifier.

Everything here consumes only the two stage-1 outputs (anomaly score, class
probabilities). None of it reads labels at inference time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nids.stages.classifier import UNKNOWN

BENIGN = "Benign"
ZERO_DAY = "Zero Day"


# ---------------------------------------------------------------------------
# Family A — discrete rules, rewritten
# ---------------------------------------------------------------------------

@dataclass
class RuleFusionConfig:
    tau_b: float = 0.0        # detector: benign vs suspicious
    tau_m: float = 0.9        # classifier: confident vs Unknown
    tau_u: float = 0.0        # extension: zero-day vs benign
    #: Second, higher classifier threshold. The manuscript has a single τ_M, so
    #: a label is either trusted outright or discarded. With two levels the rule can
    #: say "confident enough to name, but not confident enough to override a
    #: clean detector verdict", which is precisely the case the old table
    #: collapsed.
    tau_m_override: float = 0.99
    #: When the detector says benign and the classifier is only moderately
    #: confident, whose verdict wins.
    trust_detector_when_calm: bool = True


class RuleFusion:
    """Discrete rules, but with a two-level classifier confidence.

    The manuscript's table has five rows keyed on (detector verdict, classifier
    named/Unknown). Its flaw is that "named" is a single bucket: any label above
    τ_M overrides the detector completely. This version splits that bucket, so a
    merely-confident label no longer silences a detector that is calm — which is
    what makes the detector matter again.
    """

    name = "rules-v2"

    def __init__(self, cfg: RuleFusionConfig):
        self.cfg = cfg

    def fuse(self, scores: np.ndarray, labels: np.ndarray,
             certainty: np.ndarray) -> np.ndarray:
        c = self.cfg
        out = np.full(len(scores), BENIGN, dtype=object)
        suspicious = scores > c.tau_b
        named = labels != UNKNOWN
        strong = named & (certainty >= c.tau_m_override)

        # 1. Very confident label: always honoured, detector or not. Keeps the
        #    manuscript's high attack recall on classes the classifier knows.
        out[strong] = labels[strong]

        # 2. Moderately confident label AND the detector agrees something is
        #    off: honour it.
        mid = named & ~strong
        out[mid & suspicious] = labels[mid & suspicious]

        # 3. Moderately confident label but the detector is calm: this is the
        #    row that did not exist before. Trusting the detector here is what
        #    recovers the false positives the old table waved through.
        if c.trust_detector_when_calm:
            out[mid & ~suspicious] = BENIGN
        else:
            out[mid & ~suspicious] = labels[mid & ~suspicious]

        # 4/5. No usable label: the extension stage arbitrates on the score.
        unnamed = ~named
        out[unnamed & suspicious & (scores > c.tau_u)] = ZERO_DAY
        out[unnamed & suspicious & (scores <= c.tau_u)] = BENIGN
        return out


# ---------------------------------------------------------------------------
# Family B — continuous score fusion
# ---------------------------------------------------------------------------

@dataclass
class ScoreFusionConfig:
    #: Weight on the detector's evidence. 0 reproduces a classifier-only
    #: system, 1 a detector-only one.
    w_detector: float = 0.5
    #: Decision threshold on the fused maliciousness evidence.
    tau_decide: float = 0.5
    #: Above this fused evidence, a flow with no confident label is called
    #: Zero Day rather than benign.
    tau_zero_day: float = 0.8
    tau_m: float = 0.9
    #: Percentile reference for the anomaly score, computed on benign
    #: validation rows so the two evidences live on the same [0,1] scale.
    ref_scores: np.ndarray | None = field(default=None, repr=False)


class ScoreFusion:
    """Combine the raw evidences instead of thresholding them separately.

    Both stages produce a graded opinion — an anomaly score and a class
    probability — and the manuscript throws away that gradation by thresholding
    each independently before combining. Here they are mapped to a common
    [0, 1] scale (the anomaly score by its percentile against benign validation
    traffic) and blended, so a detector that is *slightly* alarmed still shifts
    a borderline decision. The detector influences every flow, not a residual.

    Still training-free: `w_detector` and the thresholds are tunable
    post-training, so the manuscript's flexibility claim survives.
    """

    name = "score-fusion"

    def __init__(self, cfg: ScoreFusionConfig):
        self.cfg = cfg

    def _pct(self, scores: np.ndarray) -> np.ndarray:
        ref = self.cfg.ref_scores
        if ref is None or len(ref) == 0:
            return np.zeros(len(scores))
        return np.searchsorted(np.sort(ref), scores) / len(ref)

    def fuse(self, scores: np.ndarray, labels: np.ndarray,
             certainty: np.ndarray) -> np.ndarray:
        c = self.cfg
        det_evidence = self._pct(scores)
        # The classifier's evidence that a flow is malicious at all: it was
        # trained only on attacks, so high confidence in *any* class is itself
        # evidence of maliciousness.
        clf_evidence = np.clip(certainty, 0.0, 1.0)
        fused = c.w_detector * det_evidence + (1 - c.w_detector) * clf_evidence

        out = np.full(len(scores), BENIGN, dtype=object)
        malicious = fused >= c.tau_decide
        named = (labels != UNKNOWN) & (certainty >= c.tau_m)
        out[malicious & named] = labels[malicious & named]
        out[malicious & ~named & (fused >= c.tau_zero_day)] = ZERO_DAY
        return out


# ---------------------------------------------------------------------------
# Family C — learned fusion
# ---------------------------------------------------------------------------

@dataclass
class LearnedFusionConfig:
    max_depth: int = 4          # shallow: the rule must stay readable
    seed: int = 0
    tau_m: float = 0.9


class LearnedFusion:
    """A shallow decision tree over the two stages' outputs.

    Trained on validation only, on features that are *outputs of stage 1*
    (anomaly percentile, top class probability, margin between the top two
    classes) — never on raw flow features, so it cannot bypass the stages and
    become a classifier in disguise.

    Depth is capped so the learned rule can be printed and audited, which
    partially preserves the interpretability the discrete table gives for free.
    The honest cost: it needs fitting, so "tunable without retraining" no longer
    holds for stage 2.
    """

    name = "learned-fusion"

    def __init__(self, cfg: LearnedFusionConfig):
        self.cfg = cfg
        self.model = None
        self.ref_scores: np.ndarray | None = None

    def _features(self, scores, proba, certainty) -> np.ndarray:
        pct = (np.searchsorted(np.sort(self.ref_scores), scores) /
               max(len(self.ref_scores), 1)) if self.ref_scores is not None \
            else np.zeros(len(scores))
        top2 = np.sort(proba, axis=1)[:, -2:] if proba.shape[1] > 1 else \
            np.c_[np.zeros(len(proba)), proba[:, 0]]
        margin = top2[:, -1] - top2[:, -2]
        return np.c_[pct, certainty, margin]

    def fit(self, scores, proba, certainty, y_true, ref_scores) -> "LearnedFusion":
        from sklearn.tree import DecisionTreeClassifier

        self.ref_scores = np.asarray(ref_scores)
        X = self._features(scores, proba, certainty)
        self.model = DecisionTreeClassifier(
            max_depth=self.cfg.max_depth, random_state=self.cfg.seed,
            class_weight="balanced").fit(X, y_true)
        return self

    def fuse(self, scores, labels, certainty, proba=None) -> np.ndarray:
        X = self._features(scores, proba, certainty)
        decision = self.model.predict(X)      # 'Benign' | 'Malicious' | 'Zero Day'
        out = np.full(len(scores), BENIGN, dtype=object)
        mal = decision != BENIGN
        named = (labels != UNKNOWN) & (certainty >= self.cfg.tau_m)
        out[mal & named] = labels[mal & named]
        out[mal & ~named] = ZERO_DAY
        return out
