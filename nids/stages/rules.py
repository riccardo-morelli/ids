"""Stage 2, four rule families that attack the measured failure.

**What the measurements say.** Rewriting the rule table, replacing it with a
continuous formula, and replacing it with a learned tree all produced the same
zero-day recall: 0.0000, 0.0012, 0.0000. Three different rule *forms* failing
identically means the form was never the problem — the input was. Two facts,
both measured:

1. The classifier hands stage 2 a label it declares confident (>= 0.80) on
   **90.8%** of a class it has never seen. Every rule that respects confidence
   therefore mislabels a novel attack as a known one.
2. The withheld class occupies a completely different band of the anomaly score
   from the known attacks: max 0.8351 against a known-malicious median of
   2.6889, while benign traffic reaches 49.27. The detector *does* separate it
   (AUROC 0.7835) — it is simply compressed low.

Fact 2 is the reason a single global threshold cannot work: tau_U is calibrated
on known attacks, which by definition do not describe where unseen ones land.
So every rule here uses the score's **position relative to benign traffic** (a
percentile) rather than its absolute value.

All four families calibrate on validation statistics only — percentiles,
prototypes, reference distributions. None fits a supervised model, so the
manuscript's claim survives in the weaker but honest form "adjustable without
retraining the *models*".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nids.stages.classifier import UNKNOWN

BENIGN = "Benign"
ZERO_DAY = "Zero Day"


def _percentile_of(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Where each value falls in the reference distribution, in [0, 1].

    This is the scale-free primitive the four families share. An absolute score
    means nothing across detectors or attack families; "higher than 97% of
    benign traffic" means the same thing everywhere.
    """
    if reference is None or len(reference) == 0:
        return np.zeros(len(values))
    return np.searchsorted(np.sort(reference), values) / len(reference)


# ---------------------------------------------------------------------------
# Family A — trust the SHAPE of the probability vector, not its maximum
# ---------------------------------------------------------------------------

@dataclass
class ShapeRulesConfig:
    """Rules on the distribution's shape rather than its peak.

    A random forest facing an unseen class can still concentrate its votes and
    report high confidence. But *how* it does so differs: on in-distribution
    input the probability mass sits on one class with the rest near zero; out of
    distribution it tends to be flatter, or split between two classes, even when
    the maximum is high. Entropy and the top-two margin see that; the maximum
    alone does not.
    """

    #: Maximum entropy (normalised to [0,1]) still considered a decisive vote.
    max_entropy: float = 0.35
    #: Minimum gap between the top two classes for a label to be trusted.
    min_margin: float = 0.30
    #: Detector percentile above which a flow is suspicious at all.
    susp_pct: float = 0.90
    #: Detector percentile above which an unnameable flow becomes Zero Day.
    zero_day_pct: float = 0.95
    ref_benign: np.ndarray | None = field(default=None, repr=False)


class ShapeRules:
    name = "A-shape"

    def __init__(self, cfg: ShapeRulesConfig):
        self.cfg = cfg

    def fuse(self, scores, labels, certainty, proba=None) -> np.ndarray:
        c = self.cfg
        pct = _percentile_of(scores, c.ref_benign)
        n = len(scores)

        if proba is not None and proba.shape[1] > 1:
            p = np.clip(proba, 1e-12, 1.0)
            ent = -(p * np.log(p)).sum(axis=1) / np.log(proba.shape[1])
            top2 = np.sort(proba, axis=1)[:, -2:]
            margin = top2[:, 1] - top2[:, 0]
        else:
            ent = np.zeros(n)
            margin = np.asarray(certainty)

        # A label is trusted only if the vote is BOTH concentrated (low entropy)
        # and decisive (wide margin). This is the test the plain confidence
        # threshold cannot express.
        trusted = (labels != UNKNOWN) & (ent <= c.max_entropy) & (margin >= c.min_margin)
        suspicious = pct >= c.susp_pct

        out = np.full(n, BENIGN, dtype=object)
        out[trusted] = labels[trusted]
        # Confident-looking but shapeless vote on an anomalous flow: exactly the
        # zero-day case the manuscript's rules route to a wrong known class.
        untrusted_anomalous = (~trusted) & (pct >= c.zero_day_pct)
        out[untrusted_anomalous] = ZERO_DAY
        out[(~trusted) & suspicious & (pct < c.zero_day_pct)] = BENIGN
        return out


# ---------------------------------------------------------------------------
# Family B — the detector gets a veto
# ---------------------------------------------------------------------------

@dataclass
class VetoRulesConfig:
    """A confident label no longer wins unconditionally.

    The manuscript's table has no row where the detector overrules the
    classifier, which is precisely why cycle 4 found the detector irrelevant at
    every threshold. Here a *strongly* anomalous flow overrides a confident
    label: the classifier may be sure, but if the flow looks nothing like
    anything benign, being sure is not enough.
    """

    #: Percentile above which the detector's objection beats the label.
    veto_pct: float = 0.99
    susp_pct: float = 0.90
    tau_m: float = 0.80
    ref_benign: np.ndarray | None = field(default=None, repr=False)


class VetoRules:
    name = "B-veto"

    def __init__(self, cfg: VetoRulesConfig):
        self.cfg = cfg

    def fuse(self, scores, labels, certainty, proba=None) -> np.ndarray:
        c = self.cfg
        pct = _percentile_of(scores, c.ref_benign)
        named = (labels != UNKNOWN) & (np.asarray(certainty) >= c.tau_m)

        out = np.full(len(scores), BENIGN, dtype=object)
        out[named] = labels[named]
        # The veto: extreme anomaly outranks a confident label.
        out[pct >= c.veto_pct] = ZERO_DAY
        # Unnameable and merely suspicious stays benign - the false-positive
        # control the extension stage was meant to provide.
        out[(~named) & (pct >= c.susp_pct) & (pct < c.veto_pct)] = BENIGN
        return out


# ---------------------------------------------------------------------------
# Family C — how well does the classifier EXPLAIN this flow?
# ---------------------------------------------------------------------------

@dataclass
class ExplainRulesConfig:
    """Ask the classifier how typical the flow is, not how sure it is.

    Confidence is a statement about the decision boundary; typicality is a
    statement about the data. A flow can sit far from every class prototype the
    forest learned and still fall on one side of a boundary with high
    confidence — which is the out-of-distribution case exactly.

    Typicality here is the distance from the per-class mean probability vector
    seen on validation: an unseen class produces a probability signature unlike
    any known class's signature, even when its argmax matches one.
    """

    #: Percentile of the typicality distance above which the label is rejected.
    atypical_pct: float = 0.95
    zero_day_pct: float = 0.95
    tau_m: float = 0.80
    ref_benign: np.ndarray | None = field(default=None, repr=False)
    prototypes: dict = field(default_factory=dict, repr=False)
    ref_distance: np.ndarray | None = field(default=None, repr=False)


class ExplainRules:
    name = "C-explain"

    def __init__(self, cfg: ExplainRulesConfig):
        self.cfg = cfg

    @staticmethod
    def fit_prototypes(proba_val: np.ndarray, y_val: np.ndarray,
                       classes: np.ndarray) -> dict:
        """Mean probability signature per known class, from validation."""
        protos = {}
        for cls in classes:
            m = y_val == cls
            if m.any():
                protos[str(cls)] = proba_val[m].mean(axis=0)
        return protos

    def _distance(self, proba, labels) -> np.ndarray:
        d = np.full(len(proba), np.inf)
        for i, lab in enumerate(labels):
            proto = self.cfg.prototypes.get(str(lab))
            if proto is not None:
                d[i] = float(np.linalg.norm(proba[i] - proto))
        finite = np.isfinite(d)
        if not finite.all():
            d[~finite] = d[finite].max() if finite.any() else 0.0
        return d

    def fuse(self, scores, labels, certainty, proba=None) -> np.ndarray:
        c = self.cfg
        pct = _percentile_of(scores, c.ref_benign)
        out = np.full(len(scores), BENIGN, dtype=object)
        if proba is None or not c.prototypes:
            return out

        dist = self._distance(proba, labels)
        dist_pct = _percentile_of(dist, c.ref_distance)
        named = (labels != UNKNOWN) & (np.asarray(certainty) >= c.tau_m)
        # Typical for its claimed class: honour the label.
        typical = named & (dist_pct < c.atypical_pct)
        out[typical] = labels[typical]
        # Claims a class but does not look like it, and the detector agrees
        # something is off: novel.
        atypical = named & (dist_pct >= c.atypical_pct) & (pct >= c.zero_day_pct)
        out[atypical] = ZERO_DAY
        out[(~named) & (pct >= c.zero_day_pct)] = ZERO_DAY
        return out


# ---------------------------------------------------------------------------
# Family D — hierarchical cascade
# ---------------------------------------------------------------------------

@dataclass
class CascadeRulesConfig:
    """Tests in order of cost and reliability, each able to decide or defer.

    Keeps the manuscript's interpretability and post-hoc tunability: every gate
    is a threshold a reader can inspect and an operator can move. The ordering
    is the design — cheap confident decisions first, expensive ambiguous ones
    last — which is also the deployment story the paper tells about tiered
    architectures.
    """

    calm_pct: float = 0.50        # gate 1: clearly normal
    strong_conf: float = 0.95     # gate 2: overwhelming label
    tau_m: float = 0.80           # gate 3: usable label
    extreme_pct: float = 0.99     # gate 4: extreme anomaly
    zero_day_pct: float = 0.95    # gate 5: novel
    max_entropy: float = 0.35
    ref_benign: np.ndarray | None = field(default=None, repr=False)


class CascadeRules:
    name = "D-cascade"

    def __init__(self, cfg: CascadeRulesConfig):
        self.cfg = cfg

    def fuse(self, scores, labels, certainty, proba=None) -> np.ndarray:
        c = self.cfg
        pct = _percentile_of(scores, c.ref_benign)
        cert = np.asarray(certainty)
        n = len(scores)
        out = np.full(n, BENIGN, dtype=object)
        decided = np.zeros(n, dtype=bool)

        if proba is not None and proba.shape[1] > 1:
            p = np.clip(proba, 1e-12, 1.0)
            ent = -(p * np.log(p)).sum(axis=1) / np.log(proba.shape[1])
        else:
            ent = np.zeros(n)

        # Gate 1 — the detector is calm and the classifier has nothing to say.
        g1 = (pct < c.calm_pct) & (labels == UNKNOWN)
        out[g1] = BENIGN; decided |= g1

        # Gate 2 — overwhelming, well-shaped label: trust it immediately.
        g2 = ~decided & (labels != UNKNOWN) & (cert >= c.strong_conf) & (ent <= c.max_entropy)
        out[g2] = labels[g2]; decided |= g2

        # Gate 3 — extreme anomaly outranks anything left. Placed AFTER the
        # overwhelming-label gate so it only fires on genuinely doubtful cases.
        g3 = ~decided & (pct >= c.extreme_pct)
        out[g3] = ZERO_DAY; decided |= g3

        # Gate 4 — usable label plus detector agreement.
        g4 = ~decided & (labels != UNKNOWN) & (cert >= c.tau_m) & (pct >= c.calm_pct)
        out[g4] = labels[g4]; decided |= g4

        # Gate 5 — nothing nameable, but anomalous: novel.
        g5 = ~decided & (pct >= c.zero_day_pct)
        out[g5] = ZERO_DAY; decided |= g5

        return out


# ---------------------------------------------------------------------------
# Family E — the signal, in the direction the data actually points
# ---------------------------------------------------------------------------

@dataclass
class OpenSetRulesConfig:
    """Built from the measurement that invalidated the four families above.

    Separability against the unseen class, benign vs never-seen:

        classifier certainty  AUROC 0.9314   (benign 0.445, unseen 0.960)
        top-2 margin          AUROC 0.9256
        detector score        AUROC 0.7835
        entropy               AUROC 0.0702  (i.e. 0.93 inverted)

    Certainty is the *best* available signal — better than the detector — but
    every rule written so far, including the manuscript's, reads it backwards.
    They treat high confidence as "this is a known class, trust the label". The
    data says it means "this resembles an attack": a classifier trained only on
    malicious traffic is uncertain on benign flows (median 0.445, nothing looks
    familiar) and confident on *any* attack, seen or unseen (median 0.960).

    So the rule inverts: certainty decides attack-vs-benign, and a second,
    orthogonal signal decides known-vs-novel. That second signal is typicality
    against prototypes built from the **known malicious** validation rows only —
    the earlier attempt averaged over all of validation, which is dominated by
    benign traffic and produced prototypes representing nothing.
    """

    #: Certainty above which a flow is an attack at all.
    attack_certainty: float = 0.80
    #: Typicality percentile above which the attack is called novel rather than
    #: assigned its argmax label.
    novel_pct: float = 0.90
    #: A strongly anomalous flow is novel regardless of typicality.
    detector_novel_pct: float = 0.99
    ref_benign: np.ndarray | None = field(default=None, repr=False)
    prototypes: dict = field(default_factory=dict, repr=False)
    ref_distance: np.ndarray | None = field(default=None, repr=False)


class OpenSetRules:
    name = "E-openset"

    def __init__(self, cfg: OpenSetRulesConfig):
        self.cfg = cfg

    @staticmethod
    def fit_prototypes(proba_mal: np.ndarray, labels_mal: np.ndarray) -> dict:
        """Probability signature per class, from KNOWN MALICIOUS rows only."""
        return {str(c): proba_mal[labels_mal == c].mean(axis=0)
                for c in np.unique(labels_mal) if (labels_mal == c).any()}

    def _distance(self, proba: np.ndarray, labels: np.ndarray) -> np.ndarray:
        d = np.zeros(len(proba))
        for i, lab in enumerate(labels):
            proto = self.cfg.prototypes.get(str(lab))
            d[i] = np.linalg.norm(proba[i] - proto) if proto is not None else 0.0
        return d

    def fuse(self, scores, labels, certainty, proba=None) -> np.ndarray:
        c = self.cfg
        cert = np.asarray(certainty)
        out = np.full(len(scores), BENIGN, dtype=object)
        if proba is None:
            return out

        # Step 1: is it an attack? Certainty, read in the direction the
        # measurement supports.
        is_attack = cert >= c.attack_certainty

        # Step 2: known or novel? Typicality against known-attack prototypes,
        # plus an extreme-anomaly override from the detector.
        dist = self._distance(proba, labels)
        dist_pct = _percentile_of(dist, c.ref_distance)
        det_pct = _percentile_of(scores, c.ref_benign)
        novel = is_attack & ((dist_pct >= c.novel_pct) |
                             (det_pct >= c.detector_novel_pct))

        known = is_attack & ~novel
        out[known] = labels[known]
        out[novel] = ZERO_DAY
        return out


# ---------------------------------------------------------------------------
# Family F — OpenMax (Bendale & Boult, CVPR 2016), the principled version of E
# ---------------------------------------------------------------------------

@dataclass
class OpenMaxConfig:
    """Extreme-value calibration of the class scores.

    Family E works — it is the only rule that lifted zero-day recall above zero
    — but its novelty test is a percentile of the prototype distance, a
    threshold picked by search. OpenMax replaces that with a Weibull fitted to
    the *tail* of the observed distances, which is what extreme value theory
    says the tail of a distance distribution looks like. The threshold stops
    being a free parameter and becomes a calibrated probability.

    That matters here for a specific reason: the measured weakness of family E
    is its dispersion across folds (0.2848), i.e. one percentile does not suit
    every attack family. A per-class Weibull adapts to each class's own tail.

    Algorithm, following the paper:
      1. Mean Activation Vector per class, from *correctly classified*
         validation rows.
      2. Distances from those rows to their class MAV.
      3. Weibull fitted to the `tail_size` largest distances per class.
      4. At inference, rescale the top `alpha` class scores by the Weibull CDF;
         the mass removed becomes the unknown-class pseudo-probability.

    Fitted on validation statistics only, so the "tunable without retraining
    the models" claim still holds.
    """

    tail_size: int = 20
    alpha: int = 2            # how many top classes get recalibrated
    unknown_threshold: float = 0.5
    attack_certainty: float = 0.64
    ref_benign: np.ndarray | None = field(default=None, repr=False)
    detector_novel_pct: float = 0.99
    mav: dict = field(default_factory=dict, repr=False)
    weibull: dict = field(default_factory=dict, repr=False)


class OpenMaxRules:
    name = "F-openmax"

    def __init__(self, cfg: OpenMaxConfig):
        self.cfg = cfg

    @staticmethod
    def fit(proba: np.ndarray, labels: np.ndarray, true_labels: np.ndarray,
            classes: np.ndarray, tail_size: int = 20) -> tuple[dict, dict]:
        """MAVs and per-class Weibull tails, from correctly classified rows."""
        from scipy.stats import weibull_min

        mav, wb = {}, {}
        for cls in classes:
            correct = (labels == cls) & (true_labels == cls)
            if correct.sum() < 5:
                continue
            v = proba[correct]
            m = v.mean(axis=0)
            mav[str(cls)] = m
            d = np.linalg.norm(v - m, axis=1)
            tail = np.sort(d)[-min(tail_size, len(d)):]
            if len(tail) < 3 or np.allclose(tail, tail[0]):
                continue
            try:
                # floc=0: distances are non-negative by construction.
                shape, loc, scale = weibull_min.fit(tail, floc=0)
                wb[str(cls)] = (shape, loc, scale)
            except Exception:
                continue
        return mav, wb

    def _unknown_prob(self, proba: np.ndarray) -> np.ndarray:
        """Pseudo-probability that a row belongs to no known class.

        Vectorised over rows. The obvious per-row implementation calls
        `weibull_min.cdf` once per (row, rank) — on 820k evaluation rows that is
        millions of scipy calls and it killed the process outright. Here the
        distance to every class MAV is computed as one matrix operation and the
        CDF is evaluated once per class over the whole column.
        """
        from scipy.stats import weibull_min

        c = self.cfg
        n, k = proba.shape
        classes = list(c.mav.keys())
        if not classes:
            return np.zeros(n)

        # (n, n_classes) distances to each class MAV, in one pass.
        M = np.stack([c.mav[cl] for cl in classes])              # (C, k)
        d = np.linalg.norm(proba[:, None, :] - M[None, :, :], axis=2)

        # Weibull CDF per class, evaluated column-wise.
        w = np.zeros_like(d)
        for ci, cl in enumerate(classes):
            if cl in c.weibull:
                shape, loc, scale = c.weibull[cl]
                w[:, ci] = weibull_min.cdf(d[:, ci], shape, loc=loc, scale=scale)

        # Recalibrate only the top-alpha classes by predicted probability,
        # weighting by rank exactly as the paper does.
        n_cls = len(classes)
        order = np.argsort(proba[:, :n_cls], axis=1)[:, ::-1]
        rows = np.arange(n)
        removed = np.zeros(n)
        for rank in range(min(c.alpha, n_cls)):
            j = order[:, rank]
            weight = (c.alpha - rank) / c.alpha
            removed += proba[rows, j] * w[rows, j] * weight
        return removed

    def fuse(self, scores, labels, certainty, proba=None) -> np.ndarray:
        c = self.cfg
        out = np.full(len(scores), BENIGN, dtype=object)
        if proba is None or not c.mav:
            return out

        cert = np.asarray(certainty)
        is_attack = cert >= c.attack_certainty
        unk = self._unknown_prob(proba)
        det_pct = _percentile_of(scores, c.ref_benign)

        novel = is_attack & ((unk >= c.unknown_threshold) |
                             (det_pct >= c.detector_novel_pct))
        known = is_attack & ~novel
        out[known] = labels[known]
        out[novel] = ZERO_DAY
        return out


# ---------------------------------------------------------------------------
# Family G — the detector gets an INDEPENDENT path to "novel"
# ---------------------------------------------------------------------------

@dataclass
class DualPathConfig:
    """Two independent routes to a Zero-Day verdict.

    **The structural defect this fixes.** Family E is sequential: first decide
    "is this an attack?" from classifier certainty alone, then decide
    "known or novel?". If the first filter rejects a flow, the detector is
    never consulted at all.

    Cycle 13 measured the cost precisely. On the slowloris fold the detector
    separates the withheld class at AUROC 0.8659 — its median percentile is
    0.8879 against benign 0.5000 — yet the system detects 0.30% of it, because
    classifier certainty on slowloris (median 0.4772) barely exceeds benign
    (0.3655) and only 0.30% clears `attack_certainty`. Improving the detector
    from 0.55 to 0.87 changed nothing end-to-end, exactly as improving the
    views did in cycle 12.

    So the detector gets its own route: a flow the detector finds *extremely*
    unusual is called novel regardless of what the classifier thinks. The
    classifier route is unchanged, so classes it handles well are unaffected.

    The two paths are deliberately asymmetric. `detector_solo_pct` should sit
    high — the detector alone must be very confident to overrule silence from
    the classifier, otherwise benign tail traffic floods the Zero-Day class,
    which is the failure mode that cost 55 points of balanced accuracy in
    cycle 10.
    """

    #: Path 1 (unchanged): certainty says attack, typicality says known/novel.
    attack_certainty: float = 0.64
    novel_pct: float = 0.90
    #: Path 2 (new): detector percentile above which a flow is novel on the
    #: detector's word alone, whatever the classifier says.
    detector_solo_pct: float = 0.995
    #: Within path 1, the detector can still force "novel".
    detector_novel_pct: float = 0.99
    ref_benign: np.ndarray | None = field(default=None, repr=False)
    prototypes: dict = field(default_factory=dict, repr=False)
    ref_distance: np.ndarray | None = field(default=None, repr=False)


class DualPathRules:
    name = "G-dualpath"

    def __init__(self, cfg: DualPathConfig):
        self.cfg = cfg

    def _distance(self, proba: np.ndarray, labels: np.ndarray) -> np.ndarray:
        d = np.zeros(len(proba))
        for i, lab in enumerate(labels):
            proto = self.cfg.prototypes.get(str(lab))
            d[i] = np.linalg.norm(proba[i] - proto) if proto is not None else 0.0
        return d

    def fuse(self, scores, labels, certainty, proba=None) -> np.ndarray:
        c = self.cfg
        cert = np.asarray(certainty)
        out = np.full(len(scores), BENIGN, dtype=object)
        if proba is None:
            return out

        det_pct = _percentile_of(scores, c.ref_benign)
        dist_pct = _percentile_of(self._distance(proba, labels), c.ref_distance)

        # Path 1 — the classifier's route, unchanged.
        is_attack = cert >= c.attack_certainty
        novel_1 = is_attack & ((dist_pct >= c.novel_pct) |
                               (det_pct >= c.detector_novel_pct))
        known = is_attack & ~novel_1
        out[known] = labels[known]
        out[novel_1] = ZERO_DAY

        # Path 2 — the detector's own route, for flows path 1 discarded.
        out[(~is_attack) & (det_pct >= c.detector_solo_pct)] = ZERO_DAY
        return out


# ---------------------------------------------------------------------------
# Family H — per-view percentile rules on top of the dual path.
# ---------------------------------------------------------------------------
@dataclass
class PerViewConfig:
    """Path 2 reads the *vector* of per-mechanism percentiles, not one score.

    **Why this is worth a separate family.** Cycle 12 measured that mechanism
    views are genuinely specialised — Brute Force is seen by `shape` at 0.914
    while `timing` reads it backwards at 0.339 — yet aggregating them with a
    weighted maximum lost to the plain single detector at equal preservation of
    known classes (0.1643 against 0.2730). The diagnosis was that a maximum
    rises on benign and known traffic too: with six views, a benign flow only
    needs one unlucky view to score high, which is the same saturation that
    made `extremeness_agg='max'` produce AUROC exactly 0.5000.

    So the aggregation here is *counting*, not maximising. A flow is novel when
    at least `min_views` mechanisms independently call it extreme. Benign tail
    traffic is extreme in one view by chance; a genuinely novel attack should
    disturb several mechanisms at once, because an attack that changes nothing
    but one incidental statistic is not much of an attack.

    `concentration` is the opposite reading, kept because the measurements do
    not decide between them a priori: a slow-DoS variant may be extreme in
    `timing` alone and *typical* everywhere else, and that contrast is itself a
    signature. It fires when one view is very high while the mean stays low.
    """

    # Path 1 — the classifier's route, identical to family G so the comparison
    # isolates the aggregator and nothing else.
    attack_certainty: float = 0.64
    novel_pct: float = 0.90
    detector_novel_pct: float = 0.99

    #: Path 2, counting rule: a view counts as extreme past this percentile,
    #: and this many views must agree.
    view_extreme_pct: float = 0.99
    min_views: int = 2
    #: Per-view orientation, +1 or -1, one entry per column of `pv`. A view
    #: with -1 is read as `1 - percentile`, because being *unusually typical*
    #: in that mechanism is the anomaly.
    #:
    #: This is not optional bookkeeping. The `control` view — TCP flags,
    #: initial window sizes, header lengths — separates the withheld class at
    #: AUROC 0.0680 on Web Attack XSS and 0.1999 on DoS Hulk: strongly
    #: informative, read backwards. Automated attacks drive protocol machinery
    #: *more regularly* than human traffic, which is the same compactness the
    #: project measured on the raw features. A `pv >= threshold` rule spends
    #: that signal in the wrong direction.
    #:
    #: `fit_orientation` estimates it from benign versus **known** attacks, so
    #: nothing about the withheld class enters. The estimate is stable: control
    #: lands between 0.2346 and 0.3429 across all five folds, the same
    #: direction it has on the class it has never seen.
    orientation: np.ndarray | None = field(default=None, repr=False)
    #: Path 2, concentration rule: fires when the top view is past
    #: `conc_top_pct` while the mean across views stays below `conc_mean_max`.
    #: Disabled when `conc_top_pct` >= 1.
    conc_top_pct: float = 1.0
    conc_mean_max: float = 0.5

    ref_benign: np.ndarray | None = field(default=None, repr=False)
    prototypes: dict = field(default_factory=dict, repr=False)
    ref_distance: np.ndarray | None = field(default=None, repr=False)


class PerViewRules:
    """Family H. `fuse` takes the per-view matrix in place of a single score."""

    name = "H-perview"

    def __init__(self, cfg: PerViewConfig):
        self.cfg = cfg

    @staticmethod
    def fit_orientation(pv_benign: np.ndarray, pv_malicious: np.ndarray) -> np.ndarray:
        """+1 where high percentiles mean attack, -1 where low ones do.

        Estimated on benign versus **known** attacks only, exactly as
        `MechanismDetector.fit_weights` does. A view whose AUROC sits below 0.5
        on every known family is measuring the opposite of what the rule
        assumes, and the fix is to read it the other way round rather than to
        silence it — at AUROC 0.29 a view carries more signal than one at 0.55.
        """
        from sklearn.metrics import roc_auc_score

        b = np.asarray(pv_benign, dtype=np.float64)
        m = np.asarray(pv_malicious, dtype=np.float64)
        y = np.r_[np.zeros(len(b)), np.ones(len(m))]
        X = np.vstack([b, m])
        return np.array([1.0 if roc_auc_score(y, X[:, j]) >= 0.5 else -1.0
                         for j in range(X.shape[1])])

    def _oriented(self, pv: np.ndarray) -> np.ndarray:
        o = self.cfg.orientation
        if o is None:
            return pv
        o = np.asarray(o, dtype=np.float64)
        return np.where(o[None, :] > 0, pv, 1.0 - pv)

    def _distance(self, proba: np.ndarray, labels: np.ndarray) -> np.ndarray:
        d = np.zeros(len(proba))
        for i, lab in enumerate(labels):
            proto = self.cfg.prototypes.get(str(lab))
            d[i] = np.linalg.norm(proba[i] - proto) if proto is not None else 0.0
        return d

    def fuse(self, pv, labels, certainty, proba=None, scores=None) -> np.ndarray:
        """`pv` is (n_flows, n_views) of percentiles within benign traffic."""
        c = self.cfg
        pv = np.asarray(pv, dtype=np.float64)
        cert = np.asarray(certainty)
        out = np.full(len(pv), BENIGN, dtype=object)
        if proba is None:
            return out

        dist_pct = _percentile_of(self._distance(proba, labels), c.ref_distance)
        # Path 1 still consults the *global* detector, as family G does, so any
        # difference measured against G comes from path 2 alone.
        det_pct = (_percentile_of(scores, c.ref_benign) if scores is not None
                   else pv.mean(axis=1))

        is_attack = cert >= c.attack_certainty
        novel_1 = is_attack & ((dist_pct >= c.novel_pct) |
                               (det_pct >= c.detector_novel_pct))
        known = is_attack & ~novel_1
        out[known] = labels[known]
        out[novel_1] = ZERO_DAY

        # Path 2 — agreement across mechanisms, each read in its own direction.
        pvo = self._oriented(pv)
        n_extreme = (pvo >= c.view_extreme_pct).sum(axis=1)
        fires = n_extreme >= c.min_views
        if c.conc_top_pct < 1.0:
            fires = fires | ((pvo.max(axis=1) >= c.conc_top_pct) &
                             (pvo.mean(axis=1) <= c.conc_mean_max))
        out[(~is_attack) & fires] = ZERO_DAY
        return out


# ---------------------------------------------------------------------------
# Family I — a light learned model over the per-view percentiles.
# ---------------------------------------------------------------------------
@dataclass
class LearnedViewConfig:
    """Path 2 is a small supervised model on the per-view percentile vector.

    **What it is trained on, and why that is legitimate.** The model learns to
    separate benign from *known* malicious traffic using only the six
    mechanism percentiles — never raw features, never the withheld class. It is
    the same discipline as `MechanismDetector.fit_weights`, which uses known
    attacks for weighting only: a linear rule over "how unusual is this flow in
    each mechanism" carries no class identity, so nothing stops it applying to
    an attack nobody has seen. The withheld class never enters `fit`.

    **Why it might beat counting.** Family H treats views symmetrically and
    thresholds each at the same percentile. In reality the views are not
    equally informative and they are correlated — `volume` and `duration` move
    together on DoS. A logistic model can down-weight a redundant view and read
    a *combination* (high timing with low volume) that no per-view threshold
    expresses.

    **Why it might not.** It has 6 inputs and is fitted on known attacks, so it
    can key on what makes *known* attacks unusual, which is exactly the
    quantity that improves while zero-day detection gets worse. That is the
    failure the project has measured repeatedly, so the model is deliberately
    tiny and regularised, and it is judged on the same double criterion.
    """

    attack_certainty: float = 0.64
    novel_pct: float = 0.90
    detector_novel_pct: float = 0.99
    #: Path 2 fires when the model's malicious probability clears this.
    learned_pct: float = 0.99
    #: Inverse regularisation strength for the logistic model.
    C: float = 1.0

    ref_benign: np.ndarray | None = field(default=None, repr=False)
    prototypes: dict = field(default_factory=dict, repr=False)
    ref_distance: np.ndarray | None = field(default=None, repr=False)
    #: Reference distribution of model outputs on benign traffic, so
    #: `learned_pct` is a percentile and not a raw probability - the scale
    #: mistake that cost cycles 6 and 12 a wasted evaluation each.
    ref_learned: np.ndarray | None = field(default=None, repr=False)


class LearnedViewRules:
    name = "I-learned"

    def __init__(self, cfg: LearnedViewConfig):
        self.cfg = cfg
        self.model = None

    def fit(self, pv_benign: np.ndarray, pv_malicious: np.ndarray) -> "LearnedViewRules":
        """Benign vs KNOWN malicious, on per-view percentiles only."""
        from sklearn.linear_model import LogisticRegression

        X = np.vstack([np.asarray(pv_benign, dtype=np.float64),
                       np.asarray(pv_malicious, dtype=np.float64)])
        y = np.r_[np.zeros(len(pv_benign)), np.ones(len(pv_malicious))]
        self.model = LogisticRegression(
            C=self.cfg.C, max_iter=2000, class_weight="balanced").fit(X, y)
        self.cfg.ref_learned = np.sort(
            self.model.predict_proba(np.asarray(pv_benign, dtype=np.float64))[:, 1])
        return self

    def _distance(self, proba: np.ndarray, labels: np.ndarray) -> np.ndarray:
        d = np.zeros(len(proba))
        for i, lab in enumerate(labels):
            proto = self.cfg.prototypes.get(str(lab))
            d[i] = np.linalg.norm(proba[i] - proto) if proto is not None else 0.0
        return d

    def fuse(self, pv, labels, certainty, proba=None, scores=None) -> np.ndarray:
        c = self.cfg
        pv = np.asarray(pv, dtype=np.float64)
        cert = np.asarray(certainty)
        out = np.full(len(pv), BENIGN, dtype=object)
        if proba is None or self.model is None:
            return out

        dist_pct = _percentile_of(self._distance(proba, labels), c.ref_distance)
        det_pct = (_percentile_of(scores, c.ref_benign) if scores is not None
                   else pv.mean(axis=1))

        is_attack = cert >= c.attack_certainty
        novel_1 = is_attack & ((dist_pct >= c.novel_pct) |
                               (det_pct >= c.detector_novel_pct))
        known = is_attack & ~novel_1
        out[known] = labels[known]
        out[novel_1] = ZERO_DAY

        p = self.model.predict_proba(pv)[:, 1]
        out[(~is_attack) & (_percentile_of(p, c.ref_learned) >= c.learned_pct)] = ZERO_DAY
        return out


# ---------------------------------------------------------------------------
# Family J — disagreement between the stages as evidence, not noise.
# ---------------------------------------------------------------------------
@dataclass
class DisagreementConfig:
    """A confident classifier contradicted by the detector is *more* suspect.

    **The measurement this comes from.** Cycle 14 asked why 49% of zero-day
    flows escape every aggregator, and found three different causes. On Web
    Attack XSS the cause is stark: the 601 lost flows have median classifier
    certainty of **1.0000** — the maximum — and 99.17% of them clear
    `attack_certainty`, so they enter path 1 and leave it wearing the name of
    a known class. Meanwhile the detector puts them at percentile 0.9063
    against benign 0.5.

    So the system's two stages maximally disagree — one is certain it is a
    known attack, the other is certain it is unlike anything benign — and
    every rule family built so far reads that as *agreement to trust the
    classifier*. Certainty acts as a pass that switches the detector off,
    which is the same structural fault cycle 13 found and the dual path only
    half-fixed: path 2 rescues flows path 1 *rejects*, never flows path 1
    accepts.

    **Why disagreement is the right signal.** A classifier trained on five
    families has no output meaning "none of these". Faced with a sixth, it
    must still distribute its probability mass, and a genuinely novel attack
    that resembles a known one along the features the classifier uses will
    produce *high* certainty for the wrong class. High certainty is therefore
    not evidence of correctness on open-set inputs — it is evidence only of
    where the flow sits relative to the training classes. The detector is the
    independent witness, and when it dissents on a flow the classifier is sure
    about, the disagreement itself is the information.

    This is the open-set argument from OpenMax (Bendale & Boult, CVPR 2016)
    applied at the fusion stage rather than inside the classifier: cycle 10
    measured that OpenMax on a 4-component random-forest vector adds nothing,
    but the *reasoning* transfers to the rules, where the detector supplies
    the outside view OpenMax lacks.

    **The cost, stated plainly.** Known classes are also predicted with high
    certainty, so a disagreement rule can strip correct labels off traffic the
    classifier handles well — the failure that cost 55 points of balanced
    accuracy in cycle 10. It is gated on the detector percentile being high in
    absolute terms (`disagree_det_pct`), so it fires only where the detector
    positively dissents rather than merely fails to confirm.
    """

    # Path 1 and path 2 as in family G, unchanged, so any difference measured
    # against G is attributable to path 3 alone.
    attack_certainty: float = 0.64
    novel_pct: float = 0.90
    detector_solo_pct: float = 0.995
    detector_novel_pct: float = 0.99

    #: Path 3: fires when the classifier is at least this certain *and* the
    #: detector puts the flow at least this far out. Both must hold — a
    #: confident classifier with a quiet detector is left alone.
    disagree_cert: float = 0.95
    disagree_det_pct: float = 0.90
    #: Set >= 1 to disable path 3 entirely, which makes family J reduce
    #: exactly to family G and gives the ablation for free.
    disagree_enabled: float = 1.0

    ref_benign: np.ndarray | None = field(default=None, repr=False)
    prototypes: dict = field(default_factory=dict, repr=False)
    ref_distance: np.ndarray | None = field(default=None, repr=False)


class DisagreementRules:
    name = "J-disagree"

    def __init__(self, cfg: DisagreementConfig):
        self.cfg = cfg

    def _distance(self, proba: np.ndarray, labels: np.ndarray) -> np.ndarray:
        d = np.zeros(len(proba))
        for i, lab in enumerate(labels):
            proto = self.cfg.prototypes.get(str(lab))
            d[i] = np.linalg.norm(proba[i] - proto) if proto is not None else 0.0
        return d

    def fuse(self, scores, labels, certainty, proba=None) -> np.ndarray:
        c = self.cfg
        cert = np.asarray(certainty)
        out = np.full(len(scores), BENIGN, dtype=object)
        if proba is None:
            return out

        det_pct = _percentile_of(scores, c.ref_benign)
        dist_pct = _percentile_of(self._distance(proba, labels), c.ref_distance)

        is_attack = cert >= c.attack_certainty
        novel_1 = is_attack & ((dist_pct >= c.novel_pct) |
                               (det_pct >= c.detector_novel_pct))
        known = is_attack & ~novel_1
        out[known] = labels[known]
        out[novel_1] = ZERO_DAY

        # Path 2 — the detector alone, for flows path 1 rejected.
        out[(~is_attack) & (det_pct >= c.detector_solo_pct)] = ZERO_DAY

        # Path 3 — the two stages contradict each other. Applies to flows
        # path 1 *accepted* and labelled, which is what paths 1 and 2 between
        # them cannot reach.
        if c.disagree_enabled < 1.0:
            out[known & (cert >= c.disagree_cert) &
                (det_pct >= c.disagree_det_pct)] = ZERO_DAY
        return out

    def explain(self, scores, labels, certainty, proba=None):
        """(verdict, which path fired) — for the explainability claim."""
        c = self.cfg
        cert = np.asarray(certainty)
        verdict = self.fuse(scores, labels, certainty, proba=proba)
        why = np.full(len(scores), "benign", dtype=object)
        if proba is None:
            return verdict, why
        det_pct = _percentile_of(scores, c.ref_benign)
        dist_pct = _percentile_of(self._distance(proba, labels), c.ref_distance)
        is_attack = cert >= c.attack_certainty
        why[is_attack] = "classificatore"
        why[is_attack & (dist_pct >= c.novel_pct)] = "atipico fra i noti"
        why[is_attack & (det_pct >= c.detector_novel_pct)] = "detector, entro via 1"
        why[(~is_attack) & (det_pct >= c.detector_solo_pct)] = "detector da solo"
        if c.disagree_enabled < 1.0:
            why[(verdict == ZERO_DAY) & is_attack & (cert >= c.disagree_cert) &
                (det_pct >= c.disagree_det_pct)] = "disaccordo fra stadi"
        return verdict, why
