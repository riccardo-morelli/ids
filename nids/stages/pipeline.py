"""Stage 2 - rule-based fusion, and the two competing architectures.

Both architectures are built from the same Detector and Classifier objects and
differ only in how stage 2 combines them. Expressing them this way is what
makes the comparison fair in the sense the brief demands: same splits, same
preprocessing budget, same tuning effort, one harness.

**Verkerken et al. (sequential, 3 stages).** The detector gates everything: a
sample scoring below tau_B is called Benign immediately and never reaches the
classifier. Only suspicious samples are classified; those the classifier
cannot name are sent to the extension stage, which re-reads the anomaly score
against tau_U to decide Zero Day vs Benign.

**Ours (parallel, 2 stages).** Detector and classifier run concurrently on
every sample, and stage 2 fuses both outputs. The key behavioural difference:
a confident attack label from the classifier is honoured *even when the
detector called the sample benign* - which is the source of both our higher
attack recall and our higher benign false-positive rate.

The inference-time claim is measured here rather than asserted. Our submitted
paper reported 1.116s vs 1.520s (a 27% reduction) but the legacy code summed
sequential stage timings, and our detector skipped the PCA transform theirs
performs. `timed_predict` records per-stage times so the parallel claim can be
stated as max(detector, classifier) + fusion with the components visible,
rather than as an unexplained total.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from nids.stages.classifier import UNKNOWN, Classifier, ClassifierConfig
from nids.stages.detector import Detector, DetectorConfig

BENIGN = "Benign"
ZERO_DAY = "Zero Day"


@dataclass
class Thresholds:
    tau_b: float          # detector: benign vs suspicious
    tau_m: float          # classifier: confident vs Unknown
    tau_u: float          # extension: zero-day vs benign
    provenance: dict = field(default_factory=dict)


@dataclass
class Timing:
    detector_s: float
    classifier_s: float
    fusion_s: float

    @property
    def sequential_total(self) -> float:
        return self.detector_s + self.classifier_s + self.fusion_s

    @property
    def parallel_total(self) -> float:
        """Stages 1a and 1b are independent, so a parallel deployment pays the
        slower of the two plus fusion - not their sum."""
        return max(self.detector_s, self.classifier_s) + self.fusion_s


class MultiStage:
    """Base: holds the two stage-1 models and the thresholds."""

    architecture = "base"

    def __init__(self, detector: Detector, classifier: Classifier,
                 thresholds: Thresholds):
        self.detector = detector
        self.classifier = classifier
        self.thresholds = thresholds

    def _stage1(self, X) -> tuple[np.ndarray, np.ndarray, Timing]:
        t0 = time.perf_counter()
        scores = self.detector.score(X)
        t1 = time.perf_counter()
        labels = self.classifier.predict_with_unknown(X, self.thresholds.tau_m)
        t2 = time.perf_counter()
        return scores, labels, Timing(t1 - t0, t2 - t1, 0.0)

    def fuse(self, scores, labels) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        return self.timed_predict(X)[0]

    def timed_predict(self, X) -> tuple[np.ndarray, Timing]:
        scores, labels, t = self._stage1(X)
        t0 = time.perf_counter()
        out = self.fuse(scores, labels)
        t.fusion_s = time.perf_counter() - t0
        return out, t


class VerkerkenPipeline(MultiStage):
    """Sequential three-stage architecture (IEEE TNSM 2023, Fig. 1)."""

    architecture = "verkerken"

    def fuse(self, scores, labels) -> np.ndarray:
        tau_b, tau_u = self.thresholds.tau_b, self.thresholds.tau_u
        out = np.full(len(scores), BENIGN, dtype=object)
        suspicious = scores > tau_b
        # Stage 2 names what it can.
        named = suspicious & (labels != UNKNOWN)
        out[named] = labels[named]
        # Stage 3 (extension) arbitrates the rest on the anomaly score alone.
        unnamed = suspicious & (labels == UNKNOWN)
        out[unnamed & (scores > tau_u)] = ZERO_DAY
        out[unnamed & (scores <= tau_u)] = BENIGN
        return out


class ParallelPipeline(MultiStage):
    """Our parallel two-stage architecture (submitted manuscript, Table 2).

    Rules, in the manuscript's order:

    | Stage 1a          | Stage 1b     | Final       |
    |-------------------|--------------|-------------|
    | Benign            | Unknown      | Benign      |
    | Benign            | Attack class | Attack class|
    | Malicious, E>tau_u| Unknown      | Zero Day    |
    | Malicious, E<=tau_u| Unknown     | Benign      |
    | Malicious         | Attack class | Attack class|

    Row 2 is the substantive difference from Verkerken: a confident attack
    label overrides a benign detector verdict, so attacks the detector misses
    can still be caught.
    """

    architecture = "parallel"

    def fuse(self, scores, labels) -> np.ndarray:
        tau_b, tau_u = self.thresholds.tau_b, self.thresholds.tau_u
        out = np.full(len(scores), BENIGN, dtype=object)
        named = labels != UNKNOWN
        # Rows 2 and 5: a confident class label wins regardless of the detector.
        out[named] = labels[named]
        # Rows 3 and 4: detector suspicious, classifier cannot name it.
        unnamed_susp = (~named) & (scores > tau_b)
        out[unnamed_susp & (scores > tau_u)] = ZERO_DAY
        out[unnamed_susp & (scores <= tau_u)] = BENIGN
        # Row 1 (detector benign, classifier Unknown) is the BENIGN default.
        return out


class ReplicaGatedPipeline(ParallelPipeline):
    """Our architecture with the detector restored as a hard gate.

    This is the manuscript's section 6.2 ablation: modify only the rule table
    so a benign detector verdict is final, and the system reproduces
    Verkerken's behaviour without retraining. Kept because it isolates how much
    of our margin comes from the rule change alone.
    """

    architecture = "parallel-gated"

    def fuse(self, scores, labels) -> np.ndarray:
        tau_b, tau_u = self.thresholds.tau_b, self.thresholds.tau_u
        out = np.full(len(scores), BENIGN, dtype=object)
        suspicious = scores > tau_b
        named = suspicious & (labels != UNKNOWN)
        out[named] = labels[named]
        unnamed = suspicious & (labels == UNKNOWN)
        out[unnamed & (scores > tau_u)] = ZERO_DAY
        return out


ARCHITECTURES = {
    "verkerken": VerkerkenPipeline,
    "parallel": ParallelPipeline,
    "parallel-gated": ReplicaGatedPipeline,
}


def build(
    architecture: str,
    *,
    detector_cfg: DetectorConfig,
    classifier_cfg: ClassifierConfig,
    thresholds: Thresholds,
    detector: Detector | None = None,
    classifier: Classifier | None = None,
) -> MultiStage:
    cls = ARCHITECTURES[architecture]
    return cls(
        detector or Detector(detector_cfg),
        classifier or Classifier(classifier_cfg),
        thresholds,
    )
