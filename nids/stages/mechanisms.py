"""Mechanism views: one detector per *physical behaviour*, not per attack family.

**Where this comes from.** The supervisor observed that different attack
families act on different features — DoS on volume, web attacks on a single
window-size metric — and proposed a per-family aggregate metric. The
measurements back that: volumetric DoS variants reach 0.85 zero-day recall
while slow DoS and web attacks sit near 0.05, `Init_Win_bytes_backward` alone
separates Web Attack at AUROC 0.9291, and cycle 3 showed 64 features *dilute*
a signal that 10 features carry.

**Why views are keyed on mechanism, not family.** Defining a view per known
family means that on a genuinely new attack *no view is its view* — the same
trap as cycle 4, where conditioning the detector on the classifier switched the
mechanism off in precisely the zero-day case it existed for. A view keyed on
*what it measures* has no such failure: a novel volumetric attack lands in the
volume view whether or not anyone can name it.

Each view is a coherent physical question about a flow:

* `volume`    — how much traffic? (bytes, packets, rates)
* `timing`    — how is it spaced in time? (inter-arrival statistics)
* `duration`  — how long-lived, how idle/active?
* `direction` — how asymmetric between the two directions?
* `shape`     — what do individual packets look like? (lengths, segment sizes)
* `control`   — what does the protocol machinery say? (flags, window sizes)

The mapping is written by hand from feature semantics, deliberately: an
unsupervised clustering of features (tried in cycle 10) produced degenerate
groups — some held a single feature — and gave no benefit. Hand-written groups
are also what makes the explanation readable: "anomalous in its timing" is
actionable for an analyst in a way "anomalous in view3" is not.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Substrings that assign a feature to a mechanism. Order matters: the first
#: matching rule wins, so more specific patterns come first.
_RULES: list[tuple[str, tuple[str, ...]]] = [
    # Protocol machinery: flags and advertised window sizes. Checked first
    # because names like "Init_Win_bytes_forward" also contain "bytes".
    ("control", ("flag", "init_win", "min_seg_size", "header length")),
    # Time between packets - the regularity signature of automated traffic.
    ("timing", ("iat",)),
    # How long the flow lived and how it alternated activity and idleness.
    ("duration", ("duration", "idle", "active")),
    # Directional asymmetry: who talks more, client or server.
    ("direction", ("down/up", "act_data_pkt")),
    # What individual packets look like, independent of how many there are.
    ("shape", ("packet length", "segment size", "average packet size")),
    # How much traffic: counts, totals, rates.
    ("volume", ("total", "subflow", "bytes/s", "packets/s", "/s")),
]


def assign_view(feature: str) -> str:
    low = feature.lower()
    for view, pats in _RULES:
        if any(p in low for p in pats):
            return view
    return "other"


def build_views(feature_cols: list[str]) -> dict[str, list[str]]:
    """Group features by mechanism. Uses names only — no data, no labels."""
    views: dict[str, list[str]] = {}
    for c in feature_cols:
        views.setdefault(assign_view(c), []).append(c)
    # A view with one or two features cannot support a density estimate and
    # becomes noise in any aggregation - cycle 10 measured exactly that failure
    # with correlation-clustered views. Fold the stragglers into 'other'.
    small = [v for v, cols in views.items() if len(cols) < 3 and v != "other"]
    for v in small:
        views.setdefault("other", []).extend(views.pop(v))
    return {v: cols for v, cols in views.items() if cols}


@dataclass
class MechanismConfig:
    """One detector per mechanism view, plus how to combine them."""

    #: 'rules' | 'learned' | 'weighted_max' - the three aggregators compared.
    aggregate: str = "weighted_max"
    #: For 'weighted_max': weight each view by how well it separated benign
    #: from KNOWN malicious traffic on validation. Corrects the cycle-10 defect
    #: where a degenerate view dominated a plain maximum.
    view_weights: dict = field(default_factory=dict)
    n_train: int = 5000
    n_neighbors: int = 9
    two_sided: bool = True
    seed: int = 0


class MechanismDetector:
    """Benign-only, per-mechanism scores plus a combined score and a *type*.

    `score` satisfies the usual stage-1a contract, so it drops into the harness
    unchanged. `explain` returns which mechanism fired, which is the
    type-identification the supervisor asked for: not just "anomalous" but
    "anomalous in its volume".
    """

    def __init__(self, cfg: MechanismConfig):
        self.cfg = cfg
        self.views: dict[str, list[str]] = {}
        self._parts: dict[str, tuple] = {}
        self._ref: dict[str, np.ndarray] = {}
        self._idx: dict[str, int] = {}

    def fit(self, X_benign: np.ndarray, cols: list[str]) -> "MechanismDetector":
        from nids.data.transforms import FeatureTransform, TransformConfig
        from nids.stages.detector import Detector, DetectorConfig

        X = np.asarray(X_benign, dtype=np.float64)
        self._idx = {c: i for i, c in enumerate(cols)}
        self.views = build_views(cols)
        for name, vcols in self.views.items():
            ix = [self._idx[c] for c in vcols]
            tr = FeatureTransform(TransformConfig(kind="quantile_uniform")).fit(X[:, ix])
            det = Detector(DetectorConfig(
                kind="knn_density", n_train=self.cfg.n_train,
                n_neighbors=self.cfg.n_neighbors, two_sided=self.cfg.two_sided,
                seed=self.cfg.seed)).fit(tr.transform(X[:, ix]))
            self._parts[name] = (vcols, tr, det)
            # Benign reference distribution, so per-view scores become
            # percentiles and are comparable across views of different width.
            self._ref[name] = np.sort(det.score(tr.transform(X[:, ix])))
        return self

    def per_view(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Percentile of each row within benign traffic, per mechanism."""
        X = np.asarray(X, dtype=np.float64)
        out = {}
        for name, (vcols, tr, det) in self._parts.items():
            ix = [self._idx[c] for c in vcols]
            s = det.score(tr.transform(X[:, ix]))
            ref = self._ref[name]
            out[name] = np.searchsorted(ref, s) / max(len(ref), 1)
        return out

    def score(self, X: np.ndarray) -> np.ndarray:
        pv = self.per_view(X)
        names = sorted(pv)
        M = np.stack([pv[n] for n in names], axis=1)
        if self.cfg.aggregate == "weighted_max":
            w = np.array([self.cfg.view_weights.get(n, 1.0) for n in names])
            return (M * w).max(axis=1)
        if self.cfg.aggregate == "mean":
            return M.mean(axis=1)
        return M.max(axis=1)

    def explain(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(score, mechanism responsible) — the type identification."""
        pv = self.per_view(X)
        names = sorted(pv)
        M = np.stack([pv[n] for n in names], axis=1)
        w = np.array([self.cfg.view_weights.get(n, 1.0) for n in names])
        Mw = M * w
        return Mw.max(axis=1), np.array(names)[Mw.argmax(axis=1)]

    @staticmethod
    def fit_weights(pv_benign: dict, pv_malicious: dict) -> dict:
        """Weight each view by how well it separates benign from known attacks.

        Uses known attacks only for *weighting*, never for the score itself, so
        a novel attack is still scored by every view. A view that cannot tell
        benign from any known attack is unlikely to recognise a new one either.
        """
        from sklearn.metrics import roc_auc_score

        w = {}
        for name in pv_benign:
            b, m = pv_benign[name], pv_malicious[name]
            if len(b) == 0 or len(m) == 0:
                w[name] = 1.0
                continue
            y = np.r_[np.zeros(len(b)), np.ones(len(m))]
            auc = roc_auc_score(y, np.r_[b, m])
            # Separability in either direction, floored so no view is silenced
            # entirely - a view useless on known attacks may still catch a
            # novel one.
            w[name] = max(0.3, abs(auc - 0.5) * 2)
        return w
