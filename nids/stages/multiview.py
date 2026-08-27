"""Multi-view detector: one score per feature group, combined without asking
the classifier anything.

**Where the idea comes from.** The supervisor proposed conditioning the
detector on the classifier: if the classifier is confident a flow is a Web
Attack, score it using only the features that separate Web Attacks. The
principle is right — cycle 3 measured that different classes are separable by
different features (`Init_Win_bytes_backward` alone gives AUROC 0.9291 on Web
Attack), so a single detector on one fixed feature subset is necessarily a
compromise, and the irrelevant features actively dilute the signal.

**Why the literal form breaks.** Conditioning stage 1a on stage 1b's verdict
destroys the premise that makes zero-day detection credible: the detector must
work on attacks nobody has ever seen, and on those the classifier — trained
only on known classes — has nothing meaningful to say. The mechanism would
switch itself off in precisely the case that justifies the architecture.

**The form implemented here.** Compute one anomaly score per *view* (a group of
features chosen by geometry, never by class label), then combine. A never-seen
attack that is geometrically similar to a Web Attack gets caught by the right
view without anyone having to name it. Views are built from benign training
data only, so nothing here can leak.

This differs from the ensemble that failed in cycle 2: there, two detectors ran
on the *same* features and shared the same blind spot (both collapsed on Web
Attack). Here the views differ by construction.

**Explainability comes free.** Because the final score decomposes into
per-view contributions, every alarm carries *which group of features made this
flow anomalous* — actionable triage information rather than a bare number. That
is a partial answer to reviewer 2 point 4, which attacks the false-positive
rate by asking how much analyst work it generates: an alarm that says why is
far cheaper to triage than one that says only "anomalous".

The explanation is a by-product, never a selection criterion: if a multi-view
detector loses on the objective metric, it is discarded regardless of how well
it explains itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration

from nids.data.transforms import FeatureTransform, TransformConfig
from nids.stages.detector import Detector, DetectorConfig


@dataclass
class ViewSpec:
    """One view: a named group of feature indices."""

    name: str
    cols: list[str]


def views_by_correlation(
    X_benign: np.ndarray, cols: list[str], *, n_views: int = 6, seed: int = 0,
) -> list[ViewSpec]:
    """Group features by correlation structure, using benign rows only.

    Features that move together describe the same aspect of a flow (packet
    sizes, timings, flag counts...), so a group is a coherent "view" of
    behaviour. Unsupervised, so it is legitimate inside a leave-one-class-out
    fold.
    """
    X = np.asarray(X_benign, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n_views = max(2, min(n_views, X.shape[1]))
    agg = FeatureAgglomeration(n_clusters=n_views).fit(X)
    out = []
    for v in range(n_views):
        members = [c for c, lab in zip(cols, agg.labels_) if lab == v]
        if members:
            out.append(ViewSpec(name=f"view{v}", cols=members))
    return out


def views_by_prefix(cols: list[str]) -> list[ViewSpec]:
    """Group features by their semantic prefix (Fwd/Bwd/Flow/IAT/Flag/...).

    Cruder than the correlation grouping but far more interpretable, which
    matters for the explanation: "anomalous in its backward-direction packet
    sizes" is something an analyst can act on.
    """
    groups: dict[str, list[str]] = {}
    for c in cols:
        low = c.lower()
        if "iat" in low:
            key = "timing"
        elif "flag" in low:
            key = "flags"
        elif low.startswith("fwd") or "fwd" in low:
            key = "forward"
        elif low.startswith("bwd") or "bwd" in low:
            key = "backward"
        elif "active" in low or "idle" in low:
            key = "activity"
        elif "len" in low or "size" in low or "byte" in low or "packet" in low:
            key = "volume"
        else:
            key = "other"
        groups.setdefault(key, []).append(c)
    return [ViewSpec(name=k, cols=v) for k, v in groups.items() if v]


@dataclass
class MultiViewConfig:
    view_mode: str = "correlation"      # 'correlation' | 'prefix'
    n_views: int = 6
    #: How per-view scores are combined. 'max' takes the most alarmed view
    #: (sensitive, and the one that names the explanation); 'mean' averages
    #: (stable); 'topk' averages the k most alarmed views.
    combine: str = "max"
    top_k: int = 2
    transform: TransformConfig = field(default_factory=
                                       lambda: TransformConfig(kind="quantile_uniform"))
    detector: DetectorConfig = field(default_factory=
                                     lambda: DetectorConfig(kind="knn_density",
                                                            n_train=5000,
                                                            n_neighbors=20,
                                                            two_sided=True))
    seed: int = 0


class MultiViewDetector:
    """Benign-only, emits a score plus a per-view explanation.

    Satisfies the same stage-1a contract as any other detector, so it drops
    into the existing harness unchanged.
    """

    def __init__(self, cfg: MultiViewConfig):
        self.cfg = cfg
        self.views: list[ViewSpec] = []
        self._parts: list[tuple[ViewSpec, FeatureTransform, Detector]] = []
        self._col_index: dict[str, int] = {}

    def fit(self, X_benign: np.ndarray, cols: list[str]) -> "MultiViewDetector":
        X = np.asarray(X_benign, dtype=np.float64)
        self._col_index = {c: i for i, c in enumerate(cols)}
        self.views = (views_by_correlation(X, cols, n_views=self.cfg.n_views,
                                           seed=self.cfg.seed)
                      if self.cfg.view_mode == "correlation"
                      else views_by_prefix(cols))
        self._parts = []
        for v in self.views:
            idx = [self._col_index[c] for c in v.cols]
            tr = FeatureTransform(self.cfg.transform).fit(X[:, idx])
            det_cfg = DetectorConfig(**{**self.cfg.detector.__dict__,
                                        "seed": self.cfg.seed})
            det = Detector(det_cfg).fit(tr.transform(X[:, idx]))
            self._parts.append((v, tr, det))
        # Per-view score distributions on the benign training rows, used to put
        # views on a common scale before combining: a raw score from a 3-column
        # view is not comparable with one from a 20-column view.
        self._ref = [np.sort(d.score(t.transform(X[:, [self._col_index[c]
                                                       for c in v.cols]])))
                     for v, t, d in self._parts]
        return self

    def _per_view(self, X: np.ndarray) -> np.ndarray:
        """(n_rows, n_views) matrix of benign-rank-normalised scores."""
        X = np.asarray(X, dtype=np.float64)
        out = np.empty((len(X), len(self._parts)))
        for j, (v, tr, det) in enumerate(self._parts):
            idx = [self._col_index[c] for c in v.cols]
            s = det.score(tr.transform(X[:, idx]))
            # Percentile against the benign training distribution: comparable
            # across views regardless of each view's native score scale.
            out[:, j] = np.searchsorted(self._ref[j], s) / max(len(self._ref[j]), 1)
        return out

    def score(self, X: np.ndarray) -> np.ndarray:
        p = self._per_view(X)
        if self.cfg.combine == "max":
            return p.max(axis=1)
        if self.cfg.combine == "mean":
            return p.mean(axis=1)
        if self.cfg.combine == "topk":
            k = min(self.cfg.top_k, p.shape[1])
            return np.sort(p, axis=1)[:, -k:].mean(axis=1)
        raise ValueError(f"unknown combine {self.cfg.combine!r}")

    def explain(self, X: np.ndarray, top: int = 3) -> pd.DataFrame:
        """Why was each row flagged? One row per input, most-alarmed views first.

        This is the by-product the supervisor asked for: instead of a bare
        anomaly score, an alarm arrives with the feature groups responsible and
        their percentile against benign traffic.
        """
        p = self._per_view(X)
        names = [v.name for v, _, _ in self._parts]
        rows = []
        for i in range(len(X)):
            order = np.argsort(p[i])[::-1][:top]
            rows.append({
                "score": float(p[i].max()),
                "reasons": ", ".join(
                    f"{names[j]} ({p[i, j]:.2f})" for j in order),
                **{f"view:{names[j]}": round(float(p[i, j]), 4)
                   for j in range(len(names))},
            })
        return pd.DataFrame(rows)

    def view_summary(self) -> pd.DataFrame:
        return pd.DataFrame([{"view": v.name, "n_features": len(v.cols),
                              "features": ", ".join(v.cols[:6]) +
                                          ("…" if len(v.cols) > 6 else "")}
                             for v, _, _ in self._parts])
