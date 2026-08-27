"""Feature transformations for the detector's input space.

The supervisor's standing instruction is that ~90% of the work is in the data,
and the diagnosis backs it: every failed leave-one-class-out fold misroutes the
held-out class to Benign, which means the anomaly score does not separate it.
A different model cannot fix a feature space in which the class is not
separable — a transformation might.

Why this matters specifically here: CICFlowMeter features are heavy-tailed by
construction. Flow durations, byte counts and packet rates span many orders of
magnitude, and network traffic is documented as heavy-tailed in the literature.
`StandardScaler` on a heavy-tailed feature leaves the tail dominating every
distance computation — and distance is exactly what OCSVM, LOF, KDE and
Mahalanobis consume. A benign flow in the tail then looks further from the
benign centre than a genuinely anomalous flow near the mode.

Every transform here is fitted on **training benign data only** and applied
unchanged to validation and test, so no transformation can leak.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


@dataclass
class TransformConfig:
    """One point in the preprocessing search space.

    `kind` selects the scaling family; the flags before it are applied first,
    in the order listed here.
    """

    #: Signed log1p. Compresses the heavy tail while preserving sign and
    #: mapping 0 to 0, which matters because many flow features are zero-heavy.
    signed_log: bool = False
    #: Clip each feature to a benign-training quantile range before scaling, so
    #: a single extreme benign outlier cannot set the scale for everything.
    clip_quantile: float | None = None
    #: Scaling family: 'standard' | 'robust' | 'quantile_uniform' |
    #: 'quantile_normal' | 'power' | 'minmax' | 'none'
    kind: str = "standard"
    #: Optional decorrelation after scaling.
    pca_components: int | None = None
    whiten: bool = False
    #: Drop features whose benign-training variance is below this, computed
    #: after scaling. Rule 22: unused features add dimensions to every distance.
    variance_floor: float | None = None
    name: str = ""
    params: dict = field(default_factory=dict)

    def label(self) -> str:
        if self.name:
            return self.name
        bits = []
        if self.signed_log:
            bits.append("slog")
        if self.clip_quantile:
            bits.append(f"clip{self.clip_quantile}")
        bits.append(self.kind)
        if self.pca_components:
            bits.append(f"pca{self.pca_components}{'w' if self.whiten else ''}")
        if self.variance_floor:
            bits.append(f"vf{self.variance_floor}")
        return "+".join(bits)


class FeatureTransform:
    """Fit on benign training rows; apply everywhere else."""

    def __init__(self, cfg: TransformConfig):
        self.cfg = cfg
        self.scaler = None
        self.pca: PCA | None = None
        self.lo_: np.ndarray | None = None
        self.hi_: np.ndarray | None = None
        self.keep_: np.ndarray | None = None

    @staticmethod
    def _signed_log(X: np.ndarray) -> np.ndarray:
        return np.sign(X) * np.log1p(np.abs(X))

    def _pre(self, X: np.ndarray, *, fit: bool) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.cfg.clip_quantile is not None:
            q = self.cfg.clip_quantile
            if fit:
                self.lo_ = np.quantile(X, 1 - q, axis=0)
                self.hi_ = np.quantile(X, q, axis=0)
            X = np.clip(X, self.lo_, self.hi_)
        if self.cfg.signed_log:
            X = self._signed_log(X)
        return X

    def _make_scaler(self, n_samples: int):
        k = self.cfg.kind
        if k == "standard":
            return StandardScaler()
        if k == "robust":
            return RobustScaler(quantile_range=(25.0, 75.0))
        if k == "quantile_uniform":
            return QuantileTransformer(
                output_distribution="uniform",
                n_quantiles=min(1000, n_samples), subsample=200_000,
                random_state=0)
        if k == "quantile_normal":
            return QuantileTransformer(
                output_distribution="normal",
                n_quantiles=min(1000, n_samples), subsample=200_000,
                random_state=0)
        if k == "power":
            # Yeo-Johnson handles zeros and negatives, unlike Box-Cox.
            return PowerTransformer(method="yeo-johnson", standardize=True)
        if k == "minmax":
            return MinMaxScaler()
        if k == "none":
            return None
        raise ValueError(f"unknown transform kind {self.cfg.kind!r}")

    def fit(self, X_benign_train: np.ndarray) -> "FeatureTransform":
        X = self._pre(X_benign_train, fit=True)
        self.scaler = self._make_scaler(len(X))
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if self.cfg.variance_floor is not None:
            var = X.var(axis=0)
            self.keep_ = var > self.cfg.variance_floor
            if not self.keep_.any():          # never drop everything
                self.keep_ = np.ones(X.shape[1], dtype=bool)
            X = X[:, self.keep_]

        if self.cfg.pca_components:
            self.pca = PCA(n_components=min(self.cfg.pca_components, X.shape[1]),
                           whiten=self.cfg.whiten, random_state=0)
            self.pca.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._pre(X, fit=False)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if self.keep_ is not None:
            X = X[:, self.keep_]
        if self.pca is not None:
            X = self.pca.transform(X)
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


#: The preprocessing search space for cycle 2, ordered simple to complex.
#: Rule 4: the incumbent goes first so every result is read as a delta from it.
CANDIDATES: list[TransformConfig] = [
    TransformConfig(kind="standard", name="baseline-standard"),
    TransformConfig(kind="robust", name="robust"),
    TransformConfig(signed_log=True, kind="standard", name="slog+standard"),
    TransformConfig(signed_log=True, kind="robust", name="slog+robust"),
    TransformConfig(kind="quantile_uniform", name="quantile-uniform"),
    TransformConfig(kind="quantile_normal", name="quantile-normal"),
    TransformConfig(kind="power", name="yeo-johnson"),
    TransformConfig(clip_quantile=0.999, kind="standard", name="clip999+standard"),
    TransformConfig(signed_log=True, clip_quantile=0.999, kind="robust",
                    name="slog+clip999+robust"),
    TransformConfig(signed_log=True, kind="standard", pca_components=20,
                    whiten=True, name="slog+standard+pca20w"),
    TransformConfig(kind="quantile_normal", pca_components=20, whiten=True,
                    name="quantile-normal+pca20w"),
    TransformConfig(signed_log=True, kind="standard", variance_floor=1e-6,
                    name="slog+standard+vf"),
]
