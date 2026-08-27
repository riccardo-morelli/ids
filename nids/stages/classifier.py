"""Stage 1b - the multi-class attack classifier.

Trained on malicious traffic only, so it can name an attack type but cannot
recognise benign traffic. It emits a predicted class plus a certainty (the max
class probability); certainty below tau_M becomes "Unknown", which is what
makes zero-day detection possible without ever having seen a zero-day.

`select_tau_m` runs on validation only. The legacy full-model notebook
hard-coded `rf_threshold = 0.9218` rather than reading the value its own
classifier notebook selected, so the pipeline and the reported selection
procedure were not connected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

UNKNOWN = "Unknown"


@dataclass
class ClassifierConfig:
    kind: str = "rf"                 # 'rf' | 'nn'
    scale: bool = False              # RF is scale-invariant; NN needs it
    # Random forest
    n_estimators: int = 500
    max_depth: int | None = None
    max_features: str | int | float = "sqrt"
    max_samples: float | None = None
    # Neural net
    hidden: int = 41
    alpha: float = 0.0379
    max_iter: int = 200
    seed: int = 0
    #: Cost-sensitive learning. 'balanced' reweights classes inversely to
    #: frequency, which is the alternative to downsampling that reviewer 2
    #: point 6 asks us to compare against. Only meaningful when the training
    #: set was NOT already balanced by downsampling.
    class_weight: str | dict | None = None
    params: dict = field(default_factory=dict)


class Classifier:
    """Malicious-only multi-class classifier with an Unknown reject option."""

    def __init__(self, cfg: ClassifierConfig):
        self.cfg = cfg
        self.scaler = StandardScaler() if cfg.scale else None
        self.model = None
        self.classes_: np.ndarray | None = None

    def _x(self, X, *, fit: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.scaler is None:
            return X
        return self.scaler.fit_transform(X) if fit else self.scaler.transform(X)

    def fit(self, X, y) -> "Classifier":
        Xt = self._x(X, fit=True)
        if self.cfg.kind == "rf":
            self.model = RandomForestClassifier(
                n_estimators=self.cfg.n_estimators,
                max_depth=self.cfg.max_depth,
                max_features=self.cfg.max_features,
                max_samples=self.cfg.max_samples,
                class_weight=self.cfg.class_weight,
                random_state=self.cfg.seed, n_jobs=-1, **self.cfg.params,
            )
        elif self.cfg.kind == "nn":
            self.model = MLPClassifier(
                hidden_layer_sizes=(self.cfg.hidden,), alpha=self.cfg.alpha,
                max_iter=self.cfg.max_iter, random_state=self.cfg.seed,
                early_stopping=True, **self.cfg.params,
            )
        elif self.cfg.kind in {"lgbm", "xgb", "catboost"}:
            # Gradient boosting is the strongest family on tabular data in
            # every recent benchmark, and Stage 1b is a tabular multi-class
            # problem. It plugs in unchanged: the rule stage only needs
            # predict_proba, which all three provide.
            k = self.cfg.kind
            if k == "lgbm":
                from lightgbm import LGBMClassifier
                self.model = LGBMClassifier(
                    n_estimators=self.cfg.n_estimators,
                    max_depth=self.cfg.max_depth or -1,
                    random_state=self.cfg.seed, n_jobs=-1, verbose=-1,
                    **self.cfg.params)
            elif k == "xgb":
                from xgboost import XGBClassifier
                # XGBoost needs integer targets, so labels are encoded here
                # and decoded through classes_ on the way out.
                from sklearn.preprocessing import LabelEncoder
                self._le = LabelEncoder().fit(y)
                y = self._le.transform(y)
                self.model = XGBClassifier(
                    n_estimators=self.cfg.n_estimators,
                    max_depth=self.cfg.max_depth or 6,
                    random_state=self.cfg.seed, n_jobs=-1,
                    tree_method="hist", **self.cfg.params)
            else:
                from catboost import CatBoostClassifier
                self.model = CatBoostClassifier(
                    iterations=self.cfg.n_estimators,
                    depth=self.cfg.max_depth or 6,
                    random_seed=self.cfg.seed, verbose=False,
                    **self.cfg.params)
        else:
            raise ValueError(f"unknown classifier kind {self.cfg.kind!r}")
        self.model.fit(Xt, y)
        if self.cfg.kind == "xgb":
            self.classes_ = self._le.classes_
        else:
            self.classes_ = np.asarray(self.model.classes_)
        return self

    def predict_with_certainty(self, X) -> tuple[np.ndarray, np.ndarray]:
        proba = self.model.predict_proba(self._x(X))
        idx = proba.argmax(axis=1)
        return self.classes_[idx], proba[np.arange(len(idx)), idx]

    def predict_with_unknown(self, X, tau_m: float) -> np.ndarray:
        labels, certainty = self.predict_with_certainty(X)
        return np.where(certainty >= tau_m, labels, UNKNOWN)


def select_tau_m(
    certainty_benign: np.ndarray,
    certainty_malicious: np.ndarray,
    *,
    n_grid: int = 500,
    objective: str = "f1",
) -> dict:
    """Choose tau_M on VALIDATION certainties.

    A good tau_M separates "confidently a known attack" from everything else.
    Benign rows should fall below it (the classifier never saw benign traffic,
    so its probabilities there should be diffuse); malicious rows above.
    """
    y = np.r_[np.zeros(len(certainty_benign)), np.ones(len(certainty_malicious))]
    c = np.r_[np.asarray(certainty_benign), np.asarray(certainty_malicious)]
    grid = np.unique(np.quantile(c, np.linspace(0, 1, n_grid)))

    # Vectorised, for the same reason as the detector's F-beta sweep: one sort
    # plus suffix sums yields TP/FP at every threshold at once. Verified
    # bit-identical to the per-threshold f1_score loop it replaces.
    order = np.argsort(c, kind="mergesort")
    c_sorted, y_sorted = c[order], y[order]
    n_pos, n_neg = float(y.sum()), float(len(y) - y.sum())

    idx = np.searchsorted(c_sorted, grid, side="left")
    tp_suffix = np.concatenate([np.cumsum(y_sorted[::-1])[::-1], [0.0]])
    tp = tp_suffix[idx]
    pred_pos = len(c) - idx
    fp = pred_pos - tp

    prec = np.divide(tp, tp + fp, out=np.zeros(len(grid)), where=(tp + fp) > 0)
    rec = tp / n_pos if n_pos else np.zeros(len(grid))
    denom = prec + rec
    f1_pos = np.divide(2 * prec * rec, denom, out=np.zeros(len(grid)),
                       where=denom > 0)

    if objective == "f1":
        score = f1_pos
    elif objective == "f1_weighted":
        # Weighted average of the positive and negative class F1s, matching
        # sklearn's average='weighted'.
        tn = n_neg - fp
        fn = n_pos - tp
        prec_n = np.divide(tn, tn + fn, out=np.zeros(len(grid)), where=(tn + fn) > 0)
        rec_n = tn / n_neg if n_neg else np.zeros(len(grid))
        den_n = prec_n + rec_n
        f1_neg = np.divide(2 * prec_n * rec_n, den_n, out=np.zeros(len(grid)),
                           where=den_n > 0)
        score = (n_pos * f1_pos + n_neg * f1_neg) / (n_pos + n_neg)
    else:
        raise ValueError(f"unknown objective {objective!r}")

    best_i = int(np.argmax(score))
    return {
        "threshold": float(grid[best_i]),
        "score": float(score[best_i]),
        "curve": list(zip(grid.tolist(), score.tolist())),
    }
