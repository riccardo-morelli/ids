"""Feature selection for the detector.

The cycle-3 diagnosis showed the detector is *diluted*, not blind: Web Attack
is separable by `Init_Win_bytes_backward` alone (AUROC 0.9291), yet the same
detector scores 0.7921 on all 63 features and 0.8909 on the best 10. In a
distance computation every irrelevant feature adds variance without adding
signal, so the informative directions get averaged away.

**The leakage trap.** Ranking features by how well they separate the held-out
class and then measuring on that class is circular — it would report a huge
gain that vanishes on a genuinely unseen attack. The selectors here therefore
come in two flavours:

* `unsupervised` — uses benign training rows only. Legitimate everywhere,
  including inside a leave-one-class-out fold.
* `known_attacks` — uses benign plus the *known* attack classes of the
  validation split. Legitimate only when the target class is withheld from
  that split, which `leave_one_class_out` guarantees. This is the honest
  analogue of what a practitioner does: tune on the attacks you have, hope it
  transfers to the ones you do not.

Nothing here may read the test partition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def variance_filter(X: np.ndarray, cols: list[str], *, floor: float = 1e-8
                    ) -> list[str]:
    """Drop near-constant columns. Cheapest possible unsupervised filter."""
    v = np.var(np.asarray(X, dtype=np.float64), axis=0)
    keep = v > floor
    return [c for c, k in zip(cols, keep) if k] or list(cols)


def correlation_filter(X: np.ndarray, cols: list[str], *, threshold: float = 0.95
                       ) -> list[str]:
    """Greedily drop one of each pair of highly correlated columns.

    Two columns carrying the same information contribute twice to every
    distance, effectively doubling that direction's weight.
    """
    df = pd.DataFrame(np.asarray(X, dtype=np.float64), columns=cols)
    corr = df.corr().abs().fillna(0.0).values
    n = len(cols)
    drop: set[int] = set()
    for i in range(n):
        if i in drop:
            continue
        for j in range(i + 1, n):
            if j not in drop and corr[i, j] >= threshold:
                drop.add(j)
    return [c for k, c in enumerate(cols) if k not in drop] or list(cols)


def unsupervised_select(
    X_benign: np.ndarray, cols: list[str], *,
    corr_threshold: float = 0.95, variance_floor: float = 1e-8,
) -> list[str]:
    """Benign-only selection: safe inside any LOCO fold."""
    keep = variance_filter(X_benign, cols, floor=variance_floor)
    idx = [cols.index(c) for c in keep]
    return correlation_filter(np.asarray(X_benign)[:, idx], keep,
                              threshold=corr_threshold)


def rank_by_known_attacks(
    benign: pd.DataFrame, malicious: pd.DataFrame, cols: list[str],
) -> pd.DataFrame:
    """Rank features by univariate separability against KNOWN attacks.

    Separability is |AUROC - 0.5| * 2, so a strongly anti-correlated feature
    (AUROC 0.08) ranks as highly as a strongly correlated one (0.92) — an
    anomaly score can use either direction.
    """
    rows = []
    for c in cols:
        a, b = benign[c].values, malicious[c].values
        if np.std(np.r_[a, b]) < 1e-12:
            continue
        auc = roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])
        rows.append({"feature": c, "auroc": float(auc),
                     "sep": abs(float(auc) - 0.5) * 2})
    return pd.DataFrame(rows).sort_values("sep", ascending=False)


def select(
    split, *, mode: str = "unsupervised", top_k: int | None = None,
    corr_threshold: float = 0.95,
) -> list[str]:
    """Return the feature subset a detector should be fitted on.

    `mode`:
      'all'           — no selection (the baseline)
      'unsupervised'  — benign-only variance + correlation filtering
      'known_attacks' — unsupervised filter, then rank by separability against
                        the known attack classes present in validation and keep
                        the best `top_k`

    In `known_attacks` mode the ranking reads `split.val_malicious`, which by
    construction excludes the held-out class inside a LOCO fold — so the
    selection never sees the class it will be judged on.
    """
    cols = list(split.feature_cols)
    if mode == "all":
        return cols

    Xb = split.train_benign[cols].values
    keep = unsupervised_select(Xb, cols, corr_threshold=corr_threshold)
    if mode == "unsupervised":
        return keep[:top_k] if top_k else keep
    if mode != "known_attacks":
        raise ValueError(f"unknown selection mode {mode!r}")

    ranked = rank_by_known_attacks(split.val_benign, split.val_malicious, keep)
    if ranked.empty:
        return keep
    return list(ranked.head(top_k or len(keep))["feature"])
