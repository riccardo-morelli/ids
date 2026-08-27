"""Metrics, dispersion, and significance tests.

Reviewer 2 point 3 is the reason this module exists: the submitted manuscript
reported classification metrics from a single train/test split with no
cross-validation, confidence intervals, standard deviations, or significance
test, while claiming a win of 0.9362 vs 0.9216 balanced accuracy.

So the harness makes the rigorous form the *only* form available:

* `evaluate` returns the full metric family for one run.
* `aggregate` turns per-seed runs into mean +/- std with a bootstrap CI.
* `mcnemar` / `wilcoxon_seeds` / `bootstrap_diff_ci` test whether a margin is
  real.

Zero-day recall gets a Wilson interval rather than a normal approximation,
because reviewer 2 point 2 objects to a proportion estimated from 47
instances - and at n=47 the normal approximation is simply wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from nids import config


@dataclass
class RunResult:
    """Metrics from a single (model, seed) evaluation."""

    seed: int
    metrics: dict[str, float]
    y_true: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)
    extra: dict = field(default_factory=dict)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    seed: int = 0,
    zero_day_label: str = "Zero Day",
    benign_label: str = "Benign",
    extra: dict | None = None,
) -> RunResult:
    """Full metric family for one run.

    Balanced accuracy is the headline in both papers, so it leads - but a
    single metric is exactly what the brief forbids resting a claim on, hence
    the rest.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(set(y_true) | set(y_pred))

    # Guard against phantom classes.
    #
    # Balanced accuracy averages per-class recall over the union of true and
    # predicted labels. When a partition structurally cannot contain a class -
    # validation never holds Zero Day rows, because withholding them is what
    # makes them zero-day - every prediction of that class invents a class with
    # recall 0 and drags the mean down by 1/n_classes. Two systems then become
    # incomparable: the one that never predicts Zero Day is flattered for free.
    #
    # This bit me twice: first in the cycle-1 stage ablation, then again in
    # Phase A, where it cost ~1/7 of the balanced accuracy and made a faithful
    # reproduction look broken. The fix lived in an ablation helper the second
    # time round, so a new caller walked straight into it. It belongs here, in
    # the one place every caller goes through.
    present = set(y_true)
    phantom = sorted(set(y_pred) - present)
    if phantom:
        real = np.isin(y_true, list(present))
        pred_folded = np.where(np.isin(y_pred, phantom), "__phantom__", y_pred)
        bal_acc = balanced_accuracy_score(y_true[real], pred_folded[real])
    else:
        bal_acc = balanced_accuracy_score(y_true, y_pred)

    m: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    # Zero-day recall, the contested headline of reviewer 2 point 2.
    zd = y_true == zero_day_label
    if zd.any():
        hits = int((y_pred[zd] == zero_day_label).sum())
        n = int(zd.sum())
        m["zero_day_recall"] = hits / n
        lo, hi = wilson_interval(hits, n)
        m["zero_day_recall_ci_lo"] = lo
        m["zero_day_recall_ci_hi"] = hi
        m["zero_day_n"] = float(n)

    # Benign false-positive rate: reviewer 2 point 4 asks precisely how many
    # benign flows survive to become analyst workload.
    ben = y_true == benign_label
    if ben.any():
        fp = int((y_pred[ben] != benign_label).sum())
        m["benign_fpr"] = fp / int(ben.sum())
        m["benign_fp_count"] = float(fp)

    # Attack recall: of everything genuinely malicious, how much was caught as
    # something other than benign (the security-relevant question).
    atk = (y_true != benign_label)
    if atk.any():
        m["attack_detection_rate"] = float((y_pred[atk] != benign_label).mean())
        m["attacks_missed"] = float((y_pred[atk] == benign_label).sum())

    if phantom:
        # Recorded so a reader of the results can see the correction happened
        # and how much of the output it touched.
        m["phantom_classes"] = float(len(phantom))
        m["phantom_predictions"] = float(np.isin(y_pred, phantom).sum())

    return RunResult(
        seed=seed, metrics=m, y_true=y_true, y_pred=y_pred, extra=extra or {}
    )


def wilson_interval(successes: int, n: int, alpha: float = config.ALPHA
                    ) -> tuple[float, float]:
    """Wilson score interval - valid for small n, unlike the normal approx."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def aggregate(runs: list[RunResult]) -> pd.DataFrame:
    """Mean and dispersion across seeds. One row per metric."""
    if not runs:
        raise ValueError("no runs to aggregate")
    keys = sorted({k for r in runs for k in r.metrics})
    rows = []
    for k in keys:
        vals = np.array([r.metrics[k] for r in runs if k in r.metrics], dtype=float)
        rows.append({
            "metric": k,
            "mean": vals.mean(),
            "std": vals.std(ddof=1) if len(vals) > 1 else 0.0,
            "min": vals.min(),
            "max": vals.max(),
            "n_seeds": len(vals),
        })
    return pd.DataFrame(rows).set_index("metric")


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar test on paired predictions over the same test rows.

    The right test when comparing two systems on one fixed test set, which is
    our situation: the test split is frozen, so A and B see identical rows.
    """
    a_ok = np.asarray(pred_a) == np.asarray(y_true)
    b_ok = np.asarray(pred_b) == np.asarray(y_true)
    n01 = int((a_ok & ~b_ok).sum())   # A right, B wrong
    n10 = int((~a_ok & b_ok).sum())   # A wrong, B right
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0, "test": "mcnemar_exact"}
    # Two-sided exact binomial test under H0: p = 0.5.
    p = float(stats.binomtest(min(n01, n10), n, 0.5).pvalue)
    return {"n01": n01, "n10": n10, "p_value": p, "test": "mcnemar_exact"}


def wilcoxon_seeds(a: list[float], b: list[float]) -> dict:
    """Wilcoxon signed-rank across paired per-seed scores."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b):
        raise ValueError("paired test needs equal-length inputs")
    if np.allclose(a, b):
        return {"statistic": 0.0, "p_value": 1.0, "test": "wilcoxon", "n": len(a)}
    st, p = stats.wilcoxon(a, b)
    return {"statistic": float(st), "p_value": float(p), "test": "wilcoxon",
            "n": len(a)}


def bootstrap_diff_ci(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric: str = "balanced_accuracy",
    n_boot: int = config.N_BOOTSTRAP,
    alpha: float = config.ALPHA,
    seed: int = 0,
) -> dict:
    """Bootstrap CI on the A-B metric difference, resampling test rows.

    A margin whose CI straddles zero is not a margin.
    """
    fn = {
        "balanced_accuracy": balanced_accuracy_score,
        "accuracy": accuracy_score,
        "f1_macro": lambda t, p: f1_score(t, p, average="macro", zero_division=0),
        "f1_weighted": lambda t, p: f1_score(t, p, average="weighted", zero_division=0),
    }[metric]

    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        diffs[i] = fn(y_true[idx], pred_a[idx]) - fn(y_true[idx], pred_b[idx])
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "metric": metric,
        "observed_diff": float(fn(y_true, pred_a) - fn(y_true, pred_b)),
        "ci_lo": float(lo), "ci_hi": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_boot": n_boot,
    }


def confusion(y_true, y_pred, labels=None) -> pd.DataFrame:
    labels = labels or sorted(set(np.asarray(y_true)) | set(np.asarray(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)
