"""Ablations answering specific reviewer objections.

Each function here exists because a reviewer asked a question the submitted
manuscript could not answer with a number.

* `stagewise_false_positives` - reviewer 2 point 4. The manuscript admits
  stage 1a flags ~14,366 of 41,857 benign samples (34% FPR) and argues stage 2
  "partially compensates" via tau_U, but never quantifies how many false
  positives survive. This computes the funnel explicitly.

* `imbalance_comparison` - reviewer 2 point 6. Downsampling discards >99% of
  DoS (321,759 -> 1,437) with no comparison against alternatives. This runs
  the same architecture under each strategy on the same split seed.

* `stage_contribution` - the ablation the definition of done requires:
  isolating what each of the three stages contributes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nids import config
from nids.data.prepare import make_split
from nids.data.schema import Dataset
from nids.stages.classifier import UNKNOWN
from nids.stages.pipeline import BENIGN, ZERO_DAY, build


def stagewise_false_positives(fitted, split, partition: str = "validation"
                              ) -> pd.DataFrame:
    """Trace benign traffic through every stage. Reviewer 2 point 4.

    Reports, for benign rows only:
      - how many stage 1a flags as suspicious (the 34% the reviewer objects to)
      - how many of those stage 1b rescues by naming a known class
      - how many the extension stage (tau_U) recovers as benign
      - how many survive as alarms, and what they are called

    The final row is the number that actually reaches a SOC analyst, which is
    what the reviewer asked for and the manuscript never gave.
    """
    feats = split.feature_cols
    if partition == "validation":
        frame = split.val_benign
    else:
        frame = split.test[split.test["Attack Type"] == BENIGN]

    X = frame[feats].values
    scores = fitted.detector.score(X)
    labels = fitted.classifier.predict_with_unknown(X, fitted.thresholds.tau_m)
    tau_b, tau_u = fitted.thresholds.tau_b, fitted.thresholds.tau_u

    n = len(X)
    flagged = scores > tau_b
    named = labels != UNKNOWN

    pipe = build(fitted.cfg.architecture, detector_cfg=fitted.cfg.detector,
                 classifier_cfg=fitted.cfg.classifier,
                 thresholds=fitted.thresholds,
                 detector=fitted.detector, classifier=fitted.classifier)
    final = pipe.predict(X)
    surviving = final != BENIGN

    rows = [
        {"stage": "0. benign rows evaluated", "count": n, "pct_of_benign": 100.0},
        {"stage": "1a. flagged suspicious (score > tau_B)",
         "count": int(flagged.sum()), "pct_of_benign": 100 * flagged.mean()},
        {"stage": "1b. named a known attack class (certainty >= tau_M)",
         "count": int(named.sum()), "pct_of_benign": 100 * named.mean()},
        {"stage": "1a flagged AND 1b could not name (-> extension stage)",
         "count": int((flagged & ~named).sum()),
         "pct_of_benign": 100 * (flagged & ~named).mean()},
        {"stage": "  of those, recovered as benign (score <= tau_U)",
         "count": int((flagged & ~named & (scores <= tau_u)).sum()),
         "pct_of_benign": 100 * (flagged & ~named & (scores <= tau_u)).mean()},
        {"stage": "  of those, raised as Zero Day (score > tau_U)",
         "count": int((flagged & ~named & (scores > tau_u)).sum()),
         "pct_of_benign": 100 * (flagged & ~named & (scores > tau_u)).mean()},
        {"stage": "FINAL. false alarms reaching an analyst",
         "count": int(surviving.sum()), "pct_of_benign": 100 * surviving.mean()},
    ]
    breakdown = pd.Series(final[surviving]).value_counts().to_dict()
    for lbl, c in sorted(breakdown.items(), key=lambda kv: -kv[1]):
        rows.append({"stage": f"    alarm labelled '{lbl}'", "count": int(c),
                     "pct_of_benign": 100 * c / n})

    df = pd.DataFrame(rows)
    df["pct_of_benign"] = df["pct_of_benign"].round(2)
    return df


def imbalance_comparison(
    ds: Dataset, cfg, *, strategies=("downsample", "weighted", "none"),
    seeds: tuple = (0, 1, 2), per_class: int | None = None,
) -> pd.DataFrame:
    """Same architecture, same split seed, different imbalance handling.

    Reviewer 2 point 6 asks why 99% of the DoS class was discarded and what it
    cost. Evaluated on validation, so unbudgeted.
    """
    import dataclasses

    from nids.experiment import evaluate_validation, fit_and_select
    from nids.eval.metrics import aggregate

    rows = []
    for strat in strategies:
        split = make_split(ds, seed=config.SPLIT_SEED, imbalance=strat,
                           per_class=per_class)
        # 'weighted' means cost-sensitive learning on the FULL imbalanced
        # training set - it is only a distinct arm if the class weights are
        # actually applied. Without this the arm is identical to 'none' and the
        # ablation answers reviewer 2 point 6 with a tautology.
        arm = cfg
        if strat == "weighted":
            arm = dataclasses.replace(
                cfg,
                classifier=dataclasses.replace(cfg.classifier,
                                               class_weight="balanced"),
            )
        runs = []
        for seed in seeds:
            fitted = fit_and_select(split, arm, seed)
            runs.append(evaluate_validation(fitted, split))
        agg = aggregate(runs)
        row = {"strategy": strat,
               "n_train_malicious": len(split.train_malicious)}
        for m in ("balanced_accuracy", "f1_macro", "f1_weighted",
                  "zero_day_recall", "benign_fpr", "attack_detection_rate"):
            if m in agg.index:
                row[m] = f"{agg.loc[m, 'mean']:.4f}±{agg.loc[m, 'std']:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def stage_contribution(fitted, split, partition: str = "validation"
                       ) -> pd.DataFrame:
    """What each stage contributes, by disabling the others.

    Required by the definition of done: "An ablation isolating the contribution
    of each of the three stages."

    Variants:
      detector only     - threshold the anomaly score, binary benign/attack
      classifier only   - trust the classifier, Unknown -> benign
      no extension      - drop tau_U (unnamed suspicious rows stay alarms)
      full              - the assembled system

    **Read the validation numbers with care.** The zero-day classes are
    withheld from training *and* validation by construction, so no validation
    row carries the 'Zero Day' label. Any variant that emits 'Zero Day' on
    validation is therefore scored as wrong every time, and balanced accuracy
    - which averages per-class recall over the union of true and predicted
    labels - charges it for a class that cannot be right. That systematically
    flatters "classifier only" (which never emits Zero Day) against the full
    system.

    `zero_day_capable` marks which rows are affected, and the
    `balanced_accuracy_known` column recomputes the metric over rows whose true
    label is a known class, ignoring Zero Day predictions, so the variants are
    comparable on the part of the problem validation can actually measure. To
    compare zero-day behaviour, use the test partition or
    `nids.eval.zeroday.leave_one_class_out`.
    """
    from nids.eval.metrics import evaluate

    feats = split.feature_cols
    if partition == "validation":
        frame = pd.concat([split.val_benign, split.val_malicious],
                          ignore_index=True)
    else:
        frame = split.test
    X = frame[feats].values
    y = frame["Attack Type"].values
    has_zero_day = bool((y == ZERO_DAY).any())

    scores = fitted.detector.score(X)
    labels = fitted.classifier.predict_with_unknown(X, fitted.thresholds.tau_m)
    tau_b, tau_u = fitted.thresholds.tau_b, fitted.thresholds.tau_u

    variants: dict[str, np.ndarray] = {}

    # Detector alone: it cannot name attacks, so every alarm is "Zero Day".
    det_only = np.where(scores > tau_b, ZERO_DAY, BENIGN)
    variants["detector only (1a)"] = det_only

    # Classifier alone: Unknown becomes benign, since without a detector there
    # is no anomaly score to arbitrate with.
    clf_only = np.where(labels != UNKNOWN, labels, BENIGN)
    variants["classifier only (1b)"] = clf_only

    # Full system minus the extension stage: unnamed suspicious rows stay
    # alarms rather than being recovered by tau_U.
    no_ext = np.where(labels != UNKNOWN, labels,
                      np.where(scores > tau_b, ZERO_DAY, BENIGN))
    variants["1a + 1b, no extension (no tau_U)"] = no_ext

    pipe = build(fitted.cfg.architecture, detector_cfg=fitted.cfg.detector,
                 classifier_cfg=fitted.cfg.classifier,
                 thresholds=fitted.thresholds,
                 detector=fitted.detector, classifier=fitted.classifier)
    variants["full system"] = pipe.predict(X)

    from sklearn.metrics import balanced_accuracy_score

    rows = []
    for name, pred in variants.items():
        r = evaluate(y, pred, seed=fitted.seed)
        row = {
            "variant": name,
            **{k: round(v, 4) for k, v in r.metrics.items()
               if k in ("balanced_accuracy", "f1_macro", "f1_weighted",
                        "zero_day_recall", "benign_fpr",
                        "attack_detection_rate")},
        }
        # Restrict to rows whose TRUE label is a known class, and fold any
        # 'Zero Day' prediction into a single wrong-but-not-phantom bucket, so
        # variants are not charged for a class the partition cannot contain.
        if not has_zero_day:
            mask = y != ZERO_DAY
            pred_known = np.where(pred[mask] == ZERO_DAY, "__alarm__", pred[mask])
            row["balanced_accuracy_known"] = round(
                balanced_accuracy_score(y[mask], pred_known), 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["partition_has_zero_day"] = has_zero_day
    return df
