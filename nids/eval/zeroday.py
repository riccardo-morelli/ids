"""Leave-one-class-out zero-day evaluation.

Reviewer 2, point 2, verbatim: "the choice to designate these two specific
attack types as 'unknown' is not rigorously justified. A more robust
evaluation protocol, such as a leave-one-class-out experiment, should be
adopted to validate zero-day detection capability in a more convincing manner.
Confidence intervals on the zero-day recall must also be reported."

Both papers evaluate zero-day detection on 47 Infiltration + Heartbleed
instances. At n=47, moving four samples swings the reported recall by 8.5
points, so the metric cannot distinguish a real capability from noise - and
the class choice is unfalsifiable because it was made once, by hand.

Leave-one-class-out fixes both problems. For each known attack class in turn:
withhold it entirely from training and validation, treat it as the unknown,
and measure whether the system flags it as Zero Day. The result is a
*distribution* of zero-day recalls across held-out classes, which says
something the single 47-sample number cannot: whether the architecture detects
novelty in general, or happened to detect Infiltration.

Each fold needs its own split (the withheld class must be absent from
training), so this is expensive. It runs on validation and is therefore
unbudgeted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nids import config
from nids.data.prepare import make_split
from nids.data.schema import Dataset
from nids.eval.metrics import wilson_interval
from nids.stages.pipeline import ZERO_DAY, build


@dataclass
class FoldResult:
    held_out: str
    n_held_out: int
    recall: float
    ci_lo: float
    ci_hi: float
    #: How the held-out class was labelled when it was NOT called Zero Day.
    misrouted_as: dict = field(default_factory=dict)


def leave_one_class_out(
    ds: Dataset,
    cfg,  # ExperimentConfig
    *,
    classes: tuple[str, ...] | None = None,
    seed: int = 0,
    split_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, list[FoldResult]]:
    """Withhold each attack class in turn and measure zero-day recall on it.

    Evaluated on the validation partition of each fold's split, so no test
    budget is consumed. Returns (summary table, per-fold detail).
    """
    from nids.experiment import fit_and_select

    split_kwargs = dict(split_kwargs or {})
    fine = ds.frame.loc[ds.frame["Attack Type"] != "Benign", "Label"]
    coarse = ds.frame.loc[ds.frame["Attack Type"] != "Benign", "Attack Type"]
    # Map coarse class -> its fine labels, so withholding "DoS" withholds every
    # DoS variant rather than leaving Hulk in while holding out GoldenEye.
    members: dict[str, list[str]] = {}
    for c, f in zip(coarse, fine):
        members.setdefault(c, [])
        if f not in members[c]:
            members[c].append(f)

    targets = classes or tuple(sorted(members))
    folds: list[FoldResult] = []

    for cls in targets:
        if cls not in members:
            continue
        held_fine = tuple(members[cls])
        split = make_split(ds, seed=config.SPLIT_SEED,
                           zero_day=held_fine, **split_kwargs)
        # The withheld class must not survive anywhere in training.
        for part in (split.train_malicious, split.val_malicious):
            assert not part["Label"].isin(held_fine).any(), (
                f"{cls} leaked into training/validation - the fold is void")

        fitted = fit_and_select(split, cfg, seed)
        feats = split.feature_cols
        pipe = build(cfg.architecture, detector_cfg=cfg.detector,
                     classifier_cfg=cfg.classifier,
                     thresholds=fitted.thresholds,
                     detector=fitted.detector, classifier=fitted.classifier)

        # Score the held-out class where it actually sits: the test partition
        # of THIS fold's split holds it, but that partition is a different
        # object from the frozen protocol test set (different zero_day
        # designation), and it is used here only to measure novelty detection.
        held = split.test[split.test["Label"].isin(held_fine)]
        if held.empty:
            continue
        pred = pipe.predict(held[feats].values)
        hits = int((pred == ZERO_DAY).sum())
        n = len(pred)
        lo, hi = wilson_interval(hits, n)
        vals, counts = np.unique(pred[pred != ZERO_DAY], return_counts=True)
        folds.append(FoldResult(
            held_out=cls, n_held_out=n, recall=hits / n, ci_lo=lo, ci_hi=hi,
            misrouted_as=dict(zip(vals.tolist(), counts.tolist())),
        ))

    table = pd.DataFrame([{
        "held_out_class": f.held_out,
        "n": f.n_held_out,
        "zero_day_recall": round(f.recall, 4),
        "ci_lo": round(f.ci_lo, 4),
        "ci_hi": round(f.ci_hi, 4),
        "top_misroute": (max(f.misrouted_as.items(), key=lambda kv: kv[1])[0]
                         if f.misrouted_as else "-"),
    } for f in folds])

    if not table.empty:
        table.loc[len(table)] = {
            "held_out_class": "MEAN",
            "n": int(table["n"].sum()),
            "zero_day_recall": round(table["zero_day_recall"].mean(), 4),
            "ci_lo": round(table["zero_day_recall"].std(ddof=1), 4)
                     if len(table) > 1 else 0.0,
            "ci_hi": float("nan"),
            "top_misroute": "(ci_lo column holds std across folds)",
        }
    return table, folds
