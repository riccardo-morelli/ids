"""Detector bench: score a (transform, detector) pair by its ability to
separate a class it has never seen.

The objective, agreed with the supervisor, is **mean leave-one-class-out
zero-day separability, reported with its dispersion**. A detector scoring 0.94
on DoS and 0.22 on Brute Force is not a good detector; it memorised the shape
of one class.

Two shortcuts make the search affordable without weakening it:

1. **AUROC of the held-out class against benign, instead of recall at a
   threshold.** Thresholds are chosen later, on validation, by the existing
   selection functions. Measuring the ranking removes threshold calibration as
   a confound while searching the feature space, which is what the diagnosis
   says is actually broken. A class with AUROC 0.5 cannot be rescued by any
   threshold.

2. **The detector never sees the held-out class**, which is what makes this a
   novelty measurement rather than a classification one. Since the detector is
   benign-only by contract, withholding a class costs nothing at fit time — so
   one fit serves every fold, and the whole sweep is one fit per (transform,
   detector, seed) rather than one per fold.

`AUROC < 0.5` is reported honestly rather than flipped. A detector that ranks
an attack class as *more normal than benign traffic* is broken in a way worth
seeing, and hiding it behind `max(a, 1-a)` would flatter every candidate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from nids import config
from nids.data.prepare import Split
from nids.data.transforms import FeatureTransform, TransformConfig
from nids.stages.detector import Detector, DetectorConfig


@dataclass
class BenchResult:
    transform: str
    detector: str
    #: AUROC per held-out class, averaged across seeds.
    per_class: dict[str, float]
    #: Mean over classes of the per-class AUROC — the objective.
    mean_auroc: float
    #: Dispersion across classes. Large std means the detector works for some
    #: attack families and not others, which is the failure the supervisor
    #: asked to fix.
    std_auroc: float
    #: Worst class. A detector is only as good as the attack it misses.
    min_auroc: float
    worst_class: str
    #: Dispersion across seeds of the mean — is the result stable at all?
    seed_std: float
    fit_seconds: float
    extra: dict = field(default_factory=dict)

    def row(self) -> dict:
        return {
            "transform": self.transform,
            "detector": self.detector,
            "mean_auroc": round(self.mean_auroc, 4),
            "std_across_classes": round(self.std_auroc, 4),
            "min_auroc": round(self.min_auroc, 4),
            "worst_class": self.worst_class,
            "seed_std": round(self.seed_std, 4),
            "fit_s": round(self.fit_seconds, 1),
            **{f"auc:{k}": round(v, 3) for k, v in sorted(self.per_class.items())},
        }


def evaluate_detector(
    split: Split,
    transform_cfg: TransformConfig,
    detector_cfg: DetectorConfig,
    *,
    seeds: tuple = (0, 1, 2),
    eval_frame: pd.DataFrame | None = None,
) -> BenchResult:
    """Fit on benign training data; rank every attack class against benign.

    `eval_frame` defaults to the validation partitions. The test partition is
    never read here — the bench is a validation-only instrument.
    """
    import time

    feats = split.feature_cols
    X_fit = split.train_benign[feats].values
    X_ben_eval = split.val_benign[feats].values

    if eval_frame is None:
        eval_frame = split.val_malicious
    classes = sorted(set(eval_frame["Attack Type"]) - {"Benign"})

    per_class_runs: dict[str, list[float]] = {c: [] for c in classes}
    means: list[float] = []
    total_fit = 0.0

    for seed in seeds:
        t0 = time.perf_counter()
        tr = FeatureTransform(transform_cfg).fit(X_fit)
        d_cfg = DetectorConfig(**{**detector_cfg.__dict__, "seed": seed})
        det = Detector(d_cfg).fit(tr.transform(X_fit))
        total_fit += time.perf_counter() - t0

        s_ben = det.score(tr.transform(X_ben_eval))
        if not np.isfinite(s_ben).all():
            # A detector producing non-finite scores is not a candidate.
            return BenchResult(transform_cfg.label(), detector_cfg.kind, {},
                               float("nan"), float("nan"), float("nan"),
                               "non-finite", float("nan"), total_fit,
                               {"error": "non-finite benign scores"})

        seed_vals = []
        for cls in classes:
            grp = eval_frame[eval_frame["Attack Type"] == cls]
            s_cls = det.score(tr.transform(grp[feats].values))
            y = np.r_[np.zeros(len(s_ben)), np.ones(len(s_cls))]
            s = np.r_[s_ben, s_cls]
            if not np.isfinite(s).all():
                auc = float("nan")
            else:
                auc = float(roc_auc_score(y, s))
            per_class_runs[cls].append(auc)
            seed_vals.append(auc)
        means.append(float(np.nanmean(seed_vals)))

    per_class = {c: float(np.nanmean(v)) for c, v in per_class_runs.items()}
    vals = np.array(list(per_class.values()), dtype=float)
    worst_idx = int(np.nanargmin(vals))
    return BenchResult(
        transform=transform_cfg.label(),
        detector=detector_cfg.kind,
        per_class=per_class,
        mean_auroc=float(np.nanmean(vals)),
        std_auroc=float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
        min_auroc=float(vals[worst_idx]),
        worst_class=list(per_class)[worst_idx],
        seed_std=float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
        fit_seconds=total_fit / len(seeds),
    )


def sweep(
    split: Split,
    transforms: list[TransformConfig],
    detectors: list[DetectorConfig],
    *,
    seeds: tuple = (0, 1, 2),
    eval_frame: pd.DataFrame | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Cross every transform with every detector. Validation only."""
    rows = []
    for t in transforms:
        for d in detectors:
            res = evaluate_detector(split, t, d, seeds=seeds,
                                    eval_frame=eval_frame)
            rows.append(res.row())
            if verbose:
                print(f"  {t.label():28s} {d.kind:14s} "
                      f"mean={res.mean_auroc:.4f} "
                      f"std={res.std_auroc:.4f} "
                      f"min={res.min_auroc:.4f} ({res.worst_class}) "
                      f"[{res.fit_seconds:.1f}s]", flush=True)
    df = pd.DataFrame(rows)
    return df.sort_values("mean_auroc", ascending=False).reset_index(drop=True)
