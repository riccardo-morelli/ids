"""The experiment runner: fit, select on validation, evaluate across seeds.

This is the only path by which a number enters `results/`. `BRIEF.md`: "Any
result produced outside that harness does not count."

The ordering inside `run_once` encodes the protocol:

    fit on train  ->  select thresholds on validation  ->  (optionally) evaluate on test

Thresholds are always selected from validation scores, never from test, and
`evaluate_test` refuses to run without a ledger entry.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from nids import config
from nids.data.prepare import Split
from nids.eval import metrics as M
from nids.eval import testguard
from nids.stages.classifier import Classifier, ClassifierConfig
from nids.stages.detector import (
    Detector,
    DetectorConfig,
    select_tau_u,
    select_threshold_fbeta,
)
from nids.stages.pipeline import Thresholds, build


@dataclass
class ExperimentConfig:
    """Everything that defines one comparable system."""

    name: str
    architecture: str                       # 'verkerken' | 'parallel' | ...
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    #: Which F-beta optimum to use for tau_B. Verkerken's three Table V configs
    #: differ mainly in this and in the tau_U quantile.
    tau_b_beta: str = "F9"
    tau_u_quantile: str = "0.95"
    tau_m_objective: str = "f1"
    #: Fix tau_M to a literal instead of selecting it on validation. Their
    #: published code (gitlab.ilabt.imec.be/mverkerk/multi-stage-hierarchical-
    #: ids, code.ipynb cell 15) hard-codes tau_M = 0.98 and never tunes it, so
    #: a faithful re-implementation must do the same. None keeps the selected
    #: value, which is what our own system uses.
    tau_m_fixed: float | None = None
    notes: str = ""


@dataclass
class FittedSystem:
    cfg: ExperimentConfig
    detector: Detector
    classifier: Classifier
    thresholds: Thresholds
    seed: int
    fit_seconds: float
    selection: dict


def fit_and_select(split: Split, cfg: ExperimentConfig, seed: int) -> FittedSystem:
    """Train both stage-1 models and choose all three thresholds.

    Every threshold comes from the validation partitions. The test partition is
    not read anywhere in this function.
    """
    feats = split.feature_cols
    d_cfg = DetectorConfig(**{**asdict(cfg.detector), "seed": seed})
    c_cfg = ClassifierConfig(**{**asdict(cfg.classifier), "seed": seed})

    t0 = time.perf_counter()
    detector = Detector(d_cfg).fit(split.train_benign[feats].values)
    classifier = Classifier(c_cfg).fit(
        split.train_malicious[feats].values,
        split.train_malicious["Attack Type"].values,
    )
    fit_seconds = time.perf_counter() - t0

    # --- threshold selection: VALIDATION ONLY ---------------------------
    s_ben = detector.score(split.val_benign[feats].values)
    s_mal = detector.score(split.val_malicious[feats].values)

    fbeta = select_threshold_fbeta(s_ben, s_mal)
    tau_b = fbeta[cfg.tau_b_beta]["threshold"]

    tau_u_candidates = select_tau_u(s_ben)
    tau_u = tau_u_candidates[cfg.tau_u_quantile]

    _, c_ben = classifier.predict_with_certainty(split.val_benign[feats].values)
    _, c_mal = classifier.predict_with_certainty(split.val_malicious[feats].values)
    tau_m_sel = select_tau_m(c_ben, c_mal, objective=cfg.tau_m_objective)
    tau_m = tau_m_sel["threshold"]
    if cfg.tau_m_fixed is not None:
        tau_m = float(cfg.tau_m_fixed)

    thresholds = Thresholds(
        tau_b=tau_b, tau_m=tau_m, tau_u=tau_u,
        provenance={
            "tau_b": {"source": "val F-beta sweep", "beta": cfg.tau_b_beta,
                      "all": {k: v["threshold"] for k, v in fbeta.items()}},
            "tau_m": ({"source": "fixed (their published code)",
                       "value": float(cfg.tau_m_fixed),
                       "selected_would_have_been": tau_m_sel["threshold"]}
                      if cfg.tau_m_fixed is not None else
                      {"source": "val certainty F1", "score": tau_m_sel["score"]}),
            "tau_u": {"source": "benign val quantile",
                      "quantile": cfg.tau_u_quantile, "all": tau_u_candidates},
        },
    )
    return FittedSystem(cfg, detector, classifier, thresholds, seed, fit_seconds,
                        thresholds.provenance)


def select_tau_m(c_ben, c_mal, objective="f1"):
    from nids.stages.classifier import select_tau_m as _f
    return _f(c_ben, c_mal, objective=objective)


def evaluate_validation(sys: FittedSystem, split: Split) -> M.RunResult:
    """Score the assembled pipeline on VALIDATION. Unlimited, free to repeat.

    This is the loop the brief says should "run hot": every architectural idea
    is judged here, and only here, until a test spend is authorised.
    """
    feats = split.feature_cols
    val = pd.concat([split.val_benign, split.val_malicious], ignore_index=True)
    pipe = build(sys.cfg.architecture, detector_cfg=sys.cfg.detector,
                 classifier_cfg=sys.cfg.classifier, thresholds=sys.thresholds,
                 detector=sys.detector, classifier=sys.classifier)
    y_pred, timing = pipe.timed_predict(val[feats].values)
    return M.evaluate(
        val["Attack Type"].values, y_pred, seed=sys.seed,
        extra={"partition": "validation", "timing": asdict(timing),
               "fit_seconds": sys.fit_seconds},
    )


def evaluate_test(
    sys: FittedSystem, split: Split, *,
    cycle: int, phase: str, purpose: str, authorised: bool = False,
) -> M.RunResult:
    """Score on the FROZEN test set. Budgeted, logged, and rare.

    Raises unless the spend is permitted by the SESSION.md policy.
    """
    testguard.spend(cycle=cycle, phase=phase, purpose=purpose,
                    model=sys.cfg.name, authorised=authorised)
    feats = split.feature_cols
    pipe = build(sys.cfg.architecture, detector_cfg=sys.cfg.detector,
                 classifier_cfg=sys.cfg.classifier, thresholds=sys.thresholds,
                 detector=sys.detector, classifier=sys.classifier)
    y_pred, timing = pipe.timed_predict(split.test[feats].values)
    return M.evaluate(
        split.test["Attack Type"].values, y_pred, seed=sys.seed,
        extra={"partition": "test", "timing": asdict(timing),
               "fit_seconds": sys.fit_seconds},
    )


def run_multiseed(
    split: Split, cfg: ExperimentConfig, *,
    seeds: tuple = config.MODEL_SEEDS, on: str = "validation",
    cycle: int = 0, phase: str = "A", purpose: str = "", authorised: bool = False,
) -> tuple[pd.DataFrame, list[M.RunResult]]:
    """Fit and evaluate across seeds. Returns (aggregate table, raw runs).

    Rule 4 of the protocol: any comparative claim carries mean and dispersion
    across seeds. This function is how that becomes automatic.
    """
    runs = []
    for seed in seeds:
        fitted = fit_and_select(split, cfg, seed)
        if on == "validation":
            runs.append(evaluate_validation(fitted, split))
        else:
            runs.append(evaluate_test(
                fitted, split, cycle=cycle, phase=phase,
                purpose=f"{purpose} (seed {seed})", authorised=authorised))
    return M.aggregate(runs), runs


def save(name: str, agg: pd.DataFrame, runs: list[M.RunResult],
         cfg: ExperimentConfig, split: Split, partition: str) -> Path:
    """Persist a result. Subagent findings must land on disk or they did not
    happen (BRIEF.md, Delegation)."""
    out = config.RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out / "aggregate.csv")
    meta = {
        "experiment": asdict(cfg),
        "partition": partition,
        "split_seed": split.seed,
        "split_meta": split.meta,
        "model_seeds": [r.seed for r in runs],
        "per_seed": [{"seed": r.seed, **r.metrics} for r in runs],
        "timings": [r.extra.get("timing") for r in runs],
        "thresholds": [r.extra.get("thresholds") for r in runs],
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    np.savez_compressed(
        out / "predictions.npz",
        **{f"seed{r.seed}_pred": r.y_pred for r in runs},
        **{f"seed{r.seed}_true": r.y_true for r in runs},
    )
    return out
