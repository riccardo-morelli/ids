"""Hyper-parameter optimisation, applied equally to every architecture.

`BRIEF.md` rule 5: "Baselines get the same treatment as us. Same splits, same
preprocessing budget, same tuning effort. A baseline that was tuned less
carefully than our method is not a baseline; it is a strawman, and the
reviewers will find it."

That is the whole reason this module exists. Verkerken tuned with Optuna's TPE
sampler using AUROC for the detector and weighted F1 for the classifier
(their section IV-C). We use the same sampler, the same validation metrics,
and - critically - the same `n_trials` for their architecture and ours. The
trial count is recorded in every result so the claim "equally tuned" is
checkable rather than asserted.

All objectives read the validation partitions only. The test set is not
reachable from this module.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import f1_score, roc_auc_score

from nids import config
from nids.data.prepare import Split
from nids.stages.classifier import Classifier, ClassifierConfig
from nids.stages.detector import Detector, DetectorConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)

#: Shared trial budget. Changing this changes it for every architecture at
#: once, which is the point.
DEFAULT_TRIALS = 40


@dataclass
class TuningResult:
    best_params: dict
    best_value: float
    n_trials: int
    metric: str
    study_name: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def tune_detector(
    split: Split,
    *,
    kind: str = "ocsvm",
    use_pca: bool = False,
    n_trials: int = DEFAULT_TRIALS,
    seed: int = 0,
    n_train: int = 10_000,
) -> TuningResult:
    """Maximise validation AUROC, as Verkerken do for stage 1.

    AUROC is chosen because it is threshold-independent: it ranks the detector
    without committing to tau_B, which is selected separately afterwards.
    """
    feats = split.feature_cols
    Xtr = split.train_benign[feats].values
    Xb = split.val_benign[feats].values
    Xm = split.val_malicious[feats].values
    y = np.r_[np.zeros(len(Xb)), np.ones(len(Xm))]

    def objective(trial: optuna.Trial) -> float:
        if kind == "ocsvm":
            cfg = DetectorConfig(
                kind="ocsvm", use_pca=use_pca, n_train=n_train, seed=seed,
                gamma=trial.suggest_float("gamma", 1e-4, 1.0, log=True),
                nu=trial.suggest_float("nu", 1e-6, 0.5, log=True),
                n_components=(
                    trial.suggest_int("n_components", 10, min(60, len(feats)))
                    if use_pca else 56
                ),
            )
        else:
            cfg = DetectorConfig(
                kind="autoencoder", use_pca=use_pca, n_train=n_train, seed=seed,
                hidden=trial.suggest_int("hidden", 4, 64),
                max_iter=trial.suggest_int("max_iter", 50, 300),
                n_components=(
                    trial.suggest_int("n_components", 10, min(60, len(feats)))
                    if use_pca else 56
                ),
            )
        try:
            det = Detector(cfg).fit(Xtr)
            s = np.r_[det.score(Xb), det.score(Xm)]
            if not np.isfinite(s).all():
                return 0.0
            return float(roc_auc_score(y, s))
        except Exception:
            # A failed trial is a bad trial, not a crashed study.
            return 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"detector-{kind}-pca{int(use_pca)}",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return TuningResult(study.best_params, study.best_value, n_trials,
                        "val_auroc", study.study_name)


def tune_classifier(
    split: Split,
    *,
    kind: str = "rf",
    n_trials: int = DEFAULT_TRIALS,
    seed: int = 0,
) -> TuningResult:
    """Maximise validation weighted F1 over known attack classes.

    Verkerken's stage-2 validation set mixes benign and malicious so the model
    is rewarded for assigning *low* probability to benign rows, which is what
    makes the Unknown reject option work. We mirror that here.
    """
    feats = split.feature_cols
    Xtr = split.train_malicious[feats].values
    ytr = split.train_malicious["Attack Type"].values

    Xb = split.val_benign[feats].values
    Xm = split.val_malicious[feats].values
    ym = split.val_malicious["Attack Type"].values
    # Balance the mixed validation set 50/50, as both papers do.
    rng = np.random.RandomState(seed)
    take = min(len(Xb), len(Xm))
    Xb = Xb[rng.choice(len(Xb), take, replace=False)]

    X_val = np.vstack([Xm, Xb])
    y_val = np.r_[ym, np.full(take, "Benign", dtype=object)]

    def objective(trial: optuna.Trial) -> float:
        if kind == "rf":
            cfg = ClassifierConfig(
                kind="rf", seed=seed,
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
                max_features=trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", 0.3, 0.5]),
                max_samples=trial.suggest_float("max_samples", 0.5, 1.0),
            )
        else:
            cfg = ClassifierConfig(
                kind="nn", scale=True, seed=seed,
                hidden=trial.suggest_int("hidden", 8, 128),
                alpha=trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                max_iter=trial.suggest_int("max_iter", 50, 300),
            )
        try:
            clf = Classifier(cfg).fit(Xtr, ytr)
            pred, _ = clf.predict_with_certainty(X_val)
            return float(f1_score(y_val, pred, average="weighted", zero_division=0))
        except Exception:
            return 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"classifier-{kind}",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return TuningResult(study.best_params, study.best_value, n_trials,
                        "val_f1_weighted", study.study_name)


def save_tuning(name: str, results: dict[str, TuningResult]) -> Path:
    out = config.RESULTS / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.json"
    p.write_text(json.dumps(
        {k: asdict(v) for k, v in results.items()}, indent=2, default=str))
    return p
