"""Shared helpers for the revision scripts.

Everything here exists so that each numbered script can be read on its own
without repeating boilerplate, and so that the environment/hardware provenance
demanded by R2.5 is recorded identically by every script.

Not a script: import only.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

REVISION = Path(__file__).resolve().parent.parent
RESULTS = REVISION / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def environment() -> dict:
    """Everything a reader needs to know whether they can reproduce a number.

    R2.5: "These omissions make it impossible to reproduce the reported
    results independently."
    """
    import sklearn
    import scipy

    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, cwd=REVISION.parent).stdout.strip()
    except Exception:
        git = "unknown"

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "git_commit": git,
    }


def write(name: str, df: pd.DataFrame, *, meta: dict | None = None) -> Path:
    """Write a result as CSV plus a sidecar meta.json.

    The brief requires results on disk as CSV *and* rendered, so numbers can be
    checked without re-running.
    """
    path = RESULTS / f"{name}.csv"
    df.to_csv(path, index=False)
    payload = {"environment": environment(), "rows": len(df)}
    payload.update(meta or {})
    (RESULTS / f"{name}.meta.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n-> {path}")
    return path


def load_clean_any(name: str, *, drop_duplicates: bool = True):
    """Load a cleaned dataset, tolerating a stale cache key.

    `drop_duplicates=False` keeps rows that are identical across all features.
    That matters more than it sounds: Port Scan flows are near-degenerate by
    nature (a handful of packets, no payload, one port apart), so exact-feature
    deduplication collapses 158,804 of them to 1,956 - 1.2% survive, against
    55-100% for every other class. See 25_dedup_effect.py.

    `nids.data.cache.load_clean` keys the parquet on a hash of the cleaning
    options. That hash formula changed after the CSE-CIC-IDS2018 cache was
    built, so the lookup misses and the loader falls back to re-parsing 6.5 GB
    of raw CSV - which exhausts memory on this machine.

    The cached frame is still valid: its sidecar meta records a full ten-file
    build with 63 features. So we accept any `<name>-clean-*.parquet` that
    carries a meta, and fall back to the normal loader when none exists.
    Recorded rather than silently worked around, because "which cache did this
    number come from" is exactly the question a reproducibility reviewer asks.
    """
    import json

    from nids import config
    from nids.data import cache, schema

    if not drop_duplicates:
        # No cache for this variant: the cache key predates the option.
        return cache.load_clean(name, drop_duplicates=False)

    hits = sorted(config.DATA_INTERIM.glob(f"{name}-clean-*.parquet"))
    for path in hits:
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        print(f"[cache] using {path.name} "
              f"({len(meta['feature_cols'])} features)")
        return schema.Dataset(
            name=name, frame=pd.read_parquet(path),
            feature_cols=meta["feature_cols"],
            zero_day_labels=tuple(meta["zero_day_labels"]),
            provenance=meta["provenance"])
    return cache.load_clean(name)


@contextmanager
def timed(label: str):
    """Record wall-clock runtime per script, as the brief requires."""
    t0 = time.perf_counter()
    print(f"[{label}] start", flush=True)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[{label}] done in {dt/60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mean_std(values) -> str:
    """Format as the tables report it: mean +- std (ddof=1)."""
    a = np.asarray(list(values), dtype=float)
    if a.size < 2:
        return f"{a.mean():.4f}"
    return f"{a.mean():.4f}+-{a.std(ddof=1):.4f}"


def wilson(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval.

    R2.2 asks for confidence intervals on zero-day recall at n=47. Wilson is
    the right choice there: the normal approximation is invalid at small n and
    at proportions near 0 or 1, both of which occur in our folds.
    """
    from scipy.stats import norm
    if n == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(1 - alpha / 2)
    p = successes / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# The adopted imbalance strategy, in one place
# ---------------------------------------------------------------------------

def smote_train_pool(ds, *, feature_cols, zero_day=None, seed: int = 0,
                     exclude_row_ids=None):
    """The malicious training pool, SMOTE-resampled.

    SMOTE is the imbalance strategy adopted for the revision (see
    revision/TUNING.md). It must change ONLY what the classifier is trained on.
    Building the evaluation split with imbalance="none" would also change the
    TEST partition (1,802,102 rows against the protocol's 44,040) and silently
    make configurations incomparable. That error was made once in this project
    and is recorded in .phd-log/DECISIONS.md N6.

    This therefore draws the pool from a SEPARATE split object. The caller
    keeps its own evaluation split untouched and only swaps the classifier.

    `feature_cols` is passed in rather than inferred, so the caller and the
    pool are guaranteed to agree on column order.

    Returns (X_resampled, y_resampled, n_before, n_after).
    """
    import pandas as pd
    from imblearn.over_sampling import SMOTE

    from nids import config as _cfg
    from nids.data import prepare as _prep

    kw = {"zero_day": zero_day} if zero_day is not None else {}
    pool = _prep.make_split(ds, seed=_cfg.SPLIT_SEED,
                            imbalance="none", **kw).train_malicious

    # `make_split` draws from one stateful RNG, so an imbalance="none" split
    # and the caller's evaluation split place the SAME row differently: 1,088
    # rows of the evaluation test partition and 761 validation rows reappear in
    # this pool. The withheld class is never among them (verified: 0 of 1,437),
    # so zero-day recall is unaffected - but per-class known metrics computed
    # on the same fold would be. Excluding by row_id removes the ambiguity
    # instead of arguing about which metrics survive it.
    if exclude_row_ids is not None and "row_id" in pool.columns:
        before = len(pool)
        pool = pool[~pool["row_id"].isin(exclude_row_ids)]
        if before != len(pool):
            print(f"[smote] excluded {before - len(pool):,} rows that appear "
                  f"in the caller's evaluation partitions", flush=True)

    X = pool[list(feature_cols)].values
    y = pool["Attack Type"].values
    n_before = len(y)
    if zero_day is not None:
        assert not pool["Label"].isin(zero_day).any(), (
            "withheld class leaked into the SMOTE training pool")
    k = max(1, min(5, int(pd.Series(y).value_counts().min()) - 1))
    Xr, yr = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y)
    return Xr, yr, n_before, len(yr)


def apply_smote_classifier(fitted, ds, *, feature_cols, zero_day=None,
                           seed: int = 0, n_estimators: int = 200,
                           exclude_row_ids=None):
    """Refit `fitted.classifier` on a SMOTE-resampled pool, in place.

    The evaluation split the caller holds is never touched.
    """
    import gc

    from nids.stages.classifier import Classifier, ClassifierConfig

    Xr, yr, n_before, n_after = smote_train_pool(
        ds, feature_cols=feature_cols, zero_day=zero_day, seed=seed,
        exclude_row_ids=exclude_row_ids)
    fitted.classifier = Classifier(
        ClassifierConfig(kind="rf", n_estimators=n_estimators,
                         seed=seed)).fit(Xr, yr)
    del Xr, yr
    gc.collect()
    return n_before, n_after


# ---------------------------------------------------------------------------
# The competitor's published configuration
# ---------------------------------------------------------------------------
#
# Verkerken et al. released their code, fitted models and test data at
#     https://gitlab.ilabt.imec.be/mverkerk/multi-stage-hierarchical-ids
# (commit 43c1f3b, non-commercial research and education licence). Running
# their models on their test set reproduces their published Table V: balanced
# accuracy 0.8954 / 0.9544 / 0.9342 against a published 0.8954 / 0.9608 /
# 0.9342, and zero-day 28 / 45 / 41 of 47.
#
# Their paper describes the architecture but not the tuned constants, which
# were found with Optuna and survive only inside the pickles. The values below
# are read out of models/stage1_ocsvm.p and models/stage2_rf.p, and out of
# code.ipynb cell 15 for tau_M. Our first re-implementation had to guess them,
# and every guess made their system weaker than it is.
#
# These are used to configure OUR re-implementation of THEIR architecture. No
# code or fitted model of theirs is executed in our results; only the numbers
# below cross over.

#: OC-SVM, from their stage-1 pipeline.
THEIR_GAMMA = 0.0632653906314333
THEIR_NU = 0.0002316646233151

#: PCA sits between the scaler and the OC-SVM in their stage 1.
THEIR_PCA_COMPONENTS = 56

#: Random forest, from their stage-2 pipeline.
THEIR_N_ESTIMATORS = 97
THEIR_MAX_FEATURES = 0.1751204590963604

#: tau_M is a literal in their notebook and is never tuned. Ours selects it on
#: validation, where it saturates at 1.000000 and sends every flow to Unknown.
THEIR_TAU_M = 0.98

#: Their three Table V rows, as (tau_B F-beta, tau_U benign quantile).
THEIR_ARMS = {"max-fscore": ("F5", "0.995"),
              "max-bacc": ("F9", "0.95"),
              "balanced": ("F5", "0.99")}

#: What their paper prints for balanced accuracy, quoted not measured.
THEIR_PUBLISHED_BACC = {"max-fscore": 0.8954, "max-bacc": 0.9608,
                        "balanced": 0.9342}

COMPETITOR_REPO = "https://gitlab.ilabt.imec.be/mverkerk/multi-stage-hierarchical-ids"
COMPETITOR_COMMIT = "43c1f3b"


def competitor_config(arm: str = "balanced", *, faithful: bool = True):
    """Our re-implementation of their architecture, at their own settings.

    `faithful=False` reproduces the configuration used in the originally
    submitted manuscript, which predates the discovery of their repository.
    """
    from nids.experiment import ExperimentConfig
    from nids.stages.classifier import ClassifierConfig
    from nids.stages.detector import DetectorConfig

    beta, quant = THEIR_ARMS[arm]
    if faithful:
        det = DetectorConfig(kind="ocsvm", use_pca=True,
                             n_components=THEIR_PCA_COMPONENTS,
                             n_train=10_000,
                             gamma=THEIR_GAMMA, nu=THEIR_NU)
        clf = ClassifierConfig(kind="rf", n_estimators=THEIR_N_ESTIMATORS,
                               max_features=THEIR_MAX_FEATURES)
        tau_m_fixed = THEIR_TAU_M
    else:
        det = DetectorConfig(kind="ocsvm", n_train=10_000,
                             gamma=2.93e-4, nu=1.3e-6)
        clf = ClassifierConfig(kind="rf", n_estimators=200)
        tau_m_fixed = None

    return ExperimentConfig(
        name=f"verkerken-{arm}", architecture="verkerken",
        detector=det, classifier=clf,
        tau_b_beta=beta, tau_u_quantile=quant, tau_m_objective="f1",
        tau_m_fixed=tau_m_fixed)
