"""Stage-1b tuning: imbalance strategy x RF hyperparameters, 5 seeds.

PRODUCES
    revision/results/tune_classifier.csv       - per-seed rows, every arm
    revision/results/tune_classifier_agg.csv   - mean +- std (ddof=1) per arm
    revision/results/tune_classifier.meta.json - environment + arm definitions

FEEDS
    Responses R2.5 (hyperparameters and the selection procedure must be stated
    precisely enough to reproduce) and R2.6 ("This drastic choice is not
    compared against alternative imbalance-handling strategies such as
    weighted downsampling, SMOTE oversampling, or cost-sensitive learning").

WHY THIS EXISTS
    08_ablations.py established at 3 seeds that SMOTE (0.9340) beats the
    manuscript's aggressive downsampling (0.8884). That leaves the question
    this script answers: once the training pool is no longer discarded, what
    is the best JOINT choice of resampling strategy and classifier
    hyperparameters? R2.6 is not satisfied by "SMOTE beats downsampling" if
    the SMOTE arm was itself untuned.

    Two corrections to the 08 arms are made here.

    1. 08's "weighted" arm never set `class_weight`, so it was identical to
       "none": `make_split` treats imbalance="weighted" and "none" the same
       (both keep the full malicious pool), and the cost-sensitive part lives
       in ClassifierConfig, which 08 left at its default of None. The genuine
       cost-sensitive arms are run here. That is the third alternative the
       reviewer names by name, and it had not actually been tested.

    2. Everything is at 5 seeds (0-4), matching the reference point in
       TUNING.md, so the comparison against bACC 0.9001 +- 0.0235 is
       like-for-like.

ARCHITECTURE IS FROZEN
    parallel; OCSVM detector (n_train=10000, gamma=2.93e-4, nu=1.3e-6);
    tau_b_beta=F9, tau_u_quantile=0.95, tau_m_objective=f1. Only the stage-1b
    training pool and the RF's own hyperparameters vary.

PROTOCOL
    VALIDATION ONLY. `split.test` is never read; every number comes from
    `nids.experiment.evaluate_validation`. Adoption follows TUNING.md: a gain
    counts only if it exceeds seed variance.

MEMORY
    One arm at a time, one split alive at a time, gc.collect() between arms
    and between seeds. Resampled matrices are freed before evaluation.

    That is still not enough to run every arm in ONE process on a 16 GB
    machine. The full-pool split (165k malicious train rows, 166k validation
    rows) plus a SMOTE expansion to ~788k rows plus an unbounded 200-tree
    forest accumulates fragmented heap across arms until the OS kills the
    process outright - no MemoryError is raised, so it cannot be caught and
    retried in-process. The `none`-split arms are therefore run in separate
    processes via --only/--append, each starting from a clean heap, and their
    rows are accumulated into the same CSV. `run_all.ps1`-style chunking is
    left to the caller so the chunk size can be tuned to the machine.

USAGE
    # one chunk per process, accumulating into tune_classifier.csv
    python revision/scripts/12_tune_classifier.py --only downsample-base,cycle9-downsample
    python revision/scripts/12_tune_classifier.py --only smote-k5-base --append --no-agg
    ...
    python revision/scripts/12_tune_classifier.py --only <last arm> --append

RUNTIME  ~50 min for the full arm set x 5 seeds on a 12-core laptop.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RESULTS, load_clean_any, timed, write  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nids import config  # noqa: E402
from nids.data import prepare  # noqa: E402
from nids.experiment import (ExperimentConfig, evaluate_validation,  # noqa: E402
                             fit_and_select)
from nids.stages.classifier import Classifier, ClassifierConfig  # noqa: E402
from nids.stages.detector import DetectorConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default=",".join(map(str, config.MODEL_SEEDS)))
ap.add_argument("--only", default="")
ap.add_argument("--dataset", default="cic-ids2017")
#: Append to an existing tune_classifier.csv instead of starting fresh. The
#: full sweep does not fit in one process (see MEMORY above), so arms are run
#: in separate processes and their rows accumulated.
ap.add_argument("--append", action="store_true")
#: Emit only the per-seed CSV and skip the aggregate, for a partial run whose
#: rows will be aggregated once the last chunk lands.
ap.add_argument("--no-agg", action="store_true")
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(",")]

DETECTOR = DetectorConfig(kind="ocsvm", n_train=10_000, gamma=2.93e-4,
                          nu=1.3e-6)

# The previously tuned classifier, results/cycle9-tuning/clf.json. It was
# selected for zero-day recall under an FPR constraint, not for balanced
# accuracy, so it enters as one arm rather than as the default.
CYCLE9 = dict(n_estimators=197, max_depth=16, max_features="log2",
              max_samples=0.5449565190789558, class_weight="balanced")

# RF settings. RF_BASE is the manuscript's stage-1b classifier.
RF_BASE = dict(n_estimators=200)
RF_DEEP = dict(n_estimators=500, max_features="sqrt")
RF_CAPPED = dict(n_estimators=200, max_depth=16)
RF_LOG2 = dict(n_estimators=200, max_features="log2")
RF_SUB = dict(n_estimators=300, max_samples=0.5, max_features="sqrt")

# resample: None (train on the split's own pool) or (kind, k_neighbors).
# imbalance: what make_split does to the malicious pool.
ARMS: dict[str, dict] = {
    # --- reference: the manuscript's own choice, re-run at 5 seeds -------
    "downsample-base": dict(imbalance="downsample", resample=None, rf=RF_BASE),
    "cycle9-downsample": dict(imbalance="downsample", resample=None, rf=CYCLE9),
    # --- keep the whole pool, no resampling -----------------------------
    "none-base": dict(imbalance="none", resample=None, rf=RF_BASE),
    "cycle9-none": dict(imbalance="none", resample=None, rf=CYCLE9),
    # --- cost-sensitive learning, the arm 08 believed it had run --------
    "weighted-base": dict(imbalance="none", resample=None,
                          rf=dict(RF_BASE, class_weight="balanced")),
    "weighted-subsample": dict(imbalance="none", resample=None,
                               rf=dict(RF_BASE,
                                       class_weight="balanced_subsample")),
    # --- SMOTE family, RF at the manuscript's settings ------------------
    "smote-k5-base": dict(imbalance="none", resample=("smote", 5), rf=RF_BASE),
    "smote-k3-base": dict(imbalance="none", resample=("smote", 3), rf=RF_BASE),
    "smote-k10-base": dict(imbalance="none", resample=("smote", 10),
                           rf=RF_BASE),
    "borderline-smote": dict(imbalance="none", resample=("borderline", 5),
                             rf=RF_BASE),
    "smote-tomek": dict(imbalance="none", resample=("smotetomek", 5),
                        rf=RF_BASE),
    # --- SMOTE x RF hyperparameters -------------------------------------
    "smote-k5-deep": dict(imbalance="none", resample=("smote", 5), rf=RF_DEEP),
    "smote-k5-capped": dict(imbalance="none", resample=("smote", 5),
                            rf=RF_CAPPED),
    "smote-k5-log2": dict(imbalance="none", resample=("smote", 5), rf=RF_LOG2),
    "smote-k5-subsample": dict(imbalance="none", resample=("smote", 5),
                               rf=RF_SUB),
    "smote-k5-cycle9": dict(imbalance="none", resample=("smote", 5),
                            rf=CYCLE9),
}

if args.only:
    keep = [a.strip() for a in args.only.split(",")]
    ARMS = {k: v for k, v in ARMS.items() if k in keep}


def make_resampler(spec, seed: int, counts: pd.Series):
    """Build the resampler, clamping k to what the smallest class supports.

    SMOTE interpolates between a minority point and one of its k nearest
    minority neighbours, so k must be < the smallest class size or the fit
    raises. The clamped value is recorded in the result row rather than
    applied silently.
    """
    kind, k_req = spec
    k = max(1, min(int(k_req), int(counts.min()) - 1))
    if kind == "smote":
        from imblearn.over_sampling import SMOTE
        return SMOTE(random_state=seed, k_neighbors=k), k
    if kind == "borderline":
        from imblearn.over_sampling import BorderlineSMOTE
        return BorderlineSMOTE(random_state=seed, k_neighbors=k), k
    if kind == "smotetomek":
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE
        return (SMOTETomek(random_state=seed,
                           smote=SMOTE(random_state=seed, k_neighbors=k)), k)
    raise ValueError(f"unknown resampler {kind!r}")


def run_arm(name: str, spec: dict, split, seed: int) -> dict:
    """One (arm, seed): fit stage 1a + 1b, select thresholds, evaluate.

    For a resampling arm the classifier is refitted on the resampled pool and
    swapped into the fitted system. tau_B and tau_U depend only on the
    detector, so they survive the swap unchanged; tau_M is chosen from the
    CLASSIFIER's validation certainties, so it is re-selected for the
    resampled model - otherwise the arm would be scored with a threshold that
    belongs to a different classifier.
    """
    feats = split.feature_cols
    rf = dict(spec["rf"])
    c_cfg = ClassifierConfig(kind="rf", seed=seed, **rf)
    cfg = ExperimentConfig(name=name, architecture="parallel",
                           detector=DETECTOR, classifier=c_cfg,
                           tau_b_beta="F9", tau_u_quantile="0.95",
                           tau_m_objective="f1")

    row: dict = {"arm": name, "seed": seed,
                 "imbalance": spec["imbalance"],
                 "resampler": ("none" if spec["resample"] is None
                               else spec["resample"][0]),
                 "k_requested": (spec["resample"][1] if spec["resample"]
                                 else None),
                 "n_train_malicious": len(split.train_malicious)}
    row.update({f"rf_{k}": v for k, v in rf.items()})

    t0 = time.perf_counter()
    if spec["resample"] is None:
        fitted = fit_and_select(split, cfg, seed)
        row["n_train_after_resample"] = len(split.train_malicious)
        row["k_used"] = None
    else:
        Xtr = split.train_malicious[feats].values
        ytr = split.train_malicious["Attack Type"].values
        counts = pd.Series(ytr).value_counts()
        sampler, k_used = make_resampler(spec["resample"], seed, counts)
        Xr, yr = sampler.fit_resample(Xtr, ytr)
        row["n_train_after_resample"] = int(len(yr))
        row["k_used"] = k_used
        del Xtr, ytr, sampler
        gc.collect()

        clf = Classifier(replace(c_cfg, seed=seed)).fit(Xr, yr)
        del Xr, yr
        gc.collect()

        fitted = fit_and_select(split, cfg, seed)
        fitted.classifier = clf

        from nids.stages.classifier import select_tau_m
        _, c_ben = clf.predict_with_certainty(split.val_benign[feats].values)
        _, c_mal = clf.predict_with_certainty(
            split.val_malicious[feats].values)
        sel = select_tau_m(c_ben, c_mal, objective="f1")
        fitted.thresholds.tau_m = sel["threshold"]
        del c_ben, c_mal, sel
        gc.collect()

    row["fit_seconds"] = time.perf_counter() - t0

    res = evaluate_validation(fitted, split)
    row.update({k: v for k, v in res.metrics.items()
                if isinstance(v, (int, float))})
    row.update({"tau_b": fitted.thresholds.tau_b,
                "tau_m": fitted.thresholds.tau_m,
                "tau_u": fitted.thresholds.tau_u})
    del fitted, res
    gc.collect()
    return row


with timed("12_tune_classifier"):
    ds = load_clean_any(args.dataset)

    rows: list[dict] = []
    if args.append and (RESULTS / "tune_classifier.csv").exists():
        prior = pd.read_csv(RESULTS / "tune_classifier.csv")
        # Replace only the (arm, seed) cells this process is about to
        # recompute. Keying on the arm alone would discard sibling seeds of
        # the same arm, which matters because an arm too heavy to finish in
        # one process is chunked one seed at a time.
        redo = {(a, s) for a in ARMS for s in SEEDS}
        keep = [not (r.arm, r.seed) in redo for r in prior.itertuples()]
        prior = prior[keep]
        rows = prior.to_dict("records")
        print(f"[append] carrying {len(rows)} rows from earlier chunks",
              flush=True)

    # Group arms by the split they need, so each split is built once and only
    # one is ever alive. This machine OOMs on two.
    for imbalance in ("downsample", "none"):
        arms = {k: v for k, v in ARMS.items() if v["imbalance"] == imbalance}
        if not arms:
            continue
        split = prepare.make_split(ds, seed=config.SPLIT_SEED,
                                   imbalance=imbalance)
        print(f"\n[split imbalance={imbalance}] "
              f"train_mal={len(split.train_malicious):,} "
              f"val_ben={len(split.val_benign):,} "
              f"val_mal={len(split.val_malicious):,}", flush=True)
        for name, spec in arms.items():
            for seed in SEEDS:
                try:
                    row = run_arm(name, spec, split, seed)
                    rows.append(row)
                    print(f"  {name:20s} s{seed} "
                          f"bACC={row.get('balanced_accuracy', float('nan')):.4f} "
                          f"F1w={row.get('f1_weighted', float('nan')):.4f} "
                          f"FPR={row.get('benign_fpr', float('nan')):.4f} "
                          f"n={row['n_train_after_resample']:,} "
                          f"({row['fit_seconds']:.0f}s)", flush=True)
                except MemoryError as exc:
                    # Reduce scope rather than retry identically.
                    print(f"  {name} s{seed} OOM -> arm dropped: {exc}",
                          flush=True)
                    rows.append({"arm": name, "seed": seed,
                                 "error": f"OOM: {exc}"})
                    gc.collect()
                    break
                except Exception as exc:
                    print(f"  {name} s{seed} FAILED: {exc}", flush=True)
                    rows.append({"arm": name, "seed": seed, "error": str(exc)})
                    gc.collect()
                gc.collect()
            # Checkpoint after every arm: a 45-minute run must not lose
            # everything to a late failure.
            pd.DataFrame(rows).to_csv(RESULTS / "tune_classifier.csv",
                                      index=False)
        del split
        gc.collect()
    del ds
    gc.collect()

raw = pd.DataFrame(rows)
write("tune_classifier", raw, meta={
    "reviewer_comment": "R2.5/R2.6",
    "partition": "validation",
    "seeds": SEEDS,
    "dataset": args.dataset,
    "architecture_frozen": {
        "architecture": "parallel",
        "detector": asdict(DETECTOR),
        "tau_b_beta": "F9", "tau_u_quantile": "0.95",
        "tau_m_objective": "f1"},
    "arms": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                 for kk, vv in v.items()} for k, v in ARMS.items()},
    "reference_point": {"arm": "ours-submitted",
                        "balanced_accuracy": "0.9001 +- 0.0235",
                        "source": "baselines_cic-ids2017.csv"},
    "note": ("08_ablations.py's 'weighted' arm did not set class_weight and "
             "was therefore identical to 'none': make_split treats "
             "'weighted' and 'none' identically and the cost-sensitive part "
             "lives in ClassifierConfig. The genuine cost-sensitive arms are "
             "weighted-base and weighted-subsample here. tau_M is "
             "re-selected for every resampled classifier."),
})

# --- aggregate ---------------------------------------------------------
if args.no_agg:
    print(f"\n[no-agg] {len(raw)} rows written; aggregate deferred to the "
          f"final chunk.")
    print("\n12-OK")
    sys.exit(0)

ok = raw[raw["error"].isna()] if "error" in raw.columns else raw
num = [c for c in ok.columns
       if c not in ("arm", "seed", "imbalance", "resampler", "error")
       and pd.api.types.is_numeric_dtype(ok[c])]
agg = ok.groupby("arm")[num].agg(["mean", "std"])
agg.columns = ["_".join(c) for c in agg.columns]
agg = agg.reset_index()
agg.insert(1, "n_seeds", ok.groupby("arm")["seed"].nunique().values)
agg = agg.sort_values("balanced_accuracy_mean", ascending=False)
write("tune_classifier_agg", agg, meta={
    "reviewer_comment": "R2.5/R2.6", "partition": "validation",
    "note": "std is ddof=1 over seeds, as TUNING.md requires."})

show = ["arm", "n_seeds", "balanced_accuracy_mean", "balanced_accuracy_std",
        "f1_weighted_mean", "benign_fpr_mean", "n_train_after_resample_mean"]
print("\n" + agg[[c for c in show if c in agg.columns]].to_string(index=False))
print("\nreference: ours-submitted bACC 0.9001 +- 0.0235 (5 seeds)")
print("\n12-OK")
