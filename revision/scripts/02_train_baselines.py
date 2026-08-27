"""All systems under ONE protocol, 5 seeds: ours, the competitor, and RF.

PRODUCES
    revision/results/baselines_<dataset>.csv      - mean +- std over 5 seeds
    revision/results/baselines_<dataset>_raw.csv  - per-seed rows
    revision/results/predictions/<arm>_s<seed>.npz - paired predictions,
        consumed by 06_significance.py

FEEDS
    Response R2.3 (statistical rigour), R2.1 (second dataset when run with
    --dataset cse-cic-ids2018), and the revised Table 5.

WHY THIS EXISTS
    R2.3: "The classification metrics ... are reported from a single
    train/test split without cross-validation, confidence intervals, or
    standard deviations."

    Every arm here shares the same split, the same preprocessing, the same
    seeds and the same tuning effort. The competitor is re-run inside our
    harness (nids.stages.pipeline.VerkerkenPipeline), never quoted from their
    paper, because a reviewer is entitled to ask whether the comparison is
    like-for-like.

    All numbers are on VALIDATION. The frozen test set is touched only by
    10_final_test.py.

USAGE
    python revision/scripts/02_train_baselines.py [--dataset cic-ids2017]
                                                  [--seeds 0,1,2,3,4]

RUNTIME  ~35 min for 3 arms x 5 seeds on CIC-IDS2017 (12-core laptop)
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (RESULTS, apply_smote_classifier, load_clean_any,  # noqa: E402
                     timed, write)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nids import config  # noqa: E402
from nids.data import cache, prepare  # noqa: E402
from nids.eval import metrics as M  # noqa: E402
from nids.experiment import ExperimentConfig, evaluate_validation, fit_and_select  # noqa: E402
from nids.stages.classifier import Classifier, ClassifierConfig  # noqa: E402
from nids.stages.detector import DetectorConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="cic-ids2017")
ap.add_argument("--seeds", default=",".join(map(str, config.MODEL_SEEDS)))
ap.add_argument("--smote", action="store_true",
                help="train the classifier on a SMOTE-resampled pool - the "
                     "imbalance strategy adopted for the revision. The "
                     "evaluation partition is unchanged either way.")
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(",")]

SUF0 = "_smote" if args.smote else ""
PRED = RESULTS / "predictions"
PRED.mkdir(parents=True, exist_ok=True)

# The three arms. "ours" and "verkerken" differ ONLY in the stage-2
# architecture: same detector family, same classifier, same thresholds
# procedure. That is what makes the comparison fair.
ARMS = {
    "ours-submitted": ExperimentConfig(
        name="ours-submitted",
        architecture="parallel",
        detector=DetectorConfig(kind="ocsvm", n_train=10_000,
                                gamma=2.93e-4, nu=1.3e-6),
        classifier=ClassifierConfig(kind="rf", n_estimators=200),
        tau_b_beta="F9", tau_u_quantile="0.95", tau_m_objective="f1"),
    "verkerken-balanced": ExperimentConfig(
        name="verkerken-balanced",
        architecture="verkerken",
        detector=DetectorConfig(kind="ocsvm", n_train=10_000,
                                gamma=2.93e-4, nu=1.3e-6),
        classifier=ClassifierConfig(kind="rf", n_estimators=200),
        tau_b_beta="F5", tau_u_quantile="0.99", tau_m_objective="f1"),
}

with timed("02_train_baselines"):
    ds = load_clean_any(args.dataset)
    split = prepare.make_split(ds, seed=config.SPLIT_SEED)
    # Kept only when --smote needs to build the resampled pool.
    ds_ref = ds if args.smote else None
    if not args.smote:
        del ds
    gc.collect()
    print(f"{args.dataset}: val_benign={len(split.val_benign):,} "
          f"val_malicious={len(split.val_malicious):,}", flush=True)

    SH = RESULTS / "shards"
    SH.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    for arm, cfg in ARMS.items():
        for seed in SEEDS:
            # Per-(arm, seed) shard: SMOTE runs are memory-heavy and this
            # machine hard-kills the interpreter. Resumable.
            _bl_shard = SH / f"_bl_{args.dataset}{SUF0}_{arm}_s{seed}.csv"
            if _bl_shard.exists():
                raw_rows.extend(pd.read_csv(_bl_shard).to_dict("records"))
                print(f"  {arm} seed={seed}: cached", flush=True)
                continue
            fitted = fit_and_select(split, cfg, seed)
            if args.smote:
                apply_smote_classifier(fitted, ds_ref,
                                       feature_cols=split.feature_cols,
                                       seed=seed)
            res = evaluate_validation(fitted, split)
            row = {"arm": arm, "seed": seed}
            row.update({k: v for k, v in res.metrics.items()
                        if isinstance(v, (int, float))})
            row.update({
                "tau_b": fitted.thresholds.tau_b,
                "tau_m": fitted.thresholds.tau_m,
                "tau_u": fitted.thresholds.tau_u,
            })
            pd.DataFrame([row]).to_csv(_bl_shard, index=False)
            raw_rows.append(row)
            np.savez_compressed(
                PRED / f"{arm}_s{seed}_{args.dataset}{SUF0}.npz",
                y=res.y_true.astype(str), pred=res.y_pred.astype(str))
            # zero_day_recall is None on the default split by design: the
            # zero-day classes live in the frozen test partition only. Zero-day
            # capability on validation is measured by 03_zero_day_protocols.py
            # via leave-one-class-out, which is what R2.2 actually asks for.
            print(f"  {arm} seed={seed} "
                  f"bACC={row.get('balanced_accuracy', float('nan')):.4f} "
                  f"F1w={row.get('f1_weighted', float('nan')):.4f} "
                  f"FPR={row.get('benign_fpr', float('nan')):.4f}", flush=True)
            del fitted
            gc.collect()

    # --- closed RF baseline: the paper's other comparison point -----------
    # It cannot emit a Zero Day label at all, which is exactly why the
    # comparison is ill-posed on that axis. Reported, with that caveat.
    feats = split.feature_cols
    Xval = pd.concat([split.val_benign, split.val_malicious],
                     ignore_index=True)
    yval = np.where(Xval["Attack Type"].values == "Benign", "Benign",
                    Xval["Attack Type"].values)
    Xtr = pd.concat([split.train_benign, split.train_malicious],
                    ignore_index=True)
    ytr = np.where(Xtr["Attack Type"].values == "Benign", "Benign",
                   Xtr["Attack Type"].values)
    for seed in SEEDS:
        clf = Classifier(ClassifierConfig(kind="rf", n_estimators=200,
                                          seed=seed))
        clf.fit(Xtr[feats].values, ytr)
        pred = clf.predict_with_certainty(Xval[feats].values)[0]
        res = M.evaluate(yval, pred, seed=seed)
        row = {"arm": "rf-closed-baseline", "seed": seed}
        row.update({k: v for k, v in res.metrics.items()
                    if isinstance(v, (int, float))})
        raw_rows.append(row)
        np.savez_compressed(
            PRED / f"rf-closed-baseline_s{seed}_{args.dataset}{SUF0}.npz",
            y=yval.astype(str), pred=pred.astype(str))
        print(f"  rf-closed seed={seed} "
              f"bACC={row.get('balanced_accuracy'):.4f}", flush=True)
        del clf
        gc.collect()

raw = pd.DataFrame(raw_rows)
SUF = "_smote" if args.smote else ""
write(f"baselines_{args.dataset}{SUF}_raw", raw)

num = [c for c in raw.columns if c not in ("arm", "seed")
       and pd.api.types.is_numeric_dtype(raw[c])]
agg = raw.groupby("arm")[num].agg(["mean", "std", "min", "max"])
agg.columns = ["_".join(c) for c in agg.columns]
agg = agg.reset_index()
agg.insert(1, "n_seeds", len(SEEDS))
write(f"baselines_{args.dataset}{SUF}", agg, meta={
    "seeds": SEEDS,
    "dataset": args.dataset,
    "partition": "validation",
    "note": ("rf-closed-baseline cannot emit a Zero Day label; its zero-day "
             "recall is structurally 0 and the comparison on that axis is "
             "ill-posed. Reported for completeness."),
})

show = ["arm", "n_seeds", "balanced_accuracy_mean", "balanced_accuracy_std",
        "f1_weighted_mean", "zero_day_recall_mean", "benign_fpr_mean"]
print(agg[[c for c in show if c in agg.columns]].to_string(index=False))
print("\n02-OK")
