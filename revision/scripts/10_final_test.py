"""THE SINGLE TEST-SET EVALUATION. Run once, last, on the chosen configuration.

PRODUCES
    revision/results/final_test.csv
    results/test_set_ledger.jsonl   (appended - the audit trail)

FEEDS
    Revised manuscript Table 5 (the headline table) and the response's
    statement of how many times the test set was touched.

WHY THIS EXISTS AND WHY IT IS GUARDED
    The test partition was split once, with SPLIT_SEED = 20260803, and has
    never been read for selection. Every threshold, every hyperparameter and
    the choice of configuration were made on validation. This script is the
    only place in the revision that reads it.

    `nids.eval.testguard.spend()` writes an append-only ledger entry BEFORE the
    data is touched, so the count is auditable after the fact rather than
    remembered. A reviewer asking "how many times did you look at the test
    set?" gets an answer backed by a file.

    DO NOT run this script twice to "check something". Each run is a spend and
    each spend contaminates the estimate. If a number here looks wrong, fix it
    on validation and accept that the test estimate stands as recorded.

CONFIGURATIONS (see revision/DECISION.md and revision/TUNING.md)
    --config submitted  the pipeline as submitted: downsampling to 1,437 per
        class, tau_3 at the 95th benign percentile.
    --config updated    the adopted configuration: SMOTE on the full malicious
        pool, tau_3 at the 99.5th benign percentile. Both changes were selected
        on validation (scripts 15 and 16) before this script saw the test set.

    Each configuration is a SEPARATE, logged spend. Both are reported in the
    paper, with the ledger count stated, because a reader is entitled to know
    how many times the frozen partition was read.

    Both are compared against our re-run of the competitor's 'balanced'
    configuration - the row of their Table V we compare to (see
    04_competitor_reproduction.py and finding A1).

USAGE
    python revision/scripts/10_final_test.py --authorise [--config updated]
        (the flag is deliberate friction; the script refuses without it)

RUNTIME  ~5 min submitted, ~25 min updated (SMOTE resampling per seed)
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import competitor_config, timed, wilson, write  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nids import config  # noqa: E402
from nids.data import cache, prepare  # noqa: E402
from nids.eval import metrics as M  # noqa: E402
from nids.eval import testguard  # noqa: E402
from nids.experiment import ExperimentConfig, fit_and_select  # noqa: E402
from nids.stages.classifier import ClassifierConfig  # noqa: E402
from nids.stages.detector import DetectorConfig  # noqa: E402
from nids.stages.pipeline import build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--authorise", action="store_true",
                help="required; each run is a logged test-set spend")
ap.add_argument("--resume", action="store_true",
                help="continue an already-logged evaluation of the same "
                     "configuration from its shards, without writing a new "
                     "ledger entry. Only legitimate when a run of THIS "
                     "configuration was already logged and died before "
                     "finishing - it completes one spend, it does not make a "
                     "new one.")
ap.add_argument("--seeds", default=",".join(map(str, config.MODEL_SEEDS)))
ap.add_argument("--no-dedup", action="store_true",
                help=("load the dataset without exact-feature row "
                      "deduplication. This CHANGES THE TEST PARTITION "
                      "(44,040 -> 59,640 rows), so results are not comparable "
                      "with earlier ledger entries and must be reported as a "
                      "separate configuration."))
ap.add_argument("--config", default="submitted",
                choices=["submitted", "updated", "smote-only"],
                help="'updated' = SMOTE + tau_u 0.995, adopted after the "
                     "validation evidence of scripts 15/16. Each choice is a "
                     "SEPARATE, logged test-set spend.")
args = ap.parse_args()

if not args.authorise:
    print("REFUSED. This script spends the frozen test set.\n"
          "Re-run with --authorise only when the configuration is final.\n"
          f"Ledger currently records {testguard.count()} spend(s).")
    sys.exit(1)

#: tau_u differs between the two configurations; SMOTE is applied to the
#: classifier's training pool below when --config updated.
_TAU_U = "0.995" if args.config == "updated" else "0.95"
_SMOTE = args.config in ("updated", "smote-only")
_OURS = {"updated": "ours-updated", "smote-only": "ours-smote-only",
         "submitted": "ours-submitted"}[args.config]

ARMS = {
    _OURS: ExperimentConfig(
        name=_OURS, architecture="parallel",
        detector=DetectorConfig(kind="ocsvm", n_train=10_000,
                                gamma=2.93e-4, nu=1.3e-6),
        classifier=ClassifierConfig(kind="rf", n_estimators=200),
        tau_b_beta="F9", tau_u_quantile=_TAU_U, tau_m_objective="f1"),
    "verkerken-balanced": ExperimentConfig(
        name="verkerken-balanced", architecture="verkerken",
        detector=DetectorConfig(kind="ocsvm", n_train=10_000,
                                gamma=2.93e-4, nu=1.3e-6),
        classifier=ClassifierConfig(kind="rf", n_estimators=200),
        tau_b_beta="F5", tau_u_quantile="0.99", tau_m_objective="f1"),
    # The same architecture at the authors' own published constants, which
    # their repository made available only after submission. Our inferred
    # values weakened them on every parameter, so the arm above understates
    # their system. See 23_baseline_fidelity.py and _common.competitor_config.
    "verkerken-balanced-faithful": competitor_config("balanced"),
}
SEEDS = [int(s) for s in args.seeds.split(",")]

with timed("10_final_test"):
    if args.resume:
        print("resuming an already-logged evaluation; no new ledger entry. "
              f"Ledger stands at {testguard.count()} spend(s).", flush=True)
    else:
        testguard.spend(
            cycle=16, phase="C",
            purpose=("Final reported evaluation of the revision, config="
                     + args.config
                     + (", NO-DEDUP dataset variant (test partition 59,640 "
                        "rows, not the 44,040 of earlier entries)"
                        if args.no_dedup else "")
                     + ". Configuration chosen on validation "
                     "(scripts 15/16); this is its single test evaluation, "
                     "plus the competitor arms it is compared against."),
            model=(_OURS + " + verkerken-balanced + "
                   "verkerken-balanced-faithful, 5 seeds"),
            authorised=True)
        print(f"ledger now records {testguard.count()} spend(s)", flush=True)

    ds = cache.load_clean("cic-ids2017", drop_duplicates=not args.no_dedup)
    # The EVALUATION split is always the protocol split (imbalance="downsample"),
    # because the imbalance strategy is a property of how the classifier is
    # TRAINED, not of the traffic it is scored on. Building the updated arm on
    # an imbalance="none" split would give it a different test partition
    # (1,802,102 rows against 44,040) and the two configurations would not be
    # comparable - and neither would be comparable with the competitor.
    split = prepare.make_split(ds, seed=config.SPLIT_SEED,
                               imbalance="downsample")
    # The updated configuration's classifier trains on the FULL malicious pool,
    # resampled by SMOTE. That pool comes from a separate split object; only its
    # training partition is used, and its test partition is never read.
    train_pool = (prepare.make_split(ds, seed=config.SPLIT_SEED,
                                     imbalance="none").train_malicious
                  if _SMOTE else None)
    del ds
    gc.collect()

    feats = split.feature_cols
    test = split.test
    y = test["Attack Type"].values
    # Zero-day rows carry their own label in the decision space.
    y = np.where(np.isin(test["Label"].values, config.DEFAULT_ZERO_DAY),
                 "Zero Day", y)
    X = test[feats].values
    print(f"test rows: {len(test):,} "
          f"(zero-day: {int((y == 'Zero Day').sum())})", flush=True)

    SHARDS = (Path(__file__).resolve().parents[1] / "results" / "shards")
    SHARDS.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm, cfg in ARMS.items():
        for seed in SEEDS:
            # Shard per (arm, seed). The SMOTE arm resamples 788k rows and this
            # machine kills the interpreter on it; without shards a memory kill
            # would force ANOTHER ledger spend to recover the missing seeds.
            _sh = SHARDS / ("_test_" + args.config + "_" + arm
                            + "_s" + str(seed) + ".csv")
            if _sh.exists():
                rows.extend(pd.read_csv(_sh).to_dict("records"))
                print("  " + arm + " s" + str(seed) + ": cached", flush=True)
                continue
            fitted = fit_and_select(split, cfg, seed)
            if _SMOTE and arm == _OURS:
                from imblearn.over_sampling import SMOTE
                from nids.stages.classifier import Classifier
                _X = train_pool[feats].values
                _y = train_pool["Attack Type"].values
                _k = max(1, min(5, int(pd.Series(_y).value_counts().min()) - 1))
                _Xr, _yr = SMOTE(random_state=seed,
                                 k_neighbors=_k).fit_resample(_X, _y)
                fitted.classifier = Classifier(
                    ClassifierConfig(kind="rf", n_estimators=200,
                                     seed=seed)).fit(_Xr, _yr)
                del _X, _y, _Xr, _yr
                gc.collect()
            pipe = build(cfg.architecture, detector_cfg=cfg.detector,
                         classifier_cfg=cfg.classifier,
                         thresholds=fitted.thresholds,
                         detector=fitted.detector, classifier=fitted.classifier)
            pred = pipe.predict(X)
            res = M.evaluate(y, pred, seed=seed)
            zd_mask = y == "Zero Day"
            hits = int((pred[zd_mask] == "Zero Day").sum())
            n_zd = int(zd_mask.sum())
            lo, hi = wilson(hits, n_zd)
            row = {"arm": arm, "seed": seed}
            row.update({k: v for k, v in res.metrics.items()
                        if isinstance(v, (int, float))})
            row.update({"zero_day_hits": hits, "zero_day_n": n_zd,
                        "zero_day_recall_exact": hits / n_zd if n_zd else np.nan,
                        "zero_day_ci_lo": lo, "zero_day_ci_hi": hi})
            pd.DataFrame([row]).to_csv(_sh, index=False)
            rows.append(row)
            print(f"  {arm} s{seed} "
                  f"bACC={row.get('balanced_accuracy', float('nan')):.4f} "
                  f"ZD={hits}/{n_zd} FPR={row.get('benign_fpr', float('nan')):.4f}",
                  flush=True)
            del fitted, pipe
            gc.collect()

raw = pd.DataFrame(rows)
write("final_test_raw" + ("_" + args.config if args.config != "submitted" else ""), raw)

num = [c for c in raw.columns if c not in ("arm", "seed")
       and pd.api.types.is_numeric_dtype(raw[c])]
agg = raw.groupby("arm")[num].agg(["mean", "std"])
agg.columns = ["_".join(c) for c in agg.columns]
agg = agg.reset_index()
write("final_test" + ("_" + args.config if args.config != "submitted" else ""), agg, meta={
    "seeds": SEEDS,
    "partition": "TEST (frozen)",
    "ledger_spends_after_this_run": testguard.count(),
    "split_seed": config.SPLIT_SEED,
    "zero_day_classes": config.DEFAULT_ZERO_DAY,
    "configuration": args.config,
    "note": ("One spend per configuration. All thresholds were chosen on "
             "validation before this script ran."),
})
print()
print(agg.to_string(index=False))
print("\n10-OK")
