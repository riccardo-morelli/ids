"""Re-implement the baseline at the competitor's own published settings.

PRODUCES
    revision/results/baseline_fidelity.csv

FEEDS
    Response R2.3 and the fairness paragraph in Section 4.7.

WHY THIS EXISTS
    Verkerken et al. released their code, fitted models and test data after the
    original submission (see _common.COMPETITOR_REPO). Their paper describes
    the architecture but not the tuned constants, so our re-implementation had
    to infer them, and every inference happened to make their system weaker:

        parameter        theirs        ours, as submitted
        gamma            0.0633        2.93e-4   (two orders of magnitude)
        nu               2.32e-4       1.3e-6
        PCA              56 components none
        RF               97 trees,     200 trees, max_features='sqrt'
                         max_features 0.175
        tau_M            0.98, fixed   selected on validation -> 1.000000

    A comparison against a weakened baseline flatters our system, and a
    reviewer who downloads their repository can see it. This script re-runs
    their architecture at their own settings, on our data and our split, so the
    comparison is like-for-like in both directions.

    Their fitted models are NOT used: they are trained on their split and on 67
    CICFlowMeter v4 features, where our CIC-IDS2017 release has 64 in v3 names
    (Protocol, SYN Flag Count and the two Subflow packet counts are absent from
    our CSVs entirely). Scoring their models on our test set would measure
    cross-dataset transfer rather than architecture. Only the constants above
    cross over; the implementation stays ours.

    Each fidelity fix is added cumulatively, so the fairness paragraph can
    attribute the correction rather than quote a single lump figure.

    VALIDATION ONLY. No test-set budget is spent here.

USAGE
    python revision/scripts/23_baseline_fidelity.py [--seeds 0,1,2,3,4]

RUNTIME  ~10 min
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (COMPETITOR_COMMIT, COMPETITOR_REPO, THEIR_ARMS,  # noqa: E402
                     THEIR_GAMMA, THEIR_MAX_FEATURES, THEIR_N_ESTIMATORS,
                     THEIR_NU, THEIR_PCA_COMPONENTS, THEIR_TAU_M,
                     competitor_config, load_clean_any, timed, write)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nids import config  # noqa: E402
from nids.data import prepare  # noqa: E402
from nids.experiment import ExperimentConfig, evaluate_validation, fit_and_select  # noqa: E402
from nids.stages.classifier import ClassifierConfig  # noqa: E402
from nids.stages.detector import DetectorConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default=",".join(map(str, config.MODEL_SEEDS)))
ap.add_argument("--arm", default="balanced", choices=sorted(THEIR_ARMS))
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(",")]

# Cumulative ladder: each rung restores one of their settings.
LADDER = {
    "as submitted":        dict(pca=False, hp=False, tau_m=None),
    "+ PCA(56)":           dict(pca=True,  hp=False, tau_m=None),
    "+ their OC-SVM/RF":   dict(pca=True,  hp=True,  tau_m=None),
    "+ tau_M 0.98 (full)": dict(pca=True,  hp=True,  tau_m=THEIR_TAU_M),
}


def cfg_for(pca: bool, hp: bool, tau_m):
    beta, quant = THEIR_ARMS[args.arm]
    gamma, nu = (THEIR_GAMMA, THEIR_NU) if hp else (2.93e-4, 1.3e-6)
    clf = (ClassifierConfig(kind="rf", n_estimators=THEIR_N_ESTIMATORS,
                            max_features=THEIR_MAX_FEATURES)
           if hp else ClassifierConfig(kind="rf", n_estimators=200))
    return ExperimentConfig(
        name=f"verkerken-{args.arm}", architecture="verkerken",
        detector=DetectorConfig(kind="ocsvm", use_pca=pca,
                                n_components=THEIR_PCA_COMPONENTS,
                                n_train=10_000, gamma=gamma, nu=nu),
        classifier=clf, tau_b_beta=beta, tau_u_quantile=quant,
        tau_m_objective="f1", tau_m_fixed=tau_m)


rows = []
with timed("23_baseline_fidelity"):
    ds = load_clean_any("cic-ids2017")
    split = prepare.make_split(ds, seed=config.SPLIT_SEED,
                               imbalance="downsample")

    for label, kw in LADDER.items():
        for seed in SEEDS:
            fitted = fit_and_select(split, cfg_for(**kw), seed)
            res = evaluate_validation(fitted, split)
            rows.append({
                "variant": label, "seed": seed, "arm": args.arm,
                "use_pca": kw["pca"], "their_hyperparams": kw["hp"],
                "tau_m": fitted.thresholds.tau_m,
                "tau_b": fitted.thresholds.tau_b,
                "tau_u": fitted.thresholds.tau_u,
                "balanced_accuracy": res.metrics["balanced_accuracy"],
                "f1_weighted": res.metrics["f1_weighted"],
                "fit_seconds": fitted.fit_seconds,
            })
            print(f"  {label:20s} seed={seed} tau_m={fitted.thresholds.tau_m:.4f} "
                  f"bACC={res.metrics['balanced_accuracy']:.4f} "
                  f"F1w={res.metrics['f1_weighted']:.4f}", flush=True)
            del fitted
            gc.collect()

df = pd.DataFrame(rows)
write("baseline_fidelity", df, meta={
    "partition": "VALIDATION",
    "seeds": SEEDS,
    "arm": args.arm,
    "competitor_repo": COMPETITOR_REPO,
    "competitor_commit": COMPETITOR_COMMIT,
    "note": ("Their published constants, re-run inside our implementation on "
             "our split. None of their fitted models is executed. No test-set "
             "access."),
})

print("\n=== validation means over seeds ===")
print(df.groupby("variant", sort=False)
        [["balanced_accuracy", "f1_weighted", "fit_seconds"]]
        .agg(["mean", "std"]).round(4).to_string())

base = df[df.variant == "as submitted"].balanced_accuracy.mean()
print("\n=== balanced accuracy vs the submitted baseline ===")
for label in LADDER:
    m = df[df.variant == label].balanced_accuracy.mean()
    print(f"  {label:20s} {m:.4f}   {m - base:+.4f}")

print("\n23-OK")
