"""Determinism check: re-run one experiment and confirm the artifact matches.

PRODUCES
    revision/results/determinism_check.csv

FEEDS
    Response R2.5 (reproducibility). Revised manuscript: the reproducibility
    statement in the experimental-setup section.

WHY THIS EXISTS
    The brief requires "a determinism check: re-run one experiment from a
    clean state and confirm the artifact matches." R2.5 charges that the
    reported results are impossible to reproduce independently; a claim of
    reproducibility that has never been tested is not worth printing.

    This re-fits the chosen configuration from scratch at a fixed seed and
    compares every metric against the value already recorded in
    results/baselines_cic-ids2017_raw.csv. Any mismatch beyond floating-point
    tolerance is a FAILURE and is reported as one.

USAGE
    python revision/scripts/09_determinism_check.py [--seed 0] [--tol 1e-9]

RUNTIME  ~2 min
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RESULTS, load_clean_any, timed, write  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nids import config  # noqa: E402
from nids.data import prepare  # noqa: E402
from nids.experiment import ExperimentConfig, evaluate_validation, fit_and_select  # noqa: E402
from nids.stages.classifier import ClassifierConfig  # noqa: E402
from nids.stages.detector import DetectorConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tol", type=float, default=1e-9)
ap.add_argument("--dataset", default="cic-ids2017")
args = ap.parse_args()

CFG = ExperimentConfig(
    name="ours-submitted", architecture="parallel",
    detector=DetectorConfig(kind="ocsvm", n_train=10_000,
                            gamma=2.93e-4, nu=1.3e-6),
    classifier=ClassifierConfig(kind="rf", n_estimators=200),
    tau_b_beta="F9", tau_u_quantile="0.95", tau_m_objective="f1")

REF = RESULTS / f"baselines_{args.dataset}_raw.csv"

with timed("09_determinism_check"):
    if not REF.exists():
        print(f"!! {REF} not found - run 02_train_baselines.py first")
        sys.exit(1)
    ref = pd.read_csv(REF)
    ref = ref[(ref.arm == "ours-submitted") & (ref.seed == args.seed)]
    if ref.empty:
        print(f"!! no reference row for seed {args.seed}")
        sys.exit(1)
    ref_row = ref.iloc[0]

    ds = load_clean_any(args.dataset)
    split = prepare.make_split(ds, seed=config.SPLIT_SEED)
    del ds
    gc.collect()

    fitted = fit_and_select(split, CFG, args.seed)
    res = evaluate_validation(fitted, split)

    rows = []
    for metric, new in res.metrics.items():
        if not isinstance(new, (int, float)):
            continue
        if metric not in ref_row.index:
            continue
        old = ref_row[metric]
        if pd.isna(old) or pd.isna(new):
            continue
        delta = abs(float(new) - float(old))
        rows.append({"metric": metric, "recorded": float(old),
                     "recomputed": float(new), "abs_delta": delta,
                     "match": bool(delta <= args.tol)})

    # Thresholds too: a reproducible metric with an irreproducible threshold
    # would mean the agreement was luck.
    for name, got in (("tau_b", fitted.thresholds.tau_b),
                      ("tau_m", fitted.thresholds.tau_m),
                      ("tau_u", fitted.thresholds.tau_u)):
        if name in ref_row.index and not pd.isna(ref_row[name]):
            delta = abs(float(got) - float(ref_row[name]))
            rows.append({"metric": name, "recorded": float(ref_row[name]),
                         "recomputed": float(got), "abs_delta": delta,
                         "match": bool(delta <= args.tol)})

df = pd.DataFrame(rows)
n_ok = int(df["match"].sum())
n = len(df)
verdict = "PASS" if n_ok == n else "FAIL"

write("determinism_check", df, meta={
    "seed": args.seed, "tolerance": args.tol, "dataset": args.dataset,
    "verdict": verdict, "matched": n_ok, "checked": n,
    "note": ("Re-fits the chosen configuration from scratch and compares "
             "every metric and every selected threshold against the recorded "
             "artifact. A single mismatch is a failure."),
})
print()
print(df.to_string(index=False))
print(f"\n{verdict}: {n_ok}/{n} quantities reproduce within {args.tol}")
print("\n09-OK" if verdict == "PASS" else "\n09-FAIL")
sys.exit(0 if verdict == "PASS" else 2)
