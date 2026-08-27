"""Swap in the strongest available detector and classifier, keep our rules.

PRODUCES
    revision/results/sota_components.csv

FEEDS
    The question of whether the architecture's contribution survives when its
    two stages are replaced by stronger components.

WHY THIS EXISTS
    Our architecture is a claim about the *rule stage*: that running an
    unsupervised detector and a supervised classifier in parallel and fusing
    them with an explicit rule table beats gating one behind the other. That
    claim is separable from the choice of the two models, and it should be
    tested with the best models we can obtain rather than only with the OC-SVM
    and random forest the submitted version happens to use.

    Stage 1a candidates (unsupervised, benign-only):
      ocsvm      the submitted detector
      ecod       PyOD's parameter-free per-feature tail probability. Leads or
                 ties on the ADBench tabular benchmark and has nothing to tune,
                 so no tuning decision can hide inside it.
      copod      the copula-based sibling of ECOD, same property.
      iforest    the standard baseline for this family.

    Stage 1b candidates (supervised, malicious-only):
      rf         the submitted classifier
      lgbm       gradient boosting, the strongest family on tabular data in
                 every recent benchmark
      xgb        the same family, different implementation

    Deep unsupervised systems reported at higher figures on this dataset
    (BLADE and similar) are NOT included: they re-label the PCAPs and classify
    windows of 50 flows rather than single flows, so their numbers answer a
    different question and are not comparable under our protocol. Saying that
    is more useful than quoting a number that cannot be reproduced here.

    VALIDATION ONLY. Zero-day is measured by leave-one-class-out, since the
    47 Infiltration/Heartbleed rows live only in the frozen test partition.

USAGE
    python revision/scripts/26_sota_components.py [--seeds 0,1] [--quick]

RUNTIME  ~40 min full, ~12 min with --quick
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_clean_any, timed, wilson, write  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nids import config  # noqa: E402
from nids.data import prepare  # noqa: E402
from nids.experiment import ExperimentConfig, fit_and_select  # noqa: E402
from nids.stages.classifier import ClassifierConfig  # noqa: E402
from nids.stages.detector import DetectorConfig  # noqa: E402
from nids.stages.pipeline import ZERO_DAY, build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="0,1")
ap.add_argument("--quick", action="store_true",
                help="one withheld class instead of all five")
ap.add_argument("--no-dedup", action="store_true",
                help="load the dataset without exact-feature row dedup")
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(",")]

DETECTORS = {
    "ocsvm": DetectorConfig(kind="ocsvm", n_train=10_000,
                            gamma=2.93e-4, nu=1.3e-6),
    "ecod": DetectorConfig(kind="ecod", n_train=10_000),
    "copod": DetectorConfig(kind="copod", n_train=10_000),
    "iforest": DetectorConfig(kind="iforest", n_train=10_000,
                              n_estimators=200),
}
CLASSIFIERS = {
    "rf": ClassifierConfig(kind="rf", n_estimators=200),
    "lgbm": ClassifierConfig(kind="lgbm", n_estimators=200),
    "xgb": ClassifierConfig(kind="xgb", n_estimators=200),
}

# The pairs worth running: our baseline, each stage improved alone, and the
# strongest of each together. A full grid would be 12 cells for little more.
PAIRS = [("ocsvm", "rf"),          # the submitted pair, the reference
         ("ecod", "rf"),           # detector swapped
         ("ocsvm", "lgbm"),        # classifier swapped
         ("ecod", "lgbm"),         # both swapped
         ("iforest", "rf"),        # the standard unsupervised baseline
         ("ocsvm", "xgb")]


def cfg_for(det: str, clf: str) -> ExperimentConfig:
    return ExperimentConfig(
        name=f"{det}+{clf}", architecture="parallel",
        detector=DETECTORS[det], classifier=CLASSIFIERS[clf],
        tau_b_beta="F9", tau_u_quantile="0.95", tau_m_objective="f1")


def run_fold(ds, held, det, clf, seed):
    """Withhold `held`, fit, measure. Own split, so no test budget."""
    split = prepare.make_split(ds, seed=config.SPLIT_SEED, zero_day=held)
    for part in (split.train_malicious, split.val_malicious):
        assert not part["Label"].isin(held).any(), "leak - fold void"

    cfg = cfg_for(det, clf)
    fitted = fit_and_select(split, cfg, seed)
    pipe = build(cfg.architecture, detector_cfg=cfg.detector,
                 classifier_cfg=cfg.classifier, thresholds=fitted.thresholds,
                 detector=fitted.detector, classifier=fitted.classifier)

    heldrows = split.test[split.test["Label"].isin(held)]
    if heldrows.empty:
        return None
    pred = pipe.predict(heldrows[split.feature_cols].values)
    hits, n = int((pred == ZERO_DAY).sum()), len(pred)
    lo, hi = wilson(hits, n)

    # Known-class performance and benign cost on VALIDATION.
    val = pd.concat([split.val_benign, split.val_malicious], ignore_index=True)
    y = val["Attack Type"].values
    pv = pipe.predict(val[split.feature_cols].values)
    per = {c: float((pv[y == c] == c).mean()) for c in sorted(set(y))}
    bacc = float(np.mean(list(per.values())))
    fpr = float((pv[y == "Benign"] != "Benign").mean())
    tau_m = fitted.thresholds.tau_m

    del fitted, pipe, split
    gc.collect()
    return {"zero_day_recall": hits / n, "zero_day_hits": hits, "n": n,
            "ci_lo": lo, "ci_hi": hi, "balanced_accuracy": bacc,
            "benign_fpr": fpr, "tau_m": tau_m,
            **{f"recall_{c}": v for c, v in per.items()}}


rows = []
with timed("26_sota_components"):
    ds = load_clean_any("cic-ids2017", drop_duplicates=not args.no_dedup)
    fine = ds.frame.loc[ds.frame["Attack Type"] != "Benign", "Label"]
    coarse = ds.frame.loc[ds.frame["Attack Type"] != "Benign", "Attack Type"]
    members: dict[str, list[str]] = {}
    for c, f in zip(coarse, fine):
        members.setdefault(c, [])
        if f not in members[c]:
            members[c].append(f)

    classes = sorted(members)
    if args.quick:
        classes = ["Botnet"] if "Botnet" in members else classes[:1]

    for det, clf in PAIRS:
        for cls in classes:
            for seed in SEEDS:
                try:
                    r = run_fold(ds, tuple(members[cls]), det, clf, seed)
                except Exception as e:                      # noqa: BLE001
                    print(f"  {det}+{clf} {cls} seed={seed} FAILED: "
                          f"{type(e).__name__}: {e}", flush=True)
                    continue
                if r is None:
                    continue
                r.update({"detector": det, "classifier": clf,
                          "withheld": cls, "seed": seed})
                rows.append(r)
                print(f"  {det:8s}+{clf:8s} {cls:12s} s{seed} "
                      f"zd={r['zero_day_recall']:.4f} "
                      f"({r['zero_day_hits']}/{r['n']}) "
                      f"bACC={r['balanced_accuracy']:.4f} "
                      f"fpr={r['benign_fpr']:.4f}", flush=True)

df = pd.DataFrame(rows)
write("sota_components", df, meta={
    "partition": "VALIDATION (leave-one-class-out; folds build own splits)",
    "seeds": SEEDS,
    "dedup": not args.no_dedup,
    "note": ("Rule stage and architecture unchanged; only the two stage-1 "
             "models vary. No test-set access."),
})

if not df.empty:
    print("\n=== mean over withheld classes and seeds ===")
    agg = (df.groupby(["detector", "classifier"])
             [["zero_day_recall", "balanced_accuracy", "benign_fpr", "tau_m"]]
             .mean().round(4).sort_values("zero_day_recall", ascending=False))
    print(agg.to_string())

    base = df[(df.detector == "ocsvm") & (df.classifier == "rf")]
    if not base.empty:
        b_zd = base.zero_day_recall.mean()
        b_ba = base.balanced_accuracy.mean()
        print(f"\nsubmitted pair: zero-day {b_zd:.4f}  bACC {b_ba:.4f}")
        print("deltas against it:")
        for (d, c), g in df.groupby(["detector", "classifier"]):
            if (d, c) == ("ocsvm", "rf"):
                continue
            print(f"  {d:8s}+{c:8s} zero-day {g.zero_day_recall.mean()-b_zd:+.4f}"
                  f"   bACC {g.balanced_accuracy.mean()-b_ba:+.4f}")

print("\n26-OK")
