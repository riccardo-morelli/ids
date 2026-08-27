"""Does exact-feature deduplication cost us the Port Scan class?

PRODUCES
    revision/results/dedup_effect.csv

FEEDS
    The reproduction gap in the fairness section, and the Port Scan row of the
    confusion matrix in Figure 2(c).

WHY THIS EXISTS
    Our cleaning drops rows that are identical across every feature, on the
    reasoning that an exact duplicate carries no new information. Comparing
    the raw CSVs against the cleaned frame shows what that costs per class:

        class              raw CSV    cleaned    kept
        Port Scan          158,930      1,956     1.2%
        DDoS               128,027    128,014   100.0%
        DoS Hulk           231,073    172,846    74.8%
        SSH-Patator          5,897      3,219    54.6%
        Bot                  1,966      1,437    73.1%

    Every class loses something; Port Scan loses almost everything. That is
    not a coincidence. A port scan is near-degenerate by construction - a
    handful of packets, no payload, differing only in the destination port,
    which our schema drops as an identifier. Two scans of adjacent ports are
    therefore identical in every surviving feature, and deduplication treats
    them as one flow.

    Three consequences, and all three are testable:

      1. The classifier trains on 1,437 Port Scan rows drawn from 1,956
         survivors rather than from 158,804 flows, so its Port Scan
         representation is built from whatever happened to survive.
      2. Port Scan is the known attack class our system classifies worst
         (377 of 431 in Figure 2c) and the class with the largest deficit
         against the competitor's published figures (0.8091 against 0.9914).
      3. The competitor's per-class support is 1,948, which our deduplicated
         Port Scan pool of 1,956 can barely supply. That is a candidate
         explanation for a reproduction gap the manuscript currently reports
         as unexplained after five refuted hypotheses.

    This script re-runs the full pipeline with and without row deduplication
    and reports per-class recall for both, on VALIDATION only.

USAGE
    python revision/scripts/25_dedup_effect.py [--seeds 0,1,2]

RUNTIME  ~25 min (the no-dedup arm carries 2.8M rows)
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import competitor_config, load_clean_any, timed, write  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nids import config  # noqa: E402
from nids.data import prepare  # noqa: E402
from nids.experiment import ExperimentConfig, fit_and_select  # noqa: E402
from nids.stages.classifier import ClassifierConfig  # noqa: E402
from nids.stages.detector import DetectorConfig  # noqa: E402
from nids.stages.pipeline import BENIGN, ZERO_DAY, build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="0,1,2")
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(",")]

CFG_OURS = ExperimentConfig(
    name="ours", architecture="parallel",
    detector=DetectorConfig(kind="ocsvm", n_train=10_000,
                            gamma=2.93e-4, nu=1.3e-6),
    classifier=ClassifierConfig(kind="rf", n_estimators=200),
    tau_b_beta="F9", tau_u_quantile="0.95", tau_m_objective="f1")

rows = []
with timed("25_dedup_effect"):
    for dedup in (True, False):
        tag = "dedup (current)" if dedup else "no dedup"
        ds = load_clean_any("cic-ids2017", drop_duplicates=dedup)
        counts = ds.frame["Label"].value_counts()
        print(f"\n=== {tag}: {len(ds.frame):,} rows, "
              f"Port Scan {int(counts.get('PortScan', 0)):,} ===", flush=True)

        for cfg, name in ((CFG_OURS, "ours"),
                          (competitor_config("balanced"), "baseline")):
            for seed in SEEDS:
                split = prepare.make_split(ds, seed=config.SPLIT_SEED,
                                           imbalance="downsample")
                fitted = fit_and_select(split, cfg, seed)
                pipe = build(cfg.architecture, detector_cfg=cfg.detector,
                             classifier_cfg=cfg.classifier,
                             thresholds=fitted.thresholds,
                             detector=fitted.detector,
                             classifier=fitted.classifier)

                # VALIDATION only. The frozen test partition is untouched.
                val = pd.concat([split.val_benign, split.val_malicious],
                                ignore_index=True)
                y = val["Attack Type"].values
                pred = pipe.predict(val[split.feature_cols].values)

                per_class = {}
                for c in sorted(set(y)):
                    m = y == c
                    per_class[c] = float((pred[m] == c).mean())
                bacc = float(np.mean(list(per_class.values())))

                rows.append({
                    "dedup": dedup, "arm": name, "seed": seed,
                    "n_rows": len(ds.frame),
                    "portscan_pool": int(counts.get("PortScan", 0)),
                    "balanced_accuracy": bacc,
                    **{f"recall_{c}": v for c, v in per_class.items()},
                })
                print(f"  {tag:16s} {name:9s} seed={seed} bACC={bacc:.4f} "
                      f"PortScan={per_class.get('Port Scan', float('nan')):.4f}",
                      flush=True)
                del fitted, pipe, split
                gc.collect()
        del ds
        gc.collect()

df = pd.DataFrame(rows)
write("dedup_effect", df, meta={
    "partition": "VALIDATION",
    "seeds": SEEDS,
    "note": ("Exact-feature row deduplication keeps 1.2% of Port Scan flows "
             "against 55-100% for every other class. No test-set access."),
})

print("\n=== validation balanced accuracy ===")
print(df.groupby(["arm", "dedup"])["balanced_accuracy"]
        .agg(["mean", "std"]).round(4).to_string())

rec = [c for c in df.columns if c.startswith("recall_")]
print("\n=== per-class recall, ours ===")
o = df[df.arm == "ours"]
print(o.groupby("dedup")[rec].mean().round(4).T.to_string())

print("\n25-OK")
