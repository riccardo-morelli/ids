"""Quantify the closed-RF defence: what does it actually score on zero-day?

PRODUCES
    revision/results/rf_zeroday.csv

FEEDS
    response R2.3, manuscript Section 6 (stage
    contribution) and the Limitations section.

WHY THIS EXISTS
    A closed Random Forest outscores the full three-stage system on known-class
    balanced accuracy (0.9583 against 0.9050). Our entire defence is that the
    RF "cannot emit a Zero Day label at all", so the comparison is ill-posed on
    the axis the architecture exists to serve.

    That defence has been asserted, not measured. A reviewer is entitled to
    ask what the RF actually does when a novel class arrives. This script
    answers it under the same leave-one-class-out protocol used for the full
    system in script 03: withhold each attack family in turn, train the closed
    RF on the rest, and record what it predicts for the withheld class.

    Two quantities matter:
      * zero-day recall: structurally 0.0000, since the label is not in its
        output space. Measured rather than assumed.
      * where the novel traffic goes instead: the RF must assign it some known
        label, and the distribution of those labels is the operational cost -
        every one is a confident, wrong, actionable alert.

    VALIDATION ONLY.

USAGE
    python revision/scripts/18_rf_zeroday.py [--seeds 0,1,2]

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
from _common import load_clean_any, timed, write  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nids import config  # noqa: E402
from nids.data import prepare  # noqa: E402
from nids.stages.classifier import Classifier, ClassifierConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="0,1,2")
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(",")]

rows = []
with timed("18_rf_zeroday"):
    ds = load_clean_any("cic-ids2017")
    fine = ds.frame.loc[ds.frame["Attack Type"] != "Benign", "Label"]
    coarse = ds.frame.loc[ds.frame["Attack Type"] != "Benign", "Attack Type"]
    members: dict[str, list[str]] = {}
    for c, f in zip(coarse, fine):
        members.setdefault(c, [])
        if f not in members[c]:
            members[c].append(f)

    for cls in sorted(members):
        held = tuple(members[cls])
        try:
            split = prepare.make_split(ds, seed=config.SPLIT_SEED,
                                       zero_day=held)
        except Exception as exc:
            print(f"  {cls}: split failed: {exc}")
            continue
        feats = split.feature_cols
        holdout = split.test[split.test["Label"].isin(held)]
        if holdout.empty:
            continue
        Xh = holdout[feats].values

        for seed in SEEDS:
            # Closed classifier: trained on benign + known attacks, exactly the
            # arm that outscores the full system on known classes.
            tr = pd.concat([split.train_benign, split.train_malicious],
                           ignore_index=True)
            ytr = np.where(tr["Attack Type"].values == "Benign", "Benign",
                           tr["Attack Type"].values)
            clf = Classifier(ClassifierConfig(kind="rf", n_estimators=200,
                                              seed=seed)).fit(tr[feats].values,
                                                              ytr)
            pred, cert = clf.predict_with_certainty(Xh)
            vals, counts = np.unique(pred, return_counts=True)
            top = vals[counts.argmax()]
            rows.append({
                "held_out": cls,
                "seed": seed,
                "n_holdout": len(pred),
                # The label does not exist in a closed classifier's output.
                "zero_day_recall": 0.0,
                "called_benign": float((pred == "Benign").mean()),
                "called_some_attack": float((pred != "Benign").mean()),
                "top_label": str(top),
                "top_label_share": float(counts.max() / counts.sum()),
                "median_certainty": float(np.median(cert)),
            })
            print(f"  {cls:14s} s{seed}: benign={rows[-1]['called_benign']:.3f} "
                  f"attack={rows[-1]['called_some_attack']:.3f} "
                  f"-> {top} ({rows[-1]['top_label_share']:.2f}), "
                  f"cert={rows[-1]['median_certainty']:.3f}", flush=True)
            del clf, tr
            gc.collect()
        del split
        gc.collect()

raw = pd.DataFrame(rows)
write("rf_zeroday", raw, meta={
    "seeds": SEEDS, "partition": "validation",
    "protocol": "leave-one-class-out, identical to 03_zero_day_protocols.py",
    "note": ("A closed classifier has no Zero Day label in its output space, "
             "so its zero-day recall is 0.0000 by construction on every fold. "
             "The operational quantity is where the novel traffic goes "
             "instead, and with what confidence."),
})

if not raw.empty:
    g = raw.groupby("held_out").agg(
        n=("n_holdout", "first"),
        zero_day_recall=("zero_day_recall", "mean"),
        called_benign=("called_benign", "mean"),
        called_some_attack=("called_some_attack", "mean"),
        median_certainty=("median_certainty", "mean"),
        top_label=("top_label", lambda x: x.mode().iloc[0])).reset_index()
    print()
    print(g.round(4).to_string(index=False))
    print(f"\nzero-day recall, every fold: {raw.zero_day_recall.max():.4f}")
    print(f"mean share misrouted to a known ATTACK label: "
          f"{raw.called_some_attack.mean():.4f}")
    print(f"median certainty on novel traffic: "
          f"{raw.median_certainty.mean():.4f}")

print("\n18-OK")
