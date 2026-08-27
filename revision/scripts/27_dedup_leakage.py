"""Why our Port Scan recall sits below the competitor's published figure.

PRODUCES
    revision/results/dedup_leakage.csv

FEEDS
    The reproduction gap in the fairness section, which the manuscript
    currently reports as unexplained after five refuted hypotheses.

WHY THIS EXISTS
    Our cleaning drops rows identical across every feature. Port Scan pays for
    that far more than any other class: 158,930 raw flows become 1,956, a 1.2%
    survival rate against 55-100% elsewhere. A port scan is near-degenerate by
    construction - a few packets, no payload, differing only in the
    destination port, which is dropped as an identifier - so scans of adjacent
    ports collapse into one row.

    Port Scan is also the class carrying our largest deficit against the
    competitor's published per-class recalls: 0.8091 against their 0.9914.

    It is tempting to conclude that deduplication is a mistake and to remove
    it. Measured on validation, removing it lifts Port Scan recall from 0.8687
    to 0.9854 and balanced accuracy from 0.8884 to 0.9099, which looks like a
    clear improvement and closes almost the whole gap.

    IT IS LEAKAGE, AND THIS SCRIPT IS THE PROOF. Without deduplication, 95.2%
    of the Port Scan rows in the test partition appear identically, feature for
    feature, in the training partition. The model is not detecting them; it has
    memorised them. Deduplication is the defence against precisely this, and it
    stays.

    That reverses the reading of the gap rather than closing it. The question
    is no longer why our number is low but why theirs is high, and their own
    published test set answers it: 387 of its 584 Port Scan rows (66.3%) are
    exact duplicates of one another. A test partition built without
    deduplication rewards memorisation, and their figure includes what ours
    excludes.

    Three quantities, each measured rather than argued:
      A. per-class survival under deduplication, from the raw CSVs
      B. train/test duplicate overlap with deduplication disabled
      C. duplicate rate inside the competitor's own published test set

    VALIDATION AND PUBLIC ARTEFACTS ONLY. The frozen test partition is read
    only for its row identities, which were already read under earlier ledger
    entries; no metric is computed on it and no new entry is written.

USAGE
    python revision/scripts/27_dedup_leakage.py

RUNTIME  ~6 min
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_clean_any, timed, write  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nids import config  # noqa: E402
from nids.data import prepare  # noqa: E402

# Their published test partition, if it has been fetched. Optional: the script
# reports what it can without it.
VK = Path("C:/Users/ricca/AppData/Local/Temp/vk")

rows = []
with timed("27_dedup_leakage"):
    # ---- A. what deduplication costs, per class ------------------------
    ded = load_clean_any("cic-ids2017", drop_duplicates=True)
    raw = load_clean_any("cic-ids2017", drop_duplicates=False)
    c_ded = ded.frame["Label"].value_counts()
    c_raw = raw.frame["Label"].value_counts()
    print("=== A. survival under deduplication ===")
    for lab in c_raw.index:
        a, b = int(c_raw[lab]), int(c_ded.get(lab, 0))
        rows.append({"measure": "survival", "key": lab, "before": a,
                     "after": b, "value": b / a if a else np.nan})
        print(f"  {lab:28s} {a:>9,} -> {b:>9,}  {100*b/a:5.1f}%")

    # ---- B. leakage when deduplication is disabled ---------------------
    split = prepare.make_split(raw, seed=config.SPLIT_SEED,
                               imbalance="downsample")
    feats = split.feature_cols

    def sig(df):
        return pd.util.hash_pandas_object(
            df[feats].round(6).astype(str).agg("|".join, axis=1), index=False)

    train = pd.concat([split.train_benign, split.train_malicious],
                      ignore_index=True)
    seen = set(sig(train))
    test = split.test
    overlap = sig(test).isin(seen)

    print("\n=== B. test rows duplicated into train, dedup OFF ===")
    for c in sorted(test["Attack Type"].unique()):
        m = (test["Attack Type"] == c).values
        frac = float(overlap.values[m].mean())
        rows.append({"measure": "train_test_overlap", "key": c,
                     "before": int(m.sum()), "after": int(overlap.values[m].sum()),
                     "value": frac})
        print(f"  {c:14s} {int(m.sum()):>6,} rows  {100*frac:5.1f}% in train")
    del raw, split, train
    import gc
    gc.collect()

    # ---- C. duplicates inside their published test set -----------------
    tp, xp = VK / "data" / "test.parquet", VK / "data" / "test_x.parquet"
    if tp.exists() and xp.exists():
        d, x = pd.read_parquet(tp), pd.read_parquet(xp)
        print("\n=== C. duplicates inside the competitor's own test set ===")
        for c in sorted(d["Y"].unique()):
            m = (d["Y"] == c).values
            if not m.sum():
                continue
            frac = float(x[m].duplicated().mean())
            rows.append({"measure": "their_test_internal_dup", "key": c,
                         "before": int(m.sum()),
                         "after": int(x[m].duplicated().sum()), "value": frac})
            print(f"  {c:14s} {int(m.sum()):>6,} rows  {100*frac:5.1f}% duplicate")
    else:
        print(f"\n=== C. skipped: {tp} not present ===")
        print("    clone gitlab.ilabt.imec.be/mverkerk/multi-stage-hierarchical-ids")

write("dedup_leakage", pd.DataFrame(rows), meta={
    "partition": ("VALIDATION plus row identities of the frozen test "
                  "partition, already read under ledger entries 1-7. No "
                  "metric computed on test; no new ledger entry."),
    "competitor_artefacts": str(VK),
    "finding": ("Removing deduplication lifts Port Scan recall from 0.8687 to "
                "0.9854, but 95.2% of test Port Scan rows then appear "
                "identically in train. Their published test set is 66.3% "
                "duplicate on the same class."),
})
print("\n27-OK")
