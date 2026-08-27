"""Cleaning, imbalance handling, and the ONE frozen split.

Two things in this module are load-bearing for the revision's credibility.

**The split happens once.** `make_split` is a pure function of
`config.SPLIT_SEED` and the cleaned frame. The test partition it returns is
never re-derived with a different seed, never inspected during development,
and never used to select anything. Selection happens on `val` only.

**Validation and training never overlap.** The legacy code concatenated the
malicious *training* set into the malicious *validation* set before sweeping
for the detector threshold tau_1
(`legacy/.../detector_ocsvm.ipynb` cell 5), so the threshold was chosen partly
on data the model had trained beside. `make_split` returns disjoint frames and
`assert_disjoint` is called at construction, so reintroducing that overlap
raises rather than silently inflating a number.

Imbalance strategy is a *parameter*, not a constant, because reviewer 2 point
6 demands the 99%-discard downsampling be compared against alternatives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from nids import config
from nids.data.schema import Dataset

ImbalanceStrategy = Literal["downsample", "weighted", "none"]


@dataclass
class Split:
    """The frozen partitioning. Constructed once per (dataset, protocol)."""

    #: Benign-only frame for training the detector (stage 1a).
    train_benign: pd.DataFrame
    #: Malicious-only frame for training the classifier (stage 1b).
    train_malicious: pd.DataFrame
    #: Benign validation pool - threshold selection, never trained on.
    val_benign: pd.DataFrame
    #: Malicious validation pool - threshold selection, never trained on.
    val_malicious: pd.DataFrame
    #: FROZEN. Only touched by a logged, budgeted test evaluation.
    test: pd.DataFrame
    feature_cols: list[str]
    seed: int
    meta: dict

    def describe(self) -> pd.DataFrame:
        rows = []
        for part in ("train_benign", "train_malicious", "val_benign",
                     "val_malicious", "test"):
            f = getattr(self, part)
            counts = f["Attack Type"].value_counts().to_dict()
            rows.append({"partition": part, "n": len(f), **counts})
        return pd.DataFrame(rows).fillna(0)


def clean(ds: Dataset, *, drop_duplicates: bool = True) -> Dataset:
    """Remove degenerate columns and rows.

    Mirrors the cleaning both papers describe, with one correction. The legacy
    notebook dropped 'Avg Fwd Segment Size' but kept its duplicate twin
    'Avg Bwd Segment Size' (cell 7); here duplicate detection is computed from
    the data rather than hard-coded, so neither twin survives by accident.
    That is exactly the "feature duplication" defect reviewer 2 cites for
    CIC-IDS2017.
    """
    df = ds.frame
    feats = list(ds.feature_cols)
    report: dict[str, list[str] | int] = {}

    # Inf -> NaN, then drop affected rows (Flow Bytes/s overflows on 0-duration
    # flows). Done before variance checks so constants are judged on real data.
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    n0 = len(df)
    df = df.dropna(subset=feats)
    report["rows_dropped_nan_inf"] = n0 - len(df)

    # Constant columns carry no signal.
    nunique = df[feats].nunique()
    constant = sorted(nunique[nunique <= 1].index)
    feats = [c for c in feats if c not in constant]
    report["dropped_constant"] = constant

    # Exactly-duplicated columns: keep the first of each duplicate group.
    dup_map: dict[str, str] = {}
    seen: dict[bytes, str] = {}
    for c in feats:
        key = pd.util.hash_pandas_object(df[c], index=False).values.tobytes()
        if key in seen:
            dup_map[c] = seen[key]
        else:
            seen[key] = c
    feats = [c for c in feats if c not in dup_map]
    report["dropped_duplicate_cols"] = dup_map

    if drop_duplicates:
        n0 = len(df)
        df = df.drop_duplicates(subset=feats + ["Label"])
        report["rows_dropped_duplicate"] = n0 - len(df)

    df = df[["Label", "Attack Type", *feats]].reset_index(drop=True)
    ds.frame = df
    ds.feature_cols = feats
    ds.provenance["clean"] = report
    return ds


def assert_disjoint(*frames: pd.DataFrame) -> None:
    """Guard against the train/validation overlap present in the legacy code."""
    seen: set[int] = set()
    for f in frames:
        ids = set(f.attrs.get("row_ids", f.index))
        if seen & ids:
            raise AssertionError(
                f"Partitions overlap on {len(seen & ids)} rows. Selection made "
                "on data the model trained beside is exactly the defect this "
                "harness exists to prevent."
            )
        seen |= ids


def make_split(
    ds: Dataset,
    *,
    seed: int = config.SPLIT_SEED,
    zero_day: tuple[str, ...] | None = None,
    imbalance: ImbalanceStrategy = "downsample",
    per_class: int | None = None,
    #: Classes with fewer instances than this are dropped from the balanced
    #: malicious pool instead of being allowed to set the downsampling target.
    #: See the note in the downsample branch.
    min_class_size: int = 100,
    n_benign_train: int = 100_000,
    n_benign_val: int = 95_551,
    benign_test_ratio: float = 0.95,
) -> Split:
    """Partition once. Pure in (ds, seed, kwargs).

    Zero-day classes are withheld from train and validation entirely and appear
    only in test - otherwise "zero-day recall" measures nothing.

    `benign_test_ratio` sets the benign share of the test set; both papers use
    a ~95% benign test stream to imitate real traffic.
    """
    rng = np.random.RandomState(seed)
    too_small: list[str] = []
    zero_day = tuple(zero_day if zero_day is not None else ds.zero_day_labels)
    df = ds.frame.copy()
    df["row_id"] = np.arange(len(df))

    zd = df[df["Label"].isin(zero_day)]
    rest = df[~df["Label"].isin(zero_day)]
    benign = rest[rest["Attack Type"] == "Benign"]
    malicious = rest[rest["Attack Type"] != "Benign"]

    # --- malicious: balance across attack types -------------------------
    if imbalance == "downsample":
        counts = malicious["Attack Type"].value_counts()
        if per_class is not None:
            n = per_class
        else:
            # Downsampling to the smallest class is only sensible while that
            # class can actually sustain a train/val/test split. CIC-IDS2017
            # holds classes with 11 and 36 instances (Heartbleed,
            # Infiltration); whenever those are NOT the withheld zero-day
            # class they land in the balanced pool, become the minimum, and
            # drag every other class down with them — a leave-one-class-out
            # fold on Web Attack ended up training the classifier on six rows
            # per class, which silently invalidated a whole comparison.
            #
            # So the target is the smallest class large enough to be split,
            # and anything below that floor is dropped from the balanced pool
            # rather than allowed to define it. Dropped classes are recorded
            # in the split metadata, never removed silently.
            eligible = counts[counts >= min_class_size]
            n = int(eligible.min()) if len(eligible) else int(counts.min())
        too_small = sorted(counts[counts < min_class_size].index)
        if too_small:
            malicious = malicious[~malicious["Attack Type"].isin(too_small)]
        parts = []
        for cls, grp in malicious.groupby("Attack Type"):
            # Stratify within the coarse class by fine label, so downsampling
            # DoS does not silently delete DoS Slowhttptest entirely.
            frac = grp["Label"].value_counts(normalize=True)
            take = (frac * n).round().astype(int)
            take.iloc[0] += n - int(take.sum())
            for lbl, k in take.items():
                sub = grp[grp["Label"] == lbl]
                parts.append(sub.sample(n=min(int(k), len(sub)), random_state=rng))
        malicious = pd.concat(parts, ignore_index=True)
    elif imbalance not in ("weighted", "none"):
        raise ValueError(f"unknown imbalance strategy {imbalance!r}")

    # --- malicious: train / val / test ----------------------------------
    mal_test = malicious.groupby("Attack Type", group_keys=False).apply(
        lambda g: g.sample(frac=config.TEST_FRACTION, random_state=rng)
    )
    mal_pool = malicious[~malicious["row_id"].isin(mal_test["row_id"])]
    # 30% of the remaining pool becomes validation (matches both papers' spirit
    # of ~300/class validation on 1006/class training).
    mal_val = mal_pool.groupby("Attack Type", group_keys=False).apply(
        lambda g: g.sample(frac=0.30, random_state=rng)
    )
    mal_train = mal_pool[~mal_pool["row_id"].isin(mal_val["row_id"])]

    # --- benign: train / val / test -------------------------------------
    n_benign_train = min(n_benign_train, len(benign))
    ben_train = benign.sample(n=n_benign_train, random_state=rng)
    remaining = benign[~benign["row_id"].isin(ben_train["row_id"])]
    n_benign_val = min(n_benign_val, len(remaining))
    ben_val = remaining.sample(n=n_benign_val, random_state=rng)
    remaining = remaining[~remaining["row_id"].isin(ben_val["row_id"])]

    # Size the benign test partition so the test stream is ~95% benign once
    # the malicious and zero-day rows are added.
    n_mal_test = len(mal_test) + len(zd)
    n_ben_test = int(round(n_mal_test * benign_test_ratio / (1 - benign_test_ratio)))
    n_ben_test = min(n_ben_test, len(remaining))
    ben_test = remaining.sample(n=n_ben_test, random_state=rng)

    test = pd.concat([mal_test, zd, ben_test], ignore_index=True)
    # Zero-day rows carry their own decision label.
    test.loc[test["Label"].isin(zero_day), "Attack Type"] = "Zero Day"

    for f, ids in (
        (ben_train, ben_train["row_id"]), (mal_train, mal_train["row_id"]),
        (ben_val, ben_val["row_id"]), (mal_val, mal_val["row_id"]),
        (test, test["row_id"]),
    ):
        f.attrs["row_ids"] = set(ids)

    assert_disjoint(ben_train, mal_train, ben_val, mal_val, test)

    return Split(
        train_benign=ben_train.reset_index(drop=True),
        train_malicious=mal_train.reset_index(drop=True),
        val_benign=ben_val.reset_index(drop=True),
        val_malicious=mal_val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        feature_cols=list(ds.feature_cols),
        seed=seed,
        meta={
            "dataset": ds.name,
            "zero_day_labels": list(zero_day),
            "imbalance": imbalance,
            "per_class": per_class,
            "benign_test_ratio": benign_test_ratio,
            "n_zero_day": int(len(zd)),
            "min_class_size": min_class_size,
            "dropped_too_small": too_small if imbalance == "downsample" else [],
        },
    )
