"""Parquet cache for cleaned datasets.

Parsing 2.8M rows of CSV takes ~2 minutes; the detector experiments need to
reload the data hundreds of times. Caching the *cleaned* frame turns that into
a couple of seconds.

The cache key includes the cleaning options, so a change to the cleaning path
produces a different file rather than silently serving a stale frame — the
failure mode this module has to avoid is an experiment reading data that no
longer matches the code that produced it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from nids import config
from nids.data import prepare, schema


def _key(name: str, nrows: int | None, drop_duplicates: bool) -> str:
    blob = json.dumps({"name": name, "nrows": nrows,
                       "drop_duplicates": drop_duplicates}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_clean(
    name: str = "cic-ids2017",
    *,
    nrows: int | None = None,
    drop_duplicates: bool = True,
    refresh: bool = False,
) -> schema.Dataset:
    """Load a dataset, cleaned, from cache when possible."""
    key = _key(name, nrows, drop_duplicates)
    path = config.DATA_INTERIM / f"{name}-clean-{key}.parquet"
    meta_path = path.with_suffix(".meta.json")

    if path.exists() and meta_path.exists() and not refresh:
        frame = pd.read_parquet(path)
        meta = json.loads(meta_path.read_text())
        return schema.Dataset(
            name=name, frame=frame, feature_cols=meta["feature_cols"],
            zero_day_labels=tuple(meta["zero_day_labels"]),
            provenance=meta["provenance"],
        )

    ds = prepare.clean(schema.load(name, nrows=nrows),
                       drop_duplicates=drop_duplicates)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.frame.to_parquet(path, index=False)
    meta_path.write_text(json.dumps({
        "feature_cols": ds.feature_cols,
        "zero_day_labels": list(ds.zero_day_labels),
        "provenance": ds.provenance,
    }, indent=2, default=str))
    return ds
