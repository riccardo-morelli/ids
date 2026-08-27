"""Prepare and verify both datasets; record provenance.

PRODUCES
    revision/results/data_provenance.csv   - row counts and class distribution
    revision/results/data_provenance.meta.json

FEEDS
    Response R2.1 (second dataset), R2.5 (reproducibility).
    Manuscript Table: "Dataset composition after cleaning" (Section 3.2).

WHY THIS EXISTS
    The manuscript states CIC-IDS2017 was downloaded in a "version updated
    2024-02-01". The files are dated 2018-06-07. This script records what is
    actually on disk, with hashes, so the corrected statement in the revised
    manuscript is backed by an artifact.

RUNTIME  ~3 min (both datasets, from the parquet caches)
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_clean_any, timed, write  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nids import config  # noqa: E402
from nids.data import cache, prepare  # noqa: E402


def file_digest(p: Path, limit: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
            if fh.tell() > limit:
                break
    return h.hexdigest()[:16]


rows = []
with timed("01_prepare_data"):
    for name in ("cic-ids2017", "cse-cic-ids2018"):
        try:
            ds = load_clean_any(name)
        except Exception as exc:                       # pragma: no cover
            print(f"!! {name} unavailable: {exc}")
            rows.append({"dataset": name, "status": f"UNAVAILABLE: {exc}"})
            continue

        counts = ds.frame["Attack Type"].value_counts()
        split = prepare.make_split(ds, seed=config.SPLIT_SEED)

        for cls, n in counts.items():
            rows.append({
                "dataset": name,
                "attack_type": cls,
                "n_after_cleaning": int(n),
                "status": "ok",
            })
        rows.append({
            "dataset": name, "attack_type": "TOTAL",
            "n_after_cleaning": int(len(ds.frame)), "status": "ok",
        })
        for part in ("train_benign", "train_malicious", "val_benign",
                     "val_malicious", "test"):
            rows.append({
                "dataset": name,
                "attack_type": f"[partition] {part}",
                "n_after_cleaning": int(len(getattr(split, part))),
                "status": "ok",
            })
        print(f"{name}: {len(ds.frame):,} rows, "
              f"{len(ds.feature_cols)} features")
        del ds, split

df = pd.DataFrame(rows)

raw = config.ROOT / "Data" / "raw"
hashes = {}
for sub in sorted(raw.glob("*")):
    if sub.is_dir():
        for f in sorted(sub.rglob("*.csv"))[:12]:
            hashes[str(f.relative_to(config.ROOT))] = file_digest(f)

write("data_provenance", df, meta={
    "split_seed": config.SPLIT_SEED,
    "test_fraction": config.TEST_FRACTION,
    "default_zero_day": config.DEFAULT_ZERO_DAY,
    "source_urls": {
        "cic-ids2017": "http://cicresearch.ca/CICDataset/CIC-IDS-2017/",
        "cse-cic-ids2018": "s3://cse-cic-ids2018/ (AWS Open Data)",
    },
    "file_sha256_prefix": hashes,
    "note": ("Manuscript section 3.1 states the CIC-IDS2017 files were "
             "'updated 2024-02-01'. The files on disk are dated 2018-06-07. "
             "The manuscript statement is incorrect and is corrected in the "
             "revision."),
})
print(df.to_string(index=False))
print("\n01-OK")
