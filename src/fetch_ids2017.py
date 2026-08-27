"""Fetch the CIC-IDS2017 archives and verify them against recorded checksums.

The two archives total 519 MB, which exceeds GitHub's per-file limit, so they
are not committed. They are recoverable instead: the SHA-256 of the exact
bytes the experiments ran on is recorded below, and this script downloads and
verifies them.

Both come from the same capture. MachineLearningCSV is the release the
experiments use (78 features, no identifiers). GeneratedLabelledFlows adds
source/destination IP, ports and timestamps; the experiments do not need it,
and it is listed here only because the manifest records it.

USAGE
    python src/fetch_ids2017.py [--dest data/raw/cic-ids2017]

If the university mirror is unreachable (it has moved more than once), the
manifest names the landing page to fetch them from by hand; drop the archives
in the destination directory and re-run to verify.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

# The bytes every reported figure was computed from.
ARCHIVES = {
    "MachineLearningCSV.zip":
        "c3f26274b36c837ccf28ffd2dbf4582941c30b3ee70a635c6e5b2f87c4727928",
    "GeneratedLabelledFlows.zip":
        "7bdbef286f8893f31c6db12105fa097fa5c2dcc6733179037a08129d150ea27a",
}

BASE = "http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/"
LANDING = "https://www.unb.ca/cic/datasets/ids-2017.html"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    # The loader reads data/raw/<dataset>/ (nids.config.DATA_RAW), so that
    # is where the archives are fetched and unpacked.
    ap.add_argument("--dest", default="data/raw/cic-ids2017")
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    failed = []
    for name, want in ARCHIVES.items():
        path = dest / name
        if not path.exists():
            url = BASE + name
            print(f"downloading {name} ...", flush=True)
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as exc:            # noqa: BLE001 - report, continue
                print(f"  FAILED: {exc}")
                print(f"  fetch it by hand from {LANDING} into {dest}/")
                failed.append(name)
                continue
        got = sha256(path)
        ok = got == want
        print(f"{'OK  ' if ok else 'BAD '} {name}  {got[:16]}...")
        if not ok:
            print(f"     expected {want[:16]}... - the file is not the one used here")
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} archive(s) missing or mismatched: {', '.join(failed)}")
        return 1

    # Unpack the release the experiments read. Extraction is skipped when
    # the CSVs are already there, so re-running never overwrites data.
    csvs = sorted(dest.rglob("*.pcap_ISCX.csv"))
    if csvs:
        print(f"{len(csvs)} CSVs already extracted under {dest}/")
    else:
        print(f"extracting MachineLearningCSV.zip into {dest}/ ...")
        with zipfile.ZipFile(dest / "MachineLearningCSV.zip") as z:
            z.extractall(dest)
        csvs = sorted(dest.rglob("*.pcap_ISCX.csv"))
        print(f"  {len(csvs)} CSVs extracted")

    if len(csvs) != 8:
        print(f"expected 8 CSVs, found {len(csvs)} - this is not the "
              "MachineLearningCSV release the experiments use")
        return 1
    print("All archives verified and extracted. "
          "Run: bash revision/run_all.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
