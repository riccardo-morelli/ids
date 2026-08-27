"""Dataset adapters.

Each dataset arrives with its own column naming and its own label vocabulary.
An adapter's job is to turn raw CSVs into one canonical frame:

    float32 feature columns + a single 'Attack Type' column drawn from
    config.CANONICAL_CLASSES, plus 'Label' holding the original fine-grained
    label (needed for leave-one-class-out and for zero-day designation).

Keeping the datasets behind one interface is what lets Phase A, Phase B and
every Phase C experiment run through a single harness, which the brief
requires: "Any result produced outside that harness does not count."

Note on CIC-IDS2017 column names: the MachineLearningCSV release ships names
with a leading space (' Label', ' Destination Port', ...). We normalise
whitespace on load rather than hard-coding either variant, because the
2024-02-01 revision named in our manuscript is not byte-identical to older
mirrors.
"""
from __future__ import annotations

import glob
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from nids import config


def _norm(name: str) -> str:
    """Collapse whitespace and unify dash-like characters in a column/label.

    CIC-IDS2017 ships 'Web Attack - Brute Force' with a corrupted separator:
    the bytes are EF BF BD, i.e. U+FFFD REPLACEMENT CHARACTER baked into the
    file itself (the release was written through a broken encoding conversion,
    so the original en-dash is already lost on disk). Depending on the codec
    used to read it, the same label surfaces as U+FFFD, as 0x96, or as an
    en-dash - so all of them are folded to a plain hyphen here rather than
    being handled at the call site.
    """
    s = str(name).replace("﻿", "")
    # Read as latin-1 (which never fails), the three UTF-8 bytes EF BF BD
    # arrive as three separate characters 'ï¿½' rather than as one U+FFFD, so
    # the multi-character form has to be folded first.
    s = s.replace("ï¿½", "-")
    for ch in ("�", "–", "—", "\x96"):
        s = s.replace(ch, "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass
class Dataset:
    """A loaded, canonicalised dataset."""

    name: str
    frame: pd.DataFrame
    feature_cols: list[str]
    #: Fine-grained labels considered zero-day candidates for this dataset.
    zero_day_labels: tuple[str, ...]
    provenance: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frame)


class Adapter:
    """Base adapter. Subclasses declare naming and label mapping only."""

    name: str = "base"
    label_col: str = "Label"
    #: Columns carrying identifiers/timestamps rather than behaviour. Dropped
    #: because they leak the capture setup rather than generalising.
    identifier_cols: tuple[str, ...] = ()
    attack_map: dict[str, str] = {}
    zero_day_labels: tuple[str, ...] = ()

    def raw_files(self) -> list[Path]:
        raise NotImplementedError

    def load(self, nrows: int | None = None) -> Dataset:
        files = self.raw_files()
        if not files:
            raise FileNotFoundError(
                f"No raw files for '{self.name}'. Expected under "
                f"{config.DATA_RAW / self.name}. See data/MANIFEST.md."
            )
        frames = []
        for f in sorted(files):
            df = pd.read_csv(f, low_memory=False, nrows=nrows, encoding="latin-1")
            df.columns = [_norm(c) for c in df.columns]
            frames.append(df)

        # Daily captures in a release do not always share a schema (CSE-CIC-
        # IDS2018 ships four files with extra identifier columns). Concatenating
        # the union would manufacture all-NaN columns for the files that lack
        # them, and a later dropna would then delete the entire dataset. Align
        # on the intersection instead, and record what was discarded.
        common = set(frames[0].columns)
        for fr in frames[1:]:
            common &= set(fr.columns)
        order = [c for c in frames[0].columns if c in common]
        dropped_ragged = sorted(
            {c for fr in frames for c in fr.columns} - common
        )
        frames = [fr[order] for fr in frames]
        df = pd.concat(frames, ignore_index=True)

        label_col = _norm(self.label_col)
        if label_col not in df.columns:
            raise KeyError(
                f"'{label_col}' not among columns for {self.name}: "
                f"{list(df.columns)[:10]}..."
            )

        df[label_col] = df[label_col].map(_norm)
        df = df.rename(columns={label_col: "Label"})

        # Map fine-grained labels to the canonical coarse taxonomy.
        amap = {_norm(k): v for k, v in self.attack_map.items()}
        unmapped = sorted(set(df["Label"]) - set(amap))
        if unmapped:
            raise ValueError(
                f"{self.name}: labels absent from attack_map: {unmapped}. "
                "Refusing to silently drop or guess - a mislabelled class "
                "would corrupt every downstream number."
            )
        df["Attack Type"] = df["Label"].map(amap)

        drop = [c for c in map(_norm, self.identifier_cols) if c in df.columns]
        feature_cols = [
            c for c in df.columns if c not in {"Label", "Attack Type", *drop}
        ]

        # Coerce features to numeric; non-numeric residue is a schema surprise
        # and should fail loudly rather than become NaN silently.
        for c in feature_cols:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df[["Label", "Attack Type", *feature_cols]]
        df[feature_cols] = df[feature_cols].astype(np.float64)

        return Dataset(
            name=self.name,
            frame=df,
            feature_cols=feature_cols,
            zero_day_labels=self.zero_day_labels,
            provenance={
                "files": [str(p) for p in sorted(files)],
                "rows_raw": len(df),
                "dropped_ragged_cols": dropped_ragged,
                "dropped_identifier_cols": drop,
            },
        )


class CICIDS2017(Adapter):
    """CIC-IDS2017, MachineLearningCSV release (8 CSVs, 78 features + Label).

    This is the dataset of record for both our paper and Verkerken et al.
    """

    name = "cic-ids2017"
    identifier_cols = ("Destination Port",)
    zero_day_labels = ("Infiltration", "Heartbleed")
    attack_map = {
        "BENIGN": "Benign",
        "DDoS": "DoS",
        "DoS Hulk": "DoS",
        "DoS GoldenEye": "DoS",
        "DoS slowloris": "DoS",
        "DoS Slowhttptest": "DoS",
        "PortScan": "Port Scan",
        "FTP-Patator": "Brute Force",
        "SSH-Patator": "Brute Force",
        "Bot": "Botnet",
        "Web Attack - Brute Force": "Web Attack",
        "Web Attack - XSS": "Web Attack",
        "Web Attack - Sql Injection": "Web Attack",
        "Infiltration": "Infiltration",
        "Heartbleed": "Heartbleed",
    }

    def raw_files(self) -> list[Path]:
        base = config.DATA_RAW / self.name
        # The release nests as MachineLearningCSV/MachineLearningCVE/*.csv;
        # accept the CSVs wherever they sit under the dataset root.
        return [Path(p) for p in glob.glob(str(base / "**" / "*.csv"), recursive=True)]


class CSECICIDS2018(Adapter):
    """CSE-CIC-IDS2018, processed CICFlowMeter CSVs from the public S3 bucket.

    Abbreviated column names ('Tot Fwd Pkts' vs 'Total Fwd Packets') and an
    extra Timestamp column relative to 2017. Verkerken use this dataset's
    127,844 infiltration flows to test zero-day robustness, which is why it is
    the second dataset here: it contests them on their own ground and answers
    reviewer 2 point 1 at the same time.
    """

    name = "cse-cic-ids2018"
    # Four of the ten daily CSVs carry these extra capture-identifier columns
    # that the others lack (Flow ID / Src IP / Dst IP / Src Port). They are
    # non-numeric, so leaving them in turns every row NaN after coercion. They
    # also encode the capture setup rather than flow behaviour, so a model that
    # used them would not generalise off this testbed.
    identifier_cols = (
        "Dst Port", "Timestamp", "Protocol",
        "Flow ID", "Src IP", "Dst IP", "Src Port",
    )
    zero_day_labels = ("Infilteration",)  # sic - the CSVs spell it this way
    attack_map = {
        "Benign": "Benign",
        "Bot": "Botnet",
        "DoS attacks-Hulk": "DoS",
        "DoS attacks-GoldenEye": "DoS",
        "DoS attacks-Slowloris": "DoS",
        "DoS attacks-SlowHTTPTest": "DoS",
        "DDOS attack-HOIC": "DoS",
        "DDOS attack-LOIC-UDP": "DoS",
        "DDoS attacks-LOIC-HTTP": "DoS",
        "FTP-BruteForce": "Brute Force",
        "SSH-Bruteforce": "Brute Force",
        "Brute Force -Web": "Web Attack",
        "Brute Force -XSS": "Web Attack",
        "SQL Injection": "Web Attack",
        "Infilteration": "Infiltration",
        "Label": "DROP",  # repeated header rows appear mid-file in this release
    }

    def raw_files(self) -> list[Path]:
        base = config.DATA_RAW / self.name
        return [Path(p) for p in glob.glob(str(base / "*.csv"))]

    def load(self, nrows: int | None = None) -> Dataset:
        ds = super().load(nrows=nrows)
        # Strip the embedded header rows flagged by the 'Label' -> DROP mapping.
        keep = ds.frame["Attack Type"] != "DROP"
        ds.frame = ds.frame[keep].reset_index(drop=True)
        return ds


ADAPTERS: dict[str, type[Adapter]] = {
    CICIDS2017.name: CICIDS2017,
    CSECICIDS2018.name: CSECICIDS2018,
}


def load(name: str, nrows: int | None = None) -> Dataset:
    if name not in ADAPTERS:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(ADAPTERS)}")
    return ADAPTERS[name]().load(nrows=nrows)
