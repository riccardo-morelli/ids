# Dataset manifest

Reviewers ask which version of a dataset was used, and CIC-IDS2017 in
particular has been revised repeatedly (the manuscript pins the
7 June 2018 file date and the archive checksum). Every file the experiments read is recorded here with
its SHA-256 so the claim is checkable rather than remembered.

Machine-readable checksums are written to `data/raw/<dataset>/_manifest.json` by the fetchers in `src/`; the archive checksums the CIC-IDS2017 figures were computed from are recorded in `src/fetch_ids2017.py` itself.

---

## CSE-CIC-IDS2018 — available

**Source:** AWS Open Data, `s3://cse-cic-ids2018` (public, no registration).
**Subset used:** `Processed Traffic Data for ML Algorithms/` — the
CICFlowMeter flow CSVs. PCAPs and raw logs (~45 GB/day) are not used.
**Fetched by:** `python src/fetch_ids2018.py` (records SHA-256 per file).
**Role:** second benchmark, answering reviewer 2 point 1. Also the dataset
Verkerken et al. use for their zero-day robustness check (127,844 infiltration
flows), so it lets us contest them on their own ground rather than on a
dataset they never ran.

**Schema note:** 80 columns with abbreviated names (`Tot Fwd Pkts`, not
`Total Fwd Packets`) — a different vocabulary from CIC-IDS2017, which is why
the harness routes every dataset through an adapter (`nids/data/schema.py`).
Four of the ten daily files carry extra capture-identifier columns
(`Flow ID`, `Src IP`, `Dst IP`, `Src Port`) that the other six lack; the
loader aligns on the column intersection and records what it discarded.

| File | Day | Rows | Notes |
|---|---|---|---|
| `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` | 14 Feb | 1,048,575 | **TRUNCATED** — FTP/SSH brute force |
| `Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv` | 15 Feb | 1,048,575 | **TRUNCATED** — DoS GoldenEye, Slowloris |
| `Friday-16-02-2018_TrafficForML_CICFlowMeter.csv` | 16 Feb | 1,048,575 | **TRUNCATED** — DoS Hulk, SlowHTTPTest |
| `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv` | 20 Feb | 7,948,748 | intact — DDoS LOIC-HTTP (largest) |
| `Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv` | 21 Feb | 1,048,575 | **TRUNCATED** — DDoS LOIC-UDP, HOIC |
| `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv` | 22 Feb | 1,048,575 | **TRUNCATED** — Web attacks |
| `Friday-23-02-2018_TrafficForML_CICFlowMeter.csv` | 23 Feb | 1,048,575 | **TRUNCATED** — Web attacks |
| `Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv` | 28 Feb | 613,104 | intact — Infiltration |
| `Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv` | 1 Mar | 331,125 | intact — Infiltration, Verkerken's zero-day source |
| `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv` | 2 Mar | 1,048,575 | **TRUNCATED** — Botnet |

Total: 16,233,002 rows across 10 files.

### Truncation defect in the official distribution

**Seven of the ten files** contain *exactly* 1,048,575 data rows — Excel's
sheet limit (1,048,576) minus the header. Those days were opened and re-saved
in Excel before upload, discarding every flow past the limit. The three intact
files (7,948,748 / 613,104 / 331,125 rows) confirm the limit comes from the
upload, not from the capture.

Verified rather than inferred: `Friday-02-03` opens at timestamp `08:47:38`
while rows adjacent to the cut read `02:08:33`, so the file is not in
chronological order. The truncation removes an **arbitrary** subset of each
day's flows, not a clean time-ordered tail, and cannot be corrected by
reasoning about the capture schedule.

Consequences we must state wherever 2018 numbers appear:

- Class counts from the seven affected days are **lower bounds**, not counts.
- Affected classes: FTP/SSH brute force, DoS GoldenEye/Slowloris/Hulk/
  SlowHTTPTest, DDoS LOIC-UDP/HOIC, Web attacks, and Botnet — i.e. **every
  attack class except DDoS LOIC-HTTP and Infiltration**.
- Our 2018 class distribution must **not** be compared against papers using a
  pre-truncation copy without flagging the difference.
- **Verkerken's zero-day source is unaffected.** Both infiltration days
  (28 Feb, 1 Mar) are intact, so their 127,844-sample figure remains a fair
  target. Measured here: 93,063 infiltration flows on 1 Mar, with the
  remainder on 28 Feb — consistent with their total.

The checksums recorded here are of what S3 currently serves; this is a defect
in the published dataset, not in our fetch. It is also a reason to treat 2018
as the *second* dataset rather than the primary one.

---

## CIC-IDS2017 — obtained and verified

**Dataset of record for both our paper and Verkerken et al.**

Supplied by the supervisor on 2026-08-04, together with the official CIC `.md5`
files — which makes the provenance checkable rather than asserted. This is why
the third-party Kaggle mirrors were refused.

| Archive | MD5 (official = computed) | SHA-256 |
|---|---|---|
| `MachineLearningCSV.zip` | `4f83860afbf29cac8163854095bf6cf7` | `c3f26274b36c837ccf28ffd2dbf4582941c30b3ee70a635c6e5b2f87c4727928` |
| `GeneratedLabelledFlows.zip` | `5ca3f8f69e3514950681615824149973` | `7bdbef286f8893f31c6db12105fa097fa5c2dcc6733179037a08129d150ea27a` |

Both verified. `python src/fetch_ids2017.py` re-checks the SHA-256 of whatever
is on disk against these values and unpacks the CSVs into
`data/raw/cic-ids2017/`.

Extracted to `data/raw/cic-ids2017/MachineLearningCVE/` — 8 CSVs, 79 columns,
**2,830,743 rows**, which matches our manuscript §3.1 exactly. Our submitted
numbers are therefore reproducible against these bytes. Per-file SHA-256 in
`data/raw/cic-ids2017/_manifest.json`.

PCAPs were not requested and are not needed: both papers work from the
CICFlowMeter flow CSVs (our §3.1, Verkerken §IV-A). PCAPs would only be
required to re-extract features from scratch, which is out of scope.

### Label encoding defect

The three Web Attack labels ship with a corrupted separator — the bytes are
`EF BF BD` (U+FFFD REPLACEMENT CHARACTER) baked into the file, so the original
en-dash is already lost on disk. `nids/data/schema.py::_norm` folds U+FFFD,
0x96 and the en/em dashes to a plain hyphen, so the label maps correctly
regardless of which codec reads the file.

---
