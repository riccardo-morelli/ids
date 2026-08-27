# Parallel multi-stage NIDS — replication package

A one-class SVM trained on benign traffic and a random forest trained on
malicious traffic run in parallel; a rule table fuses their outputs into a
benign / known-attack / zero-day verdict. This repository holds everything
needed to reproduce the figures reported in the paper, and nothing else.

## Run it

```bash
uv venv --python 3.10
uv pip install -r revision/requirements-pinned.txt
uv pip install -e .

python src/fetch_ids2017.py     # verifies SHA-256, unpacks into data/raw/
python src/fetch_ids2018.py

bash revision/run_all.sh            # ~2.5 h, validation only
bash revision/run_all.sh --final    # ... and the three test-set evaluations
```

Every reported number comes from a CSV in `revision/results/`, and every CSV
from a numbered script in `revision/scripts/`. The CSVs produced by the runs
behind the paper are committed, so a figure can be checked without re-running
anything; each has a `.meta.json` recording the package versions, the machine
and the git commit it was produced on.

## Which script produces what

| Script | Output | Reported in |
|---|---|---|
| `01_prepare_data.py` | `data_provenance.csv` | class distribution and partition sizes |
| `02_train_baselines.py` | `baselines_cic-ids2017*.csv` | validation figures, and the paired predictions the significance tests consume |
| `03_zero_day_protocols.py` | `zeroday_{class,variant,summary}.csv` | zero-day under both definitions, every family withheld in turn |
| `04_competitor_reproduction.py` | `competitor_reproduction.csv` | our re-run of the baseline's own three configurations against their published values |
| `05_second_dataset.py` | `second_dataset_*.csv` | CSE-CIC-IDS2018 |
| `06_significance.py` | `significance.csv` | McNemar, Wilcoxon, bootstrap |
| `07_latency.py` | `latency.csv`, `latency_tradeoff.csv` | inference time; **run alone on an idle machine** |
| `08_ablations.py` | `ablation_*.csv` | stage-wise false positives, imbalance handling, the τ₁ sweep |
| `09_determinism_check.py` | `determinism_check.csv` | same seed, same numbers |
| `10_final_test.py` | `final_test*.csv` | **the test-set evaluations**; each run is logged, see below |
| `12`, `13`, `14` | `tune_*.csv`, `equal_budget.csv` | tuning, and the equal-budget fairness check |
| `16_final_config.py` | `final_config*.csv` | the adopted configuration and its training cost |
| `18_rf_zeroday.py` | `rf_zeroday.csv` | a closed classifier scores 0.0000 on every withheld family |
| `20_stage2_search.py` | `stage2_search.csv`, `rule_candidates.csv` | the rule table |
| `22_figures.py` | `figure_counts.csv` | the confusion-matrix cell counts |
| `23_baseline_fidelity.py` | `baseline_fidelity.csv` | inferred constants → the authors' published constants |
| `25`, `27` | `dedup_effect.csv`, `dedup_leakage.csv` | what exact-row deduplication costs, and the leakage avoiding it would cause |
| `26_sota_components.py` | `sota_components.csv` | six detector/classifier pairs |
| `31_stage_metrics.py` | `stage_metrics.csv` | the per-stage tables and their fit/score cost |

## Test-set discipline

Selection happens on validation only. Reading the test partition goes through
a guard that appends to `results/test_set_ledger.jsonl` *before* the data are
touched, so the number of evaluations is auditable after the fact rather than
remembered. Superseded entries are annotated, never deleted.

Three configurations are reported and each is a separate logged spend:
`smote-only` (the headline figures), `submitted` (the downsampling
configuration it is compared against) and `updated` (τ₃ at the 99.5th
percentile).

## Datasets

Not committed — the two CIC-IDS2017 archives are 519 MB, over GitHub's file
limit. `src/fetch_ids2017.py` downloads them, verifies the SHA-256 of the
exact bytes every reported figure was computed from, and unpacks them where
the loader reads. `Data/MANIFEST.md` records the provenance of both datasets,
including the truncation defect affecting seven of the ten CSE-CIC-IDS2018
daily files.

## Layout

| | |
|---|---|
| `nids/` | the pipeline package: data, stages, evaluation |
| `revision/scripts/` | one numbered script per reported experiment |
| `revision/results/` | their outputs, with provenance metadata |
| `src/` | dataset fetchers, checksum-verified |
| `results/` | the test-set access ledger |
