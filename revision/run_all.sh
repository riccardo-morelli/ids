#!/usr/bin/env bash
# Reproduce every number in the revision, end to end.
#
#   bash revision/run_all.sh            # everything except the test-set spend
#   bash revision/run_all.sh --final    # ... and the single test evaluation
#
# Run from the repository root. Total runtime ~2.5 h on the reference machine
# (see README.md). Scripts are serialised deliberately: 07_latency.py must not
# share the machine with any other job, and the larger runs compete for memory.
set -euo pipefail

cd "$(dirname "$0")/.."
S=revision/scripts
mkdir -p revision/results

echo "=============================================================="
echo " NIDS revision - full reproduction"
echo " started $(date)"
echo "=============================================================="

run () {                     # run <label> <script...>
  local label="$1"; shift
  echo
  echo "--------------------------------------------------------------"
  echo ">> $label"
  echo "--------------------------------------------------------------"
  python "$@" 2>&1 | tee "revision/results/log_${label}.txt"
}

# ---- data ---------------------------------------------------------------
run 01 "$S/01_prepare_data.py"

# ---- baselines and the blocking experiments -----------------------------
# R2.3: every system, one protocol, 5 seeds.
run 02 "$S/02_train_baselines.py" --dataset cic-ids2017 --seeds 0,1,2,3,4

# R2.2 (BLOCKING): zero-day under both definitions, every class withheld.
run 03 "$S/03_zero_day_protocols.py" --seeds 0,1,2,3,4

# R2.3 + finding A1: the competitor's own Table V, re-run here.
run 04 "$S/04_competitor_reproduction.py" --seeds 0,1,2

# R2.1 (BLOCKING): the second benchmark.
run 05 "$S/05_second_dataset.py" --seeds 0,1,2

# R2.3 (BLOCKING): McNemar, Wilcoxon, bootstrap.
run 06 "$S/06_significance.py" --n-boot 1000

# ---- latency: MUST run alone on an idle machine -------------------------
echo
echo ">> 07 latency - close other applications; this measurement is serialised"
run 07 "$S/07_latency.py" --reps 30 --threads 1

# ---- ablations ----------------------------------------------------------
# R2.4 (stagewise FPs), R2.6 (imbalance incl. SMOTE), R2.5 (tau_1 sensitivity).
run 08 "$S/08_ablations.py" --seeds 0,1,2

# ---- exploitation within the frozen architecture -------------------------
# R2.5/R2.6: what the current architecture can reach when tuned.
run 12 "$S/12_tune_classifier.py"
run 13 "$S/13_tune_detector.py" --seeds 0,1,2
# Fairness: the competitor gets the same search space and budget.
run 14 "$S/14_equal_budget_competitor.py" --trials 16 --seeds 0,1,2
# The two adopted changes, measured together. SMOTE arms are memory-heavy:
# one seed per process on a 16 GB machine.
for SEED in 0 1 2 3 4; do
  python "$S/16_final_config.py" --arms A,B,C,D --seeds "$SEED"     2>&1 | tee -a revision/results/log_16.txt || true
done
run 16 "$S/16_final_config.py" --arms A,B,C,D

# ---- rule-table search (Stage 2) ----------------------------------------
run 20 "$S/20_stage2_search.py"

# ---- the fairness section's evidence ------------------------------------
run 23 "$S/23_baseline_fidelity.py"     # the ladder from inferred to published
run 25 "$S/25_dedup_effect.py"          # Port Scan 0.8687 -> 0.9854
run 27 "$S/27_dedup_leakage.py"         # 95.2% / 66.3% / 98.8%

# ---- component sweep and the closed classifier --------------------------
run 26 "$S/26_sota_components.py"       # six detector/classifier pairs
run 18 "$S/18_rf_zeroday.py"            # closed classifier, zero-day 0.0000

# ---- stage tables and figures -------------------------------------------
run 31 "$S/31_stage_metrics.py"         # Tables 3 and 4
run 22 "$S/22_figures.py"               # Figures 2 and 3

# ---- determinism check --------------------------------------------------
run 09 "$S/09_determinism_check.py"

# ---- the single test-set spend, last and only on request ----------------
if [[ "${1:-}" == "--final" ]]; then
  echo
  echo ">> 10 FINAL TEST - this spends the frozen test set and is logged"
  # Three configurations are reported, and each is a separate logged spend:
  #   smote-only  Table 5 and every headline figure (final_test_smote-only.csv)
  #   submitted   the downsampling baseline the +0.047 in Section 3.2 is against
  #   updated     tau_u at the 99.5th percentile, the trade-off in Section 5.1
  for CONFIG in smote-only submitted updated; do
    run "10_${CONFIG}" "$S/10_final_test.py" --authorise --seeds 0,1,2,3,4         --config "$CONFIG"
  done
else
  echo
  echo ">> 10 final test SKIPPED. Re-run with --final when the configuration"
  echo "   is settled. Each run is a logged, irreversible test-set spend."
fi

echo
echo "=============================================================="
echo " done $(date)"
echo " results: revision/results/*.csv  logs: revision/results/log_*.txt"
echo "=============================================================="
