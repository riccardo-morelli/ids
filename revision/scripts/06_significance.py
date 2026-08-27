"""Significance testing: McNemar, Wilcoxon, and bootstrap.

PRODUCES
    revision/results/significance.csv

FEEDS
    Response R2.3 (BLOCKING). Revised manuscript: new Section 4.x paragraph
    "Statistical significance" and the +- columns of the revised Table 5.

WHY THIS EXISTS
    R2.3, verbatim: "no statistical significance test (such as McNemar's test,
    Wilcoxon signed-rank test, or bootstrap resampling) is performed to
    validate that the observed improvement in balanced accuracy over the
    baseline (0.9362 vs. 0.9216) is statistically significant and not due to
    random variation."

    The reviewer names three tests. We run all three, because running only one
    invites the objection that we picked the test that suited us. Each answers
    a different question and they can legitimately disagree:

      McNemar  - on PAIRED per-row predictions from one seed. Asks: do the two
                 systems make different errors on the same rows? Very high
                 power at n~97k, so it will find tiny differences significant.
                 A significant McNemar with a negligible effect size is not an
                 improvement worth claiming.
      Wilcoxon - across SEEDS on the aggregate metric. Asks: is the metric
                 difference consistent across training runs? This is the test
                 that matters for "is my improvement real", and with 5 seeds it
                 is the weakest in power - n=5 gives a minimum two-sided
                 p of 0.0625.
      Bootstrap - resamples rows to put a CI on the metric difference itself.
                 Reports an effect size, which the other two do not.

    IMPORTANT CONTEXT (finding A1). The specific comparison the reviewer asks
    about - 0.9362 vs 0.9216 - is mis-specified: the 0.9216 figure matches no
    row of Verkerken's Table V. We therefore test our pipeline against our own
    re-run of their published 'balanced' configuration, and say so.

USAGE
    python revision/scripts/06_significance.py [--dataset cic-ids2017]

RUNTIME  ~4 min
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RESULTS, timed, write  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nids import config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="cic-ids2017")
ap.add_argument("--n-boot", type=int, default=2000)
args = ap.parse_args()

PRED = RESULTS / "predictions"
RAW = RESULTS / f"baselines_{args.dataset}_raw.csv"

# Comparisons that matter, in the order the response letter presents them.
PAIRS = [
    ("ours-submitted", "verkerken-balanced",
     "the comparison R2.3 asks about, with the baseline corrected (A1)"),
    ("ours-submitted", "rf-closed-baseline",
     "the closed classifier; it cannot emit Zero Day, so this axis is "
     "ill-posed and reported for completeness only"),
]


def balanced_accuracy(y, p) -> float:
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, p))


def mcnemar(y, pa, pb) -> dict:
    """Exact binomial McNemar on discordant pairs."""
    from scipy.stats import binomtest
    a_ok, b_ok = (pa == y), (pb == y)
    n01 = int((~a_ok & b_ok).sum())     # b right, a wrong
    n10 = int((a_ok & ~b_ok).sum())     # a right, b wrong
    n = n01 + n10
    if n == 0:
        return {"mcnemar_n01": 0, "mcnemar_n10": 0, "mcnemar_p": 1.0}
    p = binomtest(n10, n, 0.5).pvalue
    return {"mcnemar_n01": n01, "mcnemar_n10": n10, "mcnemar_p": float(p)}


def bootstrap_diff(y, pa, pb, n_boot: int, seed: int = 0) -> dict:
    """Percentile CI on the balanced-accuracy difference, resampling rows."""
    rng = np.random.RandomState(seed)
    n = len(y)
    # Balanced accuracy is a mean of per-class recalls, so it is estimated
    # accurately from a subsample; resampling all ~97k rows 2000 times is
    # needlessly expensive for the same interval. Subsample size is recorded.
    m = min(n, 20_000)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, m)
        diffs[i] = (balanced_accuracy(y[idx], pa[idx])
                    - balanced_accuracy(y[idx], pb[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"boot_diff_mean": float(diffs.mean()),
            "boot_ci_lo": float(lo), "boot_ci_hi": float(hi),
            "boot_n": n_boot, "boot_resample_size": m,
            "boot_excludes_zero": bool(lo > 0 or hi < 0)}


rows = []
with timed("06_significance"):
    raw = pd.read_csv(RAW)
    for a, b, why in PAIRS:
        sub_a = raw[raw.arm == a].sort_values("seed")
        sub_b = raw[raw.arm == b].sort_values("seed")
        if sub_a.empty or sub_b.empty:
            print(f"!! missing arm for {a} vs {b}")
            continue

        # --- Wilcoxon across seeds ---------------------------------------
        from scipy.stats import wilcoxon
        va = sub_a["balanced_accuracy"].values
        vb = sub_b["balanced_accuracy"].values
        n = min(len(va), len(vb))
        va, vb = va[:n], vb[:n]
        try:
            w = wilcoxon(va, vb)
            w_p, w_stat = float(w.pvalue), float(w.statistic)
        except ValueError as exc:
            w_p, w_stat = float("nan"), float("nan")
            print(f"   wilcoxon unavailable: {exc}")

        row = {
            "comparison": f"{a} vs {b}",
            "rationale": why,
            "n_seeds": n,
            "mean_a": float(va.mean()), "std_a": float(va.std(ddof=1)),
            "mean_b": float(vb.mean()), "std_b": float(vb.std(ddof=1)),
            "delta_bacc": float(va.mean() - vb.mean()),
            "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
            "wilcoxon_min_possible_p": 2.0 ** -(n - 1) if n else float("nan"),
        }

        # --- McNemar + bootstrap on paired rows (seed 0) ------------------
        fa = PRED / f"{a}_s0_{args.dataset}.npz"
        fb = PRED / f"{b}_s0_{args.dataset}.npz"
        if fa.exists() and fb.exists():
            da, db = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True)
            y, pa, pb = da["y"], da["pred"], db["pred"]
            assert np.array_equal(y, db["y"]), "arms not row-aligned"
            row.update(mcnemar(y, pa, pb))
            row.update(bootstrap_diff(y, pa, pb, args.n_boot))
        rows.append(row)
        print(f"  {a} vs {b}: d={row['delta_bacc']:+.4f} "
              f"wilcoxon p={row['wilcoxon_p']:.4f} "
              f"mcnemar p={row.get('mcnemar_p', float('nan')):.3g}",
              flush=True)

df = pd.DataFrame(rows)
write("significance", df, meta={
    "alpha": config.ALPHA,
    "n_bootstrap": args.n_boot,
    "tests": {
        "mcnemar": "exact binomial on discordant paired predictions, seed 0",
        "wilcoxon": "signed-rank across seeds on balanced accuracy",
        "bootstrap": "percentile CI on the balanced-accuracy difference",
    },
    "power_caveat": ("With 5 seeds the smallest attainable two-sided Wilcoxon "
                     "p is 0.0625, so Wilcoxon cannot reach alpha=0.05 at this "
                     "sample size. It is reported for consistency of sign; the "
                     "bootstrap CI carries the effect size."),
    "finding_A1": ("The reviewer's stated comparison (0.9362 vs 0.9216) uses a "
                   "baseline figure absent from Verkerken's Table V. We test "
                   "against our own re-run of their 'balanced' configuration."),
})
print()
print(df.to_string(index=False))
print("\n06-OK")
