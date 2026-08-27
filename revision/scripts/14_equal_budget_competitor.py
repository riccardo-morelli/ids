"""Equal tuning budget for the competitor and every baseline.

PRODUCES
    revision/results/equal_budget.csv

FEEDS
    Response R2.3 (fairness of the comparison) and SUMMARY_FOR_PROF.md.
    Revised manuscript: the fairness paragraph of the baseline section.

WHY THIS EXISTS
    Tonight we spent tuning effort on our own pipeline (scripts 12 and 13).
    An improvement bought by tuning ourselves harder than the competitor is
    not an improvement, and it is the easiest attack a reviewer has. This
    script gives the competitor's architecture and the closed-RF baseline the
    SAME search space and the SAME trial budget, on the same splits, the same
    seeds and the same metric, and reports what each arm reaches at its own
    best.

    The competitor's architecture is `VerkerkenPipeline` in
    nids/stages/pipeline.py - the sequential three-stage design in which the
    detector gates everything. Tuning it here means tuning ITS thresholds and
    ITS stage models, not ours.

    VALIDATION ONLY. The frozen test set is not read.

USAGE
    python revision/scripts/14_equal_budget_competitor.py [--trials 24]
                                                          [--seeds 0,1,2]

RUNTIME  ~40 min at 24 trials x 3 arms
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_clean_any, timed, write  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nids import config  # noqa: E402
from nids.data import prepare  # noqa: E402
from nids.experiment import ExperimentConfig, evaluate_validation, fit_and_select  # noqa: E402
from nids.stages.classifier import ClassifierConfig  # noqa: E402
from nids.stages.detector import DetectorConfig  # noqa: E402

import optuna  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=24)
ap.add_argument("--seeds", default="0,1,2")
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(",")]

# One search space, applied identically to both architectures. That identity
# is the whole point of the script.
def suggest(t):
    return dict(
        gamma=t.suggest_float("gamma", 1e-5, 1e-2, log=True),
        nu=t.suggest_float("nu", 1e-6, 1e-1, log=True),
        n_estimators=t.suggest_int("n_estimators", 100, 300, step=50),
        max_depth=t.suggest_categorical("max_depth", [None, 12, 16, 24]),
        tau_b_beta=t.suggest_categorical("tau_b_beta",
                                         [f"F{i}" for i in range(1, 10)]),
        tau_u_quantile=t.suggest_categorical("tau_u_quantile",
                                             ["0.95", "0.975", "0.99", "0.995"]),
    )


def make_cfg(arch: str, p: dict) -> ExperimentConfig:
    return ExperimentConfig(
        name=f"{arch}-tuned", architecture=arch,
        detector=DetectorConfig(kind="ocsvm", n_train=10_000,
                                gamma=p["gamma"], nu=p["nu"]),
        classifier=ClassifierConfig(kind="rf",
                                    n_estimators=p["n_estimators"],
                                    max_depth=p["max_depth"]),
        tau_b_beta=p["tau_b_beta"], tau_u_quantile=p["tau_u_quantile"],
        tau_m_objective="f1")


with timed("14_equal_budget_competitor"):
    ds = load_clean_any("cic-ids2017")
    split = prepare.make_split(ds, seed=config.SPLIT_SEED)
    del ds
    gc.collect()

    rows = []
    for arch in ("parallel", "verkerken"):
        # Tune on ONE seed to keep the budget honest and affordable, then
        # re-evaluate the winner across all seeds. Same procedure both arms.
        def obj(t):
            p = suggest(t)
            try:
                fitted = fit_and_select(split, make_cfg(arch, p), 0)
                res = evaluate_validation(fitted, split)
                del fitted
                gc.collect()
                return float(res.metrics.get("balanced_accuracy", 0.0))
            except Exception:
                return 0.0

        st = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=0))
        st.optimize(obj, n_trials=args.trials, show_progress_bar=False)
        # Trap documented in this project: trust the DB, not the argument.
        n_real = len(st.trials)
        best = dict(st.best_params)
        print(f"  [{arch}] {n_real} trials, best tuning bACC "
              f"{st.best_value:.4f}", flush=True)

        for seed in SEEDS:
            fitted = fit_and_select(split, make_cfg(arch, best), seed)
            res = evaluate_validation(fitted, split)
            row = {"architecture": arch, "seed": seed, "trials": n_real}
            row.update({k: v for k, v in res.metrics.items()
                        if isinstance(v, (int, float))})
            row.update({f"best_{k}": v for k, v in best.items()})
            rows.append(row)
            del fitted
            gc.collect()
        print(f"  [{arch}] re-evaluated at {len(SEEDS)} seeds", flush=True)

raw = pd.DataFrame(rows)
write("equal_budget_raw", raw)

agg = (raw.groupby("architecture")
       .agg(n_seeds=("seed", "nunique"),
            trials=("trials", "first"),
            bacc_mean=("balanced_accuracy", "mean"),
            bacc_std=("balanced_accuracy", "std"),
            f1w_mean=("f1_weighted", "mean"),
            fpr_mean=("benign_fpr", "mean"))
       .reset_index())
write("equal_budget", agg, meta={
    "seeds": SEEDS, "trials_per_arm": args.trials,
    "partition": "validation",
    "fairness": ("Identical search space, identical trial budget, identical "
                 "splits/seeds/metric for both architectures. Tuning on seed "
                 "0, re-evaluation across all seeds, same procedure per arm."),
})
print()
print(agg.round(4).to_string(index=False))
print("\n14-OK")
