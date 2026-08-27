"""The adopted configuration: SMOTE + tau_u=0.995, measured together.

PRODUCES
    revision/results/final_config.csv       - per-(arm, seed) rows
    revision/results/final_config_agg.csv   - mean +- std, paired deltas

FEEDS
    SUMMARY_FOR_PROF.md, and the revised manuscript's Table 5 if the professor
    adopts it. Responses R2.4 (false positives) and R2.6 (imbalance).

WHY THIS EXISTS
    Two non-architectural changes were adopted on validation evidence
    (.phd-log/DECISIONS.md N3):
      tau_u 0.95 -> 0.995   +0.0075 bACC every seed, benign FPR 4.37x lower
      downsample -> SMOTE   +0.035 bACC, 4/5 seeds, seed variance 9x lower
    Neither was measured in the presence of the other. The gains need not be
    additive: SMOTE changes the classifier's certainty distribution, which is
    what tau_m thresholds, while tau_u governs the detector path. So the
    combination is measured rather than assumed.

    Four arms, one script, same split and seeds, so every comparison is paired:
      A incumbent   downsample + tau_u 0.95    (as submitted)
      B tau_u only  downsample + tau_u 0.995
      C smote only  SMOTE      + tau_u 0.95
      D combined    SMOTE      + tau_u 0.995

    VALIDATION ONLY. Adopting D changes the final configuration and therefore
    requires a SECOND, logged test-set spend - the professor's call, recorded
    in .phd-log/TODO.md.

USAGE
    python revision/scripts/16_final_config.py [--seeds 0,1,2,3,4]
                                               [--arms A,B,C,D]

RUNTIME  ~6 min/seed for the SMOTE arms, ~1 min/seed otherwise.
         Per-(arm, seed) shards make it resumable; run one seed per process on
         a 16 GB machine.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RESULTS, load_clean_any, timed, write  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nids import config  # noqa: E402
from nids.data import prepare  # noqa: E402
from nids.experiment import ExperimentConfig, evaluate_validation, fit_and_select  # noqa: E402
from nids.stages.classifier import Classifier, ClassifierConfig  # noqa: E402
from nids.stages.detector import DetectorConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default=",".join(map(str, config.MODEL_SEEDS)))
ap.add_argument("--arms", default="A,B,C,D")
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(",")]

SPEC = {
    "A-incumbent": ("downsample", "0.95"),
    "B-tau_u": ("downsample", "0.995"),
    "C-smote": ("smote", "0.95"),
    "D-combined": ("smote", "0.995"),
}
KEY = {"A": "A-incumbent", "B": "B-tau_u", "C": "C-smote", "D": "D-combined"}


def cfg_for(tau_u: str) -> ExperimentConfig:
    return ExperimentConfig(
        name="ours", architecture="parallel",
        detector=DetectorConfig(kind="ocsvm", n_train=10_000,
                                gamma=2.93e-4, nu=1.3e-6),
        classifier=ClassifierConfig(kind="rf", n_estimators=200),
        tau_b_beta="F9", tau_u_quantile=tau_u, tau_m_objective="f1")


rows = []
(RESULTS / "shards").mkdir(parents=True, exist_ok=True)

with timed("16_final_config"):
    ds = load_clean_any("cic-ids2017")
    for a in args.arms.split(","):
        arm = KEY.get(a.strip().upper(), a.strip())
        imbalance, tau_u = SPEC[arm]
        for seed in SEEDS:
            shard = RESULTS / "shards" / ("_final_" + arm + "_s" + str(seed) + ".csv")
            if shard.exists():
                rows.extend(pd.read_csv(shard).to_dict("records"))
                print("  " + arm + " s" + str(seed) + ": cached", flush=True)
                continue
            t0 = time.perf_counter()
            if imbalance == "smote":
                sp = prepare.make_split(ds, seed=config.SPLIT_SEED,
                                        imbalance="none")
                from imblearn.over_sampling import SMOTE
                Xtr = sp.train_malicious[sp.feature_cols].values
                ytr = sp.train_malicious["Attack Type"].values
                k = max(1, min(5, int(pd.Series(ytr).value_counts().min()) - 1))
                Xr, yr = SMOTE(random_state=seed,
                               k_neighbors=k).fit_resample(Xtr, ytr)
                n_after = len(Xr)
                fitted = fit_and_select(sp, cfg_for(tau_u), seed)
                fitted.classifier = Classifier(
                    ClassifierConfig(kind="rf", n_estimators=200,
                                     seed=seed)).fit(Xr, yr)
                del Xr, yr, Xtr
            else:
                sp = prepare.make_split(ds, seed=config.SPLIT_SEED,
                                        imbalance="downsample")
                n_after = len(sp.train_malicious)
                fitted = fit_and_select(sp, cfg_for(tau_u), seed)
            train_s = time.perf_counter() - t0

            res = evaluate_validation(fitted, sp)
            row = {"arm": arm, "seed": seed, "imbalance": imbalance,
                   "tau_u_quantile": tau_u, "n_train_malicious": n_after,
                   "train_seconds": round(train_s, 2),
                   "tau_u": fitted.thresholds.tau_u,
                   "tau_m": fitted.thresholds.tau_m}
            row.update({k2: v for k2, v in res.metrics.items()
                        if isinstance(v, (int, float))})
            pd.DataFrame([row]).to_csv(shard, index=False)
            rows.append(row)
            print("  %s s%d: bACC=%.4f FPR=%.4f F1w=%.4f"
                  % (arm, seed, row["balanced_accuracy"], row["benign_fpr"],
                     row["f1_weighted"]), flush=True)
            del fitted, sp
            gc.collect()

raw = pd.DataFrame(rows)
write("final_config", raw)

agg = raw.groupby("arm").agg(
    n_seeds=("seed", "nunique"),
    bacc_mean=("balanced_accuracy", "mean"),
    bacc_std=("balanced_accuracy", "std"),
    f1w_mean=("f1_weighted", "mean"),
    fpr_mean=("benign_fpr", "mean"),
    fpr_std=("benign_fpr", "std"),
    attack_det=("attack_detection_rate", "mean"),
    train_s=("train_seconds", "mean")).reset_index().sort_values("arm")

# Paired deltas against the incumbent: every arm shares the split and the
# seeds, so the paired difference is the correct test. Marginal intervals
# overlap for reasons common to all arms and would mislead.
meta = {"seeds": SEEDS, "partition": "validation"}
try:
    from scipy.stats import ttest_rel, wilcoxon

    base = raw[raw.arm == "A-incumbent"].set_index("seed")["balanced_accuracy"]
    deltas = {}
    for arm in sorted(raw.arm.unique()):
        if arm == "A-incumbent":
            continue
        o = raw[raw.arm == arm].set_index("seed")["balanced_accuracy"]
        common = sorted(set(base.index) & set(o.index))
        if len(common) < 2:
            continue
        d = (o.loc[common] - base.loc[common]).values
        e = {"mean_delta": float(d.mean()), "wins": int((d > 0).sum()),
             "n": len(common), "per_seed": [round(float(x), 4) for x in d]}
        if len(common) >= 3:
            e["ttest_p"] = float(ttest_rel(o.loc[common], base.loc[common]).pvalue)
            e["wilcoxon_p"] = float(wilcoxon(o.loc[common], base.loc[common]).pvalue)
            e["wilcoxon_floor"] = 2.0 ** -(len(common) - 1)
        deltas[arm] = e
        print("  %s vs incumbent: %+.4f (%d/%d seeds)"
              % (arm, d.mean(), e["wins"], e["n"]), flush=True)
    meta["paired_vs_incumbent"] = deltas
except Exception as exc:                                    # pragma: no cover
    meta["paired_error"] = str(exc)

write("final_config_agg", agg, meta=meta)
print()
print(agg.round(4).to_string(index=False))
print("\n16-OK")
