"""Three ablations the reviewers asked for by name.

PRODUCES
    revision/results/ablation_stagewise_fp.csv  - R2.4
    revision/results/ablation_imbalance.csv     - R2.6
    revision/results/ablation_tau1_sensitivity.csv + .png - R2.5
    revision/results/ablation_stage_contribution.csv - which stage earns what

FEEDS
    Responses R2.4, R2.5, R2.6. Revised manuscript: new Section 5.x
    "Ablation studies" subsections.

WHY EACH EXISTS

    R2.4 - "the exact number of false positives recovered by Stage 2 is never
    explicitly quantified ... the authors should report precisely how many
    false positives remain after Stage 2 processing."
    We trace every benign flow through the pipeline and report the exact
    counts. Note there is a THIRD path the manuscript's two-path argument
    omits: benign flows the classifier confidently mislabels as a known
    attack never reach the extension stage at all, so Stage 2 cannot recover
    them. That path is reported here.

    R2.6 - "This drastic choice is not compared against alternative
    imbalance-handling strategies such as weighted downsampling, SMOTE
    oversampling, or cost-sensitive learning."
    All four arms are run: the paper's aggressive downsampling, weighted,
    none, and SMOTE. The reviewer names three alternatives; we run all three.

    R2.5 - "The threshold selection procedure for tau_1 (best F-beta score for
    beta in [1,9]) is described conceptually but without sensitivity
    analysis."
    A sensitivity analysis is NOT a record of which beta won. It is how much
    the FINAL system performance moves as beta sweeps its range. That is what
    this computes.

USAGE
    python revision/scripts/08_ablations.py [--seeds 0,1,2]

RUNTIME  ~30 min
"""
from __future__ import annotations

import argparse
import gc
from dataclasses import replace
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RESULTS, timed, write  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nids import config  # noqa: E402
from nids.data import cache, prepare  # noqa: E402
from nids.eval import metrics as M  # noqa: E402
from nids.experiment import ExperimentConfig, evaluate_validation, fit_and_select  # noqa: E402
from nids.stages.classifier import UNKNOWN, Classifier, ClassifierConfig  # noqa: E402
from nids.stages.detector import DetectorConfig  # noqa: E402
from nids.stages.pipeline import BENIGN, ZERO_DAY, build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="0,1,2")
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(",")]

CFG = ExperimentConfig(
    name="ours-submitted", architecture="parallel",
    detector=DetectorConfig(kind="ocsvm", n_train=10_000,
                            gamma=2.93e-4, nu=1.3e-6),
    classifier=ClassifierConfig(kind="rf", n_estimators=200),
    tau_b_beta="F9", tau_u_quantile="0.95", tau_m_objective="f1")

with timed("08_ablations"):
    ds = cache.load_clean("cic-ids2017")

    # =====================================================================
    # R2.4 - stagewise false-positive accounting
    # =====================================================================
    fp_rows = []
    split = prepare.make_split(ds, seed=config.SPLIT_SEED)
    feats = split.feature_cols
    for seed in SEEDS:
        fitted = fit_and_select(split, CFG, seed)
        th = fitted.thresholds
        ben = split.val_benign[feats].values
        scores = fitted.detector.score(ben)
        labels, cert = fitted.classifier.predict_with_certainty(ben)
        named = cert >= th.tau_m

        n = len(ben)
        flagged_1a = int((scores > th.tau_b).sum())
        # Path 1: confidently named as a known attack -> Stage 2 never sees it.
        path_mislabel = int(named.sum())
        # Path 2: unnamed and suspicious -> Stage 2 arbitrates on tau_u.
        unnamed_susp = (~named) & (scores > th.tau_b)
        path_zeroday = int((unnamed_susp & (scores > th.tau_u)).sum())
        path_recovered = int((unnamed_susp & (scores <= th.tau_u)).sum())

        pipe = build(CFG.architecture, detector_cfg=CFG.detector,
                     classifier_cfg=CFG.classifier, thresholds=th,
                     detector=fitted.detector, classifier=fitted.classifier)
        final = pipe.predict(ben)
        residual = int((final != BENIGN).sum())

        fp_rows.append({
            "seed": seed,
            "benign_total": n,
            "flagged_by_stage_1a": flagged_1a,
            "stage_1a_fpr": flagged_1a / n,
            "path_A_mislabelled_known_attack": path_mislabel,
            "path_B_called_zero_day": path_zeroday,
            "path_C_recovered_by_stage2": path_recovered,
            "residual_false_positives": residual,
            "end_to_end_fpr": residual / n,
            "stage2_recovery_rate": (path_recovered / flagged_1a
                                     if flagged_1a else np.nan),
        })
        print(f"  [R2.4] seed={seed} 1a_fpr={flagged_1a/n:.4f} "
              f"residual={residual} e2e_fpr={residual/n:.4f}", flush=True)
        del fitted, pipe
        gc.collect()

    fp = pd.DataFrame(fp_rows)
    write("ablation_stagewise_fp", fp, meta={
        "reviewer_comment": "R2.4",
        "note": ("Path A is the accounting the manuscript omits: benign flows "
                 "the classifier confidently mislabels as a known attack are "
                 "emitted as that attack and never reach the extension stage, "
                 "so tau_3 cannot recover them. The manuscript's two-path "
                 "argument ('stage 2 partially compensates') does not sum."),
    })

    # =====================================================================
    # R2.6 - imbalance handling, four arms
    # =====================================================================
    imb_rows = []
    for strategy in ("downsample", "weighted", "none", "smote"):
        for seed in SEEDS:
            try:
                if strategy == "smote":
                    sp = prepare.make_split(ds, seed=config.SPLIT_SEED,
                                            imbalance="none")
                    from imblearn.over_sampling import SMOTE
                    Xtr = sp.train_malicious[sp.feature_cols].values
                    ytr = sp.train_malicious["Attack Type"].values
                    counts = pd.Series(ytr).value_counts()
                    k = max(1, min(5, int(counts.min()) - 1))
                    Xr, yr = SMOTE(random_state=seed,
                                   k_neighbors=k).fit_resample(Xtr, ytr)
                    clf = Classifier(ClassifierConfig(kind="rf",
                                                      n_estimators=200,
                                                      seed=seed)).fit(Xr, yr)
                    fitted = fit_and_select(sp, CFG, seed)
                    fitted.classifier = clf
                else:
                    sp = prepare.make_split(ds, seed=config.SPLIT_SEED,
                                            imbalance=strategy)
                    # make_split treats 'weighted' and 'none' identically: it
                    # decides sampling, and weighting is the classifier's job.
                    # Without this the two arms are the same experiment.
                    cfg = CFG
                    if strategy == "weighted":
                        cfg = replace(CFG, classifier=replace(
                            CFG.classifier, class_weight="balanced"))
                    fitted = fit_and_select(sp, cfg, seed)

                res = evaluate_validation(fitted, sp)
                row = {"strategy": strategy, "seed": seed,
                       "n_train_malicious": len(sp.train_malicious)}
                row.update({k2: v for k2, v in res.metrics.items()
                            if isinstance(v, (int, float))})
                imb_rows.append(row)
                print(f"  [R2.6] {strategy} s{seed} "
                      f"bACC={row.get('balanced_accuracy', float('nan')):.4f} "
                      f"n_train={row['n_train_malicious']:,}", flush=True)
                del fitted, sp
                gc.collect()
            except Exception as exc:
                print(f"  [R2.6] {strategy} s{seed} FAILED: {exc}")
                imb_rows.append({"strategy": strategy, "seed": seed,
                                 "error": str(exc)})

    imb = pd.DataFrame(imb_rows)
    write("ablation_imbalance", imb, meta={
        "reviewer_comment": "R2.6",
        "arms": ["downsample (the manuscript's choice)", "weighted",
                 "none", "smote"],
    })

    # =====================================================================
    # R2.5 - tau_1 sensitivity: how much does the SYSTEM move with beta?
    # =====================================================================
    tau_rows = []
    for beta in range(1, 10):
        cfg_b = ExperimentConfig(
            name=f"beta{beta}", architecture="parallel",
            detector=CFG.detector, classifier=CFG.classifier,
            tau_b_beta=f"F{beta}", tau_u_quantile="0.95",
            tau_m_objective="f1")
        for seed in SEEDS:
            try:
                fitted = fit_and_select(split, cfg_b, seed)
                res = evaluate_validation(fitted, split)
                row = {"beta": beta, "seed": seed,
                       "tau_b": fitted.thresholds.tau_b}
                row.update({k2: v for k2, v in res.metrics.items()
                            if isinstance(v, (int, float))})
                tau_rows.append(row)
                del fitted
                gc.collect()
            except Exception as exc:
                print(f"  [R2.5] beta={beta} s{seed} FAILED: {exc}")
        if tau_rows and tau_rows[-1]["beta"] == beta:
            print(f"  [R2.5] beta={beta} "
                  f"bACC={tau_rows[-1].get('balanced_accuracy', float('nan')):.4f} "
                  f"tau_b={tau_rows[-1]['tau_b']:.6g}", flush=True)

    tau = pd.DataFrame(tau_rows)
    write("ablation_tau1_sensitivity", tau, meta={
        "reviewer_comment": "R2.5",
        "note": ("A sensitivity analysis reports how the FINAL system moves "
                 "as beta varies, not which beta won. The manuscript selects "
                 "F9 (tau_1 = 0.00015)."),
    })

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        g = tau.groupby("beta").agg(
            bacc=("balanced_accuracy", "mean"),
            bacc_sd=("balanced_accuracy", "std"),
            fpr=("benign_fpr", "mean")).reset_index()
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.errorbar(g["beta"], g["bacc"], yerr=g["bacc_sd"], marker="o",
                     capsize=3, label="balanced accuracy")
        ax1.set_xlabel(r"$\beta$ used to select $\tau_1$")
        ax1.set_ylabel("Balanced accuracy")
        ax2 = ax1.twinx()
        ax2.plot(g["beta"], g["fpr"], marker="s", linestyle="--",
                 color="tab:red", label="benign FPR")
        ax2.set_ylabel("Benign false-positive rate")
        ax1.axvline(9, color="grey", linestyle=":",
                    label=r"manuscript choice $\beta=9$")
        ax1.grid(alpha=0.3)
        fig.legend(loc="lower left", bbox_to_anchor=(0.12, 0.15), fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS / "ablation_tau1_sensitivity.png", dpi=200)
        print(f"-> {RESULTS / 'ablation_tau1_sensitivity.png'}")
    except Exception as exc:
        print(f"!! tau_1 plot skipped: {exc}")

    # =====================================================================
    # Stage contribution: what does each stage actually earn?
    # =====================================================================
    sc_rows = []
    val = pd.concat([split.val_benign, split.val_malicious], ignore_index=True)
    y = val["Attack Type"].values
    Xv = val[feats].values
    for seed in SEEDS:
        fitted = fit_and_select(split, CFG, seed)
        th = fitted.thresholds
        # Stage 1b alone: the closed classifier, no detector, no rule head.
        labels, cert = fitted.classifier.predict_with_certainty(Xv)
        alone = np.where(cert >= th.tau_m, labels, BENIGN)
        r_alone = M.evaluate(y, alone, seed=seed)
        # Full system.
        pipe = build(CFG.architecture, detector_cfg=CFG.detector,
                     classifier_cfg=CFG.classifier, thresholds=th,
                     detector=fitted.detector, classifier=fitted.classifier)
        r_full = M.evaluate(y, pipe.predict(Xv), seed=seed)
        for who, r in (("stage-1b alone", r_alone), ("full system", r_full)):
            row = {"system": who, "seed": seed}
            row.update({k2: v for k2, v in r.metrics.items()
                        if isinstance(v, (int, float))})
            sc_rows.append(row)
        print(f"  [stage] seed={seed} "
              f"1b_alone={r_alone.metrics.get('balanced_accuracy', float('nan')):.4f} "
              f"full={r_full.metrics.get('balanced_accuracy', float('nan')):.4f}",
              flush=True)
        del fitted, pipe
        gc.collect()

    sc = pd.DataFrame(sc_rows)
    write("ablation_stage_contribution", sc, meta={
        "note": ("Finding A9: on KNOWN-class discrimination the classifier "
                 "alone can outperform the assembled system, because the "
                 "detector adds false positives faster than the rule head "
                 "recovers them. The architecture's justification therefore "
                 "rests on zero-day capability, which a closed classifier "
                 "cannot provide at all - it has no label to emit."),
    })

print("\n08-OK")
