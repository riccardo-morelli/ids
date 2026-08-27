"""Zero-day under BOTH definitions, every class withheld in turn, with CIs.

PRODUCES
    revision/results/zeroday_class.csv     - definition (a): unseen CLASS
    revision/results/zeroday_variant.csv   - definition (b): unseen SUBCATEGORY
    revision/results/zeroday_summary.csv   - both, side by side

FEEDS
    Response R2.2 (BLOCKING). Revised manuscript: new Section 4.x
    "Disambiguating zero-day evaluation" and its two tables.

WHY THIS EXISTS
    R2.2, verbatim: "The zero-day detection capability ... is evaluated on
    only 47 instances (Infiltration and Heartbleed). With such a small sample,
    a zero-day recall of 83% ... is statistically unreliable ... A more robust
    evaluation protocol, such as a leave-one-class-out experiment, should be
    adopted ... Confidence intervals on the zero-day recall must also be
    reported."

    The brief additionally requires the two senses of "zero-day" to be
    separated, because they are different difficulties:

      (a) UNSEEN ATTACK CLASS - an entire attack family withheld from
          training (e.g. all of Botnet). This is what the manuscript's
          Infiltration+Heartbleed protocol actually is, though the manuscript
          never says so.

      (b) UNSEEN SUBCATEGORY - one variant withheld while its siblings remain
          in training (e.g. DoS slowloris withheld, DoS Hulk still trained on).
          This is the harder and more realistic case: the classifier
          confidently names the unseen variant as its known sibling, so it
          never reaches the zero-day path at all.

    Both are reported. Every recall carries a Wilson interval, which is the
    correct interval at n=47 and at proportions near 0 or 1 - both of which
    occur here.

USAGE
    python revision/scripts/03_zero_day_protocols.py [--seeds 0,1,2,3,4]

RUNTIME  ~50 min (2 protocols x ~7 folds x 5 seeds, one model fit per fold)
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import timed, wilson, write  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nids import config  # noqa: E402
from nids.data import cache, prepare  # noqa: E402
from nids.experiment import ExperimentConfig, fit_and_select  # noqa: E402
from nids.stages.classifier import ClassifierConfig  # noqa: E402
from nids.stages.detector import DetectorConfig  # noqa: E402
from nids.stages.pipeline import ZERO_DAY, build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default=",".join(map(str, config.MODEL_SEEDS)))
ap.add_argument("--dataset", default="cic-ids2017")
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(",")]

CFG = ExperimentConfig(
    name="ours-submitted", architecture="parallel",
    detector=DetectorConfig(kind="ocsvm", n_train=10_000,
                            gamma=2.93e-4, nu=1.3e-6),
    classifier=ClassifierConfig(kind="rf", n_estimators=200),
    tau_b_beta="F9", tau_u_quantile="0.95", tau_m_objective="f1")


def run_fold(ds, held_fine: tuple[str, ...], seed: int) -> dict | None:
    """Withhold `held_fine`, fit, and measure zero-day recall on it.

    The fold builds its OWN split with `zero_day=held_fine`. That is a
    different split object from the frozen protocol test set (which is defined
    by zero_day=("Infiltration","Heartbleed")), so no test budget is consumed.
    """
    split = prepare.make_split(ds, seed=config.SPLIT_SEED, zero_day=held_fine)
    for part in (split.train_malicious, split.val_malicious):
        assert not part["Label"].isin(held_fine).any(), "leak - fold void"

    fitted = fit_and_select(split, CFG, seed)
    pipe = build(CFG.architecture, detector_cfg=CFG.detector,
                 classifier_cfg=CFG.classifier,
                 thresholds=fitted.thresholds,
                 detector=fitted.detector, classifier=fitted.classifier)

    held = split.test[split.test["Label"].isin(held_fine)]
    if held.empty:
        return None
    pred = pipe.predict(held[split.feature_cols].values)
    hits, n = int((pred == ZERO_DAY).sum()), len(pred)
    lo, hi = wilson(hits, n)

    # Where do the misses go? This is the mechanism, not just the number.
    other = pred[pred != ZERO_DAY]
    vals, counts = np.unique(other, return_counts=True)
    top = vals[counts.argmax()] if len(vals) else "-"

    # Benign FPR on the same fold, so recall is never read without its cost.
    ben = split.val_benign[split.feature_cols].values
    fpr = float((pipe.predict(ben) != "Benign").mean())

    del fitted, pipe, split
    gc.collect()
    return {"n": n, "hits": hits, "recall": hits / n, "ci_lo": lo,
            "ci_hi": hi, "top_misroute": str(top), "benign_fpr": fpr}


with timed("03_zero_day_protocols"):
    ds = cache.load_clean(args.dataset)
    fine = ds.frame.loc[ds.frame["Attack Type"] != "Benign", "Label"]
    coarse = ds.frame.loc[ds.frame["Attack Type"] != "Benign", "Attack Type"]
    members: dict[str, list[str]] = {}
    for c, f in zip(coarse, fine):
        members.setdefault(c, [])
        if f not in members[c]:
            members[c].append(f)

    # ---- (a) unseen CLASS -------------------------------------------------
    rows_a = []
    for cls in sorted(members):
        for seed in SEEDS:
            r = run_fold(ds, tuple(members[cls]), seed)
            if r is None:
                continue
            rows_a.append({"protocol": "unseen-class", "held_out": cls,
                           "seed": seed, **r})
            print(f"  [class] {cls} s{seed} recall={r['recall']:.4f} "
                  f"n={r['n']} fpr={r['benign_fpr']:.4f}", flush=True)

    # ---- (b) unseen SUBCATEGORY ------------------------------------------
    # Only families with >= 2 variants can support this protocol: a sibling
    # must remain in training for the case to be "subcategory of a seen
    # family" rather than a whole unseen class.
    rows_b = []
    for cls, variants in sorted(members.items()):
        if len(variants) < 2:
            continue
        for var in variants:
            n_var = int((ds.frame["Label"] == var).sum())
            if n_var < 50:          # too few to estimate a rate at all
                continue
            for seed in SEEDS:
                r = run_fold(ds, (var,), seed)
                if r is None:
                    continue
                rows_b.append({"protocol": "unseen-subcategory",
                               "family": cls, "held_out": var,
                               "seed": seed, **r})
                print(f"  [variant] {var} s{seed} recall={r['recall']:.4f} "
                      f"n={r['n']}", flush=True)

a = pd.DataFrame(rows_a)
b = pd.DataFrame(rows_b)
write("zeroday_class", a, meta={"protocol": "unseen attack class withheld",
                                "seeds": SEEDS, "partition": "validation"})
write("zeroday_variant", b, meta={"protocol": "unseen subcategory withheld",
                                  "seeds": SEEDS, "partition": "validation"})


def summarise(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("held_out").agg(
        n=("n", "first"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        ci_lo=("ci_lo", "mean"), ci_hi=("ci_hi", "mean"),
        top_misroute=("top_misroute", lambda s: s.mode().iloc[0]),
    ).reset_index()
    g.insert(0, "protocol", label)
    tot = pd.DataFrame([{
        "protocol": label, "held_out": "MEAN over classes",
        "n": int(g["n"].sum()),
        "recall_mean": g["recall_mean"].mean(),
        "recall_std": g["recall_mean"].std(ddof=1),
        "ci_lo": np.nan, "ci_hi": np.nan,
        "top_misroute": "(recall_std = spread ACROSS held-out classes)",
    }])
    return pd.concat([g, tot], ignore_index=True)


summary = pd.concat([summarise(a, "unseen-class"),
                     summarise(b, "unseen-subcategory")], ignore_index=True)
write("zeroday_summary", summary, meta={
    "seeds": SEEDS,
    "note": ("Two distinct senses of 'zero-day'. The manuscript's 47-instance "
             "Infiltration+Heartbleed protocol is an instance of "
             "'unseen-class'. The manuscript does not distinguish the two "
             "senses; the revision does."),
})
print()
print(summary.to_string(index=False))
print("\n03-OK")
