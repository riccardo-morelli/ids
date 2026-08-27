"""Inference latency under a protocol a reviewer cannot dismiss.

PRODUCES
    revision/results/latency.csv          - per-stage, per-arm, median and p95
    revision/results/latency_tradeoff.csv - detection quality vs latency
    revision/results/latency.png          - the trade-off plot

FEEDS
    Response R2.0a (the 27% claim R2 praised) and finding A6.
    Revised manuscript: replacement Table 5 latency column, new Section 4.x
    "Inference latency protocol", and the quality-latency trade-off figure.

WHY THIS EXISTS
    The submitted manuscript reports 1.116s vs 1.520s (a 27% reduction) over
    10 runs. Two defects, both ours, both disclosed in the revision:

      A6 - the legacy code summed the two stage timings SEQUENTIALLY while the
           text claimed a parallel architecture, and the baseline arm paid a
           PCA transform and a second scaler that our arm skipped. The two arms
           were never measured on comparable work.
      A7 - some legacy timing cells referenced a stale `start_time`.

    Reviewer 2 named the 27% figure as one of three strengths of the paper.
    We therefore do not quietly change it: we re-measure it properly and
    report what survives.

PROTOCOL (stated so it can be checked, not trusted)
    - Warm-up runs discarded, then >= 30 timed repetitions per measurement.
    - MEDIAN and P95 reported, never the mean alone: tail latency is what
      matters for a detection system and a mean hides it.
    - Both BATCH (whole validation stream, one vectorised call) and PER-SAMPLE
      (single flow, the operational worst case) are measured.
    - Throughput in flows/second reported alongside.
    - INSIDE the timer: the feature transform, the stage-1 model call, and -
      for end-to-end arms - the stage-2 fusion.
      OUTSIDE the timer: data loading, model fitting, metric computation.
      Both arms pay their own transform, so the PCA asymmetry of A6 is gone.
    - Sequential total and MEASURED PARALLEL execution are both reported. The
      parallel number comes from actually running the two stages concurrently
      in a thread pool, not from computing max(d, c). The submitted manuscript
      asserted the bound; here it is measured, including contention.
    - Thread count pinned via threadpoolctl so both arms get identical BLAS
      resources. Two systems compared under different thread counts is not a
      comparison.
    - Training time reported SEPARATELY from inference latency.
    - Hardware, OS, library versions and thread counts recorded in the meta.

    MUST RUN ON AN OTHERWISE IDLE MACHINE. Serialise it: do not run other
    experiments concurrently. The meta records whether this was asserted.

USAGE
    python revision/scripts/07_latency.py [--reps 30] [--threads 1]

RUNTIME  ~25 min
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import THEIR_GAMMA, THEIR_NU, RESULTS, environment, timed, write  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nids import config  # noqa: E402
from nids.data import cache, prepare, selection  # noqa: E402
from nids.data.transforms import FeatureTransform, TransformConfig  # noqa: E402
from nids.stages.classifier import Classifier, ClassifierConfig  # noqa: E402
from nids.stages.detector import Detector, DetectorConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--per-sample-reps", type=int, default=200)
ap.add_argument("--threads", type=int, default=1)
args = ap.parse_args()

try:
    from threadpoolctl import threadpool_limits
except ImportError:                                        # pragma: no cover
    threadpool_limits = None
    print("!! threadpoolctl unavailable - thread count NOT pinned")


def timeit(fn, reps: int, warmup: int) -> dict:
    """Warm up, then time `reps` calls with a monotonic clock."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()          # monotonic
        fn()
        ts.append(time.perf_counter() - t0)
    a = np.asarray(ts)
    return {"median_s": float(np.median(a)), "p95_s": float(np.percentile(a, 95)),
            "mean_s": float(a.mean()), "std_s": float(a.std(ddof=1)),
            "min_s": float(a.min()), "reps": reps}


ARMS = [
    # name, detector cols, transform, detector cfg, architecture
    ("ours-submitted (parallel, OCSVM, 64 feat)", None,
     TransformConfig(kind="standard"),
     DetectorConfig(kind="ocsvm", n_train=10_000, gamma=2.93e-4, nu=1.3e-6)),
    ("verkerken (sequential, OCSVM+PCA, 64 feat)", None,
     TransformConfig(kind="standard", pca_components=56),
     DetectorConfig(kind="ocsvm", n_train=10_000, gamma=2.93e-4, nu=1.3e-6)),
    # The same architecture at the authors' own published constants. gamma and
    # nu determine how many support vectors the OC-SVM keeps, and therefore how
    # fast it scores, so latency has to be re-measured rather than carried over
    # from the inferred configuration above.
    ("verkerken-faithful (sequential, OCSVM+PCA, 64 feat)", None,
     TransformConfig(kind="standard", pca_components=56),
     DetectorConfig(kind="ocsvm", n_train=10_000,
                    gamma=THEIR_GAMMA, nu=THEIR_NU)),
    ("candidate (parallel, knn2s, 36 feat)", 36,
     TransformConfig(kind="quantile_uniform"),
     DetectorConfig(kind="knn_density", n_train=5_000, n_neighbors=9,
                    two_sided=True)),
]

rows = []
with timed("07_latency"):
    ds = cache.load_clean("cic-ids2017")
    split = prepare.make_split(ds, seed=config.SPLIT_SEED)
    del ds
    gc.collect()
    feats = split.feature_cols
    sel36 = selection.select(split, mode="unsupervised", top_k=36)
    X = pd.concat([split.val_benign, split.val_malicious], ignore_index=True)
    Xc = X[feats].values
    n_flows = len(X)
    print(f"flows: {n_flows:,}  threads pinned to {args.threads}", flush=True)

    limiter = (threadpool_limits(limits=args.threads)
               if threadpool_limits else None)
    try:
        for name, topk, tcfg, dcfg in ARMS:
            dcols = sel36 if topk else feats
            Xd = X[dcols].values

            # ---- training time, measured separately -----------------------
            t0 = time.perf_counter()
            tr = FeatureTransform(tcfg).fit(split.train_benign[dcols].values)
            det = Detector(DetectorConfig(**{**dcfg.__dict__, "seed": 0})).fit(
                tr.transform(split.train_benign[dcols].values))
            t_det_fit = time.perf_counter() - t0

            t0 = time.perf_counter()
            clf = Classifier(ClassifierConfig(kind="rf", n_estimators=200,
                                              seed=0)).fit(
                split.train_malicious[feats].values,
                split.train_malicious["Attack Type"].values)
            t_clf_fit = time.perf_counter() - t0

            # ---- batch inference, per stage -------------------------------
            det_fn = lambda: det.score(tr.transform(Xd))          # noqa: E731
            clf_fn = lambda: clf.predict_with_certainty(Xc)       # noqa: E731
            d = timeit(det_fn, args.reps, args.warmup)
            c = timeit(clf_fn, args.reps, args.warmup)

            # ---- fusion, inside the timer ---------------------------------
            scores = det.score(tr.transform(Xd))
            labels, cert = clf.predict_with_certainty(Xc)

            def fuse():
                out = np.full(len(scores), "Benign", dtype=object)
                named = cert >= 0.9
                out[named] = labels[named]
                susp = (~named) & (scores > 0.0)
                out[susp] = "Zero Day"
                return out
            f = timeit(fuse, args.reps, args.warmup)

            # ---- MEASURED parallel execution ------------------------------
            # Not max(d, c): the two stages actually run concurrently, so the
            # number includes scheduling and memory-bandwidth contention.
            def parallel():
                with ThreadPoolExecutor(max_workers=2) as ex:
                    fa = ex.submit(det_fn)
                    fb = ex.submit(clf_fn)
                    fa.result(), fb.result()
                return fuse()
            par = timeit(parallel, max(args.reps // 2, 10), args.warmup)

            # ---- per-sample latency ---------------------------------------
            one_d, one_c = Xd[:1], Xc[:1]
            ps_d = timeit(lambda: det.score(tr.transform(one_d)),
                          args.per_sample_reps, args.warmup)
            ps_c = timeit(lambda: clf.predict_with_certainty(one_c),
                          args.per_sample_reps, args.warmup)

            seq = d["median_s"] + c["median_s"] + f["median_s"]
            rows.append({
                "arm": name,
                "n_flows": n_flows,
                "detector_median_s": d["median_s"], "detector_p95_s": d["p95_s"],
                "classifier_median_s": c["median_s"],
                "classifier_p95_s": c["p95_s"],
                "fusion_median_s": f["median_s"],
                "sequential_total_s": seq,
                "parallel_measured_median_s": par["median_s"],
                "parallel_measured_p95_s": par["p95_s"],
                "parallel_theoretical_bound_s": (
                    max(d["median_s"], c["median_s"]) + f["median_s"]),
                "speedup_measured": seq / par["median_s"],
                "saving_measured_pct": 100 * (1 - par["median_s"] / seq),
                "throughput_flows_per_s": n_flows / par["median_s"],
                "per_sample_detector_us": 1e6 * ps_d["median_s"],
                "per_sample_classifier_us": 1e6 * ps_c["median_s"],
                "per_sample_p95_us": 1e6 * max(ps_d["p95_s"], ps_c["p95_s"]),
                "train_detector_s": t_det_fit,
                "train_classifier_s": t_clf_fit,
                "reps": args.reps,
            })
            print(f"  {name}\n     seq={seq:.3f}s "
                  f"par(measured)={par['median_s']:.3f}s "
                  f"saving={rows[-1]['saving_measured_pct']:.1f}% "
                  f"p95={par['p95_s']:.3f}s", flush=True)
            del det, clf, tr
            gc.collect()
    finally:
        if limiter is not None:
            limiter.unregister()

df = pd.DataFrame(rows)
write("latency", df, meta={
    "protocol": {
        "warmup_discarded": args.warmup,
        "timed_repetitions": args.reps,
        "per_sample_repetitions": args.per_sample_reps,
        "clock": "time.perf_counter (monotonic)",
        "threads_pinned": args.threads,
        "inside_timer": ("feature transform, stage-1 model call, stage-2 "
                         "fusion"),
        "outside_timer": "data loading, model fitting, metric computation",
        "parallel": ("MEASURED via ThreadPoolExecutor(2), not computed as "
                     "max(detector, classifier); includes contention"),
        "gpu": "not used",
        "idle_machine": ("asserted by the operator; this script must be run "
                         "serialised, with no other experiment running"),
    },
    "environment": environment(),
    "disclosure_A6": ("The submitted 27% figure came from a legacy notebook "
                      "that summed stage timings sequentially while claiming "
                      "a parallel architecture, and compared arms that paid "
                      "different preprocessing (their PCA, our none). Both "
                      "arms here pay their own transform inside the timer."),
})

# ---- trade-off table: quality against latency -----------------------------
try:
    base = pd.read_csv(RESULTS / "baselines_cic-ids2017.csv")
    key = {"ours-submitted (parallel, OCSVM, 64 feat)": "ours-submitted",
           "verkerken (sequential, OCSVM+PCA, 64 feat)": "verkerken-balanced"}
    t = []
    for _, r in df.iterrows():
        arm = key.get(r["arm"])
        m = base[base.arm == arm]
        t.append({
            "arm": r["arm"],
            "balanced_accuracy": (float(m["balanced_accuracy_mean"].iloc[0])
                                  if not m.empty else np.nan),
            "balanced_accuracy_std": (float(m["balanced_accuracy_std"].iloc[0])
                                      if not m.empty else np.nan),
            "parallel_measured_median_s": r["parallel_measured_median_s"],
            "throughput_flows_per_s": r["throughput_flows_per_s"],
        })
    tr_df = pd.DataFrame(t)
    write("latency_tradeoff", tr_df)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ok = tr_df.dropna(subset=["balanced_accuracy"])
    ax.errorbar(ok["parallel_measured_median_s"], ok["balanced_accuracy"],
                yerr=ok["balanced_accuracy_std"], fmt="o", capsize=4,
                markersize=9)
    for _, r in ok.iterrows():
        ax.annotate(r["arm"].split(" (")[0],
                    (r["parallel_measured_median_s"], r["balanced_accuracy"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel("Inference latency, measured parallel execution (s)")
    ax.set_ylabel("Balanced accuracy (mean ± std, 5 seeds)")
    ax.set_title("Detection quality against inference latency")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "latency.png", dpi=200)
    print(f"-> {RESULTS / 'latency.png'}")
except Exception as exc:                                   # pragma: no cover
    print(f"!! trade-off plot skipped: {exc}")

print()
print(df[["arm", "sequential_total_s", "parallel_measured_median_s",
          "saving_measured_pct", "throughput_flows_per_s"]].to_string(index=False))
print("\n07-OK")
