"""Command-line entry point.

The definition of done requires "everything reproducible from a clean checkout
with a documented command". This is that command.

    python -m nids.cli manifest  --dataset cic-ids2017
    python -m nids.cli split     --dataset cic-ids2017
    python -m nids.cli phase-a   --dataset cic-ids2017 --trials 40
    python -m nids.cli phase-b   --dataset cic-ids2017 --trials 40
    python -m nids.cli ledger
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

from nids import config


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def cmd_manifest(args) -> int:
    """Record checksums, sizes and row counts for a dataset's raw files."""
    base = config.DATA_RAW / args.dataset
    if not base.exists():
        print(f"No such dataset directory: {base}", file=sys.stderr)
        print("See data/MANIFEST.md for how to obtain it.", file=sys.stderr)
        return 1
    files = sorted(p for p in base.rglob("*.csv"))
    if not files:
        print(f"No CSVs under {base}", file=sys.stderr)
        return 1

    entries = {}
    for p in files:
        n_rows = sum(1 for _ in open(p, "rb")) - 1
        entries[str(p.relative_to(base))] = {
            "bytes": p.stat().st_size,
            "sha256": _sha256(p),
            "rows": n_rows,
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        print(f"{n_rows:>10,} rows  {entries[str(p.relative_to(base))]['sha256'][:16]}…  "
              f"{p.relative_to(base)}")
    out = base / "_manifest.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(entries)
    out.write_text(json.dumps(prev, indent=2))
    print(f"\nTotal rows: {sum(e['rows'] for e in entries.values()):,}")
    print(f"Written: {out}")
    return 0


def _load_split(args):
    from nids.data import prepare, schema
    ds = schema.load(args.dataset, nrows=args.nrows)
    ds = prepare.clean(ds)
    split = prepare.make_split(
        ds, seed=config.SPLIT_SEED, imbalance=args.imbalance)
    return ds, split


def cmd_split(args) -> int:
    """Materialise and describe the frozen split."""
    ds, split = _load_split(args)
    print(f"dataset: {ds.name}   features: {len(split.feature_cols)}   "
          f"seed: {split.seed}")
    print(split.describe().to_string(index=False))
    out = config.RESULTS / f"split-{ds.name}.json"
    out.write_text(json.dumps({
        "seed": split.seed, "meta": split.meta,
        "n_features": len(split.feature_cols),
        "features": split.feature_cols,
        "sizes": {k: int(len(getattr(split, k))) for k in
                  ("train_benign", "train_malicious", "val_benign",
                   "val_malicious", "test")},
        "clean_report": ds.provenance.get("clean", {}),
    }, indent=2, default=str))
    print(f"\nWritten: {out}")
    return 0


def _phase(args, which: str) -> int:
    """Shared driver for Phase A (theirs) and Phase B (ours)."""
    from nids import experiment as E, tuning
    from nids.stages.classifier import ClassifierConfig
    from nids.stages.detector import DetectorConfig

    ds, split = _load_split(args)
    print(f"[{which}] {ds.name}: {len(split.feature_cols)} features, "
          f"split seed {split.seed}")

    use_pca = which == "A"          # Verkerken apply PCA; we do not.
    print(f"[{which}] tuning detector (trials={args.trials}) …", flush=True)
    t_det = tuning.tune_detector(split, kind="ocsvm", use_pca=use_pca,
                                 n_trials=args.trials)
    print(f"        best val AUROC {t_det.best_value:.4f}  {t_det.best_params}")
    print(f"[{which}] tuning classifier (trials={args.trials}) …", flush=True)
    t_clf = tuning.tune_classifier(split, kind="rf", n_trials=args.trials)
    print(f"        best val F1w  {t_clf.best_value:.4f}  {t_clf.best_params}")
    tuning.save_tuning(f"phase{which.lower()}-{ds.name}",
                       {"detector": t_det, "classifier": t_clf})

    det = DetectorConfig(kind="ocsvm", use_pca=use_pca,
                         **{k: v for k, v in t_det.best_params.items()
                            if k in {"gamma", "nu", "n_components"}})
    clf = ClassifierConfig(kind="rf", **t_clf.best_params)

    if which == "A":
        # Verkerken's three Table V configurations differ in tau_B's F-beta
        # optimum and the tau_U quantile.
        variants = [
            ("verkerken-max-fscore", "verkerken", "F5", "0.995"),
            ("verkerken-max-bacc", "verkerken", "F9", "0.95"),
            ("verkerken-balanced", "verkerken", "F5", "0.99"),
        ]
    else:
        variants = [("ours-parallel", "parallel", "F9", "0.95")]

    rows = []
    for name, arch, beta, quant in variants:
        cfg = E.ExperimentConfig(name=name, architecture=arch, detector=det,
                                 classifier=clf, tau_b_beta=beta,
                                 tau_u_quantile=quant)
        agg, runs = E.run_multiseed(split, cfg, seeds=config.MODEL_SEEDS,
                                    on="validation")
        E.save(name, agg, runs, cfg, split, "validation")
        row = {"config": name}
        for m in ("balanced_accuracy", "f1_weighted", "f1_macro",
                  "zero_day_recall", "benign_fpr"):
            if m in agg.index:
                row[m] = f"{agg.loc[m,'mean']:.4f}±{agg.loc[m,'std']:.4f}"
        rows.append(row)
        print(f"  {name}: " + "  ".join(f"{k}={v}" for k, v in row.items()
                                        if k != "config"), flush=True)

    table = pd.DataFrame(rows)
    out = config.RESULTS / f"phase{which.lower()}-{ds.name}-validation.csv"
    table.to_csv(out, index=False)
    print(f"\nWritten: {out}")
    return 0


def cmd_phase_a(args) -> int:
    return _phase(args, "A")


def cmd_phase_b(args) -> int:
    return _phase(args, "B")


def cmd_ledger(args) -> int:
    from nids.eval import testguard
    print(testguard.summary())
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="nids", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--dataset", default="cic-ids2017")
        sp.add_argument("--nrows", type=int, default=None,
                        help="cap rows per file (development only)")
        sp.add_argument("--imbalance", default="downsample",
                        choices=["downsample", "weighted", "none"])
        return sp

    m = sub.add_parser("manifest", help="record checksums + row counts")
    m.add_argument("--dataset", default="cic-ids2017")
    m.set_defaults(func=cmd_manifest)

    common(sub.add_parser("split", help="build and describe the frozen split")
           ).set_defaults(func=cmd_split)

    for nm, fn, helptext in (("phase-a", cmd_phase_a, "reproduce Verkerken et al."),
                             ("phase-b", cmd_phase_b, "reproduce our architecture")):
        sp = common(sub.add_parser(nm, help=helptext))
        sp.add_argument("--trials", type=int, default=40,
                        help="Optuna trials; identical across architectures")
        sp.set_defaults(func=fn)

    sub.add_parser("ledger", help="show test-set spend ledger"
                   ).set_defaults(func=cmd_ledger)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
