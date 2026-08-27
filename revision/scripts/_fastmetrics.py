"""Integer-coded fast path for the subset of nids.eval.metrics.evaluate that
the tuning sweep reads.

Not a script: import only.

WHY THIS EXISTS
    `metrics.evaluate` costs ~3.3 s per call on the 97,061-row validation
    stream, because it pushes object-dtype string arrays through the whole
    sklearn metric suite. The threshold sweep in 13_tune_detector.py calls it
    2,520 times per seed, which is ~11.5 hours for 5 seeds - the measurement
    cost, not the model cost, would have dominated the experiment.

    Here the label space is encoded to small integers ONCE, and the five
    metrics the sweep actually reads are computed from a single confusion
    matrix built with np.bincount. Same definitions, ~500x faster.

FAITHFULNESS
    This includes the phantom-class guard from `evaluate` - balanced accuracy
    is computed over classes PRESENT IN y_true only, with all phantom
    predictions folded into one bucket, exactly as the shared path does. That
    guard is the correction that cost ~1/7 of the balanced accuracy twice
    before; reimplementing metrics without it is precisely the mistake the
    note in `metrics.evaluate` warns about, so `verify_against_reference`
    below asserts agreement to 1e-12 and 13_tune_detector.py calls it before
    trusting a single number.
"""
from __future__ import annotations

import numpy as np

PHANTOM = "__phantom__"


class FastEvaluator:
    """Pre-encodes y_true once; scores many y_pred arrays cheaply."""

    def __init__(self, y_true: np.ndarray, *, benign_label: str = "Benign",
                 label_space: list[str] | None = None):
        y_true = np.asarray(y_true)
        self.benign_label = benign_label
        present = sorted(set(y_true.tolist()))
        # Every label any prediction might carry. Anything outside `present`
        # is a phantom class and folds into one bucket, as evaluate() does.
        extra = [l for l in (label_space or []) if l not in present]
        self.classes = present + [PHANTOM]
        self.index = {c: i for i, c in enumerate(present)}
        self.phantom_idx = len(present)
        for l in extra:
            self.index[l] = self.phantom_idx
        self.n_class = len(self.classes)
        self.yt = np.array([self.index[v] for v in y_true], dtype=np.int64)
        self.n_present = len(present)
        self.benign_idx = self.index.get(benign_label)
        # Per-true-class counts, for recall denominators.
        self.true_counts = np.bincount(self.yt, minlength=self.n_class)
        self.support = self.true_counts[:self.n_present].astype(float)
        self.n = len(self.yt)

    def encode(self, y_pred: np.ndarray) -> np.ndarray:
        idx = self.index
        ph = self.phantom_idx
        return np.fromiter((idx.get(v, ph) for v in y_pred),
                           dtype=np.int64, count=len(y_pred))

    def evaluate(self, y_pred_enc: np.ndarray) -> dict:
        """Metrics from one confusion matrix. y_pred_enc is already encoded."""
        cm = np.bincount(self.yt * self.n_class + y_pred_enc,
                         minlength=self.n_class ** 2).reshape(self.n_class,
                                                              self.n_class)
        # Rows = true class. Only classes present in y_true count toward
        # balanced accuracy (the phantom guard).
        real = cm[:self.n_present]
        tp = np.diag(cm)[:self.n_present].astype(float)
        recall = np.divide(tp, self.support,
                           out=np.zeros(self.n_present), where=self.support > 0)
        bal_acc = float(recall[self.support > 0].mean())

        pred_counts = real.sum(axis=0).astype(float)      # over real rows only
        prec_den = pred_counts[:self.n_present]
        precision = np.divide(tp, prec_den, out=np.zeros(self.n_present),
                              where=prec_den > 0)
        den = precision + recall
        f1 = np.divide(2 * precision * recall, den,
                       out=np.zeros(self.n_present), where=den > 0)
        n_real = float(self.support.sum())
        # f1_macro follows sklearn's `average='macro'`, which averages over the
        # UNION of true and predicted labels. A phantom class (e.g. Zero Day,
        # which validation structurally cannot contain) is therefore a real
        # zero-F1 member of that average. This is deliberately NOT the phantom
        # guard applied to balanced accuracy: `metrics.evaluate` guards only
        # balanced accuracy, and the fast path must reproduce it, warts and all.
        n_pred_phantom = int(cm[:, self.phantom_idx].sum() > 0)
        f1_macro = float(f1.sum() / (self.n_present + n_pred_phantom))
        f1_weighted = float((f1 * self.support).sum() / n_real) if n_real else 0.0
        accuracy = float(tp.sum() / n_real) if n_real else 0.0

        out = {"accuracy": accuracy, "balanced_accuracy": bal_acc,
               "f1_macro": f1_macro, "f1_weighted": f1_weighted}

        if self.benign_idx is not None:
            nb = float(self.support[self.benign_idx])
            fp = float(nb - cm[self.benign_idx, self.benign_idx])
            out["benign_fpr"] = fp / nb if nb else 0.0
            out["benign_fp_count"] = fp
            atk_mask = np.ones(self.n_present, dtype=bool)
            atk_mask[self.benign_idx] = False
            n_atk = float(self.support[atk_mask].sum())
            caught = n_atk - float(real[atk_mask, self.benign_idx].sum())
            out["attack_detection_rate"] = caught / n_atk if n_atk else 0.0
            out["attacks_missed"] = n_atk - caught
        return out


def verify_against_reference(y_true, preds, *, benign_label="Benign",
                             label_space=None, tol=1e-12) -> list[str]:
    """Assert the fast path reproduces metrics.evaluate on real predictions.

    Returns the list of metric names checked. Raises on any disagreement.
    """
    from nids.eval import metrics as M

    fe = FastEvaluator(y_true, benign_label=benign_label,
                       label_space=label_space)
    checked = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
               "benign_fpr", "attack_detection_rate"]
    for p in preds:
        ref = M.evaluate(y_true, p, seed=0).metrics
        got = fe.evaluate(fe.encode(p))
        for k in checked:
            if k not in ref:
                continue
            if abs(float(ref[k]) - float(got[k])) > tol:
                raise AssertionError(
                    f"fast metrics disagree on {k}: reference={ref[k]!r} "
                    f"fast={got[k]!r}")
    return checked
