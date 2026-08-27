"""Stage 1a and Stage 1b metrics on the CURRENT pipeline and partition.

Tables 3 and 4 of the manuscript still carry the values measured for the
original submission (41,857/2,203 partition). The revision changed the
partition (41,838/2,202), the classifier training pool (SMOTE) and the
figures already reflect that; the two stage tables never did. This measures
them under exactly the construction 22_figures.py uses, seed 0, plus the
detector's threshold-free ranking metrics.

OUTPUT
    revision/results/stage_metrics.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision" / "scripts"))

from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from _common import apply_smote_classifier, load_clean_any
from nids import config
from nids.data import prepare
from nids.experiment import (ClassifierConfig, DetectorConfig,
                             ExperimentConfig, fit_and_select)

BENIGN, ZERO_DAY, UNKNOWN = "Benign", "Zero Day", "Unknown"

CFG = ExperimentConfig(
    name="ours-smote", architecture="parallel",
    detector=DetectorConfig(kind="ocsvm", n_train=10_000,
                            gamma=2.93e-4, nu=1.3e-6),
    classifier=ClassifierConfig(kind="rf", n_estimators=200),
    tau_b_beta="F9", tau_u_quantile="0.95", tau_m_objective="f1")

SEED = 0

ds = load_clean_any("cic-ids2017")
split = prepare.make_split(ds, seed=config.SPLIT_SEED, imbalance="downsample")
feats = split.feature_cols
test = split.test
y = np.where(np.isin(test["Label"].values, config.DEFAULT_ZERO_DAY),
             ZERO_DAY, test["Attack Type"].values)
X = test[feats].values
print(f"test rows: {len(test):,}", flush=True)

fitted = fit_and_select(split, CFG, SEED)
apply_smote_classifier(
    fitted, ds, feature_cols=feats, seed=SEED,
    exclude_row_ids=set(test["row_id"]) | set(split.val_malicious["row_id"])
    | set(split.val_benign["row_id"]))
th = fitted.thresholds

rows = []

# ---- Stage 1a: detector alone, benign vs malicious ------------------------
# Tables 3 and 4 quote a training and an inference time for each stage. Time
# them here, on this machine, rather than carrying figures forward: fit on the
# same 10,000-row benign sample the detector is trained on, score the same
# evaluation stream every other row of the table is measured over.
ben = split.train_benign[feats].values[:CFG.detector.n_train]
_t = time.perf_counter()
Detector0 = type(fitted.detector)
_probe = Detector0(CFG.detector).fit(ben)
fit_ocsvm_s = time.perf_counter() - _t
_t = time.perf_counter()
_probe.score(X)
score_ocsvm_s = time.perf_counter() - _t

scores = fitted.detector.score(X)
y_bin = np.where(y == BENIGN, 0, 1)           # 1 = malicious
# anomaly direction: higher score = more anomalous? tau_b comparison in the
# figure script: predict malicious where score > tau_b.
p_bin = (scores > th.tau_b).astype(int)
auroc = roc_auc_score(y_bin, scores)
aupr = average_precision_score(y_bin, scores)
rows.append({
    "stage": "1a", "tau_b": th.tau_b, "auroc": auroc, "aupr": aupr,
    "accuracy": accuracy_score(y_bin, p_bin),
    "balanced_accuracy": balanced_accuracy_score(y_bin, p_bin),
    "precision": precision_score(y_bin, p_bin),
    "recall": recall_score(y_bin, p_bin),
    "f1": f1_score(y_bin, p_bin),
    "f1_macro": f1_score(y_bin, p_bin, average="macro"),
    "f1_weighted": f1_score(y_bin, p_bin, average="weighted"),
    "fit_s": fit_ocsvm_s, "score_s": score_ocsvm_s,
})
print("1a:", {k: round(v, 4) for k, v in rows[-1].items()
              if isinstance(v, float)}, flush=True)

# ---- Stage 1b: classifier alone, its own label space ----------------------
# Stage 1b is trained on malicious traffic only and therefore cannot emit
# Benign: abstention IS its correct answer on benign and on zero-day rows,
# so both are scored as Unknown. Keeping Benign as a reachable label instead
# makes every benign row an automatic error and drives accuracy to 0.046 -
# a property of the label space, not of the classifier. Figure 2(b) keeps
# the benign row because a confusion matrix shows where mass lands; an
# aggregate metric over the same space would not mean anything.
_t = time.perf_counter()
pred_clf = fitted.classifier.predict_with_unknown(X, th.tau_m)
score_rf_s = time.perf_counter() - _t

# The classifier reported in Table 4 is the SMOTE-trained one, so time the fit
# on the resampled pool rather than on the downsampled split.
from nids.stages.classifier import Classifier                      # noqa: E402
from imblearn.over_sampling import SMOTE                           # noqa: E402
_pool = prepare.make_split(ds, seed=config.SPLIT_SEED, imbalance="none")
_Xc = _pool.train_malicious[feats].values
_yc = _pool.train_malicious["Attack Type"].values
import pandas as _pd
_k = max(1, min(5, int(_pd.Series(_yc).value_counts().min()) - 1))
_t = time.perf_counter()
_Xr, _yr = SMOTE(random_state=SEED, k_neighbors=_k).fit_resample(_Xc, _yc)
resample_rf_s = time.perf_counter() - _t
_t = time.perf_counter()
Classifier(CFG.classifier).fit(_Xr, _yr)
fit_rf_s = time.perf_counter() - _t
print(f"  [1b] SMOTE pool {len(_yr):,} rows, resample {resample_rf_s:.1f}s, "
      f"fit {fit_rf_s:.1f}s", flush=True)
y_clf = np.where((y == ZERO_DAY) | (y == BENIGN), UNKNOWN, y)
rows.append({
    "stage": "1b", "tau_m": th.tau_m,
    "accuracy": accuracy_score(y_clf, pred_clf),
    "balanced_accuracy": balanced_accuracy_score(y_clf, pred_clf),
    "f1_macro": f1_score(y_clf, pred_clf, average="macro"),
    "f1_weighted": f1_score(y_clf, pred_clf, average="weighted"),
    "fit_s": fit_rf_s, "score_s": score_rf_s,
    "resample_s": resample_rf_s, "n_train": len(_yr),
})
print("1b:", {k: round(v, 4) for k, v in rows[-1].items()
              if isinstance(v, float)}, flush=True)

# ---- Stage 1a, autoencoder column (design-choice comparison) --------------
from nids.stages.detector import Detector

_t = time.perf_counter()
ae = Detector(DetectorConfig(kind="autoencoder", n_train=10_000, seed=SEED)
              ).fit(split.train_benign[feats].values)
fit_ae_s = time.perf_counter() - _t
_t = time.perf_counter()
s_ae = ae.score(X)
score_ae_s = time.perf_counter() - _t
auroc_ae = roc_auc_score(y_bin, s_ae)
aupr_ae = average_precision_score(y_bin, s_ae)
# Table 3 reports the AE with a threshold too, selected the same way (F9 on
# validation) so the two columns are comparable.
s_ae_val = ae.score(split.val_benign[feats].values)
s_ae_val_m = ae.score(split.val_malicious[feats].values)
best_f, tau_ae = -1.0, 0.0
for t in np.quantile(np.r_[s_ae_val, s_ae_val_m], np.linspace(0.01, 0.99, 99)):
    tp = int((s_ae_val_m > t).sum()); fn = int(len(s_ae_val_m) - tp)
    fp = int((s_ae_val > t).sum())
    f = 82 * tp / (82 * tp + 81 * fn + fp + 1e-9)
    if f > best_f:
        best_f, tau_ae = f, float(t)
p_ae = (s_ae > tau_ae).astype(int)
rows.append({
    "stage": "1a-autoencoder", "tau_b": tau_ae, "auroc": auroc_ae,
    "aupr": aupr_ae,
    "accuracy": accuracy_score(y_bin, p_ae),
    "balanced_accuracy": balanced_accuracy_score(y_bin, p_ae),
    "precision": precision_score(y_bin, p_ae, zero_division=0),
    "recall": recall_score(y_bin, p_ae),
    "f1": f1_score(y_bin, p_ae),
    "f1_macro": f1_score(y_bin, p_ae, average="macro"),
    "f1_weighted": f1_score(y_bin, p_ae, average="weighted"),
    "fit_s": fit_ae_s, "score_s": score_ae_s,
})
print("1a-AE:", {"auroc": round(auroc_ae, 4), "aupr": round(aupr_ae, 4)},
      flush=True)
print(f"timings: ocsvm fit={fit_ocsvm_s:.3f}s score={score_ocsvm_s:.3f}s | "
      f"rf fit={fit_rf_s:.3f}s score={score_rf_s:.3f}s | "
      f"ae fit={fit_ae_s:.3f}s score={score_ae_s:.3f}s", flush=True)

pd.DataFrame(rows).to_csv(ROOT / "revision" / "results" / "stage_metrics.csv",
                          index=False)
print("31-OK")
