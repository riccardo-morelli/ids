"""Stage 1a - the anomaly detector.

Trained on benign traffic only, emits an anomaly score. Both architectures use
this stage; they differ in preprocessing (Verkerken apply PCA, we do not) and
in how the score is consumed downstream.

Threshold selection is the part the legacy code got wrong twice, so it is
handled here explicitly:

1. The F-beta sweep runs on the **validation** frames only. The legacy notebook
   folded `train_balanced_malicious` into the malicious validation set first
   (`detector_ocsvm.ipynb` cell 5), selecting the threshold partly on data the
   model trained beside.
2. The selected value is *returned*, not hard-coded. The legacy notebook printed
   a sweep table and then set `threshold = 0.000150` as a literal on the next
   line (cell 17), so the reported "value maximising F9" was never actually
   wired to the sweep.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity, LocalOutlierFactor, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from nids import config


@dataclass
class DetectorConfig:
    """A candidate for stage 1a.

    The `kind` families differ in what they assume "anomalous" means, which is
    the crux of this problem. The diagnosis in cycle 2 showed the undetected
    attack classes are *more compact and closer to the benign centre* than
    benign traffic is — so every distance-based family ("anomalous = far from
    the centre") is structurally unable to separate them, however it is tuned.

    | family | assumption | expected to work here |
    |---|---|---|
    | ocsvm, mahalanobis, lof | anomalous = far / low local density | no, by the diagnosis |
    | iforest | anomalous = easy to isolate | partly — isolation is not distance |
    | kde, knn_density | anomalous = *unusual density*, high or low | plausible |
    | autoencoder, pca_recon | anomalous = hard to reconstruct | plausible, no distance assumption |
    """

    kind: str = "ocsvm"
    use_pca: bool = False            # Verkerken: True. Ours: False.
    n_components: int = 56           # Verkerken's tuned value
    n_train: int = 10_000            # subsample of benign training rows
    # OC-SVM
    gamma: float | str = 0.05
    nu: float = 2.75e-05
    # Autoencoder / PCA reconstruction
    hidden: int = 29
    max_iter: int = 200
    # Isolation Forest
    n_estimators: int = 200
    max_samples: int | float = 256
    contamination: float = 0.01
    # LOF / kNN density
    n_neighbors: int = 20
    # KDE
    bandwidth: float = 1.0
    #: For 'extremeness': how the per-feature extremeness scores combine.
    #: 'mean' averages them; 'n_extreme' counts how many features sit past the
    #: 99th benign percentile. NOT 'max' — that saturates, because 100% of
    #: benign flows are past the extreme quantile in at least one of 36
    #: dimensions, so the maximum carries no information at all (measured in
    #: cycle 13: AUROC exactly 0.5000 for every class).
    extremeness_agg: str = "mean"
    #: For 'extremeness': per-feature signs, +1 or -1, one per column. A -1
    #: feature has its extremeness read as `1 - e`, because on that feature
    #: being unusually *typical* is what marks an attack.
    #:
    #: The plain mean assumes every feature points the same way. Cycle 14
    #: measured that roughly 10 of 36 do not: the `control` group — TCP flags,
    #: initial window sizes, header lengths — separates the withheld class at
    #: AUROC 0.068 on Web Attack XSS, strongly informative but backwards.
    #: Automated attacks drive protocol machinery *more regularly* than human
    #: traffic, so their extremeness is low precisely where it matters, and in
    #: a plain mean those features cancel against the rest.
    #:
    #: Orienting them lifts mean AUROC from 0.7684 to 0.8331 across all five
    #: folds, with the largest gains on the folds that were worst (DDoS
    #: 0.6717 → 0.7947, FTP-Patator 0.5683 → 0.6597). Set it with
    #: `fit_orientation`, which estimates the signs from benign versus
    #: **known** attacks — never from the withheld class, so the benign-only
    #: training contract of stage 1a is untouched: `fit` still sees no
    #: malicious data, and the signs are a separate, later calibration.
    orientation: np.ndarray | None = None
    # PCA minor/Shyu: fraction of variance defining the "major" subspace, and
    # the eigenvalue below which a component counts as "minor" (Shyu et al.
    # use 50% and 0.20 respectively on standardised data).
    major_var: float = 0.50
    minor_eigen: float = 0.20
    #: Flag *both* tails of the score distribution, not just the high one.
    #:
    #: The cycle-2 diagnosis found attack traffic sitting in regions that are
    #: MORE typical than typical benign traffic - more compact and closer to
    #: the centre. Every one-sided score ("bigger = more anomalous") ranks such
    #: a class as maximally normal, which is how Brute Force reached AUROC
    #: 0.22. Verified on a synthetic control where a compact cluster sits
    #: inside the normal cloud: one-sided scores give AUROC 0.000 for every
    #: family tried (ocsvm, pca_recon, pca_minor, pca_shyu), while the
    #: two-sided form gives 0.984 without losing anything on a distant
    #: anomaly (1.000).
    #:
    #: Applies to any family whose raw score is a scalar per row.
    two_sided: bool = False
    seed: int = 0
    params: dict = field(default_factory=dict)


class Detector:
    """Benign-only anomaly detector. Higher score = more anomalous."""

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.model = None

    def _transform(self, X: np.ndarray, *, fit: bool = False) -> np.ndarray:
        X = self.scaler.fit_transform(X) if fit else self.scaler.transform(X)
        if self.cfg.use_pca:
            if fit:
                self.pca = PCA(
                    n_components=min(self.cfg.n_components, X.shape[1]),
                    random_state=self.cfg.seed,
                )
                X = self.pca.fit_transform(X)
            else:
                X = self.pca.transform(X)
        return X

    def fit(self, X_benign: np.ndarray) -> "Detector":
        rng = np.random.RandomState(self.cfg.seed)
        X = np.asarray(X_benign, dtype=np.float64)
        if self.cfg.n_train and len(X) > self.cfg.n_train:
            X = X[rng.choice(len(X), self.cfg.n_train, replace=False)]
        Xt = self._transform(X, fit=True)

        k = self.cfg.kind
        if k == "ocsvm":
            self.model = OneClassSVM(
                kernel="rbf", gamma=self.cfg.gamma, nu=self.cfg.nu,
                tol=1e-3, shrinking=True, **self.cfg.params,
            ).fit(Xt)
        elif k == "autoencoder":
            # An MLP regressing its own input is a plain autoencoder; the
            # reconstruction error is the anomaly score. Verkerken use a Keras
            # DAE - sklearn keeps the harness single-dependency, and the stage
            # contract (benign-only -> score) is identical.
            self.model = MLPRegressor(
                hidden_layer_sizes=(self.cfg.hidden,),
                activation="relu", solver="adam",
                max_iter=self.cfg.max_iter, random_state=self.cfg.seed,
                early_stopping=True, **self.cfg.params,
            ).fit(Xt, Xt)
        elif k == "pca_recon":
            # Reconstruction error from a low-rank benign subspace. The linear
            # analogue of the autoencoder: cheap, deterministic, and it makes
            # no assumption that anomalies are far from the centre.
            self.model = PCA(n_components=min(self.cfg.hidden, Xt.shape[1]),
                             random_state=self.cfg.seed).fit(Xt)
        elif k in {"pca_minor", "pca_shyu"}:
            # Shyu et al., "A Novel Anomaly Detection Scheme Based on Principal
            # Component Classifier": the MINOR components - those with small
            # eigenvalues - reveal anomalies the major components cannot, and
            # combining both beats either.
            #
            # This is the family the cycle-2 diagnosis points at. An attack
            # class that is *more compact* than benign traffic has low variance
            # along the very directions that distinguish it, so keeping only
            # high-variance components (what plain PCA does) discards the
            # signal. `pca_minor` scores on the residual subspace alone;
            # `pca_shyu` sums the normalised major and minor distances.
            self.model = PCA(n_components=None, random_state=self.cfg.seed).fit(Xt)
            ev = self.model.explained_variance_
            cum = np.cumsum(ev) / ev.sum()
            # Major: components covering the first `major_var` of variance.
            self._n_major = int(np.searchsorted(cum, self.cfg.major_var) + 1)
            self._n_major = min(max(self._n_major, 1), len(ev) - 1)
            # Minor: components whose eigenvalue falls below the threshold.
            self._minor_mask = ev < self.cfg.minor_eigen
            if not self._minor_mask.any():
                # Fall back to the smallest quarter so the score stays defined.
                cut = max(1, len(ev) // 4)
                self._minor_mask = np.zeros(len(ev), dtype=bool)
                self._minor_mask[-cut:] = True
            self._ev = np.maximum(ev, 1e-12)
        elif k == "extremeness":
            # Nessun modello da addestrare: bastano i quantili benigni.
            self._srt = np.sort(Xt, axis=0)
            self.model = "extremeness"
        elif k in {"ecod", "copod"}:
            # PyOD's calibrated versions of the per-feature tail probability
            # that `extremeness` approximates by hand. Both are parameter-free,
            # which is the point: no bandwidth to tune, so no threshold of ours
            # can quietly absorb a tuning decision.
            from pyod.models.copod import COPOD
            from pyod.models.ecod import ECOD
            self.model = (ECOD() if k == "ecod" else COPOD())
            self.model.fit(Xt)
        elif k == "iforest":
            self.model = IsolationForest(
                n_estimators=self.cfg.n_estimators,
                max_samples=self.cfg.max_samples,
                contamination=self.cfg.contamination,
                random_state=self.cfg.seed, n_jobs=-1, **self.cfg.params,
            ).fit(Xt)
        elif k == "lof":
            self.model = LocalOutlierFactor(
                n_neighbors=self.cfg.n_neighbors, novelty=True, n_jobs=-1,
                **self.cfg.params).fit(Xt)
        elif k == "mahalanobis":
            # Robust covariance: MinCovDet resists the heavy benign tail that
            # would otherwise inflate the covariance and hide everything.
            est = MinCovDet(random_state=self.cfg.seed, support_fraction=0.9) \
                if self.cfg.params.get("robust", True) else EmpiricalCovariance()
            self.model = est.fit(Xt)
        elif k == "kde":
            self.model = KernelDensity(
                bandwidth=self.cfg.bandwidth, kernel="gaussian").fit(Xt)
        elif k == "knn_density":
            # Distance to the k-th nearest benign neighbour. Large = sparse
            # region; small = unusually dense region. With two_sided, both ends
            # count as anomalous, which is what the compact attack classes need.
            self.model = NearestNeighbors(
                n_neighbors=self.cfg.n_neighbors, n_jobs=-1).fit(Xt)
        else:
            raise ValueError(f"unknown detector kind {k!r}")

        # A two-sided score needs a reference for what "typical" means, taken
        # from the benign training rows the detector was fitted on.
        if self.cfg.two_sided:
            self._ref_median = float(np.median(self._raw_score(Xt)))
        return self

    def _raw_score(self, Xt: np.ndarray) -> np.ndarray:
        """One-sided score in the model's native orientation.

        Higher means more anomalous for every family except the density ones,
        where the two-sided wrapper in `score` handles the orientation.
        """
        k = self.cfg.kind
        if k == "ocsvm":
            return -self.model.decision_function(Xt)
        if k == "autoencoder":
            return ((Xt - self.model.predict(Xt)) ** 2).sum(axis=1)
        if k == "pca_recon":
            recon = self.model.inverse_transform(self.model.transform(Xt))
            return ((Xt - recon) ** 2).sum(axis=1)
        if k == "extremeness":
            # Media dell'estremita' per-feature rispetto ai quantili benigni.
            #
            # Il ciclo 13 ha misurato che le classi che il sistema non vede
            # (slowloris, Slowhttptest) SONO separabili da feature esistenti -
            # slowloris ha separabilita' massima 0.7800 contro 0.8632 di DoS
            # Hulk, che invece viene rilevato all'85%. Il segnale c'e' e la
            # densita' in 36 dimensioni lo annega: e' la diluizione del ciclo 3
            # vista dal lato del detector.
            #
            # Qui ogni feature contribuisce la propria estremita' bilaterale
            # (0 alla mediana benigna, 1 alle code) e si aggrega. Una classe
            # che e' moderatamente insolita su molte feature emerge, mentre la
            # distanza in 36 dimensioni la media via.
            #
            # NON si usa il massimo: satura, perche' il 100% dei flussi benigni
            # e' oltre il quantile estremo in almeno una dimensione.
            out = self._per_feature_extremeness(Xt)
            # Ogni feature nella propria direzione, quando e' nota. Senza
            # questo, le ~10 feature su 36 che segnalano al contrario si
            # cancellano contro le altre nella media (ciclo 14: AUROC medio
            # 0.7684 senza, 0.8331 con, e nessun fold peggiora).
            o = self.cfg.orientation
            if o is not None:
                o = np.asarray(o, dtype=np.float64)
                out = np.where(o[None, :] > 0, out, 1.0 - out)
            if self.cfg.extremeness_agg == "n_extreme":
                return (out > 0.99).mean(axis=1)
            return out.mean(axis=1)
        if k in {"pca_minor", "pca_shyu"}:
            # Project onto the benign principal axes and normalise each score
            # by its eigenvalue, so every direction contributes comparably
            # rather than the high-variance ones dominating.
            Y = self.model.transform(Xt)
            norm = (Y ** 2) / self._ev
            minor = norm[:, self._minor_mask].sum(axis=1)
            if k == "pca_minor":
                return minor
            major = norm[:, : self._n_major].sum(axis=1)
            return major + minor
        if k in {"ecod", "copod"}:
            # decision_function is already "higher = more anomalous".
            return self.model.decision_function(Xt)
        if k == "iforest":
            return -self.model.score_samples(Xt)
        if k == "lof":
            return -self.model.score_samples(Xt)
        if k == "mahalanobis":
            return self.model.mahalanobis(Xt)
        if k == "kde":
            # Log-density: low density = anomalous in the one-sided reading.
            return -self.model.score_samples(Xt)
        if k == "knn_density":
            d, _ = self.model.kneighbors(Xt)
            return d[:, -1]
        raise ValueError(f"unknown detector kind {k!r}")

    def fit_orientation(self, X_benign: np.ndarray,
                        X_known_malicious: np.ndarray) -> np.ndarray:
        """Estimate and store per-feature signs for the 'extremeness' kind.

        Called *after* `fit`, and deliberately separate from it: `fit` sees
        benign traffic only, which is the premise that makes the zero-day
        claim credible. The signs are a calibration on top, using known
        attacks in the same way `MechanismDetector.fit_weights` does — to
        learn *how to read* each feature, never what any particular class
        looks like. A withheld class contributes nothing here, so a sign
        learned from Port Scan applies unchanged to an attack nobody has seen.
        """
        from sklearn.metrics import roc_auc_score

        eb = self._per_feature_extremeness(
            self._transform(np.asarray(X_benign, dtype=np.float64)))
        em = self._per_feature_extremeness(
            self._transform(np.asarray(X_known_malicious, dtype=np.float64)))
        y = np.r_[np.zeros(len(eb)), np.ones(len(em))]
        E = np.vstack([eb, em])
        self.cfg.orientation = np.array(
            [1.0 if roc_auc_score(y, E[:, j]) >= 0.5 else -1.0
             for j in range(E.shape[1])])
        return self.cfg.orientation

    def _per_feature_extremeness(self, Xt: np.ndarray) -> np.ndarray:
        """Two-sided extremeness of each feature within benign quantiles."""
        out = np.empty_like(Xt)
        for j in range(Xt.shape[1]):
            p = np.searchsorted(self._srt[:, j], Xt[:, j]) / len(self._srt)
            out[:, j] = np.abs(p - 0.5) * 2
        return out

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score; higher means more anomalous."""
        Xt = self._transform(np.asarray(X, dtype=np.float64))
        s = self._raw_score(Xt)

        if self.cfg.two_sided:
            # Deviation from the *typical* benign score in either direction:
            # a region that is anomalously ordinary is as suspicious as one
            # that is anomalously extreme. See the note on `two_sided` in
            # DetectorConfig for the evidence.
            s = np.abs(s - self._ref_median)
        return s


def select_threshold_fbeta(
    scores_benign: np.ndarray,
    scores_malicious: np.ndarray,
    *,
    betas: range | tuple = range(1, 10),
    n_grid: int = 1000,
) -> dict:
    """Pick tau_B by maximising F-beta on VALIDATION scores.

    Returns every beta's optimum so the choice is inspectable, plus the
    selected threshold per beta. Both papers sweep beta in [1, 9]: higher beta
    weights recall, which is what a security setting wants.
    """
    y = np.r_[np.zeros(len(scores_benign)), np.ones(len(scores_malicious))]
    s = np.r_[np.asarray(scores_benign), np.asarray(scores_malicious)]

    # Grid over observed quantiles rather than a fixed numeric window: the
    # legacy code searched a hard-coded [-0.001, 0.001], which silently
    # excludes the optimum whenever the score scale differs (e.g. under PCA,
    # or for an autoencoder whose errors are positive and unbounded).
    grid = np.unique(np.quantile(s, np.linspace(0, 1, n_grid)))

    # Vectorised sweep. Sorting once and taking suffix sums gives TP and FP at
    # every candidate threshold simultaneously, which is exactly what looping
    # fbeta_score over the grid computes one threshold at a time. Same answer,
    # but O(n log n + |grid|) instead of O(|grid| * n) - and this runs for
    # every beta, every seed, every fold of leave-one-class-out.
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    n_pos = float(y.sum())

    # For threshold t, predictions are (s >= t). Using searchsorted on the
    # sorted scores, the count of predicted-positive is n - idx.
    idx = np.searchsorted(s_sorted, grid, side="left")
    tp_suffix = np.concatenate([np.cumsum(y_sorted[::-1])[::-1], [0.0]])
    tp = tp_suffix[idx]
    predicted_pos = len(s) - idx
    fp = predicted_pos - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float),
                          where=(tp + fp) > 0)
    recall = tp / n_pos if n_pos else np.zeros_like(tp, dtype=float)

    out = {}
    for beta in betas:
        b2 = float(beta) ** 2
        denom = b2 * precision + recall
        f = np.divide((1 + b2) * precision * recall, denom,
                      out=np.zeros_like(denom, dtype=float), where=denom > 0)
        best = int(np.argmax(f))
        out[f"F{beta}"] = {"threshold": float(grid[best]), "score": float(f[best])}
    return out


def select_tau_u(scores_benign_val: np.ndarray,
                 quantiles: tuple = (0.95, 0.975, 0.99, 0.995),
                 scores_malicious_val: np.ndarray | None = None) -> dict:
    """Zero-day threshold candidates as quantiles of BENIGN validation scores.

    Setting tau_U from benign scores bounds the false-positive rate directly:
    the q-quantile admits at most (1-q) of benign traffic as zero-day.

    **The trap this hit.** A benign quantile only separates anything if the
    malicious score distribution actually extends past it. That holds for the
    OCSVM score the criterion was designed around, but not for every score: with
    the two-sided knn_density detector, tau_U at the 0.95 benign quantile came
    out at 2.333 and *no* held-out attack row reached it, so the zero-day branch
    was dead by construction and every fusion rule scored 0.0000 — a failure
    that looked like a modelling problem and was a scale mismatch.

    Passing `scores_malicious_val` makes the mismatch visible: each candidate
    reports the fraction of known-malicious validation rows it would admit.
    A candidate admitting ~none of them cannot flag a novel class either, and
    `usable` marks it so callers can refuse it instead of silently reporting
    zero recall.
    """
    s = np.asarray(scores_benign_val)
    out: dict = {}
    for q in quantiles:
        tau = float(np.quantile(s, q))
        entry = {"threshold": tau}
        if scores_malicious_val is not None:
            m = np.asarray(scores_malicious_val)
            frac = float((m > tau).mean()) if len(m) else 0.0
            entry["malicious_above"] = frac
            # A threshold no known attack clears will not be cleared by an
            # unknown one either. 5% is a low bar deliberately: it flags a
            # broken scale, not a merely strict threshold.
            entry["usable"] = frac >= 0.05
        out[str(q)] = entry
    # Back-compatible: callers that just want the number keep working.
    for k, v in list(out.items()):
        out[k] = v if scores_malicious_val is not None else v["threshold"]
    return out


def pick_tau_u(scores_benign_val: np.ndarray,
               scores_malicious_val: np.ndarray,
               preferred: str = "0.95") -> float:
    """Pick a usable tau_U, falling back when the preferred quantile is dead.

    Prefers the requested quantile; if no known attack clears it, walks down to
    looser quantiles and finally to a quantile of the *malicious* distribution,
    so the zero-day branch is always reachable by something.
    """
    cands = select_tau_u(scores_benign_val,
                         quantiles=(0.95, 0.9, 0.8, 0.7, 0.5),
                         scores_malicious_val=scores_malicious_val)
    if preferred in cands and cands[preferred]["usable"]:
        return cands[preferred]["threshold"]
    for q in ("0.95", "0.9", "0.8", "0.7", "0.5"):
        if q in cands and cands[q]["usable"]:
            return cands[q]["threshold"]
    # Nothing on the benign side works: anchor on the malicious side instead,
    # accepting a higher false-positive rate over a dead branch.
    return float(np.quantile(np.asarray(scores_malicious_val), 0.25))
