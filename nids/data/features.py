"""Derived features aimed at the failure the diagnosis exposed.

The cycle-2 diagnosis found that the attack classes the detector misses are
*more compact and closer to the benign centre* than benign traffic is: Brute
Force has 5.6x less spread than benign and sits at Mahalanobis distance 3.13
against benign's 3.32. Automated traffic is regular by construction; human
traffic is bursty and heavy-tailed.

CICFlowMeter already ships the raw material for measuring that regularity —
means and standard deviations of inter-arrival times, packet lengths, active
and idle periods — but only as *absolute* quantities. A standard deviation of
1000 microseconds means something completely different for a 10ms flow than
for a 10s flow, so in absolute form the regularity signal is entangled with
flow duration and drowned by the heavy tail.

The fix is elementary and is the standard statistic for exactly this question:
the **coefficient of variation** (std / mean), which is dimensionless and
scale-free. A bot polling on a fixed interval has CV near zero; a human
browsing has CV near or above one. Literature on IoT botnet detection reports
precisely this contrast — compromised devices show fixed inter-packet waits
while ordinary devices show exponentially distributed gaps.

These are *directly observed* features in the sense of Rules of ML rule 17:
computed from what the flow already records, explainable to a reviewer, and
cheap. They are tried before any learned representation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: (name, mean_col, std_col) triples for which a coefficient of variation is
#: meaningful. Each pair is already present in the CIC-IDS2017 feature set.
_CV_PAIRS = [
    ("cv_flow_iat", "Flow IAT Mean", "Flow IAT Std"),
    ("cv_fwd_iat", "Fwd IAT Mean", "Fwd IAT Std"),
    ("cv_bwd_iat", "Bwd IAT Mean", "Bwd IAT Std"),
    ("cv_active", "Active Mean", "Active Std"),
    ("cv_idle", "Idle Mean", "Idle Std"),
    ("cv_pkt_len", "Packet Length Mean", "Packet Length Std"),
    ("cv_fwd_pkt_len", "Fwd Packet Length Mean", "Fwd Packet Length Std"),
    ("cv_bwd_pkt_len", "Bwd Packet Length Mean", "Bwd Packet Length Std"),
]

#: (name, numerator, denominator) ratios. Directional asymmetry and packet
#: shape are behavioural signatures that survive scaling, unlike raw counts.
_RATIOS = [
    ("ratio_bytes_fwd_bwd", "Total Length of Fwd Packets",
     "Total Length of Bwd Packets"),
    ("ratio_pkts_fwd_bwd", "Total Fwd Packets", "Total Backward Packets"),
    ("bytes_per_pkt_fwd", "Total Length of Fwd Packets", "Total Fwd Packets"),
    ("bytes_per_pkt_bwd", "Total Length of Bwd Packets", "Total Backward Packets"),
    ("iat_span_ratio", "Flow IAT Max", "Flow IAT Min"),
    ("pkt_len_span_ratio", "Max Packet Length", "Min Packet Length"),
]


def _safe_div(a: pd.Series, b: pd.Series) -> np.ndarray:
    """a / b with zero denominators mapped to 0 rather than inf.

    Returning 0 (not NaN, not inf) keeps the column finite so downstream
    scalers and distance computations stay well-defined; a flow with no
    backward bytes is a real behaviour, not missing data.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.zeros_like(a)
    nz = np.abs(b) > 1e-12
    out[nz] = a[nz] / b[nz]
    return np.clip(out, -1e12, 1e12)


def add_derived(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    cv: bool = True,
    ratios: bool = True,
    log_duration: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Append derived columns. Returns (frame, new feature list).

    Purely row-wise: no statistic is computed across rows, so this cannot leak
    information between train, validation and test.
    """
    frame = frame.copy()
    new_cols: list[str] = []

    if cv:
        for name, mean_c, std_c in _CV_PAIRS:
            if mean_c in frame.columns and std_c in frame.columns:
                frame[name] = _safe_div(frame[std_c], frame[mean_c])
                new_cols.append(name)

    if ratios:
        for name, num, den in _RATIOS:
            if num in frame.columns and den in frame.columns:
                frame[name] = _safe_div(frame[num], frame[den])
                new_cols.append(name)

    if log_duration and "Flow Duration" in frame.columns:
        # Duration drives most absolute timing features; its log is the scale
        # against which regularity should be read.
        frame["log_flow_duration"] = np.log1p(
            np.clip(frame["Flow Duration"].values, 0, None))
        new_cols.append("log_flow_duration")

    return frame, list(feature_cols) + new_cols
