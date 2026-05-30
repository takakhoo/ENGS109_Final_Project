"""Separation quality metrics: SI-SDR, SDR, and a museval wrapper.

SI-SDR (scale-invariant SDR, Le Roux et al. 2019) is the headline number: it is
invariant to a global gain on the estimate, which matters because masking and
NMF reconstructions are only correct up to scale.
"""

from __future__ import annotations

import numpy as np


def si_sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-9) -> float:
    """Scale-invariant SDR in dB.

    s_target = <est, ref> / <ref, ref> * ref ;  e = est - s_target.
    SI-SDR = 10 log10( ||s_target||^2 / ||e||^2 ).
    """
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    n = min(len(reference), len(estimate))
    reference, estimate = reference[:n], estimate[:n]
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + eps)
    s_target = alpha * reference
    e_noise = estimate - s_target
    return float(10.0 * np.log10((np.sum(s_target ** 2) + eps) / (np.sum(e_noise ** 2) + eps)))


def sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-9) -> float:
    """Plain signal-to-distortion ratio in dB (no scale invariance)."""
    n = min(len(reference), len(estimate))
    reference, estimate = reference[:n], estimate[:n]
    err = estimate - reference
    return float(10.0 * np.log10((np.sum(reference ** 2) + eps) / (np.sum(err ** 2) + eps)))


def evaluate_pair(reference_vocals, est_vocals, reference_acc, est_acc) -> dict:
    """Both-source metric bundle for a vocals/accompaniment split."""
    return {
        "vocals_si_sdr": si_sdr(reference_vocals, est_vocals),
        "vocals_sdr": sdr(reference_vocals, est_vocals),
        "acc_si_sdr": si_sdr(reference_acc, est_acc),
        "acc_sdr": sdr(reference_acc, est_acc),
    }


def aggregate(records: list[dict]) -> dict:
    """Median and IQR across a list of per-track metric dicts."""
    if not records:
        return {}
    keys = records[0].keys()
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in records if np.isfinite(r[k])])
        if len(vals) == 0:
            continue
        out[f"{k}_median"] = float(np.median(vals))
        out[f"{k}_q25"] = float(np.percentile(vals, 25))
        out[f"{k}_q75"] = float(np.percentile(vals, 75))
        out[f"{k}_mean"] = float(np.mean(vals))
    return out
