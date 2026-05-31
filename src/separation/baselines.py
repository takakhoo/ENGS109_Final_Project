"""Non-learning separation baselines for comparison.

These are standard signal-processing separators that require no training, used as
reference points against the sparse-coding pipeline:

  hpss_median  : median-filtering harmonic/percussive separation (Fitzgerald 2010)
  repet_sim    : REPET-SIM repeating-background separation (Rafii & Pardo 2012)

Both treat the repeating / harmonic component as accompaniment and the residual
as the vocal proxy.
"""

from __future__ import annotations

import librosa
import numpy as np

from .features import stft, istft, magphase, SR, HOP, N_FFT


def hpss_median(y: np.ndarray, margin: float = 3.0) -> dict:
    """Median-filtering HPSS. Harmonic -> accompaniment, percussive -> vocal proxy."""
    Y = stft(y)
    H, P = librosa.decompose.hpss(Y, margin=margin)
    y_h = istft(H, length=len(y))
    y_p = istft(P, length=len(y))
    return {"accompaniment": y_h, "vocals": y_p}


def repet_sim(y: np.ndarray, eps: float = 1e-9) -> dict:
    """REPET-SIM: estimate the repeating background via a self-similarity nn-filter.

    The non-local median of similar frames captures the repeating accompaniment;
    the soft residual mask captures the non-repeating vocal.
    """
    Y = stft(y)
    M, phase = magphase(Y)
    # Self-similarity nearest-neighbour filter on the magnitude (Rafii-Pardo).
    rec = librosa.segment.recurrence_matrix(
        M, mode="affinity", metric="cosine", sparse=True, width=3
    )
    M_bg = librosa.decompose.nn_filter(M, rec=rec, aggregate=np.average)
    M_bg = np.minimum(M, M_bg)
    # Soft masks (Wiener) from the repeating background vs foreground.
    margin_bg, margin_fg = 2.0, 4.0
    mask_bg = M_bg / (M + eps)
    mask_bg = (mask_bg >= (margin_bg / (1 + margin_bg))).astype(float) * mask_bg
    M_fg = M - M_bg
    mask_fg = M_fg / (M + eps)
    y_bg = istft(mask_bg * M * phase, length=len(y))
    y_fg = istft(mask_fg * M * phase, length=len(y))
    return {"accompaniment": y_bg, "vocals": y_fg}
