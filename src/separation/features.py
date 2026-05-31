"""Time-frequency front end: STFT, inverse STFT, mel, CQT, masking helpers.

The STFT is the sparsifying transform (L2 "Preliminary Math": the synthesis
dictionary Psi). Every separation stage operates on the magnitude spectrogram
M = |STFT(y)| and reuses the mixture phase for resynthesis.
"""

from __future__ import annotations

import librosa
import numpy as np

# Defaults used across all stages. 22.05 kHz keeps spectrograms small enough
# for the classical solvers to run interactively.
SR = 22050
N_FFT = 2048
HOP = 512
WIN = 2048


def stft(y: np.ndarray, n_fft: int = N_FFT, hop: int = HOP, win: int = WIN) -> np.ndarray:
    """Complex STFT. Rows = frequency bins (F), cols = time frames (T)."""
    return librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win)


def istft(Y: np.ndarray, hop: int = HOP, win: int = WIN, length: int | None = None) -> np.ndarray:
    """Inverse STFT."""
    return librosa.istft(Y, hop_length=hop, win_length=win, length=length)


def magphase(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a complex spectrogram into (magnitude, unit-modulus phase)."""
    mag = np.abs(Y)
    phase = np.exp(1j * np.angle(Y))
    return mag, phase


def apply_soft_mask(mask: np.ndarray, Y: np.ndarray, hop: int = HOP,
                    win: int = WIN, length: int | None = None) -> np.ndarray:
    """Apply a soft mask in [0,1] to the magnitude, keep the mixture phase, invert."""
    mag, phase = magphase(Y)
    return istft(mask * mag * phase, hop=hop, win=win, length=length)


def wiener_masks(*mags: np.ndarray, power: float = 2.0, eps: float = 1e-9) -> list[np.ndarray]:
    """Wiener-style soft masks from a set of nonnegative source magnitude estimates.

    mask_c = mag_c^power / sum_k mag_k^power. Masks sum to 1 by construction.
    """
    powed = [np.maximum(m, 0.0) ** power for m in mags]
    denom = sum(powed) + eps
    return [p / denom for p in powed]


def mel_spectrogram(y: np.ndarray, n_mels: int = 64, sr: int = SR,
                    n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Log-compressed mel spectrogram, used as the compact input to SparseNet."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                         n_mels=n_mels, power=2.0)
    return np.log1p(mel)


def cqt(y: np.ndarray, sr: int = SR, hop: int = HOP, n_bins: int = 84,
        bins_per_octave: int = 12) -> np.ndarray:
    """Constant-Q transform magnitude. Harmonic structure aligns to a log-freq grid."""
    C = librosa.cqt(y, sr=sr, hop_length=hop, n_bins=n_bins, bins_per_octave=bins_per_octave)
    return np.abs(C)


def to_db(mag: np.ndarray, ref=np.max) -> np.ndarray:
    """Amplitude -> dB for plotting."""
    return librosa.amplitude_to_db(np.maximum(mag, 1e-9), ref=ref)


def stack_context(X: np.ndarray, context: int) -> np.ndarray:
    """Stack a +/-context window of neighbouring frames into each input vector.

    X: (T, F) magnitude frames. Returns (T, (2*context+1)*F). The center frame is
    surrounded by its temporal neighbours so the model sees local time structure,
    while the prediction target stays the center frame. Edge frames are repeated.
    """
    if context <= 0:
        return X
    T, F = X.shape
    padded = np.pad(X, ((context, context), (0, 0)), mode="edge")
    cols = [padded[c : c + T] for c in range(2 * context + 1)]
    return np.concatenate(cols, axis=1)
