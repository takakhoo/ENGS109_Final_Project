"""Stage B: NMF with class-conditional dictionaries (L15).

Train one nonnegative dictionary per source on isolated stems using Lee-Seung
multiplicative updates (L15). At test time, factor the mixture against the
concatenated dictionary [D_v | D_a] holding D fixed, split the activations, and
build Wiener masks from the per-source reconstructions.

This is supervised NMF source separation (Smaragdis 2007; Bryan-Smaragdis 2013).
"""

from __future__ import annotations

import numpy as np

from .features import stft, istft, magphase, wiener_masks


def train_dictionary(M_train: np.ndarray, K: int = 64, max_iter: int = 300,
                     beta: str = "kl", seed: int = 0,
                     eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Lee-Seung multiplicative-update NMF: M ~= D H, D,H >= 0.

    beta = "kl" uses the Kullback-Leibler divergence (better for audio); "fro"
    uses Frobenius. Returns (D in R^{F x K}, H in R^{K x T}).
    """
    rng = np.random.default_rng(seed)
    F, T = M_train.shape
    D = rng.random((F, K)) + eps
    H = rng.random((K, T)) + eps
    M = np.maximum(M_train, 0.0) + eps

    for _ in range(max_iter):
        if beta == "kl":
            # KL multiplicative updates (Lee-Seung 2001).
            DH = D @ H + eps
            H *= (D.T @ (M / DH)) / (D.T @ np.ones_like(M) + eps)
            DH = D @ H + eps
            D *= ((M / DH) @ H.T) / (np.ones_like(M) @ H.T + eps)
        else:
            # Frobenius multiplicative updates.
            H *= (D.T @ M) / (D.T @ D @ H + eps)
            D *= (M @ H.T) / (D @ H @ H.T + eps)
        # Normalize dictionary columns to unit L1 to fix the scale ambiguity.
        norm = np.sum(D, axis=0, keepdims=True) + eps
        D /= norm
        H *= norm.T
    return D, H


def _transform(M: np.ndarray, D: np.ndarray, max_iter: int = 200,
               beta: str = "kl", l1: float = 0.0, eps: float = 1e-9) -> np.ndarray:
    """Solve for activations H with the dictionary D held fixed."""
    rng = np.random.default_rng(0)
    K = D.shape[1]
    T = M.shape[1]
    H = rng.random((K, T)) + eps
    M = np.maximum(M, 0.0) + eps
    for _ in range(max_iter):
        if beta == "kl":
            DH = D @ H + eps
            H *= (D.T @ (M / DH)) / (D.T @ np.ones_like(M) + l1 + eps)
        else:
            H *= (D.T @ M) / (D.T @ D @ H + l1 + eps)
    return H


def separate(y: np.ndarray, D_vocals: np.ndarray, D_acc: np.ndarray,
             beta: str = "kl", l1: float = 1e-2) -> dict:
    """Separate a mixture using two pre-trained class dictionaries."""
    Y = stft(y)
    M, phase = magphase(Y)
    Kv = D_vocals.shape[1]
    D = np.hstack([D_vocals, D_acc])
    H = _transform(M, D, beta=beta, l1=l1)
    Mv = D_vocals @ H[:Kv]
    Ma = D_acc @ H[Kv:]
    mask_v, mask_a = wiener_masks(Mv, Ma)
    y_v = istft(mask_v * M * phase, length=len(y))
    y_a = istft(mask_a * M * phase, length=len(y))
    return {"vocals": y_v, "accompaniment": y_a,
            "mask_vocals": mask_v, "mask_accompaniment": mask_a}
