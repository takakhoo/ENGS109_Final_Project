"""Stage C: K-SVD discriminative dictionaries + sparse-representation classification (L14, L8).

Train one K-SVD dictionary per source on spectrogram frames (L14). At test time,
sparse-code each mixture frame against the union dictionary [D_v | D_a] with OMP
(PS3), then assign time-frequency energy to whichever sub-dictionary explains it,
following the L8 SRC residual rule c* = argmin_c ||y - D_c alpha_c||_2.
"""

from __future__ import annotations

import numpy as np

from .features import stft, istft, magphase, wiener_masks


def omp(D: np.ndarray, y: np.ndarray, n_nonzero: int, eps: float = 1e-9) -> np.ndarray:
    """Orthogonal Matching Pursuit (L4, reused from PS3). Returns sparse code x."""
    K = D.shape[1]
    residual = y.astype(np.float64).copy()
    support: list[int] = []
    x = np.zeros(K)
    for _ in range(n_nonzero):
        corr = D.T @ residual
        idx = int(np.argmax(np.abs(corr)))
        if idx in support:
            break
        support.append(idx)
        Ds = D[:, support]
        x_s, *_ = np.linalg.lstsq(Ds, y, rcond=None)
        x[:] = 0.0
        x[support] = x_s
        residual = y - Ds @ x_s
        if np.linalg.norm(residual) < eps:
            break
    return x


def ksvd(Y: np.ndarray, K: int = 128, n_nonzero: int = 8, n_iter: int = 30,
         seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """K-SVD dictionary learning (Aharon-Elad-Bruckstein 2006, L14).

    Y: (F, L) data matrix of spectrogram frames. Returns (D in R^{F x K}, codes).
    """
    rng = np.random.default_rng(seed)
    F, L = Y.shape
    idx = rng.choice(L, size=min(K, L), replace=False)
    D = Y[:, idx].astype(np.float64)
    if D.shape[1] < K:  # pad if not enough frames
        D = np.hstack([D, rng.standard_normal((F, K - D.shape[1]))])
    D /= (np.linalg.norm(D, axis=0, keepdims=True) + 1e-9)
    X = np.zeros((K, L))

    for _ in range(n_iter):
        for j in range(L):
            X[:, j] = omp(D, Y[:, j], n_nonzero)
        for k in range(K):
            using = np.flatnonzero(X[k])
            if len(using) == 0:
                continue
            # Residual excluding atom k, restricted to frames that use it.
            E_k = Y[:, using] - D @ X[:, using] + np.outer(D[:, k], X[k, using])
            U, s, Vt = np.linalg.svd(E_k, full_matrices=False)
            D[:, k] = U[:, 0]
            X[k, using] = s[0] * Vt[0]
    return D, X


def separate(y: np.ndarray, D_vocals: np.ndarray, D_acc: np.ndarray,
             n_nonzero: int = 10) -> dict:
    """Per-frame OMP on the union dictionary, then SRC-style class assignment."""
    Y = stft(y)
    M, phase = magphase(Y)
    F, T = M.shape
    Kv = D_vocals.shape[1]
    D = np.hstack([D_vocals, D_acc])

    Mv = np.zeros_like(M)
    Ma = np.zeros_like(M)
    for t in range(T):
        code = omp(D, M[:, t], n_nonzero)
        Mv[:, t] = D_vocals @ code[:Kv]
        Ma[:, t] = D_acc @ code[Kv:]

    mask_v, mask_a = wiener_masks(np.abs(Mv), np.abs(Ma))
    y_v = istft(mask_v * M * phase, length=len(y))
    y_a = istft(mask_a * M * phase, length=len(y))
    return {"vocals": y_v, "accompaniment": y_a,
            "mask_vocals": mask_v, "mask_accompaniment": mask_a}
