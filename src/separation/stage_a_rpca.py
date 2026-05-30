"""Stage A: Robust PCA for harmonic-percussive separation (L13).

Solves the L13 P_R1 problem
    min  ||X||_* + lambda ||E||_1   s.t.   M = X + E,   lambda = 1/sqrt(max(F,T))
with the inexact Augmented Lagrange Multiplier method (Lin-Chen-Ma 2010).

X = low-rank harmonic content (sustained pitches form rank-deficient spectrograms).
E = sparse residual (transients, drums, vocal sibilance).
This is the Huang-Mysore-Smaragdis (2012) HPSS recipe, training-free.
"""

from __future__ import annotations

import numpy as np

from .features import stft, istft, magphase, wiener_masks


def shrink(X: np.ndarray, tau: float) -> np.ndarray:
    """Soft-thresholding: prox of the L1 norm. tau_sigma in L10 slides."""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def svt(X: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Singular value thresholding: prox of the nuclear norm (L13).

    Returns (thresholded matrix, thresholded singular value spectrum).
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_th = np.maximum(s - tau, 0.0)
    return (U * s_th) @ Vt, s_th


def rpca(M: np.ndarray, lam: float | None = None, tol: float = 1e-7,
         max_iter: int = 200, verbose: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """Inexact-ALM Robust PCA.

    Returns (low_rank X, sparse E, info dict with residual history and rank).
    """
    n1, n2 = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(n1, n2))
    norm_two = np.linalg.norm(M, 2)
    norm_inf = np.linalg.norm(M.ravel(), np.inf) / lam
    dual_norm = max(norm_two, norm_inf)
    Y = M / dual_norm                      # dual variable
    mu = 1.25 / norm_two
    rho = 1.5
    X = np.zeros_like(M)
    E = np.zeros_like(M)
    residuals = []
    final_rank = 0
    for k in range(max_iter):
        X, s_th = svt(M - E + Y / mu, 1.0 / mu)
        E = shrink(M - X + Y / mu, lam / mu)
        Y = Y + mu * (M - X - E)
        mu = mu * rho
        res = np.linalg.norm(M - X - E, "fro") / (np.linalg.norm(M, "fro") + 1e-12)
        residuals.append(res)
        final_rank = int(np.sum(s_th > 1e-9))
        if verbose and (k % 10 == 0):
            print(f"  iter {k:3d}  residual {res:.2e}  rank {final_rank}")
        if res < tol:
            break
    info = {"iters": k + 1, "final_residual": residuals[-1],
            "residuals": residuals, "rank": final_rank, "lambda": lam}
    return np.maximum(X, 0.0), np.maximum(E, 0.0), info


def separate(y: np.ndarray, max_iter: int = 120) -> dict:
    """Run Stage A on a mono mixture. Returns harmonic/percussive waveforms + info.

    For the vocals/accompaniment task we treat the sparse residual E as the vocal
    proxy (sibilance and transients are sparse) and the low-rank X as accompaniment.
    """
    Y = stft(y)
    M, _ = magphase(Y)
    X, E, info = rpca(M, max_iter=max_iter)
    mask_h, mask_p = wiener_masks(X, E)
    y_harm = istft(mask_h * np.abs(Y) * np.exp(1j * np.angle(Y)), length=len(y))
    y_perc = istft(mask_p * np.abs(Y) * np.exp(1j * np.angle(Y)), length=len(y))
    return {
        "harmonic": y_harm,       # low-rank -> accompaniment proxy
        "percussive": y_perc,     # sparse  -> vocals/transient proxy
        "accompaniment": y_harm,
        "vocals": y_perc,
        "info": info,
        "mask_harmonic": mask_h,
        "mask_percussive": mask_p,
    }
