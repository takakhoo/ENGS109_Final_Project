"""Stage E: Multiple-Measurement-Vector joint sparsity for stereo (L12).

The left and right channels of a stereo recording share the same active sources
at every instant, so their sparse codes are jointly supported (row-sparse). We
recover that shared support with Simultaneous OMP (Tropp 2006, L12), which scores
atoms by the L2 norm across channels of the residual correlation.

Used as a refinement post-pass: a dictionary trained by Stage C is applied jointly
to the two stereo channels so the masks are forced to agree on which atoms fire.
"""

from __future__ import annotations

import numpy as np

from .features import stft, istft, magphase, wiener_masks


def somp(D: np.ndarray, Y: np.ndarray, n_nonzero: int, q: float = 2.0,
         eps: float = 1e-9) -> np.ndarray:
    """Simultaneous OMP (L12). Y: (F, C) measurements. Returns row-sparse X: (K, C)."""
    K = D.shape[1]
    C = Y.shape[1]
    R = Y.astype(np.float64).copy()
    support: list[int] = []
    X = np.zeros((K, C))
    for _ in range(n_nonzero):
        scores = np.linalg.norm(D.T @ R, ord=q, axis=1)  # row-norm across channels
        idx = int(np.argmax(scores))
        if idx in support:
            break
        support.append(idx)
        Ds = D[:, support]
        Xs, *_ = np.linalg.lstsq(Ds, Y, rcond=None)
        X[:] = 0.0
        X[support] = Xs
        R = Y - Ds @ Xs
        if np.linalg.norm(R) < eps:
            break
    return X


def separate_stereo(y_stereo: np.ndarray, D_vocals: np.ndarray, D_acc: np.ndarray,
                    n_nonzero: int = 10) -> dict:
    """Joint-sparse stereo separation. y_stereo: (samples, 2). Returns stereo stems."""
    Kv = D_vocals.shape[1]
    D = np.hstack([D_vocals, D_acc])
    n = y_stereo.shape[0]

    YL = stft(np.ascontiguousarray(y_stereo[:, 0]))
    YR = stft(np.ascontiguousarray(y_stereo[:, 1]))
    ML, phaseL = magphase(YL)
    MR, phaseR = magphase(YR)
    F, T = ML.shape

    out = {"vocals": np.zeros((n, 2), dtype=np.float32),
           "accompaniment": np.zeros((n, 2), dtype=np.float32)}
    MvL = np.zeros_like(ML); MvR = np.zeros_like(MR)
    MaL = np.zeros_like(ML); MaR = np.zeros_like(MR)
    for t in range(T):
        Ymeas = np.stack([ML[:, t], MR[:, t]], axis=1)  # (F, 2)
        X = somp(D, Ymeas, n_nonzero)                   # (K, 2) row-sparse
        MvL[:, t] = D_vocals @ X[:Kv, 0]; MvR[:, t] = D_vocals @ X[:Kv, 1]
        MaL[:, t] = D_acc @ X[Kv:, 0];    MaR[:, t] = D_acc @ X[Kv:, 1]

    for (Mv, Ma, M, phase, ch) in [(MvL, MaL, ML, phaseL, 0), (MvR, MaR, MR, phaseR, 1)]:
        mask_v, mask_a = wiener_masks(np.abs(Mv), np.abs(Ma))
        out["vocals"][:, ch] = istft(mask_v * M * phase, length=n)
        out["accompaniment"][:, ch] = istft(mask_a * M * phase, length=n)
    return out
