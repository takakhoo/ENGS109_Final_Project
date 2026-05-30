"""Stage D: SparseNet mask predictor (L16, Deep Sparse Coding Networks).

This is the novel contribution. We adapt Chin's composite-sparse-coding module
(L16, Sun-Nasrabadi-Tran) from image classification to spectrogram-domain dense
regression. Each module unrolls T_fista steps of non-negative FISTA (L10) for a
fat upsampling sparse code D1 followed by a tall downsampling sparse code D2; the
whole stack is trained end-to-end by backprop, with dictionaries and shrinkage
parameters as learnable weights.

Inference per module (L16 slides, verbatim):
  a1 = argmin_{a>=0} 1/2 ||x  - D1 a||^2 + lam1 ||a||_1 + lam2/2 ||a||^2
  a2 = argmin_{a>=0} 1/2 ||a1 - D2 a||^2 + lam1 ||a||_1 + lam2/2 ||a||^2

Everything is matmul + relu so it runs on Apple MPS (no SVD on device).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .features import stft, istft, magphase, mel_spectrogram, SR, N_FFT, HOP


def power_iteration_sq_norm(D: torch.Tensor, n_iter: int = 20) -> torch.Tensor:
    """Largest squared singular value of D (Lipschitz constant of D^T D), MPS-safe."""
    v = torch.randn(D.shape[1], device=D.device, dtype=D.dtype)
    v = v / (v.norm() + 1e-9)
    for _ in range(n_iter):
        v = D.T @ (D @ v)
        v = v / (v.norm() + 1e-9)
    Dv = D @ v
    return (Dv @ Dv) / (v @ v + 1e-9)


def fista_nonneg(X: torch.Tensor, D: torch.Tensor, lam1: torch.Tensor,
                 lam2: torch.Tensor, n_iter: int = 25) -> torch.Tensor:
    """Batched non-negative FISTA (L10). X: (B, F_in), D: (F_in, K). Returns A: (B, K).

    Solves, per row of X,
        min_{a>=0} 1/2 ||x - D a||^2 + lam1 ||a||_1 + lam2/2 ||a||^2.
    Fully differentiable w.r.t. D, lam1, lam2.
    """
    B = X.shape[0]
    K = D.shape[1]
    Lf = power_iteration_sq_norm(D).detach() + lam2.detach() + 1e-6
    A = torch.zeros(B, K, device=X.device, dtype=X.dtype)
    Z = A.clone()
    t = 1.0
    lam1_eff = torch.clamp(lam1, min=0.0)
    for _ in range(n_iter):
        resid = Z @ D.T - X                 # (B, F_in)
        grad = resid @ D + lam2 * Z         # (B, K)
        A_new = torch.relu(Z - grad / Lf - lam1_eff / Lf)
        t_new = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
        Z = A_new + ((t - 1.0) / t_new) * (A_new - A)
        A, t = A_new, t_new
    return A


class CompositeModule(nn.Module):
    """L16 composite sparse coding module: fat upsampling then tall downsampling."""

    def __init__(self, n_in: int, n_up: int, n_down: int,
                 lam1: float = 0.05, lam2: float = 0.05, n_fista: int = 20):
        super().__init__()
        self.D1 = nn.Parameter(torch.randn(n_in, n_up) * (1.0 / np.sqrt(n_in)))
        self.D2 = nn.Parameter(torch.randn(n_up, n_down) * (1.0 / np.sqrt(n_up)))
        self.lam1 = nn.Parameter(torch.tensor(float(lam1)))
        self.lam2 = nn.Parameter(torch.tensor(float(lam2)))
        self.n_fista = n_fista

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a1 = fista_nonneg(X, self.D1, self.lam1, self.lam2, self.n_fista)
        a2 = fista_nonneg(a1, self.D2, self.lam1, self.lam2, self.n_fista)
        return a2, a1  # return a1 too for the sparsity penalty


class SparseNet(nn.Module):
    """Stack of composite modules + a linear mask head with sigmoid output.

    Operates frame-wise: input is a batch of magnitude spectrogram columns
    (B, F_in); output is a soft mask (B, F_out) over the same frequency axis.
    """

    def __init__(self, f_in: int, n_modules: int = 3, n_up: int = 128,
                 n_down: int = 64, n_fista: int = 20, f_out: int | None = None):
        super().__init__()
        f_out = f_out or f_in
        dims = [f_in] + [n_down] * n_modules
        self.modules_ = nn.ModuleList(
            CompositeModule(dims[i], n_up, dims[i + 1], n_fista=n_fista)
            for i in range(n_modules)
        )
        self.head = nn.Linear(n_down, f_out)
        self.f_in = f_in
        self.f_out = f_out

    def forward(self, X: torch.Tensor):
        codes = []
        h = X
        for mod in self.modules_:
            h, a1 = mod(h)
            codes.append(h)
            codes.append(a1)
        mask = torch.sigmoid(self.head(h))
        return mask, codes


def sparsity_fraction(codes: list[torch.Tensor], thresh: float = 1e-4) -> float:
    """Fraction of (near) zero entries across all intermediate codes."""
    total = 0
    zeros = 0
    for c in codes:
        total += c.numel()
        zeros += int((c.abs() < thresh).sum().item())
    return zeros / max(total, 1)


def device_auto() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def separate(y: np.ndarray, model: SparseNet, device: torch.device | None = None) -> dict:
    """Run a trained SparseNet on a mono mixture to produce vocals/accompaniment."""
    device = device or next(model.parameters()).device
    Y = stft(y)
    M, phase = magphase(Y)               # M: (F, T)
    X = torch.tensor(M.T, dtype=torch.float32, device=device)  # (T, F)
    mask, _ = model(X)                    # (T, F)
    mask_v = mask.T.cpu().numpy()
    y_v = istft(mask_v * M * phase, length=len(y))
    y_a = istft((1.0 - mask_v) * M * phase, length=len(y))
    return {"vocals": y_v, "accompaniment": y_a, "mask_vocals": mask_v}
