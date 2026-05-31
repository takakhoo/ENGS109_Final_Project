"""Correctness tests for the core solvers on synthetic data with known ground truth.

Run:
    PYTHONPATH=src python tests/test_solvers.py
"""

from __future__ import annotations

import numpy as np

from separation.stage_a_rpca import rpca
from separation.stage_c_ksvd import omp
from separation.stage_e_mmv import somp


def test_rpca_recovers_low_rank_plus_sparse():
    rng = np.random.default_rng(0)
    N1, N2, r = 60, 80, 3
    L = rng.standard_normal((N1, r)) @ rng.standard_normal((r, N2))
    mask = rng.random((N1, N2)) < 0.05
    E = np.zeros((N1, N2)); E[mask] = rng.uniform(-6, 6, mask.sum())
    X, Ehat, info = rpca(L + E, max_iter=200)
    rel = np.linalg.norm(X - L, "fro") / np.linalg.norm(L, "fro")
    assert rel < 1e-3, f"RPCA failed to recover low-rank part: rel={rel:.2e}"
    assert info["rank"] <= r + 1, f"recovered rank {info['rank']} too high"
    print(f"[PASS] RPCA: rel error {rel:.2e}, rank {info['rank']} (true {r})")


def test_omp_recovers_sparse_signal():
    rng = np.random.default_rng(1)
    F, K, S = 60, 120, 5
    D = rng.standard_normal((F, K)); D /= np.linalg.norm(D, axis=0)
    support = rng.choice(K, S, replace=False)
    x_true = np.zeros(K); x_true[support] = rng.uniform(1, 3, S) * rng.choice([-1, 1], S)
    y = D @ x_true
    x_hat = omp(D, y, S)
    rec_support = set(np.flatnonzero(np.abs(x_hat) > 1e-6))
    assert set(support) <= rec_support, "OMP missed a support atom"
    rel = np.linalg.norm(x_hat - x_true) / np.linalg.norm(x_true)
    assert rel < 1e-6, f"OMP coefficient error too high: {rel:.2e}"
    print(f"[PASS] OMP: exact support recovery, coef rel error {rel:.2e}")


def test_somp_shares_row_support():
    rng = np.random.default_rng(2)
    F, K, S, C = 50, 100, 4, 3
    D = rng.standard_normal((F, K)); D /= np.linalg.norm(D, axis=0)
    support = rng.choice(K, S, replace=False)
    Xtrue = np.zeros((K, C)); Xtrue[support] = rng.standard_normal((S, C))
    Y = D @ Xtrue
    Xhat = somp(D, Y, S)
    rec = set(np.flatnonzero(np.linalg.norm(Xhat, axis=1) > 1e-6))
    assert set(support) <= rec, "SOMP missed shared support"
    print(f"[PASS] SOMP: recovered shared row support {sorted(support)}")


if __name__ == "__main__":
    test_rpca_recovers_low_rank_plus_sparse()
    test_omp_recovers_sparse_signal()
    test_somp_shares_row_support()
    print("\nAll solver tests passed.")
