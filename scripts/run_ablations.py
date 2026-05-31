"""Ablation studies: NMF dictionary size, K-SVD sparsity, RPCA lambda scale.

Writes experiments/results/ablation_{nmf,ksvd,rpca}.csv

Usage:
    PYTHONPATH=src python scripts/run_ablations.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from separation.features import SR, stft, magphase, istft, wiener_masks
from separation.io import get_musdb, track_to_mono
from separation.metrics import si_sdr
from separation import stage_a_rpca, stage_b_nmf, stage_c_ksvd

REPO = Path(__file__).resolve().parents[1]
DICTS = REPO / "experiments" / "dictionaries"
RESULTS = REPO / "experiments" / "results"
SECONDS = 6.0


def collect_frames(tracks, source, max_frames):
    cols = []
    for tr in tracks:
        stems = track_to_mono(tr)
        if source not in stems:
            continue
        y = stems[source][: int(SECONDS * SR)]
        if np.max(np.abs(y)) < 1e-4:
            continue
        M, _ = magphase(stft(y))
        cols.append(M)
    X = np.hstack(cols)
    e = np.linalg.norm(X, axis=0)
    X = X[:, e > np.percentile(e, 20)]
    if X.shape[1] > max_frames:
        idx = np.random.default_rng(0).choice(X.shape[1], max_frames, replace=False)
        X = X[:, idx]
    return X


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    n = int(SECONDS * SR)
    train = list(get_musdb(subsets="train"))[:10]
    test = list(get_musdb(subsets="test"))[:12]
    ksvd = np.load(DICTS / "ksvd.npz")
    Dv_ksvd, Da_ksvd = ksvd["Dv"], ksvd["Da"]

    def eval_tracks(sep_fn):
        vs, accs = [], []
        for tr in test:
            stems = track_to_mono(tr)
            if "vocals" not in stems or np.max(np.abs(stems["vocals"][:n])) < 1e-4:
                continue
            res = sep_fn(stems["mix"][:n])
            vs.append(si_sdr(stems["vocals"][:n], res["vocals"]))
            accs.append(si_sdr(stems["accompaniment"][:n], res["accompaniment"]))
        return float(np.median(vs)), float(np.median(accs))

    # NMF K sweep.
    print("=== NMF K ===")
    Xv = collect_frames(train, "vocals", 800)
    Xa = collect_frames(train, "accompaniment", 800)
    rows = []
    for K in [16, 32, 64, 128]:
        Dv, _ = stage_b_nmf.train_dictionary(Xv, K=K, max_iter=150, beta="kl")
        Da, _ = stage_b_nmf.train_dictionary(Xa, K=K, max_iter=150, beta="kl")
        v, a = eval_tracks(lambda y: stage_b_nmf.separate(y, Dv, Da, beta="kl"))
        rows.append({"K": K, "vocals_si_sdr": round(v, 3), "acc_si_sdr": round(a, 3)})
        print(f"  K={K:4d}  voc {v:.3f}  acc {a:.3f}")
    with open(RESULTS / "ablation_nmf.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["K", "vocals_si_sdr", "acc_si_sdr"]); w.writeheader(); w.writerows(rows)

    # K-SVD S sweep.
    print("=== K-SVD S ===")
    rows = []
    for S in [3, 5, 10, 15]:
        v, a = eval_tracks(lambda y: stage_c_ksvd.separate(y, Dv_ksvd, Da_ksvd, n_nonzero=S))
        rows.append({"S": S, "vocals_si_sdr": round(v, 3), "acc_si_sdr": round(a, 3)})
        print(f"  S={S:3d}  voc {v:.3f}  acc {a:.3f}")
    with open(RESULTS / "ablation_ksvd.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["S", "vocals_si_sdr", "acc_si_sdr"]); w.writeheader(); w.writerows(rows)

    # RPCA lambda-scale sweep.
    print("=== RPCA lambda scale ===")
    rows = []
    for scale in [0.5, 1.0, 1.5, 2.0]:
        def sep(y, scale=scale):
            Y = stft(y); M, ph = magphase(Y)
            lam = scale / np.sqrt(max(M.shape))
            X, E, _ = stage_a_rpca.rpca(M, lam=lam, max_iter=70)
            mh, mp = wiener_masks(np.maximum(X, 0), np.maximum(E, 0))
            return {"vocals": istft(mp * M * ph, length=n),
                    "accompaniment": istft(mh * M * ph, length=n)}
        v, a = eval_tracks(sep)
        rows.append({"lambda_scale": scale, "vocals_si_sdr": round(v, 3), "acc_si_sdr": round(a, 3)})
        print(f"  scale={scale}  voc {v:.3f}  acc {a:.3f}")
    with open(RESULTS / "ablation_rpca.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lambda_scale", "vocals_si_sdr", "acc_si_sdr"]); w.writeheader(); w.writerows(rows)

    print(f"\nAblations written to {RESULTS}")


if __name__ == "__main__":
    main()
