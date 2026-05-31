"""Train class-conditional NMF and K-SVD dictionaries on MUSDB18 training stems.

Collects vocal-only and accompaniment-only spectrogram frames from a subset of
training tracks, trains one dictionary per source per method, and saves them to
experiments/dictionaries/.

Usage:
    PYTHONPATH=src python scripts/train_dictionaries.py --n_train 15 --K 64
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from separation.features import stft, magphase, SR
from separation.io import get_musdb, track_to_mono
from separation import stage_b_nmf, stage_c_ksvd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments" / "dictionaries"


def collect_frames(tracks, source: str, max_frames: int, seconds: float = 6.0) -> np.ndarray:
    """Stack magnitude-spectrogram frames from one source across tracks."""
    cols = []
    for tr in tracks:
        stems = track_to_mono(tr)
        if source not in stems:
            continue
        y = stems[source][: int(seconds * SR)]
        if np.max(np.abs(y)) < 1e-4:
            continue
        M, _ = magphase(stft(y))
        cols.append(M)
    X = np.hstack(cols)
    # Drop near-silent frames, then subsample.
    energy = np.linalg.norm(X, axis=0)
    X = X[:, energy > np.percentile(energy, 20)]
    if X.shape[1] > max_frames:
        idx = np.random.default_rng(0).choice(X.shape[1], max_frames, replace=False)
        X = X[:, idx]
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=15)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--ksvd_K", type=int, default=128)
    ap.add_argument("--ksvd_frames", type=int, default=2000)
    ap.add_argument("--nmf_iter", type=int, default=250)
    ap.add_argument("--ksvd_iter", type=int, default=15)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    mus = get_musdb(subsets="train")
    tracks = list(mus)[: args.n_train]
    print(f"Training on {len(tracks)} tracks")

    print("Collecting vocal frames...")
    Xv = collect_frames(tracks, "vocals", args.ksvd_frames)
    print("Collecting accompaniment frames...")
    Xa = collect_frames(tracks, "accompaniment", args.ksvd_frames)
    print(f"  vocal frames: {Xv.shape}, accompaniment frames: {Xa.shape}")

    # --- NMF dictionaries (KL divergence) ---
    print(f"Training NMF dictionaries (K={args.K}, KL)...")
    t0 = time.time()
    Dv_nmf, _ = stage_b_nmf.train_dictionary(Xv, K=args.K, max_iter=args.nmf_iter, beta="kl")
    Da_nmf, _ = stage_b_nmf.train_dictionary(Xa, K=args.K, max_iter=args.nmf_iter, beta="kl")
    print(f"  done in {time.time()-t0:.1f}s")
    np.savez(OUT / "nmf.npz", Dv=Dv_nmf, Da=Da_nmf, K=args.K)

    # --- K-SVD dictionaries ---
    print(f"Training K-SVD dictionaries (K={args.ksvd_K}, S=8)...")
    t0 = time.time()
    Dv_ksvd, _ = stage_c_ksvd.ksvd(Xv, K=args.ksvd_K, n_nonzero=8, n_iter=args.ksvd_iter)
    Da_ksvd, _ = stage_c_ksvd.ksvd(Xa, K=args.ksvd_K, n_nonzero=8, n_iter=args.ksvd_iter)
    print(f"  done in {time.time()-t0:.1f}s")
    np.savez(OUT / "ksvd.npz", Dv=Dv_ksvd, Da=Da_ksvd, K=args.ksvd_K)

    print(f"Saved dictionaries to {OUT}")


if __name__ == "__main__":
    main()
