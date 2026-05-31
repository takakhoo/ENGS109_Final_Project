"""Train the Stage D SparseNet vocal-mask predictor on MUSDB18 (L16).

Target: the ideal ratio mask IRM_v = |V| / (|V| + |A|) per time-frequency bin.
Input:  mixture magnitude spectrogram frames.
Loss:   masked-magnitude L1 reconstruction + explicit L1 sparsity on the codes.

Runs on Apple MPS if available. Saves checkpoint to experiments/sparsenet/model.pth.

Usage:
    PYTHONPATH=src python scripts/train_sparsenet.py --n_train 20 --epochs 40
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from separation.features import stft, magphase, SR
from separation.io import get_musdb, track_to_mono
from separation.stage_d_scn import SparseNet, sparsity_fraction, device_auto

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments" / "sparsenet"


def build_frames(tracks, seconds: float = 6.0, eps: float = 1e-9):
    """Return (X mix frames, Y target masks) as float32 arrays, shape (N, F)."""
    Xs, Ys = [], []
    for tr in tracks:
        stems = track_to_mono(tr)
        n = int(seconds * SR)
        mix = stems["mix"][:n]
        voc = stems.get("vocals")
        acc = stems.get("accompaniment")
        if voc is None or acc is None:
            continue
        Mmix, _ = magphase(stft(mix))
        Mv, _ = magphase(stft(voc[:n]))
        Ma, _ = magphase(stft(acc[:n]))
        T = min(Mmix.shape[1], Mv.shape[1], Ma.shape[1])
        Mmix, Mv, Ma = Mmix[:, :T], Mv[:, :T], Ma[:, :T]
        irm = Mv / (Mv + Ma + eps)            # ideal ratio mask for vocals
        Xs.append(Mmix.T)                     # (T, F)
        Ys.append(irm.T)
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    Y = np.concatenate(Ys, axis=0).astype(np.float32)
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_modules", type=int, default=3)
    ap.add_argument("--n_up", type=int, default=128)
    ap.add_argument("--n_down", type=int, default=64)
    ap.add_argument("--n_fista", type=int, default=15)
    ap.add_argument("--mu_sparse", type=float, default=1e-3)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = device_auto()
    print(f"Device: {device}")

    mus = get_musdb(subsets="train")
    tracks = list(mus)[: args.n_train]
    print(f"Building frames from {len(tracks)} tracks...")
    X, Y = build_frames(tracks)
    print(f"  frames: X={X.shape}, Y={Y.shape}")
    F = X.shape[1]

    Xt = torch.tensor(X, device=device)
    Yt = torch.tensor(Y, device=device)
    n = Xt.shape[0]

    model = SparseNet(f_in=F, n_modules=args.n_modules, n_up=args.n_up,
                      n_down=args.n_down, n_fista=args.n_fista).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SparseNet params: {n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    history = []
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        ep_sparsity = 0.0
        n_batches = 0
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            xb, yb = Xt[idx], Yt[idx]
            mask, codes = model(xb)
            recon_loss = (mask * xb - yb * xb).abs().mean()
            sparse_loss = sum(c.abs().mean() for c in codes) / len(codes)
            loss = recon_loss + args.mu_sparse * sparse_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(recon_loss.item())
            ep_sparsity += sparsity_fraction(codes)
            n_batches += 1
        history.append({"epoch": epoch, "loss": ep_loss / n_batches,
                        "sparsity": ep_sparsity / n_batches})
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  recon_loss {ep_loss/n_batches:.5f}  "
                  f"code_sparsity {ep_sparsity/n_batches:.3f}")

    torch.save({"state_dict": model.state_dict(),
                "config": {"f_in": F, "n_modules": args.n_modules, "n_up": args.n_up,
                           "n_down": args.n_down, "n_fista": args.n_fista},
                "history": history, "n_params": n_params}, OUT / "model.pth")
    np.save(OUT / "history.npy", history)
    print(f"Saved model to {OUT/'model.pth'}")


if __name__ == "__main__":
    main()
