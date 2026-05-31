"""Evaluate one or more SparseNet checkpoints on the MUSDB18 test set.

Usage:
    PYTHONPATH=src python scripts/eval_model.py --models model.pth model_ctx.pth --n_test 25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from separation.features import SR
from separation.io import get_musdb, track_to_mono
from separation.metrics import si_sdr
from separation.stage_d_scn import SparseNet, separate as scn_sep, device_auto

REPO = Path(__file__).resolve().parents[1]
SND = REPO / "experiments" / "sparsenet"


def load(path):
    ckpt = torch.load(SND / path, map_location="cpu", weights_only=False)
    model = SparseNet(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device_auto())
    return model, ckpt.get("n_params", 0), ckpt["config"].get("context", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["model.pth"])
    ap.add_argument("--n_test", type=int, default=25)
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()

    tracks = list(get_musdb(subsets="test"))[: args.n_test]
    n = int(args.seconds * SR)
    data = []
    for tr in tracks:
        stems = track_to_mono(tr)
        if "vocals" not in stems or np.max(np.abs(stems["vocals"][:n])) < 1e-4:
            continue
        data.append((stems["mix"][:n], stems["vocals"][:n], stems["accompaniment"][:n]))
    print(f"Evaluating on {len(data)} tracks")

    for mp in args.models:
        try:
            model, nparams, ctx = load(mp)
        except Exception as e:
            print(f"{mp}: load failed ({e})"); continue
        vs, accs = [], []
        for y, voc, acc in data:
            res = scn_sep(y, model)
            vs.append(si_sdr(voc, res["vocals"]))
            accs.append(si_sdr(acc, res["accompaniment"]))
        print(f"{mp:16s} ctx={ctx} params={nparams:>9,} "
              f"voc {np.median(vs):+.3f}  acc {np.median(accs):+.3f}")


if __name__ == "__main__":
    main()
