"""Comprehensive experiment suite for the paper.

Produces, in experiments/results/:
  main_results.csv       per-track SI-SDR/SDR for all methods + oracles
  main_summary.json      aggregated median/IQR per method
  ablation_nmf.csv       NMF dictionary size K sweep
  ablation_ksvd.csv      K-SVD sparsity S sweep
  ablation_rpca.csv      RPCA lambda-scale sweep
  timing.csv             per-method runtime per 6s clip

Methods compared:
  oracle_irm  ideal ratio mask (soft ceiling)
  oracle_ibm  ideal binary mask
  hpss        median-filtering harmonic/percussive (Fitzgerald 2010)
  repet       REPET-SIM repeating-background (Rafii-Pardo 2012)
  rpca        Robust PCA  (ours, Stage A)
  nmf         class-conditional NMF (ours, Stage B)
  ksvd        K-SVD + SRC (ours, Stage C)
  scn         SparseNet  (ours, Stage D)

Usage:
    PYTHONPATH=src python scripts/run_experiments.py --n_test 25
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from separation.features import SR, stft, istft, magphase
from separation.io import get_musdb, track_to_mono
from separation.metrics import si_sdr, sdr, aggregate
from separation import stage_a_rpca, stage_b_nmf, stage_c_ksvd, baselines

REPO = Path(__file__).resolve().parents[1]
DICTS = REPO / "experiments" / "dictionaries"
RESULTS = REPO / "experiments" / "results"


def oracle_mask(stems, seconds, binary=False, eps=1e-9):
    n = int(seconds * SR)
    Y = stft(stems["mix"][:n]); M, ph = magphase(Y)
    Mv = np.abs(stft(stems["vocals"][:n])); Ma = np.abs(stft(stems["accompaniment"][:n]))
    T = min(M.shape[1], Mv.shape[1], Ma.shape[1])
    M, ph, Mv, Ma = M[:, :T], ph[:, :T], Mv[:, :T], Ma[:, :T]
    if binary:
        mask = (Mv > Ma).astype(float)
    else:
        mask = Mv / (Mv + Ma + eps)
    yv = istft(mask * M * ph, length=n)
    ya = istft((1 - mask) * M * ph, length=n)
    return {"vocals": yv, "accompaniment": ya}


def metrics_row(voc, acc, res):
    return {
        "vocals_si_sdr": si_sdr(voc, res["vocals"]),
        "acc_si_sdr": si_sdr(acc, res["accompaniment"]),
        "vocals_sdr": sdr(voc, res["vocals"]),
        "acc_sdr": sdr(acc, res["accompaniment"]),
    }


def load_scn():
    try:
        import torch
        from separation.stage_d_scn import SparseNet, device_auto, separate
        ckpt = torch.load(REPO / "experiments" / "sparsenet" / "model.pth",
                          map_location="cpu", weights_only=False)
        model = SparseNet(**ckpt["config"]); model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, separate
    except Exception as e:
        print(f"SparseNet unavailable: {e}")
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_test", type=int, default=25)
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    nmf = np.load(DICTS / "nmf.npz"); ksvd = np.load(DICTS / "ksvd.npz")
    Dv_nmf, Da_nmf = nmf["Dv"], nmf["Da"]
    Dv_ksvd, Da_ksvd = ksvd["Dv"], ksvd["Da"]
    model, scn_sep = load_scn()

    mus = get_musdb(subsets="test")
    tracks = [t for t in list(mus)[: args.n_test]]
    n = int(args.seconds * SR)
    print(f"Main comparison on {len(tracks)} tracks")

    methods = ["oracle_irm", "oracle_ibm", "hpss", "repet", "rpca", "nmf", "ksvd"]
    if model is not None:
        methods.append("scn")

    per_track = []
    timing = {m: [] for m in methods}
    for ti, tr in enumerate(tracks):
        stems = track_to_mono(tr)
        if "vocals" not in stems or "accompaniment" not in stems:
            continue
        y = stems["mix"][:n]; voc = stems["vocals"][:n]; acc = stems["accompaniment"][:n]
        if np.max(np.abs(voc)) < 1e-4:
            continue
        row = {"track": tr.name}
        for mth in methods:
            t0 = time.time()
            if mth == "oracle_irm":   res = oracle_mask(stems, args.seconds, binary=False)
            elif mth == "oracle_ibm": res = oracle_mask(stems, args.seconds, binary=True)
            elif mth == "hpss":       res = baselines.hpss_median(y)
            elif mth == "repet":      res = baselines.repet_sim(y)
            elif mth == "rpca":       res = stage_a_rpca.separate(y, max_iter=80)
            elif mth == "nmf":        res = stage_b_nmf.separate(y, Dv_nmf, Da_nmf, beta="kl")
            elif mth == "ksvd":       res = stage_c_ksvd.separate(y, Dv_ksvd, Da_ksvd, n_nonzero=10)
            elif mth == "scn":        res = scn_sep(y, model)
            timing[mth].append(time.time() - t0)
            m = metrics_row(voc, acc, res)
            for k, v in m.items():
                row[f"{mth}_{k}"] = round(v, 3)
        per_track.append(row)
        print(f"[{ti+1}/{len(tracks)}] {tr.name[:34]:34s} "
              + " ".join(f"{mth}:{row.get(f'{mth}_vocals_si_sdr','-')}" for mth in ["rpca","nmf","ksvd"] + (["scn"] if model else [])))

    # Write per-track + summary.
    keys = ["track"] + sorted({k for r in per_track for k in r if k != "track"})
    with open(RESULTS / "main_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(per_track)

    summary = {}
    for mth in methods:
        recs = [{k.replace(f"{mth}_", ""): v for k, v in r.items() if k.startswith(f"{mth}_")}
                for r in per_track]
        summary[mth] = aggregate([r for r in recs if r])
        summary[mth]["time_mean_s"] = round(float(np.mean(timing[mth])), 3)
    with open(RESULTS / "main_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== MAIN SUMMARY (median SI-SDR dB | time s) ===")
    for mth in methods:
        s = summary[mth]
        print(f"  {mth:11s} voc {s.get('vocals_si_sdr_median','-'):>7} "
              f"acc {s.get('acc_si_sdr_median','-'):>7}  | {s['time_mean_s']:>6}s")

    # ---- Ablation: NMF dictionary size K ----
    print("\n=== Ablation: NMF K ===")
    abl = []
    for K in [16, 32, 64, 128]:
        # subselect K atoms from the trained K=64 dict where possible, else retrain quick
        from scripts.train_dictionaries import collect_frames
        train = list(get_musdb(subsets="train"))[:8]
        Xv = collect_frames(train, "vocals", 800)
        Xa = collect_frames(train, "accompaniment", 800)
        Dv, _ = stage_b_nmf.train_dictionary(Xv, K=K, max_iter=150, beta="kl")
        Da, _ = stage_b_nmf.train_dictionary(Xa, K=K, max_iter=150, beta="kl")
        vs = []
        for tr in tracks[:12]:
            stems = track_to_mono(tr)
            if "vocals" not in stems: continue
            y = stems["mix"][:n]
            res = stage_b_nmf.separate(y, Dv, Da, beta="kl")
            vs.append(si_sdr(stems["vocals"][:n], res["vocals"]))
        abl.append({"K": K, "vocals_si_sdr_median": round(float(np.median(vs)), 3)})
        print(f"  K={K:4d}  vocals {abl[-1]['vocals_si_sdr_median']}")
    with open(RESULTS / "ablation_nmf.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["K", "vocals_si_sdr_median"]); w.writeheader(); w.writerows(abl)

    # ---- Ablation: K-SVD sparsity S ----
    print("\n=== Ablation: K-SVD S ===")
    abl2 = []
    for S in [3, 5, 10, 15]:
        vs = []
        for tr in tracks[:12]:
            stems = track_to_mono(tr)
            if "vocals" not in stems: continue
            y = stems["mix"][:n]
            res = stage_c_ksvd.separate(y, Dv_ksvd, Da_ksvd, n_nonzero=S)
            vs.append(si_sdr(stems["vocals"][:n], res["vocals"]))
        abl2.append({"S": S, "vocals_si_sdr_median": round(float(np.median(vs)), 3)})
        print(f"  S={S:3d}  vocals {abl2[-1]['vocals_si_sdr_median']}")
    with open(RESULTS / "ablation_ksvd.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["S", "vocals_si_sdr_median"]); w.writeheader(); w.writerows(abl2)

    # ---- Ablation: RPCA lambda scale ----
    print("\n=== Ablation: RPCA lambda scale ===")
    abl3 = []
    for scale in [0.5, 1.0, 1.5, 2.0]:
        vs, accs = [], []
        for tr in tracks[:12]:
            stems = track_to_mono(tr)
            if "vocals" not in stems: continue
            y = stems["mix"][:n]
            Y = stft(y); M, ph = magphase(Y)
            lam = scale / np.sqrt(max(M.shape))
            X, E, _ = stage_a_rpca.rpca(M, lam=lam, max_iter=70)
            from separation.features import wiener_masks
            mh, mp = wiener_masks(np.maximum(X, 0), np.maximum(E, 0))
            yv = istft(mp * M * ph, length=n)
            vs.append(si_sdr(stems["vocals"][:n], yv))
        abl3.append({"lambda_scale": scale, "vocals_si_sdr_median": round(float(np.median(vs)), 3)})
        print(f"  scale={scale}  vocals {abl3[-1]['vocals_si_sdr_median']}")
    with open(RESULTS / "ablation_rpca.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lambda_scale", "vocals_si_sdr_median"]); w.writeheader(); w.writerows(abl3)

    print(f"\nAll results written to {RESULTS}")


if __name__ == "__main__":
    main()
