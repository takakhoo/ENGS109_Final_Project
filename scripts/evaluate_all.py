"""Full evaluation harness: run every stage on a MUSDB18 test subset.

Computes SI-SDR / SDR for vocals and accompaniment per track, aggregates median
and IQR, and writes:
  experiments/results/metrics_per_track.csv
  experiments/results/summary.json

Optionally writes a few separated WAVs for the audio-examples table.

Usage:
    PYTHONPATH=src python scripts/evaluate_all.py --n_test 15 --write_audio 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from separation.features import SR
from separation.io import get_musdb, track_to_mono, write_wav
from separation.metrics import evaluate_pair, aggregate
from separation import stage_a_rpca, stage_b_nmf, stage_c_ksvd

REPO = Path(__file__).resolve().parents[1]
DICTS = REPO / "experiments" / "dictionaries"
RESULTS = REPO / "experiments" / "results"
AUDIO = REPO / "audio_examples" / "out"


def ideal_ratio_mask_baseline(stems, seconds):
    """Oracle upper bound: separate using the true IRM. Tells us the ceiling."""
    from separation.features import stft, istft, magphase
    n = int(seconds * SR)
    mix = stems["mix"][:n]
    Y = stft(mix)
    M, phase = magphase(Y)
    Mv, _ = magphase(stft(stems["vocals"][:n]))
    Ma, _ = magphase(stft(stems["accompaniment"][:n]))
    T = min(M.shape[1], Mv.shape[1], Ma.shape[1])
    irm = Mv[:, :T] / (Mv[:, :T] + Ma[:, :T] + 1e-9)
    yv = istft(irm * M[:, :T] * phase[:, :T], length=n)
    ya = istft((1 - irm) * M[:, :T] * phase[:, :T], length=n)
    return {"vocals": yv, "accompaniment": ya}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_test", type=int, default=15)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--write_audio", type=int, default=3)
    ap.add_argument("--stages", type=str, default="oracle,A,B,C,D")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(",")

    nmf = np.load(DICTS / "nmf.npz")
    ksvd = np.load(DICTS / "ksvd.npz")
    Dv_nmf, Da_nmf = nmf["Dv"], nmf["Da"]
    Dv_ksvd, Da_ksvd = ksvd["Dv"], ksvd["Da"]

    # Optional Stage D.
    model = None
    if "D" in stages:
        try:
            import torch
            from separation.stage_d_scn import SparseNet, device_auto, separate as scn_separate
            ckpt = torch.load(REPO / "experiments" / "sparsenet" / "model.pth",
                              map_location="cpu", weights_only=False)
            cfg = ckpt["config"]
            model = SparseNet(**cfg)
            model.load_state_dict(ckpt["state_dict"])
            model.eval().to(device_auto())
            print(f"Loaded SparseNet ({ckpt['n_params']:,} params)")
        except Exception as e:
            print(f"Stage D unavailable ({e}); skipping.")
            stages = [s for s in stages if s != "D"]

    mus = get_musdb(subsets="test")
    tracks = list(mus)[: args.n_test]
    print(f"Evaluating on {len(tracks)} test tracks, stages={stages}")

    per_track = []
    n = int(args.seconds * SR)
    for ti, tr in enumerate(tracks):
        stems = track_to_mono(tr)
        if "vocals" not in stems or "accompaniment" not in stems:
            continue
        y = stems["mix"][:n]
        voc = stems["vocals"][:n]
        acc = stems["accompaniment"][:n]
        row = {"track": tr.name}

        for stage in stages:
            t0 = time.time()
            if stage == "oracle":
                res = ideal_ratio_mask_baseline(stems, args.seconds)
            elif stage == "A":
                res = stage_a_rpca.separate(y, max_iter=80)
            elif stage == "B":
                res = stage_b_nmf.separate(y, Dv_nmf, Da_nmf, beta="kl")
            elif stage == "C":
                res = stage_c_ksvd.separate(y, Dv_ksvd, Da_ksvd, n_nonzero=10)
            elif stage == "D" and model is not None:
                from separation.stage_d_scn import separate as scn_separate
                res = scn_separate(y, model)
            else:
                continue
            m = evaluate_pair(voc, res["vocals"], acc, res["accompaniment"])
            for k, v in m.items():
                row[f"{stage}_{k}"] = round(v, 3)
            row[f"{stage}_sec"] = round(time.time() - t0, 2)

            if ti < args.write_audio:
                tag = tr.name.split(" - ")[-1][:20].replace(" ", "_")
                write_wav(AUDIO / f"{stage}_{tag}_vocals.wav", res["vocals"])
                write_wav(AUDIO / f"{stage}_{tag}_accomp.wav", res["accompaniment"])

        print(f"[{ti+1}/{len(tracks)}] {tr.name[:40]:40s} "
              + " ".join(f"{s}:{row.get(f'{s}_vocals_si_sdr','-')}" for s in stages))
        per_track.append(row)

    # Write per-track CSV.
    import csv
    keys = sorted({k for r in per_track for k in r})
    keys = ["track"] + [k for k in keys if k != "track"]
    with open(RESULTS / "metrics_per_track.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(per_track)

    # Aggregate per stage.
    summary = {}
    for stage in stages:
        recs = []
        for r in per_track:
            rec = {k.replace(f"{stage}_", ""): v for k, v in r.items()
                   if k.startswith(f"{stage}_") and k != f"{stage}_sec"}
            if rec:
                recs.append(rec)
        summary[stage] = aggregate(recs)
    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY (median SI-SDR, dB) ===")
    for stage in stages:
        s = summary.get(stage, {})
        print(f"  Stage {stage:6s}  vocals {s.get('vocals_si_sdr_median','-'):>7}  "
              f"accomp {s.get('acc_si_sdr_median','-'):>7}")
    print(f"\nWrote {RESULTS/'metrics_per_track.csv'} and summary.json")


if __name__ == "__main__":
    main()
