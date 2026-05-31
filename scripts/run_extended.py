"""Extended experiments for the paper: trivial baselines, mask-type study,
method fusion, oracle-gap, per-track distributions, and the stereo MMV method.

Writes, in experiments/results/:
  extended_per_track.csv   per-track SI-SDR for every method (for boxplots)
  extended_summary.json    aggregated medians + oracle-gap percentages
  fusion.json              mask-ensemble result
  masktype.json            soft vs binary mask for each learned method
  stereo_mmv.json          Stage E stereo joint-sparsity result

Methods added beyond the main comparison:
  mixture     return the mixture unchanged for both stems (trivial floor)
  lowrank     truncated-SVD rank-r accompaniment, residual = vocals
  random      random soft mask (chance floor)
  fusion      geometric-mean ensemble of NMF, K-SVD, SCN masks
  *_binary    binarized version of each learned method's mask

Usage:
    PYTHONPATH=src python scripts/run_extended.py --n_test 25
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from separation.features import SR, stft, istft, magphase, wiener_masks
from separation.io import get_musdb, track_to_mono, track_to_stereo
from separation.metrics import si_sdr, aggregate
from separation import stage_a_rpca, stage_b_nmf, stage_c_ksvd, stage_e_mmv, baselines

REPO = Path(__file__).resolve().parents[1]
DICTS = REPO / "experiments" / "dictionaries"
RESULTS = REPO / "experiments" / "results"


def mask_to_audio(mask, Y, n):
    M, ph = magphase(Y)
    return istft(mask * M * ph, length=n)


def trivial_mixture(y, **_):
    return {"vocals": y.copy(), "accompaniment": y.copy()}


def trivial_lowrank(y, rank=4, **_):
    Y = stft(y); M, ph = magphase(Y)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s2 = s.copy(); s2[rank:] = 0.0
    L = (U * s2) @ Vt
    Eres = np.maximum(M - L, 0.0)
    mh, mp = wiener_masks(np.maximum(L, 0.0), Eres)
    n = len(y)
    return {"accompaniment": istft(mh * M * ph, length=n),
            "vocals": istft(mp * M * ph, length=n)}


def trivial_random(y, seed=0, **_):
    Y = stft(y); M, ph = magphase(Y)
    rng = np.random.default_rng(seed)
    mask = rng.random(M.shape)
    n = len(y)
    return {"vocals": istft(mask * M * ph, length=n),
            "accompaniment": istft((1 - mask) * M * ph, length=n)}


def get_vocal_mask(res, Y):
    """Recover the vocal soft mask from a separation result (for fusion/binary)."""
    if "mask_vocals" in res:
        return res["mask_vocals"]
    M = np.abs(Y)
    Vmag = np.abs(stft(res["vocals"]))
    T = min(M.shape[1], Vmag.shape[1])
    m = np.zeros_like(M)
    m[:, :T] = Vmag[:, :T] / (M[:, :T] + 1e-9)
    return np.clip(m, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_test", type=int, default=25)
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    nmf = np.load(DICTS / "nmf.npz"); ksvd = np.load(DICTS / "ksvd.npz")
    Dv_nmf, Da_nmf = nmf["Dv"], nmf["Da"]
    Dv_ksvd, Da_ksvd = ksvd["Dv"], ksvd["Da"]

    # SparseNet.
    model = None
    try:
        import torch
        from separation.stage_d_scn import SparseNet, separate as scn_sep
        ckpt = torch.load(REPO / "experiments" / "sparsenet" / "model.pth",
                          map_location="cpu", weights_only=False)
        model = SparseNet(**ckpt["config"]); model.load_state_dict(ckpt["state_dict"]); model.eval()
    except Exception as e:
        print(f"SparseNet unavailable: {e}")

    mus = get_musdb(subsets="test")
    tracks = list(mus)[: args.n_test]
    n = int(args.seconds * SR)

    learned = {
        "nmf": lambda y: stage_b_nmf.separate(y, Dv_nmf, Da_nmf, beta="kl"),
        "ksvd": lambda y: stage_c_ksvd.separate(y, Dv_ksvd, Da_ksvd, n_nonzero=10),
    }
    if model is not None:
        learned["scn"] = lambda y: scn_sep(y, model)

    trivial = {"mixture": trivial_mixture, "lowrank": trivial_lowrank, "random": trivial_random}

    per_track = []
    fusion_v, fusion_a = [], []
    masktype = {f"{k}_{t}": [] for k in learned for t in ["soft", "binary"]}
    print(f"Extended comparison on {len(tracks)} tracks")
    for ti, tr in enumerate(tracks):
        stems = track_to_mono(tr)
        if "vocals" not in stems or np.max(np.abs(stems["vocals"][:n])) < 1e-4:
            continue
        y = stems["mix"][:n]; voc = stems["vocals"][:n]; acc = stems["accompaniment"][:n]
        Y = stft(y)
        row = {"track": tr.name}

        # Trivial baselines.
        for name, fn in trivial.items():
            res = fn(y)
            row[f"{name}_vocals"] = round(si_sdr(voc, res["vocals"]), 3)
            row[f"{name}_acc"] = round(si_sdr(acc, res["accompaniment"]), 3)

        # Learned methods + soft/binary mask study + collect masks for fusion.
        masks = {}
        for name, fn in learned.items():
            res = fn(y)
            row[f"{name}_vocals"] = round(si_sdr(voc, res["vocals"]), 3)
            row[f"{name}_acc"] = round(si_sdr(acc, res["accompaniment"]), 3)
            mv = get_vocal_mask(res, Y)
            masks[name] = mv
            # soft vs binary
            yv_soft = mask_to_audio(mv, Y, n)
            yv_bin = mask_to_audio((mv > 0.5).astype(float), Y, n)
            masktype[f"{name}_soft"].append(si_sdr(voc, yv_soft))
            masktype[f"{name}_binary"].append(si_sdr(voc, yv_bin))

        # Fusion: geometric mean of available learned masks.
        if masks:
            stack = np.stack([masks[k][:, :min(m.shape[1] for m in masks.values())]
                              for k in masks])
            fused = np.exp(np.mean(np.log(stack + 1e-6), axis=0))
            Tt = fused.shape[1]
            M, ph = magphase(Y)
            yv = istft(fused * M[:, :Tt] * ph[:, :Tt], length=n)
            ya = istft((1 - fused) * M[:, :Tt] * ph[:, :Tt], length=n)
            fusion_v.append(si_sdr(voc, yv)); fusion_a.append(si_sdr(acc, ya))

        per_track.append(row)
        print(f"[{ti+1}/{len(tracks)}] {tr.name[:30]:30s} "
              + " ".join(f"{k}:{row.get(f'{k}_vocals','-')}" for k in learned))

    # Write per-track.
    keys = ["track"] + sorted({k for r in per_track for k in r if k != "track"})
    with open(RESULTS / "extended_per_track.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(per_track)

    # Oracle-gap summary: fraction of IRM oracle achieved (using main_summary).
    main = json.load(open(RESULTS / "main_summary.json"))
    irm_v = main["oracle_irm"]["vocals_si_sdr_median"]
    irm_a = main["oracle_irm"]["acc_si_sdr_median"]
    summary = {}
    for name in list(trivial) + list(learned):
        vs = [r[f"{name}_vocals"] for r in per_track if f"{name}_vocals" in r]
        accs = [r[f"{name}_acc"] for r in per_track if f"{name}_acc" in r]
        summary[name] = {
            "vocals_median": round(float(np.median(vs)), 3),
            "acc_median": round(float(np.median(accs)), 3),
        }
    with open(RESULTS / "extended_summary.json", "w") as f:
        json.dump({"summary": summary, "oracle_irm_vocals": irm_v,
                   "oracle_irm_acc": irm_a}, f, indent=2)

    # Fusion result.
    fusion = {"vocals_median": round(float(np.median(fusion_v)), 3),
              "acc_median": round(float(np.median(fusion_a)), 3)}
    json.dump(fusion, open(RESULTS / "fusion.json", "w"), indent=2)

    # Mask-type result.
    mt = {k: round(float(np.median(v)), 3) for k, v in masktype.items() if v}
    json.dump(mt, open(RESULTS / "masktype.json", "w"), indent=2)

    # Stereo MMV (Stage E) on a few tracks.
    print("=== Stage E stereo MMV ===")
    mmv_v, mmv_a = [], []
    for tr in tracks[:10]:
        st = track_to_stereo(tr)
        if "vocals" not in st:
            continue
        ys = st["mix"][:n]
        res = stage_e_mmv.separate_stereo(ys, Dv_ksvd, Da_ksvd, n_nonzero=10)
        vref = st["vocals"][:n].mean(axis=1); vest = res["vocals"].mean(axis=1)
        aref = st["accompaniment"][:n].mean(axis=1); aest = res["accompaniment"].mean(axis=1)
        mmv_v.append(si_sdr(vref, vest)); mmv_a.append(si_sdr(aref, aest))
    stereo = {"vocals_median": round(float(np.median(mmv_v)), 3),
              "acc_median": round(float(np.median(mmv_a)), 3),
              "n_tracks": len(mmv_v)}
    json.dump(stereo, open(RESULTS / "stereo_mmv.json", "w"), indent=2)

    print("\n=== EXTENDED SUMMARY ===")
    print(f"oracle IRM: voc {irm_v}  acc {irm_a}")
    for name in list(trivial) + list(learned):
        s = summary[name]
        gapv = 100 * s["vocals_median"] / irm_v if irm_v else 0
        print(f"  {name:9s} voc {s['vocals_median']:>7}  acc {s['acc_median']:>7}  ({gapv:.0f}% of IRM voc)")
    print(f"  fusion    voc {fusion['vocals_median']:>7}  acc {fusion['acc_median']:>7}")
    print(f"  stereo MMV voc {stereo['vocals_median']:>6}  acc {stereo['acc_median']:>7} (n={stereo['n_tracks']})")
    print("Mask type (vocal SI-SDR, soft vs binary):")
    for k in learned:
        print(f"  {k:5s} soft {mt.get(k+'_soft','-'):>7}  binary {mt.get(k+'_binary','-'):>7}")
    print(f"\nWrote extended results to {RESULTS}")


if __name__ == "__main__":
    main()
