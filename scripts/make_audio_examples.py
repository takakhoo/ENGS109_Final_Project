"""Render a curated set of playable before/after WAVs for the repo and report.

For two representative MUSDB test tracks, run every stage and write:
  audio_examples/curated/<track>_mix.wav
  audio_examples/curated/<track>_<stage>_vocals.wav
  audio_examples/curated/<track>_<stage>_accomp.wav
plus an oracle reference. These are the audio the README and report point to.

Usage:
    PYTHONPATH=src python scripts/make_audio_examples.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from separation.features import SR, stft, istft, magphase
from separation.io import get_musdb, track_to_mono, write_wav
from separation import stage_a_rpca, stage_b_nmf, stage_c_ksvd
from separation.metrics import si_sdr

REPO = Path(__file__).resolve().parents[1]
DICTS = REPO / "experiments" / "dictionaries"
OUT = REPO / "audio_examples" / "curated"
SECONDS = 6.0


def oracle(stems):
    n = int(SECONDS * SR)
    Y = stft(stems["mix"][:n]); M, ph = magphase(Y)
    Mv = np.abs(stft(stems["vocals"][:n])); Ma = np.abs(stft(stems["accompaniment"][:n]))
    T = min(M.shape[1], Mv.shape[1], Ma.shape[1])
    irm = Mv[:, :T] / (Mv[:, :T] + Ma[:, :T] + 1e-9)
    yv = istft(irm * M[:, :T] * ph[:, :T], length=n)
    ya = istft((1 - irm) * M[:, :T] * ph[:, :T], length=n)
    return {"vocals": yv, "accompaniment": ya}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nmf = np.load(DICTS / "nmf.npz"); ksvd = np.load(DICTS / "ksvd.npz")

    # Optional SparseNet.
    model = None
    try:
        import torch
        from separation.stage_d_scn import SparseNet, device_auto, separate as scn_sep
        ckpt = torch.load(REPO / "experiments" / "sparsenet" / "model.pth",
                          map_location="cpu", weights_only=False)
        model = SparseNet(**ckpt["config"]); model.load_state_dict(ckpt["state_dict"])
        model.eval()
    except Exception as e:
        print(f"SparseNet not available: {e}")

    mus = get_musdb(subsets="test")
    picks = []
    for hint in ["Cristina Vane", "Al James"]:
        for tr in mus:
            if hint.lower() in tr.name.lower():
                picks.append(tr); break

    n = int(SECONDS * SR)
    lines = []
    for tr in picks:
        stems = track_to_mono(tr)
        tag = tr.name.split(" - ")[-1][:18].replace(" ", "_")
        y = stems["mix"][:n]
        voc = stems["vocals"][:n]; acc = stems["accompaniment"][:n]
        write_wav(OUT / f"{tag}_mix.wav", y)

        runners = {
            "A": lambda: stage_a_rpca.separate(y, max_iter=80),
            "B": lambda: stage_b_nmf.separate(y, nmf["Dv"], nmf["Da"], beta="kl"),
            "C": lambda: stage_c_ksvd.separate(y, ksvd["Dv"], ksvd["Da"], n_nonzero=10),
            "oracle": lambda: oracle(stems),
        }
        if model is not None:
            from separation.stage_d_scn import separate as scn_sep
            runners["D"] = lambda: scn_sep(y, model)

        for stage, fn in runners.items():
            res = fn()
            write_wav(OUT / f"{tag}_{stage}_vocals.wav", res["vocals"])
            write_wav(OUT / f"{tag}_{stage}_accomp.wav", res["accompaniment"])
            v = si_sdr(voc, res["vocals"]); a = si_sdr(acc, res["accompaniment"])
            lines.append(f"{tr.name:40s} | stage {stage:6s} | vocals {v:+6.2f} dB | accomp {a:+6.2f} dB")
            print(lines[-1])

    (OUT / "INDEX.txt").write_text("\n".join(lines) + "\n")
    print(f"\nCurated audio + INDEX.txt in {OUT}")


if __name__ == "__main__":
    main()
