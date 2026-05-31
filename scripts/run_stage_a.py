"""Minimal entry point: run Stage A (Robust PCA) on one MUSDB track and report SI-SDR.

Usage:
    PYTHONPATH=src python scripts/run_stage_a.py
"""

from __future__ import annotations

from pathlib import Path

from separation.features import SR
from separation.io import get_musdb, track_to_mono, write_wav
from separation import stage_a_rpca
from separation.metrics import evaluate_pair

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "audio_examples" / "out"


def main():
    mus = get_musdb(subsets="test")
    track = list(mus)[0]
    stems = track_to_mono(track)
    n = 6 * SR
    y = stems["mix"][:n]

    print(f"Track: {track.name}")
    res = stage_a_rpca.separate(y, max_iter=80)
    print(f"RPCA: rank {res['info']['rank']}, {res['info']['iters']} iters")

    m = evaluate_pair(stems["vocals"][:n], res["vocals"],
                      stems["accompaniment"][:n], res["accompaniment"])
    print("SI-SDR:", {k: round(v, 2) for k, v in m.items() if "si_sdr" in k})

    write_wav(OUT / "stageA_vocals.wav", res["vocals"])
    write_wav(OUT / "stageA_accomp.wav", res["accompaniment"])
    write_wav(OUT / "stageA_mix.wav", y)
    print(f"Wrote WAVs to {OUT}")


if __name__ == "__main__":
    main()
