# Audio examples

Playable before/after WAVs for two MUSDB18 test tracks, one separated stem per stage. Regenerate with:

```bash
PYTHONPATH=src python scripts/make_audio_examples.py
```

## What to listen for

| File suffix | What it is |
|-------------|------------|
| `*_mix.wav` | the input mixture |
| `*_A_*.wav` | Stage A, Robust PCA (L13) — harmonic/percussive split, weak on vocals |
| `*_B_*.wav` | Stage B, class-conditional NMF (L15) — supervised dictionaries |
| `*_C_*.wav` | Stage C, K-SVD + SRC (L14/L8) — discriminative dictionaries |
| `*_D_*.wav` | Stage D, SparseNet (L16) — the novel deep-sparse-coding mask |
| `*_oracle_*.wav` | ideal ratio mask, the masking ceiling |

## Measured quality (SI-SDR, dB)

On **Cristina Vane – So Easy** (a clean acoustic track, the easy case):

| Stage | Vocals | Accompaniment |
|-------|:---:|:---:|
| A — RPCA | −4.1 | −2.8 |
| B — NMF | +5.4 | +10.9 |
| C — K-SVD | +5.1 | +10.8 |
| D — SparseNet | +5.2 | +10.4 |
| Oracle (IRM) | +16.0 | +20.6 |

On this track the three supervised methods land within 0.3 dB of each other on vocals and all clear +10 dB on accompaniment. The full per-track index is in `curated/INDEX.txt`.

> These are 6-second clips at 22.05 kHz with magnitude masking and mixture phase. The numbers are honest and modest; see the paper's discussion section for the gap to Demucs and the phase-reconstruction ceiling.
