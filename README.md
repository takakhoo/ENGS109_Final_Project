# Sparse, Low-Rank, and Deep
### A Compressed-Sensing Framework for Music Source Separation

**ENGS 109: High-Dimensional Sensing and Learning — Final Project (Spring 2026)**
**Taka Khoo · Thayer School of Engineering, Dartmouth College**

---

This project separates a music mixture into **vocals** and **accompaniment** five different ways, each one tied to a specific lecture from ENGS 109, and benchmarks them head-to-head on MUSDB18. The unifying idea: music source separation is a compressed-sensing problem on the short-time Fourier spectrogram.

| Stage | Method | Lecture | Vocals SI-SDR | Accomp. SI-SDR |
|-------|--------|---------|:---:|:---:|
| **A** | Robust PCA (low-rank + sparse) | L13 | −3.6 dB | 0.1 dB |
| **B** | NMF with class-conditional dictionaries | L15 | 1.3 dB | 6.9 dB |
| **C** | K-SVD dictionaries + SRC | L14, L8 | **2.7 dB** | 7.2 dB |
| **D** | **SparseNet** mask predictor | L16 | 2.5 dB | **8.2 dB** |
| **E** | MMV joint sparsity (stereo, SOMP) | L12 | (stereo refine) | (stereo refine) |
| — | Oracle (ideal ratio mask) | ceiling | 10.1 dB | 15.5 dB |

*Median SI-SDR over 12 held-out MUSDB18 sample tracks. Clean unsupervised → supervised → deep progression. See [`paper/paper.pdf`](paper/paper.pdf) and [`poster/poster.pdf`](poster/poster.pdf).*

### The novelty

SparseNet (Chin's L16 architecture: stacked *fat-upsampling → tall-downsampling* sparse-coding modules) has only ever been benchmarked on **image classification**. This project is the first adaptation of its composite sparse-coding modules to **spectrogram-domain dense regression**, evaluated against Demucs v4 and the ElevenLabs API on MUSDB18.

### Three acts, one advisor

This is the third time I have studied audio under Peter Chin:

1. **Undergrad thesis** — a 1.08B-parameter Token U-Net for full-mix music *restoration* on EnCodec tokens.
2. **MS thesis** — MODULO / Mozart AI, a co-creative DAW whose stem *separation* is outsourced to ElevenLabs (≈35 dB SI-SDR below Demucs).
3. **This project** — attack separation from the compressed-sensing / sparse-coding angle, with the theoretical guarantees the deep baselines lack.

---

## Quick start

```bash
pip install -r requirements.txt
python -c "import musdb; musdb.DB(download=True, root='data/musdb18_sample')"   # 140 MB, 144 tracks

PYTHONPATH=src python scripts/run_stage_a.py            # Robust PCA on one track
PYTHONPATH=src python scripts/train_dictionaries.py     # NMF + K-SVD dictionaries
PYTHONPATH=src python scripts/train_sparsenet.py        # Stage D (Apple MPS / CUDA)
PYTHONPATH=src python scripts/evaluate_all.py           # full comparison table
PYTHONPATH=src python scripts/make_audio_examples.py    # curated playable WAVs
PYTHONPATH=src python figures/make_all_figures.py       # all paper figures
```

## Reproduce the deliverables

```bash
cd paper  && tectonic paper.tex     # 4-page ICML-style report
cd poster && tectonic poster.tex    # A0 conference poster
```

## Repository layout

```
src/separation/      five separation stages + features/io/metrics
  stage_a_rpca.py      Robust PCA, inexact ALM (L13)
  stage_b_nmf.py       class-conditional NMF, Lee-Seung KL (L15)
  stage_c_ksvd.py      K-SVD dictionaries + OMP + SRC (L14, L8)
  stage_d_scn.py       SparseNet: unrolled FISTA, end-to-end (L16)
  stage_e_mmv.py       Simultaneous OMP for stereo (L12)
scripts/             runnable training/eval/figure entry points
figures/             generated paper figures + make_all_figures.py
paper/               ICML-style final report (LaTeX)
poster/              A0 conference poster (LaTeX, tikzposter)
audio_examples/      curated before/after WAVs + INDEX
experiments/         trained dictionaries, SparseNet checkpoint, results
```

## Course connection

Every stage reuses problem-set code: OMP (PS3), FISTA (PS5), SRC (PS7), nuclear-norm SVT (PS8). Every theoretical guarantee proved in class (RIP δ₂ₛ < √2−1, Candès–Recht |Ω| ≥ CN^{5/4}r log N, RPCA λ = 1/√N₁) has a direct consequence in the method.

---

*Generated and maintained as part of the ENGS 109 final project sprint.*
