# Sparse, Low-Rank, and Deep
### A Compressed-Sensing Framework for Music Source Separation

**ENGS 109: High-Dimensional Sensing and Learning — Final Project (Spring 2026)**
**Taka Khoo · Thayer School of Engineering, Dartmouth College**

---

This project separates a music mixture into **vocals** and **accompaniment** five different ways, each one tied to a specific lecture from ENGS 109, and benchmarks them head-to-head on MUSDB18. The unifying idea: music source separation is a compressed-sensing problem on the short-time Fourier spectrogram.

| Stage | Method | Lecture | Status |
|-------|--------|---------|--------|
| **A** | Robust PCA (low-rank + sparse) | L13 Matrix Completion / RPCA | working |
| **B** | NMF with class-conditional dictionaries | L15 NMF | working |
| **C** | K-SVD dictionaries + Sparse-Representation Classification | L14 Dict. Learning, L8 SRC | working |
| **D** | **SparseNet** mask predictor (deep sparse coding) | L16 Deep Sparsity | working |
| **E** | MMV joint sparsity across stereo (SOMP) | L12 Multi-Measurement | working |

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
PYTHONPATH=src python scripts/run_stage_a.py    # Robust PCA, writes WAVs + SDR
```

## Repository layout

```
src/separation/      five separation stages + features/io/metrics
scripts/             runnable experiment entry points
notebooks/           per-stage walkthroughs (PS-style)
figures/             generated paper figures
paper/               ICML-style final report (LaTeX)
poster/              conference poster (LaTeX)
audio_examples/      curated before/after WAVs
experiments/         result CSVs and configs
```

## Course connection

Every stage reuses problem-set code: OMP (PS3), FISTA (PS5), SRC (PS7), nuclear-norm SVT (PS8). Every theoretical guarantee proved in class (RIP δ₂ₛ < √2−1, Candès–Recht |Ω| ≥ CN^{5/4}r log N, RPCA λ = 1/√N₁) has a direct consequence in the method.

---

*Generated and maintained as part of the ENGS 109 final project sprint.*
