# Unmixing the Music
### Sparse, Low-Rank, and Deep: A Compressed-Sensing Approach to Music Source Separation

**ENGS 109: High-Dimensional Sensing and Learning — Final Project**
**Taka Khoo · Thayer School of Engineering, Dartmouth College**

---

Music vocal/accompaniment separation, cast as a single structured-recovery problem on the short-time Fourier spectrogram. Classical decompositions and deep networks are treated as points on one axis: recovery of sparse or low-rank representations. Four separators are built on a shared STFT front end and benchmarked against training-free baselines and oracle masks on MUSDB18.

| Method | Family | Vocals SI-SDR | Accomp. SI-SDR | Time/clip |
|--------|--------|:---:|:---:|:---:|
| HPSS (median filtering) | training-free | −11.9 | −1.4 | 0.25 s |
| REPET-SIM | training-free | −0.7 | 2.9 | 0.08 s |
| Robust PCA | low-rank + sparse | −3.1 | 0.2 | 1.14 s |
| Supervised NMF | factorization | 0.5 | 5.1 | 0.46 s |
| K-SVD + SRC | dictionary learning | 0.8 | 7.1 | 0.28 s |
| **Deep sparse coder** | **unrolled / learned** | **2.0** | **8.3** | **0.12 s** |
| Oracle IRM / IBM | ceiling | 10.2 | 15.8 | — |

*Median SI-SDR over 25 held-out MUSDB18 test tracks. The unrolled deep sparse coding network is the strongest non-oracle separator on both sources and the fastest learned model at inference.*

### Method

Each separator produces a soft time-frequency mask and resynthesizes with the mixture phase.

- **Robust PCA** — principal component pursuit `min ‖X‖_* + λ‖E‖₁ s.t. X+E=M` splits the magnitude spectrogram into a low-rank accompaniment and a sparse vocal residual.
- **Supervised NMF** — per-source non-negative dictionaries trained by Lee-Seung KL multiplicative updates; activations split at test time.
- **K-SVD + SRC** — overcomplete sparsifying dictionaries with OMP coding; time-frequency energy routed by the sparse-representation-classification residual rule.
- **Deep sparse coding network** — predicts the mask by unrolling a non-negative fast proximal-gradient solver and learning the synthesis dictionaries end-to-end (a LISTA-style construction specialized to non-negative codes and a masking output).
- **MMV / SOMP** — joint-sparsity extension for stereo: simultaneous OMP forces both channels to share an atom support.

### The unifying observation

The spectrogram magnitude of a musical source is approximately sparse in a source-adapted basis, and sustained harmonic content is approximately low-rank. These are exactly the structural assumptions of compressed sensing and sparse coding, and deep masking networks can be read as unrolled sparse-recovery solvers with learned operators. Increasing the amount of structure-learning, from training-free decomposition to supervised factorization to discriminative dictionaries to an end-to-end unrolled solver, produces a monotone improvement in separation quality.

### Deliverables

- [`Khoo_Taka_FinalProject.ipynb`](Khoo_Taka_FinalProject.ipynb): the showcase notebook, already fully run. Every figure, every SI-SDR table, and three to four playable before/after audio examples per method are embedded inline.
- [`paper/paper.pdf`](paper/paper.pdf): 5-page report (official ICML 2025 format; references run past p.5)
- [`poster/poster.pdf`](poster/poster.pdf): A0 landscape conference poster
- [`walkthrough/walkthrough.pdf`](walkthrough/walkthrough.pdf): long plain-English presenter's companion (analogies, scripts, every figure and table broken down)
- [`audio_examples/`](audio_examples/): curated before/after WAVs, all stages
- [`figures/`](figures/): all paper figures, regenerable from artifacts

---

## Quick start

```bash
pip install -r requirements.txt
python -c "import musdb; musdb.DB(download=True, root='data/musdb18_sample')"   # 140 MB, 144 tracks

PYTHONPATH=src python scripts/run_stage_a.py            # Robust PCA on one track
PYTHONPATH=src python scripts/train_dictionaries.py     # NMF + K-SVD dictionaries
PYTHONPATH=src python scripts/train_sparsenet.py        # deep sparse coder (MPS/CUDA)
PYTHONPATH=src python scripts/run_experiments.py        # full comparison + baselines
PYTHONPATH=src python scripts/run_ablations.py          # hyperparameter sweeps
PYTHONPATH=src python scripts/make_audio_examples.py    # curated playable WAVs
PYTHONPATH=src python figures/make_all_figures.py       # all figures
PYTHONPATH=src python tests/test_solvers.py             # solver correctness tests
```

## Reproduce the documents

```bash
cd paper  && tectonic paper.tex
cd poster && tectonic poster.tex
```

## Repository layout

```
src/separation/      separation methods + features/io/metrics/baselines
  stage_a_rpca.py      robust PCA (principal component pursuit)
  stage_b_nmf.py       class-conditional NMF (KL multiplicative updates)
  stage_c_ksvd.py      K-SVD dictionaries + OMP + SRC routing
  stage_d_scn.py       deep sparse coding network (unrolled FISTA, end-to-end)
  stage_e_mmv.py       simultaneous OMP for stereo joint sparsity
  baselines.py         median-filtering HPSS, REPET-SIM
scripts/             training / evaluation / ablation / figure entry points
figures/             generated figures + make_all_figures.py
paper/               ICML-format paper (LaTeX + bib)
poster/              A0 conference poster (beamerposter)
audio_examples/      curated before/after WAVs
experiments/         trained dictionaries, network checkpoint, result CSVs
tests/               solver correctness tests (RPCA, OMP, SOMP)
```

## Evaluation

Median scale-invariant SDR and SDR on 25 held-out MUSDB18 test tracks at 22.05 kHz, 2048-point STFT, hop 512. Two oracle masks (ideal ratio and ideal binary) bound any magnitude-masking method. Solver correctness is verified on synthetic ground truth: RPCA recovers a low-rank-plus-sparse matrix to relative error 1e-8, OMP recovers a known sparse support to 5e-16, and SOMP recovers a shared row support exactly.
