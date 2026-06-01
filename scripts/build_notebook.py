"""Build the comprehensive walkthrough notebook: Khoo_Taka_FinalProject.ipynb.

A homework-style, deeply technical notebook that reproduces the whole project:
the course equations worked through, every analysis figure generated inline, all
four separators run on several songs, and described, playable audio at each step.
Path detection makes it run from the repo (artifacts at root) or the submission
(artifacts under src/).

    python scripts/build_notebook.py
"""

from __future__ import annotations

from pathlib import Path
import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "Khoo_Taka_FinalProject.ipynb"

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))
def code(t): cells.append(nbf.v4.new_code_cell(t.strip("\n")))

# ===================================================================
md(r"""
# Unmixing the Music
## Sparse, Low-Rank, and Deep: A Compressed-Sensing Approach to Music Source Separation

**ENGS 109 - High-Dimensional Sensing and Learning - Final Project**
**Taka Khoo, Thayer School of Engineering, Dartmouth College**

---

This notebook *is* the project. It reproduces every result end to end and explains
the mathematics as it goes, in the language of the course. Given a finished song,
we recover its two hidden tracks, the **vocals** and the **accompaniment**, four
different ways, and show that all four are the same idea: recover a *structured*
(sparse or low-rank) representation of the time-frequency spectrogram.

The entire course lives in one underdetermined system,
$$ \mathbf{y} = \mathbf{A}\mathbf{x}, \qquad \mathbf{x}\ \text{sparse}, \qquad \mathbf{A}\in\mathbb{R}^{m\times N},\ m\ll N. $$
There are more unknowns than equations, so the null space is nontrivial and there
are infinitely many solutions, **unless** we ask for the *sparsest* one. The first
half of the subject **fixes the dictionary $\mathbf{A}$ and recovers the code**;
dictionary learning, K-SVD, and NMF instead **learn $\mathbf{A}$ from data**; and a
deep sparse coding network sits at the learned end of the same continuum, an
unrolled solver that is, in a precise sense, a learned autoencoder. A sparse vector
and a low-rank matrix are *two sides of the same coin*: the rank of a matrix is the
$\ell_0$ norm of its singular spectrum, and the nuclear norm is the $\ell_1$
relaxation.

**Roadmap.** (0) setup; (1) the data, where sparsity and low rank live;
(2) the recovery theory (spark, RIP, coherence); (3-6) the four separators, each with
its math, implementation, figures, and **played, described audio on several songs**;
(7) stereo as joint sparsity; (8) the full comparison; (9) per-track spread and soft
vs binary masks; (10) ablations; (11) the honest coherence result; (12) conclusion.
""")

# ===================================================================
md(r"""
## 0. Setup

We add the code package to the path and locate the trained artifacts (dictionaries,
the network checkpoint, cached results). The free MUSDB18 7-second sample set
(144 tracks) downloads on first run. The helper functions defined here keep every
later cell short.
""")

code(r"""
import sys, json, csv, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# --- locate code and artifacts (works from the repo or the submission's src/) ---
def find_base(name):
    for b in ("", "src"):
        p = Path(b) / name
        if p.exists():
            return p
    return Path("src") / name           # default to the submission layout

sys.path.insert(0, "src" if Path("src/separation").exists() else ".")
EXP  = find_base("experiments")
DATA = find_base("data") if find_base("data").exists() else (Path("src") / "data")

import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
from IPython.display import Audio, display, Markdown

from separation.features import SR, N_FFT, HOP, stft, istft, magphase, to_db
from separation.io import get_musdb, track_to_mono, track_to_stereo
from separation.metrics import si_sdr
from separation import stage_a_rpca, stage_b_nmf, stage_c_ksvd, stage_e_mmv, baselines

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.titlesize": 11})
print(f"sample rate {SR} Hz | n_fft {N_FFT} | hop {HOP}")
print(f"artifacts in: {EXP}")
""")

code(r"""
# Download the MUSDB18 7-second sample set if needed, then open the test split.
if not (DATA / "test").exists():
    import musdb
    musdb.DB(download=True, root=str(DATA))
mus = list(get_musdb(subsets="test", root=DATA))
print(f"{len(mus)} test tracks available")
""")

code(r"""
# ---- helpers used throughout ----
CLIP = 6 * SR   # 6-second clips keep the classical solvers interactive

def load_song(hint, idx_fallback=0):
    track = next((t for t in mus if hint.lower() in t.name.lower()), mus[idx_fallback])
    s = track_to_mono(track)
    return {"name": track.name,
            "mix": s["mix"][:CLIP], "voc": s["vocals"][:CLIP], "acc": s["accompaniment"][:CLIP]}

def spec(sig):
    return to_db(np.abs(stft(sig)))

def show_specs(sigs, titles, suptitle=None, figsize=(11, 2.8), ylim=(40, 8000)):
    fig, ax = plt.subplots(1, len(sigs), figsize=figsize, sharey=True)
    ax = np.atleast_1d(ax)
    for a, sg, t in zip(ax, sigs, titles):
        im = librosa.display.specshow(spec(sg), sr=SR, hop_length=HOP, x_axis="time",
                                      y_axis="log", ax=a, cmap="magma")
        a.set_title(t); a.set_ylim(*ylim)
    if suptitle: fig.suptitle(suptitle, y=1.04)
    plt.tight_layout(); plt.show()

def play(sig, desc):
    display(Markdown(f"&#9834; **{desc}**"))
    display(Audio(np.asarray(sig, dtype=float), rate=SR))

def evaluate_on(songs, sep_fn, source="vocals"):
    '''Run a separator on every song; return SI-SDR per song and the outputs.'''
    rows, outs = [], []
    for s in songs:
        res = sep_fn(s["mix"])
        outs.append(res)
        rows.append((s["name"],
                     si_sdr(s["voc"], res["vocals"]),
                     si_sdr(s["acc"], res["accompaniment"])))
    return rows, outs

def sisdr_table(rows, title):
    print(f"{title}")
    print(f"  {'track':38s}{'vocals':>9}{'accomp.':>9}  (SI-SDR, dB)")
    for name, v, a in rows:
        print(f"  {name[:38]:38s}{v:>9.2f}{a:>9.2f}")
    print(f"  {'median':38s}{np.median([r[1] for r in rows]):>9.2f}"
          f"{np.median([r[2] for r in rows]):>9.2f}")

# Four demonstration songs, chosen for varied instrumentation.
SONGS = [load_song("Al James", 1),
         load_song("Angels In Amplifiers", 2),
         load_song("Cristina Vane", 0),
         load_song("Detsky Sad", 3)]
print("Demo songs:")
for s in SONGS: print("  -", s["name"])
""")

# ===================================================================
md(r"""
## 1. The data: a song is a picture of sound

We never touch the raw waveform. The **short-time Fourier transform (STFT)** slides
a window across $y(t)$ and takes a DFT in each window, producing a complex matrix
$$ \mathbf{Y} = \mathrm{STFT}(y) \in \mathbb{C}^{F\times T},\qquad
   \mathbf{Y} = \mathbf{M}\odot e^{i\angle\mathbf{Y}}, \quad \mathbf{M}=|\mathbf{Y}|, $$
with $F=N_{\mathrm{FFT}}/2+1$ frequency bins and $T$ time frames. The magnitude
$\mathbf{M}$ is a nonnegative image (brightness = energy), the phase $\angle\mathbf{Y}$
holds the waveform alignment, and the transform is invertible.

Because the STFT is linear, the mixture spectrogram is approximately the sum of the
stem spectrograms. Two regularities, visible by eye below, make separation possible:

- **Vocals are sparse**: a sung note lights up a few horizontal harmonic lines.
- **Accompaniment is low-rank**: repeating chords and drums reuse a few templates.

These are exactly the structures compressed sensing recovers. We look at all four
demo songs, then a chromagram (energy folded onto the 12 pitch classes) for one.
""")

code(r"""
for s in SONGS:
    show_specs([s["mix"], s["voc"], s["acc"]],
               ["Mixture $Y$", "Vocals $X_v$ (truth)", "Accompaniment $X_a$ (truth)"],
               suptitle=s["name"])
""")

code(r"""
# Chromagram of one accompaniment: the harmony folded onto 12 pitch classes.
s = SONGS[0]
chroma = librosa.feature.chroma_cqt(y=s["acc"], sr=SR, hop_length=HOP)
plt.figure(figsize=(7, 2.4))
librosa.display.specshow(chroma, sr=SR, hop_length=HOP, x_axis="time", y_axis="chroma", cmap="viridis")
plt.colorbar(); plt.title(f"Chromagram of the accompaniment: {s['name']}"); plt.tight_layout(); plt.show()

print("Listen to the four mixtures we will un-mix (full song + band together):")
for s in SONGS:
    play(s["mix"], f"{s['name']} - the original mixture (vocals + accompaniment)")
""")

# ===================================================================
md(r"""
## 2. Preliminaries: when can we recover a sparse signal?

The honest objective is the sparsest explanation,
$$ (P_0)\quad \min_{\mathbf{z}} \|\mathbf{z}\|_0 \ \ \text{s.t.}\ \ \mathbf{A}\mathbf{z}=\mathbf{y}, $$
but counting nonzeros is NP-hard, so we **chicken out** to the convex $\ell_1$
relaxation, a linear program,
$$ (P_1)\quad \min_{\mathbf{z}} \|\mathbf{z}\|_1 \ \ \text{s.t.}\ \ \mathbf{A}\mathbf{z}=\mathbf{y}. $$
Three sufficient conditions certify that $(P_1)$ returns the true $s$-sparse $\mathbf{x}$:

$$ \underbrace{\mathrm{spark}(\mathbf{A})>2s}_{\text{uniqueness}},\qquad
   \underbrace{(1-\delta_s)\|\mathbf{x}\|_2^2\le\|\mathbf{A}\mathbf{x}\|_2^2\le(1+\delta_s)\|\mathbf{x}\|_2^2,\ \delta_{2s}<\sqrt{2}-1}_{\text{RIP}},\qquad
   \underbrace{s<\tfrac12\!\left(1+\tfrac1{\mu(\mathbf{A})}\right)}_{\text{coherence}}. $$

The **spark** is the size of the smallest linearly dependent column subset; **RIP**
says every small set of columns is nearly an isometry (length is *sandwiched*
between $1-\delta$ and $1+\delta$); and the **coherence**
$\mu(\mathbf{A})=\max_{i\ne j}|\langle\mathbf{a}_i,\mathbf{a}_j\rangle|$ measures how
similar two atoms are. A greedy alternative to $(P_1)$ is **orthogonal matching
pursuit (OMP)**: repeatedly pick the atom most correlated with the residual, refit
by least squares, and subtract. We use OMP inside K-SVD below.

Let us verify the $\ell_1$ idea on a tiny synthetic problem before touching audio.
""")

code(r"""
# Compressed-sensing sanity check: recover a sparse signal from few measurements.
rng = np.random.default_rng(0)
N, m, s = 200, 60, 5
A = rng.standard_normal((m, N)); A /= np.linalg.norm(A, axis=0, keepdims=True)
support = rng.choice(N, s, replace=False)
x = np.zeros(N); x[support] = rng.standard_normal(s)
y = A @ x
x_hat = stage_c_ksvd.omp(A, y, s)              # the project's OMP, reused from PS3
print(f"measurements m={m} << N={N}, sparsity s={s}")
print(f"support recovered exactly: {set(support) <= set(np.flatnonzero(np.abs(x_hat)>1e-6))}")
print(f"coefficient error ||x-x_hat||_2 = {np.linalg.norm(x-x_hat):.2e}")
plt.figure(figsize=(7, 2.2)); plt.stem(x, markerfmt="o", linefmt="C0-", basefmt=" ", label="truth")
plt.stem(x_hat, markerfmt="x", linefmt="C1--", basefmt=" ", label="OMP")
plt.legend(); plt.title("OMP recovers a sparse signal from few measurements"); plt.tight_layout(); plt.show()
""")

# ===================================================================
md(r"""
## 3. Method A: Robust PCA (low-rank + sparse), no training

The first separator needs **no training at all**. On the magnitude spectrogram
$\mathbf{M}$ we solve principal component pursuit,
$$ \min_{\mathbf{X},\mathbf{E}}\ \|\mathbf{X}\|_* + \lambda\|\mathbf{E}\|_1
   \quad\text{s.t.}\quad \mathbf{X}+\mathbf{E}=\mathbf{M},\qquad \lambda=\tfrac{1}{\sqrt{\max(F,T)}}, $$
where the **nuclear norm** $\|\mathbf{X}\|_*=\sum_i\sigma_i(\mathbf{X})$ is the
$\ell_1$ of the singular values (our convex stand-in for low rank) and
$\|\mathbf{E}\|_1$ is the stand-in for sparse. We solve it by inexact ALM, which
alternates two proximal operators: the **singular-value threshold**
$\mathrm{SVT}_\tau(\mathbf{Z})=\mathbf{U}\,\mathrm{soft}_\tau(\mathbf{\Sigma})\mathbf{V}^\top$
(shrinks small singular values to zero) and the **soft-threshold**
$\mathrm{soft}_\tau(x)=\mathrm{sgn}(x)\max(|x|-\tau,0)$:
$$ \mathbf{X}\leftarrow\mathrm{SVT}_{1/\mu}(\mathbf{M}-\mathbf{E}+\tfrac1\mu\mathbf{\Lambda}),\quad
   \mathbf{E}\leftarrow\mathrm{soft}_{\lambda/\mu}(\mathbf{M}-\mathbf{X}+\tfrac1\mu\mathbf{\Lambda}),\quad
   \mathbf{\Lambda}\leftarrow\mathbf{\Lambda}+\mu(\mathbf{M}-\mathbf{X}-\mathbf{E}). $$
The low-rank $\mathbf{X}$ is the repeating accompaniment; the sparse $\mathbf{E}$ is
the voice. It is the same trick that pulls a moving foreground out of a static
surveillance background, or a CD scratch (sparse noise) out of the music.
""")

code(r"""
# Decompose one song and look at the low-rank vs sparse parts.
s = SONGS[0]
Mmag, ph = magphase(stft(s["mix"]))
X, E, info = stage_a_rpca.rpca(Mmag, max_iter=100)
print(f"{s['name']}: ALM converged to residual {info['final_residual']:.1e} in "
      f"{info['iters']} iters; recovered rank {info['rank']}")
fig, ax = plt.subplots(1, 3, figsize=(11, 2.8), sharey=True)
for a, Z, t in zip(ax, [Mmag, X, E],
                   ["Mixture $|Y|$", "Low-rank $X$ (accompaniment)", "Sparse $E$ (voice)"]):
    librosa.display.specshow(to_db(np.maximum(Z,1e-9)), sr=SR, hop_length=HOP,
                             x_axis="time", y_axis="log", ax=a, cmap="magma"); a.set_title(t); a.set_ylim(40,8000)
plt.tight_layout(); plt.show()

# Convergence + the singular spectrum that justifies the low-rank prior.
U, sig, Vt = np.linalg.svd(Mmag, full_matrices=False)
fig, ax = plt.subplots(1, 2, figsize=(8, 2.6))
ax[0].semilogy(info["residuals"], color="#0B4F2F"); ax[0].set_xlabel("ALM iteration")
ax[0].set_ylabel(r"$\|M-X-E\|_F/\|M\|_F$"); ax[0].set_title("RPCA converges"); ax[0].grid(alpha=.3)
ax[1].semilogy(sig[:80]/sig[0], color="#7A1F8B"); ax[1].axvline(info["rank"], color="orange", ls="--")
ax[1].set_xlabel("singular value index"); ax[1].set_ylabel(r"$\sigma_i/\sigma_1$")
ax[1].set_title("Spectrum collapses (low rank)"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
""")

code(r"""
rows_A, outs_A = evaluate_on(SONGS, lambda y: stage_a_rpca.separate(y, max_iter=80))
sisdr_table(rows_A, "Robust PCA (training-free)")
# Spectrograms of the recovered vocals across all four songs.
show_specs([o["vocals"] for o in outs_A], [s["name"][:18] for s in SONGS],
           suptitle="Robust PCA: recovered vocals (sparse residual)", figsize=(12, 2.6))
print("\\nListen. RPCA has no idea what a voice is; it just calls the sparse leftover the voice,")
print("so the accompaniment is clean but the vocal is rough. Compare three songs:")
for s, o in zip(SONGS[:3], outs_A[:3]):
    play(o["accompaniment"], f"{s['name']} - RPCA accompaniment (the low-rank part; should sound full and clean)")
play(outs_A[0]["vocals"], f"{SONGS[0]['name']} - RPCA 'vocals' (the sparse part; thin, with bleed)")
""")

# ===================================================================
md(r"""
## 4. Method B: supervised NMF (learned per-source dictionaries)

Now we **learn** $\mathbf{A}$. Non-negative matrix factorization approximates a
nonnegative spectrogram as $\mathbf{M}\approx\mathbf{D}\mathbf{H}$ with
$\mathbf{D},\mathbf{H}\ge 0$: the columns of $\mathbf{D}$ are spectral **atoms**
(templates), and $\mathbf{H}$ holds their time-varying activations. Because you can
only *add* atoms, never subtract, NMF learns **parts**. We minimize the
Kullback-Leibler divergence with the Lee-Seung **multiplicative updates**
$$ \mathbf{H}\leftarrow\mathbf{H}\odot\frac{\mathbf{D}^\top(\mathbf{M}\oslash\mathbf{D}\mathbf{H})}{\mathbf{D}^\top\mathbf{1}},\qquad
   \mathbf{D}\leftarrow\mathbf{D}\odot\frac{(\mathbf{M}\oslash\mathbf{D}\mathbf{H})\mathbf{H}^\top}{\mathbf{1}\mathbf{H}^\top}, $$
which are a *majorization-minimization* scheme (built with Jensen's inequality) and
provably cannot increase the loss. We pre-train one dictionary per source,
$\mathbf{D}_v$ and $\mathbf{D}_a$; at test time we **fix** the dictionary
$\mathbf{D}=[\mathbf{D}_v\mid\mathbf{D}_a]$, solve only for the activations, and split
$\widehat{\mathbf{M}}_v=\mathbf{D}_v\mathbf{H}_v$, $\widehat{\mathbf{M}}_a=\mathbf{D}_a\mathbf{H}_a$.
""")

code(r"""
nmf = np.load(EXP / "dictionaries" / "nmf.npz")
Dv_nmf, Da_nmf = nmf["Dv"], nmf["Da"]
freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
def sort_atoms(D):
    cen = (freqs[:, None] * D).sum(0) / (D.sum(0) + 1e-9)
    return D[:, np.argsort(cen)]
fig, ax = plt.subplots(2, 1, figsize=(8, 3.4))
for a, D, t in zip(ax, [sort_atoms(Dv_nmf), sort_atoms(Da_nmf)],
                   ["Vocal atoms $D_v$ (sorted by pitch)", "Accompaniment atoms $D_a$"]):
    a.imshow(librosa.amplitude_to_db(D+1e-6, ref=np.max), aspect="auto", origin="lower",
             cmap="magma", extent=[0, D.shape[1], 0, freqs[-1]]); a.set_ylim(0,6000); a.set_title(t); a.set_ylabel("Hz")
plt.tight_layout(); plt.show()
print("Vocal atoms concentrate harmonic ridges in the formant region; accompaniment atoms spread lower.")
""")

code(r"""
rows_B, outs_B = evaluate_on(SONGS, lambda y: stage_b_nmf.separate(y, Dv_nmf, Da_nmf, beta="kl"))
sisdr_table(rows_B, "Supervised NMF (KL divergence, K=64 atoms/source)")
show_specs([o["vocals"] for o in outs_B], [s["name"][:18] for s in SONGS],
           suptitle="Supervised NMF: recovered vocals", figsize=(12, 2.6))
print("\\nListen. Now the model knows what a voice looks like, so the isolated vocal is far cleaner:")
for s, o in zip(SONGS, outs_B):
    play(o["vocals"], f"{s['name']} - NMF vocals (lead voice lifted off the band)")
""")

# ===================================================================
md(r"""
## 5. Method C: K-SVD + sparse-representation classification

K-SVD learns a sharper, **overcomplete** dictionary that sparsely spells out each
source. It alternates (i) sparse coding every training frame with OMP, and (ii)
updating one atom at a time by a **rank-one SVD** of the residual restricted to the
frames that use it; that update is optimal in the Eckart-Young sense, so it cannot
increase the error. At test time we sparse-code each mixture frame on the combined
dictionary $[\mathbf{D}_v\mid\mathbf{D}_a]$ and route the energy by the
**sparse-representation-classification** residual rule,
$$ c^\star=\arg\min_c \|\mathbf{y}-\mathbf{D}_c\boldsymbol{\alpha}_c\|_2, $$
keeping whichever source spells the frame more cleanly. Whether this per-frame code
is unique is governed by the coherence bound from Section 2; we return to that in
Section 11.
""")

code(r"""
ksvd = np.load(EXP / "dictionaries" / "ksvd.npz")
Dv_ksvd, Da_ksvd = ksvd["Dv"], ksvd["Da"]
rows_C, outs_C = evaluate_on(SONGS, lambda y: stage_c_ksvd.separate(y, Dv_ksvd, Da_ksvd, n_nonzero=10))
sisdr_table(rows_C, "K-SVD + SRC (K=96, sparsity s=10)")
show_specs([o["vocals"] for o in outs_C], [s["name"][:18] for s in SONGS],
           suptitle="K-SVD + SRC: recovered vocals", figsize=(12, 2.6))
print("\\nListen. A discriminative alphabet per source sharpens the routing:")
for s, o in zip(SONGS, outs_C):
    play(o["vocals"], f"{s['name']} - K-SVD vocals (each frame spelled in the cleaner source alphabet)")
""")

# ===================================================================
md(r"""
## 6. Method D: the deep sparse coding network (the new method)

A neural network *is* a sparse-recovery solver with its iterations **unrolled** into
layers. We keep that structure but **learn** the dictionaries by backpropagation.
Each composite module is an expand-then-reduce **hourglass**: a *fat* dictionary
lifts the frame to an over-complete code $\boldsymbol{\alpha}^{(1)}$, a *tall*
dictionary compresses it to $\boldsymbol{\alpha}^{(2)}$, each a non-negative
elastic-net code
$$ \boldsymbol{\alpha}^{(\ell)}=\arg\min_{\boldsymbol{\alpha}\ge0}
   \tfrac12\|\mathbf{u}-\mathbf{D}^{(\ell)}\boldsymbol{\alpha}\|_2^2
   +\lambda_1\|\boldsymbol{\alpha}\|_1+\tfrac{\lambda_2}{2}\|\boldsymbol{\alpha}\|_2^2, $$
solved by $K$ steps of non-negative **FISTA**, each a matrix multiply, a gradient
step, and a clamp,
$$ \boldsymbol{\alpha}_k=\Big[\mathbf{z}_k-\tfrac1L\mathbf{D}^\top(\mathbf{D}\mathbf{z}_k-\mathbf{u})-\tfrac{\lambda_1}{L}\Big]_+,\qquad
   \mathbf{z}_{k+1}=\boldsymbol{\alpha}_k+\tfrac{t_k-1}{t_{k+1}}(\boldsymbol{\alpha}_k-\boldsymbol{\alpha}_{k-1}). $$
Everything is differentiable, so the dictionaries $\mathbf{D}^{(\ell)}$ and the
shrinkages $\lambda_1,\lambda_2$ train end-to-end against the ideal ratio mask,
$\mathcal{L}=\|(\mathbf{M}_v-\mathbf{M}_v^{\mathrm{IRM}})\odot\mathbf{M}\|_1+\beta\sum_\ell\|\boldsymbol{\alpha}^{(\ell)}\|_1$.
This is the **LISTA** construction specialized to non-negative codes and a masking
output. We stack four modules with a $\pm 2$-frame temporal context window (1.44M
parameters) and load the trained network.
""")

code(r"""
import torch
from separation.stage_d_scn import SparseNet, separate as scn_separate
ckpt = torch.load(EXP / "sparsenet" / "model.pth", map_location="cpu", weights_only=False)
model = SparseNet(**ckpt["config"]); model.load_state_dict(ckpt["state_dict"]); model.eval()
print(f"SparseNet: {ckpt['n_params']:,} parameters, {ckpt['config']['n_modules']} modules, "
      f"context {ckpt['config'].get('context',0)} frames")

# Training curve: the masked-magnitude loss falls while the codes stay sparse.
hist = np.load(EXP / "sparsenet" / "history.npy", allow_pickle=True)
ep=[h['epoch'] for h in hist]; loss=[h['loss'] for h in hist]; sp=[h['sparsity'] for h in hist]
fig, ax1 = plt.subplots(figsize=(5.4, 2.6))
ax1.plot(ep, loss, color="#0B4F2F"); ax1.set_xlabel("epoch"); ax1.set_ylabel("recon $\\ell_1$ loss", color="#0B4F2F")
ax2 = ax1.twinx(); ax2.plot(ep, sp, "--", color="#7A1F8B"); ax2.set_ylabel("fraction zero codes", color="#7A1F8B")
ax1.set_title("Deep coder: loss down, codes stay sparse"); plt.tight_layout(); plt.show()
""")

code(r"""
rows_D, outs_D = evaluate_on(SONGS, lambda y: scn_separate(y, model))
sisdr_table(rows_D, "Deep sparse coding network (ours)")
show_specs([o["vocals"] for o in outs_D], [s["name"][:18] for s in SONGS],
           suptitle="Deep sparse coder: recovered vocals", figsize=(12, 2.6))
print("\\nListen. The learned, unrolled solver gives the cleanest vocals of any non-oracle method:")
for s, o in zip(SONGS, outs_D):
    play(o["vocals"], f"{s['name']} - deep coder vocals (trained end-to-end against the ideal mask)")
""")

# ===================================================================
md(r"""
## 7. Stereo as joint sparsity (multiple measurement vectors)

A stereo recording has two channels that share the same active sources at each
instant; only the left/right balance differs. So their sparse codes share a **row
support**: the same atoms are on in both channels. This is the *multiple measurement
vector* (MMV) problem, and **simultaneous OMP** recovers the shared support by scoring
atoms with the cross-channel row norm $\|(\mathbf{D}^\top\mathbf{R})_{n,:}\|_2$. The
shared support is extra information, so identifiability *improves*:
$ s<\tfrac12(\mathrm{spark}(\mathbf{A})-1+\mathrm{rank}(\mathbf{Y})) $.
""")

code(r"""
tr = next((t for t in mus if "Al James" in t.name), mus[1])
st = track_to_stereo(tr); ys = st["mix"][:CLIP]
res_E = stage_e_mmv.separate_stereo(ys, Dv_ksvd, Da_ksvd, n_nonzero=10)
vref = st["vocals"][:CLIP].mean(1); aref = st["accompaniment"][:CLIP].mean(1)
print(f"Stereo MMV on {tr.name}:  vocals {si_sdr(vref, res_E['vocals'].mean(1)):+.2f} dB   "
      f"accomp. {si_sdr(aref, res_E['accompaniment'].mean(1)):+.2f} dB")
play(res_E["vocals"].mean(1), f"{tr.name} - stereo joint-sparsity vocals (both channels share the active atoms)")
""")

# ===================================================================
md(r"""
## 8. The full comparison (25 held-out test tracks)

The cells above ran on a handful of songs. Here is the headline result over all 25
held-out test tracks (median SI-SDR), loaded from the saved evaluation, with the
trivial floors and the oracle ceilings. Read the table as a ladder, **training-free
< supervised < deep**. Note that even *returning the mixture unchanged* scores about
$+4$ dB on accompaniment because the band dominates the energy, so accompaniment
numbers must be read against that floor.
""")

code(r"""
summary = json.load(open(EXP / "results" / "main_summary.json"))
rows = [("hpss","HPSS (median filtering)"),("repet","REPET-SIM"),("rpca","Robust PCA"),
        ("nmf","Supervised NMF"),("ksvd","K-SVD + SRC"),("scn","Deep coder (ours)"),
        ("oracle_irm","Oracle IRM (ceiling)")]
print(f"{'Method':26s}{'Vocals':>9}{'Accomp.':>9}{'Time(s)':>9}")
print("-"*53)
for k, name in rows:
    d = summary.get(k, {})
    print(f"{name:26s}{d.get('vocals_si_sdr_median',float('nan')):>9.1f}"
          f"{d.get('acc_si_sdr_median',float('nan')):>9.1f}{str(d.get('time_mean_s','-')):>9}")

keys = [k for k in ["hpss","repet","rpca","nmf","ksvd","scn","oracle_irm","oracle_ibm"] if k in summary]
lab = {"hpss":"HPSS","repet":"REPET","rpca":"RPCA","nmf":"NMF","ksvd":"K-SVD",
       "scn":"SCN\n(ours)","oracle_irm":"IRM\n(oracle)","oracle_ibm":"IBM\n(oracle)"}
voc=[summary[k]["vocals_si_sdr_median"] for k in keys]; acc=[summary[k]["acc_si_sdr_median"] for k in keys]
x=np.arange(len(keys)); w=0.4
fig,axb=plt.subplots(figsize=(8,3)); axb.bar(x-w/2,voc,w,label="vocals",color="#7A1F8B")
axb.bar(x+w/2,acc,w,label="accompaniment",color="#0B4F2F"); axb.axhline(0,color="k",lw=.6)
axb.set_xticks(x); axb.set_xticklabels([lab[k] for k in keys],fontsize=8); axb.set_ylabel("median SI-SDR (dB)")
axb.set_title("Separation quality by method"); axb.legend(); plt.tight_layout(); plt.show()
""")

# ===================================================================
md(r"""
## 9. Per-track spread and the soft-vs-binary mask

Medians hide variance, so we plot the per-track distribution; the deep model shifts
the whole distribution rightward, not only the easy songs. We also compare **soft**
(ratio) masks against **hard binary** masks: a soft value keeps the partial-overlap
cells a 0/1 decision throws away, which mirrors the gap between the ideal-ratio and
ideal-binary oracles.
""")

code(r"""
pt = list(csv.DictReader(open(EXP / "results" / "main_results.csv")))
methods = [("hpss","HPSS"),("repet","REPET"),("rpca","RPCA"),("nmf","NMF"),("ksvd","K-SVD"),("scn","SCN\n(ours)")]
data = [[float(r[f"{k}_vocals_si_sdr"]) for r in pt if r.get(f"{k}_vocals_si_sdr")] for k,_ in methods]
fig, ax = plt.subplots(figsize=(7.5, 3))
bp = ax.boxplot(data, patch_artist=True, widths=.6, medianprops=dict(color="k", lw=1.4))
for patch, c in zip(bp["boxes"], ["#9AA0A6","#9AA0A6","#7A1F8B","#7A1F8B","#7A1F8B","#B5651D"]):
    patch.set_facecolor(c); patch.set_alpha(.6)
ax.axhline(0, color="k", lw=.6, ls=":"); ax.set_xticklabels([m[1] for m in methods])
ax.set_ylabel("vocal SI-SDR (dB)"); ax.set_title("Per-track vocal SI-SDR distribution (25 tracks)")
plt.tight_layout(); plt.show()

mt = json.load(open(EXP / "results" / "masktype.json"))
ks = [k for k in ["nmf","ksvd","scn"] if f"{k}_soft" in mt]
soft=[mt[f"{k}_soft"] for k in ks]; binr=[mt[f"{k}_binary"] for k in ks]
x=np.arange(len(ks)); w=.38
fig,ax=plt.subplots(figsize=(4.8,2.8)); ax.bar(x-w/2,soft,w,label="soft (ratio) mask",color="#0B4F2F")
ax.bar(x+w/2,binr,w,label="binary mask",color="#B5651D"); ax.set_xticks(x)
ax.set_xticklabels([k.upper() for k in ks]); ax.set_ylabel("vocal SI-SDR (dB)")
ax.set_title("Soft masks beat binary masks"); ax.legend(fontsize=8); plt.tight_layout(); plt.show()
""")

# ===================================================================
md(r"""
## 10. Ablations: what each knob does

One setting changed at a time, so we learn *why* each method works. The findings:
**smaller NMF dictionaries generalize better** (a small dictionary is a stronger
prior when data is limited); K-SVD improves with the sparsity budget up to
$s\approx10$ then saturates; and a larger RPCA penalty $\lambda$ pushes more energy
into the low-rank band and leaves a cleaner sparse vocal.
""")

code(r"""
fig, axs = plt.subplots(1, 3, figsize=(11, 2.6))
for a, (fn, xcol, xl) in zip(axs, [("ablation_nmf.csv","K","NMF atoms $K$"),
                                   ("ablation_ksvd.csv","S","K-SVD sparsity $s$"),
                                   ("ablation_rpca.csv","lambda_scale",r"RPCA $\lambda$ scale")]):
    rws = list(csv.DictReader(open(EXP / "results" / fn)))
    xs=[float(r[xcol]) for r in rws]; vv=[float(r["vocals_si_sdr"]) for r in rws]; aa=[float(r["acc_si_sdr"]) for r in rws]
    a.plot(xs, vv, "o-", color="#7A1F8B", label="vocals"); a.plot(xs, aa, "s--", color="#0B4F2F", label="accomp.")
    a.set_xlabel(xl); a.grid(alpha=.3)
axs[0].set_ylabel("median SI-SDR (dB)"); axs[0].legend(fontsize=8)
fig.suptitle("Hyperparameter ablations", y=1.03); plt.tight_layout(); plt.show()
""")

# ===================================================================
md(r"""
## 11. An honest negative result: dictionary coherence

The recovery theory of Section 2 promises exact recovery when the coherence
$\mu=\max_{i\ne j}|\langle\mathbf{d}_i,\mathbf{d}_j\rangle|$ is small, specifically for
$s<\tfrac12(1+1/\mu)$. But a dictionary **trained for reconstruction** ends up
**coherent**, its atoms overlap. We measure $\mu$ on the learned K-SVD vocal
dictionary and find the worst-case theorem certifies almost nothing, yet the method
runs at $s=10$ and works. The guarantees are worst-case; real music frames are far
from that adversarial case. Average-case behaviour, not the worst-case bound, governs
performance.
""")

code(r"""
D = Dv_ksvd / (np.linalg.norm(Dv_ksvd, axis=0, keepdims=True) + 1e-9)
G = np.abs(D.T @ D); np.fill_diagonal(G, 0); mu = G.max()
off = G[np.triu_indices_from(G, k=1)]; s_rec = 0.5*(1+1/mu)
plt.figure(figsize=(5.4,2.6)); plt.hist(off, bins=50, color="#0B4F2F", alpha=.8)
plt.axvline(mu, color="red", ls="--", label=f"$\\mu$ = {mu:.2f}")
plt.title(f"K-SVD coherence: worst-case certifies only $s < {s_rec:.1f}$")
plt.xlabel(r"$|\langle d_i, d_j\rangle|$"); plt.ylabel("count"); plt.legend(); plt.tight_layout(); plt.show()
print(f"coherence mu = {mu:.3f}  ->  exact recovery certified only for s < {s_rec:.1f},")
print(f"yet the method operates at s = 10 and separates well: average-case != worst-case.")
""")

# ===================================================================
md(r"""
## 12. Conclusion

On one STFT front end, four methods that look unrelated, a low-rank-plus-sparse
split, two learned dictionaries, and an unrolled deep solver, are one family that
all recover a structured spectrogram. The amount of structure learned from data
climbs from left to right, and quality climbs with it: the deep sparse coding
network is the strongest non-oracle separator on both sources
(**2.0 dB vocals, 8.3 dB accompaniment**) and the fastest learned model, while the
classical recovery theory turns out to be a guide rather than a guarantee for
trained, coherent dictionaries.

The five-page report (`FINAL_REPORT.pdf`), the poster (`FINAL_POSTER.pdf`), all
code, the trained models, and the curated audio accompany this notebook. Full
history: <https://github.com/takakhoo/ENGS109_Final_Project>.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, str(OUT))
print(f"wrote {OUT.name} with {len(cells)} cells")
