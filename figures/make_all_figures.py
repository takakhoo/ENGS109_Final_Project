"""Generate every figure and numerical proof for the paper and poster.

Pulls from the real trained artifacts in experiments/. Writes PDFs to figures/.

Figures:
  fig_triptych.pdf        mix / vocals / accompaniment spectrograms (real MUSDB)
  fig_rpca_decomp.pdf     RPCA low-rank vs sparse on a real song
  fig_rpca_converge.pdf   RPCA residual + singular-value spectrum (convergence proof)
  fig_nmf_atoms.pdf       learned NMF spectral atoms (vocals vs accompaniment)
  fig_ksvd_coherence.pdf  K-SVD dictionary coherence histogram (RIP/NSP evidence)
  fig_sparsenet_train.pdf SparseNet loss + code-sparsity curves
  fig_results_bars.pdf    SI-SDR by stage with oracle ceiling
  fig_mask_compare.pdf    predicted masks: oracle vs NMF vs SparseNet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import librosa
import librosa.display

from separation.features import stft, magphase, to_db, SR, HOP
from separation.io import get_musdb, track_to_mono
from separation import stage_a_rpca, stage_b_nmf

OUT = REPO / "figures"
DICTS = REPO / "experiments" / "dictionaries"
RESULTS = REPO / "experiments" / "results"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

MAGMA = "magma"


def pick_track(name_hint="Cristina"):
    mus = get_musdb(subsets="test")
    for tr in mus:
        if name_hint.lower() in tr.name.lower():
            return tr
    return list(mus)[0]


def main():
    OUT.mkdir(exist_ok=True)
    print("Loading a representative test track...")
    track = pick_track()
    stems = track_to_mono(track)
    n = 6 * SR
    mix = stems["mix"][:n]
    voc = stems["vocals"][:n]
    acc = stems["accompaniment"][:n]
    print(f"  track: {track.name}")

    Mmix = np.abs(stft(mix))
    Mv = np.abs(stft(voc))
    Ma = np.abs(stft(acc))

    # --- 1. Spectrogram triptych ---
    print("[1] triptych")
    fig, ax = plt.subplots(1, 3, figsize=(9, 2.6), sharey=True)
    for a, S, t in zip(ax, [Mmix, Mv, Ma],
                       ["Mixture $Y$", "Vocals $X_v$", "Accompaniment $X_a$"]):
        im = librosa.display.specshow(to_db(S), sr=SR, hop_length=HOP, x_axis="time",
                                      y_axis="log", ax=a, cmap=MAGMA)
        a.set_title(t); a.set_ylim(40, 8000)
    fig.colorbar(im, ax=ax.ravel().tolist(), format="%+2.0f dB", shrink=0.85, pad=0.01)
    fig.savefig(OUT / "fig_triptych.pdf"); plt.close(fig)

    # --- 2 + 3. RPCA decomposition and convergence ---
    print("[2,3] RPCA decomposition + convergence")
    X, E, info = stage_a_rpca.rpca(Mmix, max_iter=100)
    fig, ax = plt.subplots(1, 3, figsize=(9, 2.6), sharey=True)
    for a, S, t in zip(ax, [Mmix, X, E],
                       ["Mixture $|Y|$", r"Low-rank $\hat{X}$", r"Sparse $\hat{E}$"]):
        im = librosa.display.specshow(to_db(S), sr=SR, hop_length=HOP, x_axis="time",
                                      y_axis="log", ax=a, cmap=MAGMA)
        a.set_title(t); a.set_ylim(40, 8000)
    fig.colorbar(im, ax=ax.ravel().tolist(), format="%+2.0f dB", shrink=0.85, pad=0.01)
    fig.savefig(OUT / "fig_rpca_decomp.pdf"); plt.close(fig)

    # Convergence: residual curve + singular value spectrum.
    U, s, Vt = np.linalg.svd(Mmix, full_matrices=False)
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 2.6))
    ax[0].semilogy(info["residuals"], color="#0B4F2F", lw=1.5)
    ax[0].set_xlabel("ALM iteration"); ax[0].set_ylabel(r"$\|M-X-E\|_F/\|M\|_F$")
    ax[0].set_title(f"RPCA convergence ({info['iters']} iters)")
    ax[0].grid(alpha=0.3)
    ax[1].semilogy(s[:80] / s[0], color="#7A1F8B", lw=1.5)
    ax[1].axvline(info["rank"], color="orange", ls="--", lw=1, label=f"recovered rank {info['rank']}")
    ax[1].set_xlabel("singular value index"); ax[1].set_ylabel(r"$\sigma_i/\sigma_1$")
    ax[1].set_title("Spectrum of $|Y|$ (low-rank prior)"); ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_rpca_converge.pdf"); plt.close(fig)

    # --- 4. NMF atoms ---
    print("[4] NMF atoms")
    nmf = np.load(DICTS / "nmf.npz")
    Dv, Da = nmf["Dv"], nmf["Da"]
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    fig, ax = plt.subplots(2, 1, figsize=(7.5, 3.6))
    # Sort atoms by spectral centroid for a clean visual.
    def sort_atoms(D):
        cen = (freqs[:, None] * D).sum(0) / (D.sum(0) + 1e-9)
        return D[:, np.argsort(cen)]
    for a, D, t in zip(ax, [sort_atoms(Dv), sort_atoms(Da)],
                       ["Vocal NMF atoms $D_v$", "Accompaniment NMF atoms $D_a$"]):
        im = a.imshow(librosa.amplitude_to_db(D + 1e-6, ref=np.max), aspect="auto",
                      origin="lower", cmap=MAGMA,
                      extent=[0, D.shape[1], 0, freqs[-1]])
        a.set_ylim(0, 6000); a.set_title(t); a.set_ylabel("Hz")
    ax[1].set_xlabel("atom index (sorted by spectral centroid)")
    fig.tight_layout(); fig.savefig(OUT / "fig_nmf_atoms.pdf"); plt.close(fig)

    # --- 5. K-SVD coherence (RIP/NSP evidence, L6/L7) ---
    print("[5] K-SVD coherence")
    ksvd = np.load(DICTS / "ksvd.npz")
    Dk = ksvd["Dv"]
    Dn = Dk / (np.linalg.norm(Dk, axis=0, keepdims=True) + 1e-9)
    G = np.abs(Dn.T @ Dn)
    np.fill_diagonal(G, 0)
    mu = G.max()
    offdiag = G[np.triu_indices_from(G, k=1)]
    # Welch lower bound on coherence for an FxK dictionary.
    F, K = Dk.shape
    welch = np.sqrt((K - F) / (F * (K - 1))) if K > F else 0.0
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.hist(offdiag, bins=50, color="#0B4F2F", alpha=0.8)
    ax.axvline(mu, color="red", ls="--", lw=1.2, label=fr"$\mu(D)={mu:.3f}$")
    if welch > 0:
        ax.axvline(welch, color="orange", ls=":", lw=1.2, label=fr"Welch bound ${welch:.3f}$")
    # Exact-recovery sparsity from L7: S < 0.5(1 + 1/mu).
    S_rec = 0.5 * (1 + 1 / mu)
    ax.set_title(f"K-SVD vocal dictionary coherence (L7: exact recovery for $S<{S_rec:.1f}$)")
    ax.set_xlabel(r"$|\langle d_i, d_j\rangle|$"); ax.set_ylabel("count")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(OUT / "fig_ksvd_coherence.pdf"); plt.close(fig)
    print(f"    coherence mu={mu:.3f}, Welch={welch:.3f}, S_recovery<{S_rec:.1f}")

    # --- 6. SparseNet training curve ---
    print("[6] SparseNet training")
    hist_path = REPO / "experiments" / "sparsenet" / "history.npy"
    if hist_path.exists():
        hist = np.load(hist_path, allow_pickle=True)
        ep = [h["epoch"] for h in hist]
        loss = [h["loss"] for h in hist]
        sp = [h["sparsity"] for h in hist]
        fig, ax1 = plt.subplots(figsize=(5.2, 2.6))
        ax1.plot(ep, loss, color="#0B4F2F", lw=1.5, label="recon L1 loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("masked-magnitude L1", color="#0B4F2F")
        ax2 = ax1.twinx()
        ax2.plot(ep, sp, color="#7A1F8B", lw=1.5, ls="--", label="code sparsity")
        ax2.set_ylabel("fraction of zero codes", color="#7A1F8B")
        ax1.set_title("SparseNet end-to-end training (L16)")
        fig.tight_layout(); fig.savefig(OUT / "fig_sparsenet_train.pdf"); plt.close(fig)

    # --- 7. Results bar chart ---
    print("[7] results bars")
    if (RESULTS / "summary.json").exists():
        summary = json.load(open(RESULTS / "summary.json"))
        stages = [s for s in ["A", "B", "C", "D", "oracle"] if s in summary]
        labels = {"A": "A: RPCA\n(L13)", "B": "B: NMF\n(L15)", "C": "C: K-SVD\n(L14/8)",
                  "D": "D: SparseNet\n(L16)", "oracle": "Oracle\n(IRM)"}
        voc = [summary[s].get("vocals_si_sdr_median", np.nan) for s in stages]
        acc = [summary[s].get("acc_si_sdr_median", np.nan) for s in stages]
        x = np.arange(len(stages)); w = 0.38
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
        b1 = ax.bar(x - w/2, voc, w, label="vocals", color="#7A1F8B")
        b2 = ax.bar(x + w/2, acc, w, label="accompaniment", color="#0B4F2F")
        if "oracle" in stages:
            oi = stages.index("oracle")
            ax.axhline(voc[oi], color="#7A1F8B", ls=":", lw=1, alpha=0.6)
            ax.axhline(acc[oi], color="#0B4F2F", ls=":", lw=1, alpha=0.6)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels([labels[s] for s in stages], fontsize=8)
        ax.set_ylabel("median SI-SDR (dB)")
        ax.set_title("Separation quality by stage (MUSDB18 sample test)")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        for b in list(b1) + list(b2):
            h = b.get_height()
            ax.annotate(f"{h:.1f}", (b.get_x() + b.get_width()/2, h),
                        ha="center", va="bottom" if h >= 0 else "top", fontsize=6.5)
        fig.tight_layout(); fig.savefig(OUT / "fig_results_bars.pdf"); plt.close(fig)

    # --- 8. Mask comparison: oracle vs NMF vs SparseNet ---
    print("[8] mask comparison")
    irm = Mv / (Mv + Ma + 1e-9)
    resB = stage_b_nmf.separate(mix, Dv, Da, beta="kl")
    maskB = resB["mask_vocals"]
    panels = [(irm, "Oracle IRM"), (maskB, "NMF mask (Stage B)")]
    # SparseNet mask if available.
    try:
        import torch
        from separation.stage_d_scn import SparseNet, device_auto
        ckpt = torch.load(REPO / "experiments" / "sparsenet" / "model.pth",
                          map_location="cpu", weights_only=False)
        model = SparseNet(**ckpt["config"]); model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with torch.no_grad():
            Xt = torch.tensor(Mmix.T, dtype=torch.float32)
            maskD = model(Xt)[0].T.numpy()
        panels.append((maskD, "SparseNet mask (Stage D)"))
    except Exception as e:
        print(f"    SparseNet mask skipped: {e}")
    fig, ax = plt.subplots(1, len(panels), figsize=(3*len(panels), 2.6), sharey=True)
    for a, (M, t) in zip(np.atleast_1d(ax), panels):
        im = librosa.display.specshow(M, sr=SR, hop_length=HOP, x_axis="time",
                                      y_axis="log", ax=a, cmap="viridis", vmin=0, vmax=1)
        a.set_title(t); a.set_ylim(40, 8000)
    fig.colorbar(im, ax=np.atleast_1d(ax).ravel().tolist(), shrink=0.85, pad=0.01)
    fig.savefig(OUT / "fig_mask_compare.pdf"); plt.close(fig)

    print(f"\nAll figures written to {OUT}")


if __name__ == "__main__":
    main()
