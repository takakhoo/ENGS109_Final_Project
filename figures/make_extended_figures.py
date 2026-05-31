"""Figures for the extended experiments: per-track distributions, soft-vs-binary
mask study, and the oracle-gap view.

Writes:
  fig_boxplot.pdf     per-track vocal SI-SDR distribution per method
  fig_masktype.pdf    soft vs binary mask, vocal SI-SDR
  fig_oracle_gap.pdf  fraction of the IRM oracle achieved per method
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "experiments" / "results"
OUT = REPO / "figures"

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 10,
                     "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})

GREEN = "#0B4F2F"; PURPLE = "#7A1F8B"; ACCENT = "#B5651D"; GREY = "#9AA0A6"


def load_main_per_track():
    rows = list(csv.DictReader(open(RESULTS / "main_results.csv")))
    return rows


def main():
    # --- Boxplot: per-track vocal SI-SDR distribution ---
    rows = load_main_per_track()
    methods = [("hpss", "HPSS"), ("repet", "REPET"), ("rpca", "RPCA"),
               ("nmf", "NMF"), ("ksvd", "K-SVD"), ("scn", "SCN\n(ours)")]
    data = []
    for key, _ in methods:
        col = f"{key}_vocals_si_sdr"
        vals = [float(r[col]) for r in rows if r.get(col) not in (None, "", "-")]
        data.append(vals)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", lw=1.5),
                    flierprops=dict(marker="o", ms=3, alpha=0.5))
    colors = [GREY, GREY, PURPLE, PURPLE, PURPLE, ACCENT]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xticklabels([m[1] for m in methods])
    ax.set_ylabel("vocal SI-SDR (dB)")
    ax.set_title("Per-track vocal SI-SDR distribution (25 tracks)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_boxplot.pdf"); plt.close(fig)
    print("wrote fig_boxplot.pdf")

    # --- Soft vs binary mask ---
    mt = json.load(open(RESULTS / "masktype.json"))
    keys = [k for k in ["nmf", "ksvd", "scn"] if f"{k}_soft" in mt]
    labels = {"nmf": "NMF", "ksvd": "K-SVD", "scn": "SCN (ours)"}
    soft = [mt[f"{k}_soft"] for k in keys]
    binr = [mt[f"{k}_binary"] for k in keys]
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    ax.bar(x - w/2, soft, w, label="soft (ratio) mask", color=GREEN)
    ax.bar(x + w/2, binr, w, label="binary mask", color=ACCENT)
    ax.set_xticks(x); ax.set_xticklabels([labels[k] for k in keys])
    ax.set_ylabel("vocal SI-SDR (dB)")
    ax.set_title("Soft masks beat binary masks")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    for i, (s, b) in enumerate(zip(soft, binr)):
        ax.annotate(f"{s:.1f}", (i - w/2, s), ha="center", va="bottom", fontsize=7)
        ax.annotate(f"{b:.1f}", (i + w/2, b), ha="center", va="bottom", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig_masktype.pdf"); plt.close(fig)
    print("wrote fig_masktype.pdf")

    # --- Oracle gap (fraction of IRM accompaniment achieved) ---
    ext = json.load(open(RESULTS / "extended_summary.json"))
    main = json.load(open(RESULTS / "main_summary.json"))
    irm_a = ext["oracle_irm_acc"]
    methods2 = [("repet", "REPET"), ("rpca", "RPCA"), ("nmf", "NMF"),
                ("ksvd", "K-SVD"), ("scn", "SCN (ours)")]
    fracs = []
    for key, _ in methods2:
        acc = main[key]["acc_si_sdr_median"]
        fracs.append(100.0 * acc / irm_a)
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    ys = np.arange(len(methods2))
    cols = [PURPLE]*4 + [ACCENT]
    ax.barh(ys, fracs, color=cols, alpha=0.75)
    ax.set_yticks(ys); ax.set_yticklabels([m[1] for m in methods2])
    ax.set_xlabel("% of IRM oracle accompaniment SI-SDR")
    ax.set_title("How close to the oracle ceiling?")
    ax.grid(axis="x", alpha=0.3); ax.invert_yaxis()
    for y, f in zip(ys, fracs):
        ax.annotate(f"{f:.0f}%", (f, y), ha="left", va="center", fontsize=8, xytext=(3,0),
                    textcoords="offset points")
    fig.tight_layout(); fig.savefig(OUT / "fig_oracle_gap.pdf"); plt.close(fig)
    print("wrote fig_oracle_gap.pdf")


if __name__ == "__main__":
    main()
