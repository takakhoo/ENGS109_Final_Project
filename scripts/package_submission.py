"""Package the final-project submission into one clean, runnable ZIP.

Assembles the 5-page report, the fully-run notebook, all source code, the trained
artifacts the notebook needs, figures, curated audio, the poster, and a run-focused
README, then zips it. The 140 MB MUSDB sample is excluded (the notebook downloads it
on first run); the saved notebook already contains all outputs for viewing.

    python scripts/package_submission.py
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STAGE = REPO / "_submission" / "ENGS109_Unmixing_the_Music_Khoo"
ZIP = REPO / "ENGS109_Unmixing_the_Music_Khoo.zip"

SUBMISSION_README = r"""# Unmixing the Music
## Sparse, Low-Rank, and Deep: A Compressed-Sensing Approach to Music Source Separation

**ENGS 109 - High-Dimensional Sensing and Learning - Final Project**
**Taka Khoo, Thayer School of Engineering, Dartmouth College**

---

### What is in this submission

| File / folder | What it is |
|---|---|
| `report.pdf` | The 5-page final report (ICML format; references beyond p.5). |
| `Unmixing_the_Music.ipynb` | The reproducible walkthrough notebook, **already fully run** with every figure, table, and playable audio clip inline. Open it to see all results without running anything. |
| `poster.pdf` | The A0 conference poster. |
| `walkthrough.pdf` | A long plain-English presenter's companion (analogies + scripts). |
| `src/separation/` | The five separation methods + features / io / metrics / baselines. |
| `scripts/` | Training, evaluation, ablation, figure, and packaging entry points. |
| `experiments/` | The trained dictionaries (`*.npz`), the deep network checkpoint (`model.pth`), and the saved result tables. The notebook loads these so it runs in minutes. |
| `figures/` | All generated figures. |
| `audio_examples/curated/` | Curated before/after WAVs (mixture + each method). |
| `tests/` | Solver correctness tests (RPCA, OMP, SOMP on synthetic ground truth). |
| `paper/` | LaTeX source of the report + bibliography. |
| `requirements.txt` | Python dependencies. |

### Just want to see the results?

Open **`Unmixing_the_Music.ipynb`** in Jupyter (or VS Code). It is already executed,
so every plot, table, and audio player is visible without running a single cell.
`report.pdf` is the written report.

### To re-run everything yourself

```bash
pip install -r requirements.txt
jupyter notebook Unmixing_the_Music.ipynb     # then Kernel -> Restart & Run All
```

On the first run the notebook downloads the free MUSDB18 7-second sample set
(144 tracks, ~140 MB) into `data/`. It loads the pre-trained dictionaries and
network from `experiments/`, so a full pass takes a couple of minutes on a laptop
(CPU is fine; Apple MPS / CUDA is used automatically for the deep coder if present).

To reproduce the artifacts from scratch instead:

```bash
PYTHONPATH=src python scripts/train_dictionaries.py     # NMF + K-SVD dictionaries
PYTHONPATH=src python scripts/train_sparsenet.py --context 2 --n_modules 4   # deep coder
PYTHONPATH=src python scripts/run_experiments.py        # full 25-track comparison
PYTHONPATH=src python tests/test_solvers.py             # solver correctness
```

### Result in one line

Four separators on one STFT front end, from a training-free low-rank-plus-sparse
split to a learned unrolled deep coder, are one family; the deep coder is the
strongest non-oracle separator (vocal SI-SDR 2.0 dB, accompaniment 8.3 dB) and the
fastest learned model on 25 held-out MUSDB18 tracks.

Full project history: https://github.com/takakhoo/ENGS109_Final_Project
"""


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE.parent)
    STAGE.mkdir(parents=True)

    # --- top-level deliverables ---
    shutil.copy(REPO / "paper" / "paper.pdf", STAGE / "report.pdf")
    shutil.copy(REPO / "Unmixing_the_Music.ipynb", STAGE / "Unmixing_the_Music.ipynb")
    shutil.copy(REPO / "poster" / "poster.pdf", STAGE / "poster.pdf")
    if (REPO / "walkthrough" / "walkthrough.pdf").exists():
        shutil.copy(REPO / "walkthrough" / "walkthrough.pdf", STAGE / "walkthrough.pdf")
    shutil.copy(REPO / "requirements.txt", STAGE / "requirements.txt")
    (STAGE / "README.md").write_text(SUBMISSION_README)

    # --- source code ---
    def copytree(src, dst, ignore_pyc=True):
        ig = shutil.ignore_patterns("__pycache__", "*.pyc") if ignore_pyc else None
        shutil.copytree(src, dst, ignore=ig)

    copytree(REPO / "src", STAGE / "src")
    copytree(REPO / "scripts", STAGE / "scripts")
    copytree(REPO / "tests", STAGE / "tests")
    copytree(REPO / "figures", STAGE / "figures")

    # --- paper source (tex + bib + style) ---
    pdst = STAGE / "paper"; pdst.mkdir()
    for f in ["paper.tex", "references.bib", "icml2025.sty", "icml2025.bst",
              "fancyhdr.sty", "algorithm.sty", "algorithmic.sty"]:
        if (REPO / "paper" / f).exists():
            shutil.copy(REPO / "paper" / f, pdst / f)

    # --- trained artifacts needed by the notebook ---
    edst = STAGE / "experiments"; edst.mkdir()
    shutil.copytree(REPO / "experiments" / "dictionaries", edst / "dictionaries")
    shutil.copytree(REPO / "experiments" / "results", edst / "results")
    (edst / "sparsenet").mkdir()
    shutil.copy(REPO / "experiments" / "sparsenet" / "model.pth", edst / "sparsenet" / "model.pth")
    shutil.copy(REPO / "experiments" / "sparsenet" / "history.npy", edst / "sparsenet" / "history.npy")

    # --- curated audio ---
    shutil.copytree(REPO / "audio_examples" / "curated", STAGE / "audio_examples" / "curated")

    # --- empty data dir with a note (notebook downloads here) ---
    ddst = STAGE / "data"; ddst.mkdir()
    (ddst / "README.txt").write_text(
        "The MUSDB18 7-second sample set downloads here on the notebook's first run.\n")

    # --- zip it ---
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in STAGE.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(STAGE.parent))

    size_mb = ZIP.stat().st_size / 1e6
    n_files = sum(1 for _ in STAGE.rglob("*") if _.is_file())
    print(f"Packaged {n_files} files -> {ZIP.name}  ({size_mb:.1f} MB)")
    print(f"Staging dir: {STAGE}")


if __name__ == "__main__":
    main()
