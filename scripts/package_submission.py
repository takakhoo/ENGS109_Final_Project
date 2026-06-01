"""Package the final-project submission into one clean, runnable ZIP.

Top of the ZIP holds only the four things a reader wants first: the README, the
5-page report, the A0 poster, and the fully-run notebook. Everything else (all
source code, trained artifacts, figures, tests, paper source, curated audio)
lives neatly under a single ``src/`` folder, laid out exactly the way the
notebook's path detection expects (``src/separation`` for imports,
``src/experiments`` for the trained dictionaries and network, ``src/data`` for
the dataset it downloads on first run).

The 140 MB MUSDB sample is excluded: the notebook downloads it on first run, and
the saved notebook already contains every figure, table, and playable clip.

    python scripts/package_submission.py
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB = "Khoo_Taka_FinalProject.ipynb"
STAGE = REPO / "_submission" / "Khoo_Taka_FinalProject"
ZIP = REPO / "Khoo_Taka_FinalProject.zip"

SUBMISSION_README = r"""# Unmixing the Music
## Sparse, Low-Rank, and Deep: A Compressed-Sensing Approach to Music Source Separation

**ENGS 109 - High-Dimensional Sensing and Learning - Final Project**
**Taka Khoo, Thayer School of Engineering, Dartmouth College**

---

### Start here

| File | What it is |
|---|---|
| `Khoo_Taka_FinalProject.ipynb` | **The main deliverable.** A fully-executed walkthrough: every figure, every SI-SDR table, and three to four playable before/after audio examples per method are already embedded inline. Open it to see all results without running anything. |
| `FINAL_REPORT.pdf` | The 5-page report (official ICML format; references run past p.5). |
| `FINAL_POSTER.pdf` | The A0 landscape conference poster. |
| `src/` | All source code, trained artifacts, figures, tests, paper source, and curated audio. Structure below. |

### Just want to see the results?

Open **`Khoo_Taka_FinalProject.ipynb`** in Jupyter or VS Code. It is already run
end to end, so every spectrogram, plot, table, and audio player is visible
without executing a single cell. Four demonstration songs are pushed through all
four separators, and the before/after audio is described in line so you know what
you are hearing.

### Inside `src/`

| Path | Contents |
|---|---|
| `src/separation/` | The five separators plus the shared features / io / metrics / baselines. |
| `src/scripts/` | Training, evaluation, ablation, figure, and packaging entry points. |
| `src/experiments/` | Trained dictionaries (`*.npz`), the deep-network checkpoint (`model.pth`), and the saved result tables. The notebook loads these, so it runs in minutes. |
| `src/figures/` | Every paper figure (PDF). |
| `src/audio_examples/curated/` | Curated before/after WAVs (mixture + each method). |
| `src/tests/` | Solver correctness tests (RPCA, OMP, SOMP on synthetic ground truth). |
| `src/paper/` | LaTeX source of the report and the bibliography. |
| `src/data/` | Empty on arrival; the MUSDB18 7-second sample downloads here on first run. |
| `src/requirements.txt` | Python dependencies. |

### To re-run the notebook yourself

```bash
pip install -r src/requirements.txt
jupyter notebook Khoo_Taka_FinalProject.ipynb     # then Kernel -> Restart & Run All
```

On the first run the notebook downloads the free MUSDB18 7-second sample set
(144 tracks, ~140 MB) into `src/data/`. It loads the pre-trained dictionaries and
network from `src/experiments/`, so a full pass takes a couple of minutes on a
laptop (CPU is fine; Apple MPS / CUDA is used automatically for the deep coder if
present).

### To reproduce the artifacts from scratch

```bash
cd src
PYTHONPATH=. python scripts/train_dictionaries.py                          # NMF + K-SVD dictionaries
PYTHONPATH=. python scripts/train_sparsenet.py --context 2 --n_modules 4   # deep coder
PYTHONPATH=. python scripts/run_experiments.py                            # full 25-track comparison
PYTHONPATH=. python tests/test_solvers.py                                 # solver correctness
```

### Result in one line

Four separators on one STFT front end, from a training-free low-rank-plus-sparse
split to a learned unrolled deep coder, are one family; the deep coder is the
strongest non-oracle separator (vocal SI-SDR 2.0 dB, accompaniment 8.3 dB) and the
fastest learned model on 25 held-out MUSDB18 tracks.

Full project history: https://github.com/takakhoo/ENGS109_Final_Project
"""


def copytree(src: Path, dst: Path):
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def main():
    if STAGE.parent.exists():
        shutil.rmtree(STAGE.parent)
    STAGE.mkdir(parents=True)

    # --- top level: README + report + poster + the fully-run notebook ---
    (STAGE / "README.md").write_text(SUBMISSION_README)
    shutil.copy(REPO / "paper" / "paper.pdf", STAGE / "FINAL_REPORT.pdf")
    shutil.copy(REPO / "poster" / "poster.pdf", STAGE / "FINAL_POSTER.pdf")
    shutil.copy(REPO / NB, STAGE / NB)

    # --- everything else under src/ ---
    src = STAGE / "src"
    src.mkdir()

    copytree(REPO / "src" / "separation", src / "separation")
    copytree(REPO / "scripts", src / "scripts")
    copytree(REPO / "tests", src / "tests")
    copytree(REPO / "figures", src / "figures")
    copytree(REPO / "audio_examples" / "curated", src / "audio_examples" / "curated")
    shutil.copy(REPO / "requirements.txt", src / "requirements.txt")

    # paper source (tex + bib + style)
    pdst = src / "paper"
    pdst.mkdir()
    for f in ["paper.tex", "references.bib", "icml2025.sty", "icml2025.bst",
              "fancyhdr.sty", "algorithm.sty", "algorithmic.sty"]:
        if (REPO / "paper" / f).exists():
            shutil.copy(REPO / "paper" / f, pdst / f)

    # trained artifacts the notebook loads
    edst = src / "experiments"
    edst.mkdir()
    copytree(REPO / "experiments" / "dictionaries", edst / "dictionaries")
    copytree(REPO / "experiments" / "results", edst / "results")
    (edst / "sparsenet").mkdir()
    shutil.copy(REPO / "experiments" / "sparsenet" / "model.pth", edst / "sparsenet" / "model.pth")
    shutil.copy(REPO / "experiments" / "sparsenet" / "history.npy", edst / "sparsenet" / "history.npy")

    # empty data dir with a note (notebook downloads the sample here on first run)
    ddst = src / "data"
    ddst.mkdir()
    (ddst / "README.txt").write_text(
        "The MUSDB18 7-second sample set downloads here on the notebook's first run.\n")

    # --- zip it ---
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(STAGE.parent))

    size_mb = ZIP.stat().st_size / 1e6
    n_files = sum(1 for _ in STAGE.rglob("*") if _.is_file())
    print(f"Packaged {n_files} files -> {ZIP.name}  ({size_mb:.1f} MB)")
    print(f"Staging dir: {STAGE}")


if __name__ == "__main__":
    main()
