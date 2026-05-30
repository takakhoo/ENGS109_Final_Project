"""Data I/O: MUSDB18 iteration, resampling, and WAV writing.

We use the MUSDB18 7-second sample set (144 tracks, free) for development and
proof-of-concept. The same code points at full MUSDB18-HQ by changing the root.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .features import SR

REPO = Path(__file__).resolve().parents[2]
MUSDB_ROOT = REPO / "data" / "musdb18_sample"


def get_musdb(subsets="train", split=None, root: Path | None = None):
    """Return a musdb.DB handle. Lazy import so non-MUSDB code stays light."""
    import musdb

    root = root or MUSDB_ROOT
    return musdb.DB(root=str(root), subsets=subsets, split=split)


def track_to_mono(track, target_sr: int = SR):
    """Extract mixture and stems from a MUSDB track, downmixed to mono at target_sr.

    Returns a dict with keys: mix, vocals, drums, bass, other, accompaniment.
    Each value is a 1-D float32 array at target_sr.
    """
    src_sr = track.rate

    def prep(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)  # stereo -> mono
        if src_sr != target_sr:
            x = librosa.resample(x, orig_sr=src_sr, target_sr=target_sr)
        return x

    out = {"mix": prep(track.audio)}
    for name in ["vocals", "drums", "bass", "other", "accompaniment"]:
        if name in track.targets:
            out[name] = prep(track.targets[name].audio)
    return out


def track_to_stereo(track, target_sr: int = SR):
    """Same as track_to_mono but preserves L/R channels for Stage E (MMV)."""
    src_sr = track.rate

    def prep(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = np.stack([x, x], axis=1)
        if src_sr != target_sr:
            x = np.stack([librosa.resample(x[:, c], orig_sr=src_sr, target_sr=target_sr)
                          for c in range(x.shape[1])], axis=1)
        return x

    out = {"mix": prep(track.audio)}
    for name in ["vocals", "accompaniment"]:
        if name in track.targets:
            out[name] = prep(track.targets[name].audio)
    return out


def peak_norm(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return x / max(np.max(np.abs(x)), eps)


def write_wav(path: Path | str, x: np.ndarray, sr: int = SR, normalize: bool = True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), peak_norm(x) if normalize else x, sr)
    return path
