"""Intervensi gambar deterministik yang dipakai bersama oleh evaluator."""
from __future__ import annotations

import hashlib

import cv2
import numpy as np


def _seed_for(sample_id: str | int, seed: int) -> int:
    raw = f"{seed}|{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def _derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Permutasi tanpa fixed point; deterministik untuk RNG yang diberikan."""
    base = np.arange(n)
    for _ in range(1000):
        perm = rng.permutation(n)
        if np.all(perm != base):
            return perm
    # Fallback deterministik yang selalu merupakan derangement untuk n > 1.
    return np.roll(base, 1)


def acak_petak(image: np.ndarray, sample_id: str | int, seed: int = 0,
               n: int = 4, return_metadata: bool = False):
    """Acak seluruh petak n x n tanpa menyisakan strip atau petak tetap.

    Batas petak memakai linspace agar semua piksel tercakup saat tinggi/lebar
    tidak habis dibagi n. Petak sumber di-resize hanya bila bentuk sel tujuan
    berbeda satu piksel.
    """
    if n < 2:
        raise ValueError("n minimal 2")
    h, w = image.shape[:2]
    if h < n or w < n:
        raise ValueError(f"gambar {h}x{w} terlalu kecil untuk grid {n}x{n}")
    ys = np.linspace(0, h, n + 1, dtype=int)
    xs = np.linspace(0, w, n + 1, dtype=int)
    cells = [(int(ys[i]), int(ys[i + 1]), int(xs[j]), int(xs[j + 1]))
             for i in range(n) for j in range(n)]
    tiles = [image[y0:y1, x0:x1].copy() for y0, y1, x0, x1 in cells]
    rng = np.random.default_rng(_seed_for(sample_id, seed))
    permutation = _derangement(len(cells), rng)
    out = np.empty_like(image)
    resized = 0
    for target, source in enumerate(permutation.tolist()):
        y0, y1, x0, x1 = cells[target]
        th, tw = y1 - y0, x1 - x0
        tile = tiles[source]
        if tile.shape[:2] != (th, tw):
            tile = cv2.resize(tile, (tw, th), interpolation=cv2.INTER_AREA)
            resized += 1
        out[y0:y1, x0:x1] = tile
    metadata = {
        "sample_id": str(sample_id), "seed": int(seed), "grid": [n, n],
        "permutation": permutation.tolist(),
        "fixed_points": int(np.sum(permutation == np.arange(len(cells)))),
        "resized_tiles": resized, "coverage_pixels": int(h * w),
        "total_pixels": int(h * w),
    }
    return (out, metadata) if return_metadata else out
