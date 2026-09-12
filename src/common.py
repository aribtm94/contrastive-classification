"""
Utilitas bersama untuk seluruh pipeline.

Berisi: pembacaan config, seed, pemilihan device, dan yang terpenting
fungsi pembatas ukuran input classifier ke 224x224 (letterbox / resize).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

# Root project = folder di atas src/
ROOT = Path(__file__).resolve().parents[1]

LABEL_NAMES = {0: "alive", 1: "dead"}   # 0 = ayam hidup, 1 = ayam mati


# --------------------------------------------------------------------------- #
# Config & lingkungan
# --------------------------------------------------------------------------- #
def load_config(path: str | Path | None = None) -> dict:
    """Baca configs/config.yaml (atau path lain kalau diberikan)."""
    p = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Kunci semua sumber acak supaya hasil bisa diulang."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def resolve(p: str | Path) -> Path:
    """Path relatif dianggap relatif terhadap root project."""
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def imread(path: str | Path) -> np.ndarray | None:
    """cv2.imread yang tahan path non-ASCII di Windows."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite(path: str | Path, img: np.ndarray) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
    return bool(ok)


# --------------------------------------------------------------------------- #
# Pembatas ukuran input classifier -> 224 x 224
# --------------------------------------------------------------------------- #
def letterbox(img: np.ndarray, size: int = 224,
              color: tuple = (114, 114, 114)) -> np.ndarray:
    """
    Perkecil/perbesar gambar sampai muat di kotak size x size TANPA merusak
    rasio aspek, lalu sisa ruangnya dipadding abu-abu.

    Dipakai supaya ayam yang bentuknya memanjang (mis. ayam mati tergeletak)
    tidak jadi gepeng/melar seperti kalau di-resize paksa.
    """
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))

    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def to_square(img: np.ndarray, size: int = 224,
              mode: str = "letterbox") -> np.ndarray:
    """
    KETENTUAN: input classifier dibatasi size x size (default 224x224).
    Kalau crop lebih besar -> dikecilkan; lebih kecil -> dibesarkan.

    mode 'letterbox' -> jaga rasio aspek (disarankan)
    mode 'resize'    -> paksa jadi kotak (rasio aspek berubah)
    """
    if mode == "letterbox":
        return letterbox(img, size)
    interp = cv2.INTER_AREA if max(img.shape[:2]) > size else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


# --------------------------------------------------------------------------- #
# Bantuan bounding box
# --------------------------------------------------------------------------- #
def box_iou(a, b) -> float:
    """IoU dua kotak format [x1, y1, x2, y2]."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def pad_box(box, pad_ratio: float, W: int, H: int):
    """
    Lebarkan bbox sedikit supaya classifier dapat konteks di sekitar ayam
    (lantai, ayam lain), lalu dipotong agar tidak keluar dari gambar.
    """
    x1, y1, x2, y2 = box
    dw, dh = (x2 - x1) * pad_ratio, (y2 - y1) * pad_ratio
    x1, y1 = max(0, x1 - dw), max(0, y1 - dh)
    x2, y2 = min(W, x2 + dw), min(H, y2 + dh)
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def crop_box(img: np.ndarray, box, pad_ratio: float = 0.0):
    """Ambil potongan gambar dari bbox. Return None kalau kosong."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = pad_box(box, pad_ratio, W, H)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return img[y1:y2, x1:x2].copy()
