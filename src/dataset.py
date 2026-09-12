"""
Dataset & augmentasi untuk classifier.

Semua gambar yang keluar dari sini SUDAH dibatasi 224x224 (ketentuan).
Crop di disk memang sudah 224x224, tapi to_square() tetap dipanggil ulang
sebagai jaring pengaman kalau ada crop dari sumber lain.

Dua mode pengambilan data:
    ClassificationDataset -> 1 gambar per sampel  (untuk CE & linear probe)
    TwoViewDataset        -> 2 versi augmentasi dari gambar yang SAMA
                             (untuk metode kontrastif)

Kenapa dua view?
    Inti contrastive learning adalah: dua versi augmentasi dari gambar yang
    sama harus punya embedding berdekatan, sedangkan gambar berbeda harus
    berjauhan. Jadi tiap sampel wajib menghasilkan sepasang view.
"""
from __future__ import annotations

import csv
import random
import zlib

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common import resolve, to_square

# Statistik normalisasi ImageNet (backbone-nya pretrained di ImageNet)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Augmentasi: registry op yang digerakkan config
# --------------------------------------------------------------------------- #
# Tiap op berupa fungsi op_xxx(img, rng, **params) -> (img, catatan).
# "catatan" adalah string pendek yang dipakai sebagai caption di gambar
# laporan (src/aug_report.py); kembalikan None kalau op tidak jadi menyala.
#
# ATURAN YANG TIDAK BOLEH DILANGGAR: TIDAK ADA ROTASI.
# Ketiga paper sumber (SimCLR, SupCon, Zhang) memakai rotasi/shear, tapi di
# dataset ini orientasi justru SINYAL UTAMA yang membedakan ayam mati
# (tergeletak horizontal) dan hidup (berdiri tegak). Memutar gambar akan
# menghapus sinyal itu. Karena itu op rotate/shear sengaja tidak disediakan,
# dan kolam RandAugment pun disaring (lihat _RANDAUG_POOL).
# --------------------------------------------------------------------------- #

def _odd(k: int) -> int:
    """Kernel Gaussian harus ganjil dan minimal 3."""
    k = max(3, int(k))
    return k if k % 2 == 1 else k + 1


def op_hflip(img, rng, p=0.5):
    if rng.random() >= p:
        return img, None
    return cv2.flip(img, 1), "hflip"


def op_rrc(img, rng, p=1.0, scale=(0.20, 1.00), ratio=(0.75, 1.3333)):
    """
    Random resized crop. Versi lama hanya isotropik (nh, nw = h*s, w*s);
    di sini rasio aspek ikut diacak seperti di paper.
    """
    if rng.random() >= p:
        return img, None
    h, w = img.shape[:2]
    area = h * w
    for _ in range(10):                      # coba 10x cari kotak yang muat
        a = area * rng.uniform(scale[0], scale[1])
        logr = rng.uniform(float(np.log(ratio[0])), float(np.log(ratio[1])))
        r = float(np.exp(logr))
        nw, nh = int(round(np.sqrt(a * r))), int(round(np.sqrt(a / r)))
        if 0 < nw <= w and 0 < nh <= h:
            x0 = rng.randint(0, w - nw)
            y0 = rng.randint(0, h - nh)
            out = cv2.resize(img[y0:y0 + nh, x0:x0 + nw], (w, h),
                             interpolation=cv2.INTER_LINEAR)
            return out, "rrc%.2f" % (a / area)
    return img, None                          # gagal 10x -> biarkan apa adanya


def op_rrc_iso(img, rng, p=0.7, scale=(0.70, 1.00)):
    """Crop isotropik persis seperti pipeline lama (dipakai policy 'legacy')."""
    if rng.random() >= p:
        return img, None
    h, w = img.shape[:2]
    s = rng.uniform(scale[0], scale[1])
    nh, nw = int(h * s), int(w * s)
    y0 = rng.randint(0, max(0, h - nh))
    x0 = rng.randint(0, max(0, w - nw))
    out = cv2.resize(img[y0:y0 + nh, x0:x0 + nw], (w, h),
                     interpolation=cv2.INTER_LINEAR)
    return out, "crop%.2f" % s


def op_bright_contrast(img, rng, p=0.8, alpha=(0.7, 1.3), beta=(-30, 30)):
    if rng.random() >= p:
        return img, None
    a = rng.uniform(alpha[0], alpha[1])
    b = rng.uniform(beta[0], beta[1])
    return cv2.convertScaleAbs(img, alpha=a, beta=b), "brightness"


def op_hsv(img, rng, p=0.5, sat=(0.6, 1.4), hue=(-8, 8)):
    if rng.random() >= p:
        return img, None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(sat[0], sat[1]), 0, 255)
    hsv[..., 0] = (hsv[..., 0] + rng.randint(int(hue[0]), int(hue[1]))) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), "hsv"


def op_color_jitter(img, rng, p=0.8, strength=1.0):
    """
    Color distortion ala SimCLR (Chen et al. 2020, Appendix A).
    Semantik torchvision ColorJitter(0.8s, 0.8s, 0.8s, 0.2s):
      brightness/contrast/saturation  faktor ~ U[1-0.8s, 1+0.8s]
      hue                             geseran ~ U[-0.2s, 0.2s] putaran penuh
    Keempatnya diterapkan dalam URUTAN ACAK, sesuai implementasi paper.
    """
    if rng.random() >= p:
        return img, None
    s = float(strength)
    lo, hi = max(0.0, 1.0 - 0.8 * s), 1.0 + 0.8 * s
    fb, fc, fs = (rng.uniform(lo, hi) for _ in range(3))
    # hue torchvision dalam putaran [-0.5, 0.5]; OpenCV memakai 0..179
    fh = rng.uniform(-0.2 * s, 0.2 * s) * 180.0

    def _brightness(x):
        return cv2.convertScaleAbs(x, alpha=fb, beta=0)

    def _contrast(x):
        m = float(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).mean())
        return cv2.convertScaleAbs(x, alpha=fc, beta=(1.0 - fc) * m)

    def _saturation(x):
        hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * fs, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _hue(x):
        hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + int(round(fh))) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    fns = [_brightness, _contrast, _saturation, _hue]
    order = [0, 1, 2, 3]
    rng.shuffle(order)
    out = img
    for i in order:
        out = fns[i](out)
    return out, "color%.1f" % s


def op_gray(img, rng, p=0.2):
    if rng.random() >= p:
        return img, None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "gray"


def op_blur(img, rng, p=0.5, kernel=None, kernel_frac=None, sigma=None):
    """
    Gaussian blur.
      kernel      : daftar kernel px yang dipilih acak  -> perilaku lama (3/5)
      kernel_frac : pecahan dari sisi gambar            -> SimCLR (0.10 -> 23px)
      sigma       : (lo, hi); kalau None, OpenCV menghitung sendiri dari kernel
    """
    if rng.random() >= p:
        return img, None
    if kernel_frac is not None:
        k = _odd(int(float(kernel_frac) * min(img.shape[:2])))
    elif kernel:
        k = _odd(rng.choice(list(kernel)))
    else:
        k = 3
    sx = rng.uniform(sigma[0], sigma[1]) if sigma else 0
    return cv2.GaussianBlur(img, (k, k), sx), "blur%d" % k


# Kolam op RandAugment. Rotate/ShearX/ShearY SENGAJA TIDAK ADA -> lihat
# catatan di atas. Sisanya 11 op (identitas + translasi + fotometrik).
_RANDAUG_POOL = ["Identity", "TranslateX", "TranslateY", "Brightness",
                 "Color", "Contrast", "Sharpness", "Posterize", "Solarize",
                 "AutoContrast", "Equalize"]


def _randaug_apply(pil, op: str, magnitude: int, rng):
    """
    Satu op RandAugment pada gambar PIL. Magnitude 0..30 (M=9 default).

    Op "signed" (enhance + translate) menarik tanda acak, persis seperti
    RandAugment asli dan torchvision (`magnitude *= -1` separuh waktu).
    Tanpa itu kecerahan/kontras cuma pernah NAIK - itu pergeseran sistematis,
    bukan augmentasi.
    """
    import torchvision.transforms.functional as TF
    v = magnitude / 30.0
    sign = 1.0 if rng.random() < 0.5 else -1.0
    if op == "Identity":
        return pil
    if op == "TranslateX":
        return TF.affine(pil, angle=0, translate=[int(sign * v * 60), 0],
                         scale=1.0, shear=[0.0, 0.0])
    if op == "TranslateY":
        return TF.affine(pil, angle=0, translate=[0, int(sign * v * 60)],
                         scale=1.0, shear=[0.0, 0.0])
    if op == "Brightness":
        return TF.adjust_brightness(pil, 1.0 + sign * v * 0.9)
    if op == "Color":
        return TF.adjust_saturation(pil, 1.0 + sign * v * 0.9)
    if op == "Contrast":
        return TF.adjust_contrast(pil, 1.0 + sign * v * 0.9)
    if op == "Sharpness":
        return TF.adjust_sharpness(pil, 1.0 + sign * v * 0.9)
    if op == "Posterize":                       # tak bertanda
        return TF.posterize(pil, 8 - int(v * 4))
    if op == "Solarize":                        # tak bertanda
        return TF.solarize(pil, 255 - int(v * 255))
    if op == "AutoContrast":
        return TF.autocontrast(pil)
    if op == "Equalize":
        return TF.equalize(pil)
    raise ValueError("op RandAugment tidak dikenal: " + str(op))


def op_randaug(img, rng, p=1.0, n=2, magnitude=9, stacked=False):
    """
    RandAugment (Cubuk dkk.) seperti yang dipakai Khosla dkk. NeurIPS 2020.
    stacked=True -> "Stacked RandAugment": rangkaian n-op dijalankan DUA KALI
    berurutan. Ini varian terbaik untuk ResNet dalam menurut paper SupCon.

    RNG-nya memakai random.Random milik kita sendiri (bukan torch.manual_seed)
    supaya determinisme per-sampel tetap terjaga walau num_workers > 0.
    """
    if rng.random() >= p:
        return img, None
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    picked = []
    for _ in range(2 if stacked else 1):
        for _ in range(int(n)):
            op = rng.choice(_RANDAUG_POOL)
            pil = _randaug_apply(pil, op, int(magnitude), rng)
            picked.append(op)
    out = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return out, "RA:" + "+".join(o[:4] for o in picked)


OPS = {
    "hflip": op_hflip,
    "rrc": op_rrc,
    "rrc_iso": op_rrc_iso,
    "bright_contrast": op_bright_contrast,
    "hsv": op_hsv,
    "color_jitter": op_color_jitter,
    "gray": op_gray,
    "blur": op_blur,
    "randaug": op_randaug,
}


def _op_rng(base_seed: int, op_name: str, slot: int) -> random.Random:
    """
    RNG terpisah untuk tiap op.

    Kenapa bukan satu aliran RNG saja? Karena dengan aliran bersama,
    menambah/menghapus satu op menggeser SEMUA undian sesudahnya. Akibatnya
    "seed sama, kebijakan beda" menghasilkan kotak crop yang sama sekali
    lain, sehingga gambar perbandingan antar-kebijakan membandingkan
    potongan ayam yang tidak berhubungan. Dengan RNG turunan, op 'rrc'
    menarik kotak yang sama di setiap kebijakan yang memuatnya - yang
    terlihat berbeda hanyalah op yang memang berbeda.
    """
    h = zlib.crc32(op_name.encode() + b"#" + str(slot).encode())
    return random.Random((base_seed ^ (h * 2654435761)) & 0xFFFFFFFF)


def _legacy_augment(img: np.ndarray, rng: random.Random,
                    strong: bool) -> np.ndarray:
    """
    Pipeline augmentasi VERSI LAMA, dipertahankan apa adanya.

    Ini kontrak kompatibilitas: hasil tiga run yang sudah ada di
    outputs/runs/ harus tetap bisa direproduksi bit-per-bit. Jangan diubah.
    """
    out = img.copy()
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    if strong:
        if rng.random() < 0.7:                      # random resized crop
            h, w = out.shape[:2]
            s = rng.uniform(0.70, 1.0)
            nh, nw = int(h * s), int(w * s)
            y0 = rng.randint(0, max(0, h - nh))
            x0 = rng.randint(0, max(0, w - nw))
            out = cv2.resize(out[y0:y0 + nh, x0:x0 + nw], (w, h),
                             interpolation=cv2.INTER_LINEAR)
        if rng.random() < 0.8:                      # brightness/contrast
            out = cv2.convertScaleAbs(out, alpha=rng.uniform(0.7, 1.3),
                                      beta=rng.uniform(-30, 30))
        if rng.random() < 0.5:                      # HSV sat+hue
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.6, 1.4), 0, 255)
            hsv[..., 0] = (hsv[..., 0] + rng.randint(-8, 8)) % 180
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if rng.random() < 0.2:                      # grayscale
            g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        if rng.random() < 0.3:                      # blur
            k = rng.choice([3, 5])
            out = cv2.GaussianBlur(out, (k, k), 0)
    return out


def augment(img: np.ndarray, rng: random.Random,
            policy: dict | None = None, *,
            strong: bool = True, seed: int | None = None,
            return_params: bool = False):
    """
    Augmentasi satu gambar.

      policy=None  -> jalankan pipeline LAMA persis apa adanya (memakai `rng`,
                      dikendalikan `strong`). Ini jalur kompatibilitas.
      policy=dict  -> jalankan op sesuai daftar di config, berurutan. `strong`
                      diabaikan. `seed` dipakai menurunkan RNG per-op; kalau
                      None, diambil dari `rng` supaya tetap deterministik.

    return_params=True -> kembalikan (img, {"ops": [...], "level": i}) untuk
    caption di gambar laporan.
    """
    if policy is None:
        out = _legacy_augment(img, rng, strong)
        return (out, {"ops": [], "level": None}) if return_params else out

    if seed is None:
        seed = rng.getrandbits(32)

    ops = policy.get("ops", [])

    # Zhang & Ma (CVPR 2022): komposisi "add-one" berjenjang. Tiap sampel
    # menarik tingkat i ~ U{1..4} lalu HANYA menjalankan op dengan level <= i.
    # Efeknya sebagian pasangan view cuma beda crop+warna (invariansi lemah)
    # dan sebagian beda penuh - meniru "expanded views" paper dalam 2 view.
    level = None
    if policy.get("level_sampling"):
        levels = sorted({int(o.get("level", 0)) for o in ops
                         if int(o.get("level", 0)) > 0})
        if levels:
            level = _op_rng(seed, "__level__", 0).choice(levels)

    out, fired = img, []
    for slot, spec in enumerate(ops):
        spec = dict(spec)
        name = spec.pop("op")
        lv = int(spec.pop("level", 0))
        if level is not None and lv > level:
            continue
        if name not in OPS:
            raise ValueError(
                "op augmentasi '%s' tidak dikenal. Yang tersedia: %s"
                % (name, sorted(OPS)))
        spec = {k: (tuple(v) if isinstance(v, list) else v)
                for k, v in spec.items()}
        out, tag = OPS[name](out, _op_rng(seed, name, slot), **spec)
        if tag:
            fired.append(tag)

    if return_params:
        return out, {"ops": fired, "level": level}
    return out


def resolve_policy(cfg: dict, method: str, stage: str) -> dict | None:
    """
    Ambil kebijakan augmentasi untuk (metode, tahap) dari config.
    stage: 'contrastive' | 'head'.  None = tanpa augmentasi sama sekali.

    Sengaja MELEMPAR ERROR untuk nama kebijakan yang tidak dikenal. Kalau
    salah ketik diam-diam jatuh ke kebijakan lain, satu sweep penuh bisa
    terbuang tanpa jejak apa pun di log.
    """
    acfg = cfg.get("augmentation")
    if not acfg:
        return None

    by = acfg.get("by_method", {}).get(method, {})
    name = by.get(stage)

    # Probe mengukur kualitas ENCODER. Kalau augmentasi tahap probe ikut
    # berbeda per metode, angkanya mencampur dua efek sekaligus; jadi untuk
    # metode kontrastif tahap probe dikunci ke 'minimal'.
    if stage == "head" and acfg.get("lock_probe_policy", True) and method != "ce":
        name = "minimal"

    if name is None:
        return None
    return policy_by_name(cfg, name)


def policy_by_name(cfg: dict, name: str | None) -> dict | None:
    """Ambil satu kebijakan langsung lewat namanya (dipakai --aug & laporan)."""
    if name in (None, "none", "null"):
        return None
    pol = cfg.get("augmentation", {}).get("policies", {}).get(name)
    if pol is None:
        raise ValueError(
            "kebijakan augmentasi '%s' tidak ada di config. Tersedia: %s"
            % (name, sorted(cfg.get("augmentation", {}).get("policies", {}))))
    return dict(pol, name=name)

def to_tensor(img: np.ndarray, size: int, mode: str) -> torch.Tensor:
    """BGR uint8 -> tensor CHW ternormalisasi, dipastikan size x size."""
    if img.shape[0] != size or img.shape[1] != size:
        img = to_square(img, size, mode)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())


# --------------------------------------------------------------------------- #
# Pembacaan manifest
# --------------------------------------------------------------------------- #
def read_manifest(cfg: dict, split: str | None = None) -> list[dict]:
    """Baca manifest.csv hasil build_crops.py, boleh disaring per split."""
    path = resolve(cfg["crops"]["manifest"])
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest belum ada: {path}\nJalankan dulu: python src/build_crops.py")

    root = resolve(cfg["crops"]["out_dir"])
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if split and r["split"] != split:
                continue
            r["label"] = int(r["label"])
            r["abspath"] = str(root / r["path"])
            rows.append(r)
    return rows


class _Base(Dataset):
    def __init__(self, rows: list[dict], cfg: dict, seed: int = 42,
                 policy: dict | None = None):
        if not rows:
            raise ValueError("Dataset kosong.")
        self.rows = rows
        self.size = int(cfg["classifier"]["image_size"])
        self.mode = cfg["classifier"]["resize_mode"]
        self.seed = seed
        # policy=None -> pipeline augmentasi lama (jalur kompatibilitas)
        self.policy = policy

    def __len__(self) -> int:
        return len(self.rows)

    def _read(self, i: int) -> np.ndarray:
        p = self.rows[i]["abspath"]
        buf = np.fromfile(p, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Gagal membaca crop: {p}")
        return img

    def _seed(self, i: int) -> int:
        # Benih per-item. _rng() dan _op_rng() berbagi rumus yang sama ini.
        return (self.seed * 1_000_003 + i) & 0xFFFFFFFF

    def _rng(self, i: int) -> random.Random:
        # RNG per-item supaya hasil tetap bisa diulang walau num_workers > 0
        return random.Random(self._seed(i))


class ClassificationDataset(_Base):
    """Satu gambar + label. Dipakai untuk CE, linear probe, dan evaluasi."""

    def __init__(self, rows, cfg, train: bool = False, seed: int = 42,
                 policy: dict | None = None):
        super().__init__(rows, cfg, seed, policy)
        self.train = train
        self.epoch = 0

    def set_epoch(self, e: int) -> None:
        """Ganti benih tiap epoch supaya augmentasinya tidak berulang sama."""
        self.epoch = e

    def __getitem__(self, i):
        img = self._read(i)
        if self.train:
            j = i + self.epoch * 7919
            # strong=False hanya berlaku di jalur lama (policy=None);
            # kalau ada policy, isinya yang menentukan.
            img = augment(img, self._rng(j), self.policy,
                          strong=False, seed=self._seed(j))
        x = to_tensor(img, self.size, self.mode)
        return x, self.rows[i]["label"], i


class TwoViewDataset(_Base):
    """
    Dua augmentasi berbeda dari gambar yang sama -> untuk contrastive learning.

    Mengembalikan (view1, view2, label, index). Label ikut dibawa karena
    metode SupCon membutuhkannya; metode SimCLR mengabaikannya.
    """

    def __init__(self, rows, cfg, seed: int = 42,
                 policy: dict | None = None):
        super().__init__(rows, cfg, seed, policy)
        self.epoch = 0

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __getitem__(self, i):
        img = self._read(i)
        j1 = i + self.epoch * 7919
        j2 = j1 + 104_729          # view kedua: benih digeser jauh
        v1 = augment(img, self._rng(j1), self.policy,
                     strong=True, seed=self._seed(j1))
        v2 = augment(img, self._rng(j2), self.policy,
                     strong=True, seed=self._seed(j2))
        return (to_tensor(v1, self.size, self.mode),
                to_tensor(v2, self.size, self.mode),
                self.rows[i]["label"], i)


def class_weights(rows: list[dict]) -> torch.Tensor:
    """
    Bobot kelas untuk menangani ketimpangan jumlah (mati jauh lebih banyak
    daripada hidup). Bobot = N / (jumlah_kelas * n_kelas_itu).
    """
    n0 = sum(1 for r in rows if r["label"] == 0)
    n1 = sum(1 for r in rows if r["label"] == 1)
    n = n0 + n1
    w0 = n / (2 * n0) if n0 else 1.0
    w1 = n / (2 * n1) if n1 else 1.0
    return torch.tensor([w0, w1], dtype=torch.float32)
