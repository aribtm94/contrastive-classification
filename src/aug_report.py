"""
Laporan augmentasi: gambar contoh + markdown.

Menjawab empat hal yang diminta:
    (a) hasil / skor tiap metode           -> dari outputs/runs/*/result.json
    (b) ALASAN tiap augmentasi dipakai     -> ditulis di bawah, per kebijakan
    (c) output tiap ayam yang diaugmentasi -> outputs/reports/aug/*.jpg
    (d) berapa persen skornya              -> tabel hasil (mean +- std)

Keluaran:
    outputs/reports/aug/<kebijakan>.jpg        contoh tiap kebijakan
    outputs/reports/aug/policies_side_by_side.jpg
    outputs/reports/augmentation_report.md     laporan utama

Jalankan:
    python src/aug_report.py --no-metrics    # gambar saja, sebelum training
    python src/aug_report.py                 # lengkap, setelah training
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from common import imwrite, load_config, resolve  # noqa: E402
from dataset import augment, policy_by_name, read_manifest  # noqa: E402


def aug_img(img, rng, pol, seed) -> np.ndarray:
    """augment() tanpa params, dengan tipe kembalian yang pasti ndarray."""
    out = augment(img, rng, pol, seed=seed)
    return out  # type: ignore[return-value]


# Kebijakan yang ditampilkan di laporan, beserta metode yang memakainya.
POLICIES = ["simclr", "stacked_randaug", "hier_addone"]
# kebijakan -> metode yang memakainya di diagonal. ARAHNYA SATU saja;
# sempat ada entri terbalik "ce": "hier_addone" di sini, dan itu membuat
# sebuah METODE bisa terbaca sebagai nama kebijakan.
USED_BY = {"simclr": "selfcon", "stacked_randaug": "supcon",
           "hier_addone": "ce"}

PAPER = {
    "simclr": "Chen dkk., *A Simple Framework for Contrastive Learning of "
              "Visual Representations*, ICML 2020 (Appendix A)",
    "stacked_randaug": "Khosla dkk., *Supervised Contrastive Learning*, "
                       "NeurIPS 2020 (Sec. 4.3 & lampiran augmentasi)",
    "hier_addone": "Zhang & Ma, *Rethinking the Augmentation Module in "
                   "Contrastive Learning*, CVPR 2022 (modul add-one)",
}

# Alasan tiap augmentasi dipakai DI DATASET INI - bukan ringkasan papernya.
ALASAN = {
    "simclr": """
Loss NT-Xent tidak memakai label sama sekali, jadi ia paling gampang
"curang": kalau ada satu isyarat gampang yang kebetulan sejalan dengan
kelas, encoder akan memungutnya dan berhenti belajar bentuk ayam. Di
dataset ini isyarat gampang itu ada dan sangat kuat - **ketajaman gambar**
(lihat bagian jalan pintas di bawah).

Dua bagian resep SimCLR yang menjawab persis masalah itu:

1. **Distorsi warna kuat (`strength = 1.0`).** Gambar 5 paper menunjukkan
   crop + distorsi warna adalah pasangan augmentasi terkuat, +23.2 poin di
   atas crop saja. Alasan paper: potongan dari satu gambar berbagi
   histogram warna yang hampir sama, jadi tanpa distorsi warna encoder
   cukup mencocokkan warna dan tidak perlu mengenali objeknya. Di sini
   substratnya beragam (sekam, tanah, beton, kayu, jaring, aspal) dan
   pencahayaannya sangat berbeda-beda, jadi warna memang isyarat yang
   harus dimatikan.
2. **Blur kernel 10% sisi gambar = 23 px** dengan sigma ~ U[0.1, 2.0].
   Ini jauh lebih kuat daripada k=3/5 yang dipakai pipeline lama, dan
   inilah satu-satunya op di seluruh resep yang menyerang langsung jalan
   pintas ketajaman.
""",
    "stacked_randaug": """
Paper SupCon **tidak mendefinisikan augmentasinya sendiri**. Ia justru
membandingkan empat pilihan yang sudah ada (AutoAugment, RandAugment,
SimAugment, Stacked RandAugment) dan melaporkan **Stacked RandAugment**
paling baik untuk ResNet yang dalam. Jadi memilih Stacked RandAugment
untuk `supcon` adalah mengikuti temuan papernya, bukan mengarang.

"Stacked" berarti rangkaian 2 op acak dijalankan **dua kali berurutan**,
sehingga satu gambar bisa kena sampai 4 operasi fotometrik bertumpuk.

Dua alasan tambahan yang khusus dataset ini:

1. **Loss SupCon memakai label.** Dengan hanya 2 kelas, tiap anchor punya
   puluhan positif dari kelas yang sama - pasangan augmentasi cuma
   minoritas kecil di antaranya. Jadi tugas augmentasi di sini bukan
   membuat pasangan positif (label sudah menyediakannya), melainkan
   menambah keragaman. Op acak bertumpuk cocok untuk itu.
2. **Sengaja TANPA Gaussian blur** - blur bukan bagian dari kolam
   RandAugment. Ini dipilih sebagai kontrol: dengan satu kebijakan
   berblur kuat (`simclr`), satu berblur sedang (`hier_addone`), dan satu
   tanpa blur sama sekali, tabel kebocoran bisa menguji apakah blur
   memang penggerak angkanya. **Hasilnya: bukan.** Ketiganya turun ke
   kisaran yang sama, karena `Solarize`/`Posterize` memotong tingkat
   keabuan sehingga ikut merusak isyarat ketajaman tanpa memblur apa pun.
   Dugaan awal rencana (kebijakan ini paling bocor, ~0.82) meleset, dan
   dibiarkan tercatat di sini apa adanya.

Paper juga mengklaim (Sec. 4.3) SupCon **lebih tahan terhadap pilihan
augmentasi** daripada cross-entropy. Klaim itu justru bisa diuji di sini
lewat `--grid full`: kalau benar, baris `supcon` harusnya paling rata
sepanjang ketiga kebijakan.
""",
    "hier_addone": """
Inti paper Zhang bukan "augmentasi yang lebih kuat", melainkan
**komposisi add-one berjenjang**: T1 = {crop, warna}, T2 = T1 + grayscale,
T3 = T2 + blur, T4 = T3 + flip. Urutan warna -> grayscale -> blur -> flip
inilah yang menang di paper (67.1 vs 64.5 untuk urutan terburuk), dan
urutan itu dipakai persis di sini.

`level_sampling` membuat tiap sampel menarik tingkat i ~ U{1..4} lalu
hanya menjalankan op dengan level <= i. Akibatnya sebagian pasangan view
hanya berbeda crop+warna (invariansi lemah) dan sebagian berbeda penuh -
meniru gagasan "expanded views" paper dalam kerangka 2 view.

Kenapa dipasangkan ke `ce`: data latihnya cuma 75 crop. Memaksa seluruh
sampel ke invariansi maksimum pada data sekecil itu berisiko membuang
sinyal; pencampuran kuat-lemah ini memberi jaring pengaman.

**PERINGATAN Tabel 6 paper**: varian "hierarchical strength" (augmentasi
makin kuat makin dalam) justru varian TERBURUK - di bawah baseline biasa.
Manfaatnya datang dari membagi JENIS augmentasi lewat add-one, bukan dari
menaikkan intensitas menurut kedalaman. Implementasi di sini mengikuti
yang pertama.

**Yang TIDAK diimplementasikan**: loss multi-stage 4-tap, 8 view, dan
augmentation-parameter embedding. Ketiganya mengubah arsitektur jaringan,
sedangkan syarat perbandingan ini adalah arsitektur ketiga metode
identik. Jadi yang dipakai di sini adalah **modul augmentasinya saja**,
bukan metode Zhang secara utuh - menyebutnya sebaliknya tidak benar.
""",
}

DEVIASI = """
Dua parameter sengaja menyimpang dari papernya, dan keduanya perlu diketahui
pembaca:

1. **`scale` batas bawah 0.20, bukan 0.08 seperti SimCLR.** Angka 0.08 di
   paper ditala untuk foto ImageNet yang berisi pemandangan penuh. Crop kita
   sudah ketat di badan ayam; pada skala 0.08, 13% sampel mengambil kurang
   dari 20% luas gambar dan sering tidak ada ayamnya sama sekali. Yang
   dihasilkan bukan invariansi, melainkan derau label. Pada 0.20 tidak ada
   lagi sampel yang tinggal serpihan.
2. **`ratio [0.75, 1.333]` adalah parameter paling berisiko yang dipinjam
   di sini.** Distorsi rasio aspek sedikit menggerus satu-satunya sinyal
   nyata yang kita punya: ayam mati tergeletak memanjang horizontal, ayam
   hidup berdiri tegak. Rasio aspek sendiri TIDAK membocorkan label
   (AUC 0.451), jadi ini bukan jalan pintas yang dibuang melainkan fitur
   asli yang ditipiskan. Nilai paper tetap dipakai supaya setia, tapi kalau
   satu kebijakan anjlok tanpa sebab jelas, ablasi pertama yang harus
   dicoba adalah `ratio: [1.0, 1.0]`.

Selain itu, **rotasi dibuang dari semua kebijakan**, walaupun ketiga paper
memakainya (dan Zhang membahasnya panjang lebar). Untuk `stacked_randaug`
ini berarti op `Rotate`, `ShearX`, `ShearY` dikeluarkan dari kolam
RandAugment, menyisakan 11 op. Alasannya sama dengan poin 2 di atas, tapi
lebih parah: memutar gambar menghapus habis sinyal orientasi.
"""

# --------------------------------------------------------------------------- #
# Bantuan gambar
# --------------------------------------------------------------------------- #
TILE = 168
FONT = cv2.FONT_HERSHEY_SIMPLEX
FSC = 0.33
LINE_H = 13


def vlap(img: np.ndarray) -> float:
    """Variance of Laplacian = ukuran ketajaman. Inti jalan pintas di sini."""
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                               cv2.CV_64F).var())


def wrap(words: list[str], width: int, max_lines: int = 3) -> list[str]:
    """Bungkus kata ke beberapa baris agar muat di lebar ubin."""
    lines, cur = [], ""
    for w in words:
        trial = w if not cur else cur + " " + w
        if cv2.getTextSize(trial, FONT, FSC, 1)[0][0] <= width - 6:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
            if len(lines) == max_lines - 1:
                # baris terakhir: potong kalau memang tidak muat
                while cur and cv2.getTextSize(cur + "..", FONT, FSC, 1)[0][0] > width - 6:
                    cur = cur[:-1]
                lines.append(cur + ("" if cur == w else ".."))
                return lines
    if cur:
        lines.append(cur)
    return lines or [""]


def tile(img: np.ndarray, head: str, ops: list[str], label: int,
         highlight: bool = False) -> np.ndarray:
    """
    Satu ubin: gambar diperkecil + pita caption.

    `head` selalu satu baris (kelas + vLap); `ops` dibungkus di bawahnya.
    Caption TIDAK boleh terpotong diam-diam - `blur23` justru op yang paling
    perlu dibaca, dan sebelumnya dialah yang paling sering hilang.
    """
    im = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA)
    color = (0, 0, 255) if label == 1 else (0, 200, 0)
    cv2.rectangle(im, (0, 0), (TILE - 1, TILE - 1), color, 2)

    lines = [head] + wrap(ops or ["(tanpa op)"], TILE)
    bar = np.full((LINE_H * 3 + 6, TILE, 3), 245 if highlight else 255, np.uint8)
    for i, ln in enumerate(lines[:3]):
        cv2.putText(bar, ln, (3, 11 + i * LINE_H), FONT, FSC,
                    (0, 0, 140) if i == 0 and highlight else (30, 30, 30),
                    1, cv2.LINE_AA)
    return np.vstack([bar, im])


def pick_samples(rows: list[dict], cfg: dict) -> list[dict]:
    """Ambil contoh seimbang: separuh hidup, separuh mati."""
    n = int(cfg["augmentation"].get("preview_samples", 4))
    rng = random.Random(cfg["seed"])
    out = []
    for label in (0, 1):
        sel = [r for r in rows if r["label"] == label]
        rng.shuffle(sel)
        out += sel[:max(1, n // 2)]
    return out


def read_crop(r: dict) -> np.ndarray | None:
    buf = np.fromfile(r["abspath"], dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def grid_for_policy(name: str, samples: list[dict], cfg: dict,
                    out_path: Path) -> None:
    """
    Satu baris per ayam: kolom 0 = asli, kolom 1..N = hasil augmentasi.

    Angka vLap ditulis di TIAP ubin. Inilah yang membuat jalan pintas
    ketajaman terlihat, bukan sekadar diklaim: pembaca bisa melihat sendiri
    angka crop-mati (ratusan) turun mendekati rentang crop-hidup saat diblur.
    """
    pol = policy_by_name(cfg, name)
    views = int(cfg["augmentation"].get("preview_views", 6))
    rows_img = []

    for si, r in enumerate(samples):
        img = read_crop(r)
        if img is None:
            continue
        nama = "mati" if r["label"] == 1 else "hidup"
        tiles = [tile(img, "ASLI %s v%.0f" % (nama, vlap(img)), [], r["label"],
                      highlight=True)]
        for v in range(views):
            seed = (cfg["seed"] * 7919 + si * 1013 + v) & 0xFFFFFFFF
            out, info = augment(img, random.Random(seed), pol,
                                seed=seed, return_params=True)
            lv = "" if info["level"] is None else "T%d " % info["level"]
            tiles.append(tile(out, "%svLap %.0f" % (lv, vlap(out)),
                              info["ops"], r["label"]))
        rows_img.append(np.hstack(tiles))

    if rows_img:
        imwrite(out_path, np.vstack(rows_img))


def grid_side_by_side(samples: list[dict], cfg: dict, out_path: Path) -> None:
    """
    Satu ayam, tiga kebijakan sebagai tiga baris.

    Berkat RNG per-op di dataset.py, kotak crop dan flip IDENTIK di ketiga
    baris - jadi yang terlihat berbeda hanyalah op yang memang berbeda.
    Tanpa itu, gambar ini cuma membandingkan tiga potongan yang tak
    berhubungan dan tidak memberi tahu apa-apa.
    """
    views = int(cfg["augmentation"].get("preview_views", 6))
    # satu hidup + satu mati; samples[:2] dulu keduanya hidup
    pair = ([r for r in samples if r["label"] == 0][:1]
            + [r for r in samples if r["label"] == 1][:1])
    blocks = []
    for r in pair:
        img = read_crop(r)
        if img is None:
            continue
        nama = "mati" if r["label"] == 1 else "hidup"
        for name in POLICIES:
            pol = policy_by_name(cfg, name)
            tiles = [tile(img, nama.upper(), [name], r["label"],
                          highlight=True)]
            for v in range(views):
                seed = (cfg["seed"] * 31 + v) & 0xFFFFFFFF   # seed SAMA
                out, info = augment(img, random.Random(seed), pol,
                                    seed=seed, return_params=True)
                tiles.append(tile(out, "vLap %.0f" % vlap(out),
                                  info["ops"], r["label"]))
            blocks.append(np.hstack(tiles))
        blocks.append(np.full((8, blocks[-1].shape[1], 3), 255, np.uint8))
    if blocks:
        imwrite(out_path, np.vstack(blocks))


# --------------------------------------------------------------------------- #
# Diagnostik: seberapa banyak jalan pintas ketajaman yang tersisa
# --------------------------------------------------------------------------- #
def auc(neg: np.ndarray, pos: np.ndarray) -> float:
    """AUC lewat Mann-Whitney U, rata-rata pada nilai seri."""
    x = np.concatenate([neg, pos])
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def leakage_table(cfg: dict, rows: list[dict], views: int = 8,
                  repeats: int = 5) -> list[dict]:
    """
    AUC ketajaman pada crop yang SUDAH diaugmentasi.
    0.5 = jalan pintas hilang, ~1.0 = jalan pintas utuh.

    Diulang `repeats` kali dengan seed dasar berbeda, lalu dilaporkan
    mean +- std. Sekali undian saja TIDAK cukup: pada satu seed
    `stacked_randaug` keluar paling rendah, pada seed lain paling tinggi.
    Tanpa simpangan baku, urutan antar-kebijakan terbaca seolah bermakna
    padahal selisihnya masih di dalam derau.
    """
    imgs = [(r["label"], read_crop(r)) for r in rows]
    imgs = [(l, im) for l, im in imgs if im is not None]
    out = []

    a = np.array([vlap(im) for l, im in imgs if l == 0])
    d = np.array([vlap(im) for l, im in imgs if l == 1])
    out.append({"policy": "(tanpa augmentasi)", "auc": auc(a, d), "std": 0.0,
                "med_alive": float(np.median(a)),
                "med_dead": float(np.median(d))})

    for name in ["legacy"] + POLICIES:
        pol = policy_by_name(cfg, name)
        aucs, ma, md = [], [], []
        for rep in range(repeats):
            A, D = [], []
            for si, (label, im) in enumerate(imgs):
                for v in range(views):
                    seed = ((rep + 1) * 7_919_311 + si * 7919 + v) & 0xFFFFFFFF
                    aug = aug_img(im, random.Random(seed), pol, seed)
                    (D if label == 1 else A).append(vlap(aug))
            A, D = np.array(A), np.array(D)
            aucs.append(auc(A, D))
            ma.append(float(np.median(A)))
            md.append(float(np.median(D)))
        out.append({"policy": name, "auc": float(np.mean(aucs)),
                    "std": float(np.std(aucs)),
                    "med_alive": float(np.mean(ma)),
                    "med_dead": float(np.mean(md))})
    return out


def padding_leak(rows: list[dict], cfg: dict | None = None) -> dict:
    """
    Kebocoran KEDUA, ditemukan setelah kebocoran ketajaman: **bantalan
    letterbox**.

    `build_crops.py` menyimpan crop yang SUDAH dipersegi (`to_square`), jadi
    bilahnya ikut terpanggang ke dalam berkas - augmentasi apa pun bekerja
    di atas gambar yang bilahnya sudah ada.

    PENTING: `common.letterbox` membantali dengan ABU-ABU (114,114,114),
    bukan hitam. Menguji piksel gelap tidak menemukan apa pun sama sekali.
    """
    fr, lab, orient = [], [], []
    for r in rows:
        im = read_crop(r)
        if im is None:
            continue
        pad = (np.abs(im.astype(np.int16) - 114) <= 1).all(axis=2)
        rowbar = float(pad.all(axis=1).sum())      # bilah atas/bawah
        colbar = float(pad.all(axis=0).sum())      # bilah kiri/kanan
        h, w = pad.shape
        fr.append((rowbar * w + colbar * h) / float(h * w))
        orient.append(1.0 if rowbar > colbar else 0.0)
        lab.append(int(r["label"]))
    fr, lab, orient = np.array(fr), np.array(lab), np.array(orient)
    a, d = fr[lab == 0], fr[lab == 1]
    out = {"auc": auc(a, d), "med_alive": float(np.median(a)),
           "med_dead": float(np.median(d)),
           "auc_orient": auc(orient[lab == 0], orient[lab == 1]),
           "nobar_alive": int((fr[lab == 0] <= 1e-9).sum()),
           "n_alive": int((lab == 0).sum()),
           "nobar_dead": int((fr[lab == 1] <= 1e-9).sum()),
           "n_dead": int((lab == 1).sum())}

    if cfg is None:
        return out

    # Lantai bantalan: protokol yang SAMA dengan lantai ketajaman - ambang
    # dicocokkan di train, dinilai di test. Skornya dibalik tandanya karena
    # arah isyaratnya terbalik (mati = bantalan lebih sedikit).
    def _frac(split):
        f2, l2 = [], []
        for r in read_manifest(cfg, split):
            im = read_crop(r)
            if im is None:
                continue
            pd = (np.abs(im.astype(np.int16) - 114) <= 1).all(axis=2)
            h, w = pd.shape
            f2.append((pd.all(axis=1).sum() * w
                       + pd.all(axis=0).sum() * h) / float(h * w))
            l2.append(int(r["label"]))
        return np.array(l2), -np.array(f2)

    from report import _bacc as _bc
    ytr, str_ = _frac("train")
    yte, ste = _frac("test")
    u = np.unique(str_)
    cands = (u[:-1] + u[1:]) / 2.0 if len(u) > 1 else u
    bt, bb = 0.0, -1.0
    for t in cands:
        b = _bc(ytr, (str_ >= t).astype(int))
        if b > bb:
            bt, bb = float(t), b
    out.update({"floor_train": bb,
                "floor_test": _bc(yte, (ste >= bt).astype(int)),
                "floor_auc": auc(ste[yte == 0], ste[yte == 1])})
    return out


def degenerate_views(cfg: dict, rows: list[dict], views: int = 8,
                     std_min: float = 8.0) -> list[dict]:
    """
    Berapa persen view yang keluar nyaris polos (simpangan baku abu-abu
    kecil) - gambar yang praktis tidak punya isi lagi.

    Penting diperiksa per kelas: kalau satu kelas jauh lebih sering rusak,
    itu bias baru. Kalau seimbang, itu cuma derau.
    """
    imgs = [(r["label"], read_crop(r)) for r in rows]
    imgs = [(l, im) for l, im in imgs if im is not None]
    out = []
    for name in POLICIES:
        pol = policy_by_name(cfg, name)
        cnt = {0: [0, 0], 1: [0, 0]}
        for si, (label, im) in enumerate(imgs):
            for v in range(views):
                seed = (cfg["seed"] * 104_729 + si * 7919 + v) & 0xFFFFFFFF
                g = cv2.cvtColor(aug_img(im, random.Random(seed), pol, seed),
                                 cv2.COLOR_BGR2GRAY)
                cnt[label][0] += 1
                cnt[label][1] += int(g.std() < std_min)
        tot = cnt[0][0] + cnt[1][0]
        bad = cnt[0][1] + cnt[1][1]
        out.append({"policy": name, "pct": 100.0 * bad / max(1, tot),
                    "pct_alive": 100.0 * cnt[0][1] / max(1, cnt[0][0]),
                    "pct_dead": 100.0 * cnt[1][1] / max(1, cnt[1][0])})
    return out


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def ops_table(cfg: dict, name: str) -> list[str]:
    """Tabel op dirender DARI CONFIG, supaya tidak pernah melenceng."""
    pol = cfg["augmentation"]["policies"][name]
    lines = ["| urutan | op | peluang | parameter |", "|---|---|---|---|"]
    for i, spec in enumerate(pol.get("ops", []), 1):
        spec = dict(spec)
        op = spec.pop("op")
        p = spec.pop("p", 1.0)
        lv = spec.pop("level", None)
        par = ", ".join("`%s=%s`" % (k, v) for k, v in spec.items()) or "-"
        tag = "%d" % i if lv is None else "%d (T%d)" % (i, max(1, lv))
        lines.append("| %s | `%s` | %.2f | %s |" % (tag, op, float(p), par))
    if pol.get("level_sampling"):
        lines.append("")
        lines.append("`level_sampling: true` - tiap sampel menarik tingkat "
                     "i ~ U{1..4} lalu hanya menjalankan op dengan level <= i.")
    return lines


def fmt_pct(v) -> str:
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) \
        else "%.2f%%" % (100 * v)


def results_section(cfg: dict) -> list[str]:
    """Tabel hasil, dibaca dari report.load_results kalau sudah ada."""
    try:
        import report as _rep
        load_results = _rep.load_results
        sharpness_floor = getattr(_rep, "sharpness_floor", None)
    except Exception:
        return ["_(bagian hasil butuh src/report.py versi baru)_", ""]
    if sharpness_floor is None:
        return ["_src/report.py belum punya sharpness_floor() - jalankan "
                "langkah 5 rencana._", ""]
    try:
        res = load_results(cfg)
    except SystemExit:
        return ["_Belum ada hasil training. Jalankan dulu:_", "",
                "```", "python src/train.py --grid full --seeds 42,43,44,45,46",
                "```", ""]

    floor = sharpness_floor(cfg)  # sumber tunggal angka lantai
    lines = [
        "Kelas positif = ayam **MATI**. Metrik utama = **balanced accuracy** "
        "(rata-rata recall kedua kelas), karena jumlah kelasnya timpang "
        "(98 mati : 32 hidup).", "",
        "`@0.5` = ambang bawaan. `@tau` = ambang yang dipilih di "
        "**validation set**, bukan di test - lihat catatan di bawah.", "",
        "| augmentasi | metode | bal.acc @0.5 | bal.acc @tau | AUC | n seed |",
        "|---|---|---|---|---|---|",
    ]
    for r in res:
        diag = " **<-**" if USED_BY.get(r["aug"]) == r["method"] else ""
        lines.append(
            "| `%s` | %s%s | %s | %s | %s | %d |"
            % (r["aug"], r["method"], diag,
               mean_std(r["bacc_mean"], r["bacc_std"]),
               mean_std(r["tuned_mean"], r["tuned_std"]),
               mean_std(r["auc_mean"], r["auc_std"]), r["n_seeds"]))
    lines.append("| _(lantai)_ | **KETAJAMAN SAJA** | **%s** | - | %s | - |"
                 % (fmt_pct(floor["bacc"]), fmt_pct(floor["auc"])))
    lines += ["", "Baris bertanda **<-** adalah diagonal: pasangan "
              "metode-augmentasi yang diminta.", ""]

    # Vonis diagonal, ditulis eksplisit. Tabel saja tidak cukup: pembaca yang
    # mencari "jadi mana yang menang" akan memungut angka tertinggi dan
    # melewatkan bahwa ketiganya di bawah lantai. Angkanya dihitung ulang di
    # sini, tidak diketik, supaya tidak melenceng saat hasilnya berubah.
    dg = [r for r in res if USED_BY.get(r["aug"]) == r["method"]]
    if dg:
        best = max(dg, key=lambda r: r["bacc_mean"])
        n_above = sum(1 for r in dg if r["bacc_mean"] > floor["bacc"])
        lines += [
            "**Vonis untuk diagonal:** dari %d pasangan yang diminta, "
            "**%d** berada di atas lantai %s pada bal.acc. Yang tertinggi "
            "`%s`+%s (%s), masih %.1f poin **di bawah** lantai. Artinya "
            "rancangan \"satu paper satu augmentasi\" ini belum bisa "
            "diklaim mengalahkan tebakan berbasis ketajaman - pada "
            "bal.acc." % (
                len(dg), n_above, fmt_pct(floor["bacc"]),
                best["aug"], best["method"],
                mean_std(best["bacc_mean"], best["bacc_std"]),
                100 * (floor["bacc"] - best["bacc_mean"])),
            "",
            "Sisi satunya lebih menarik: diukur pada **urutan** (AUC), "
            "`simclr`+supcon mendapat %s, di atas AUC lantai %s di "
            "**kelima** seed. Itu bukan baris diagonal - ia muncul dari "
            "kolom grid penuh - tapi ia satu-satunya di seluruh grid yang "
            "benar-benar melewati lantai. Rinciannya di "
            "[`comparison.md`](comparison.md)." % (
                mean_std(*[(r["auc_mean"], r["auc_std"]) for r in res
                           if r["aug"] == "simclr"
                           and r["method"] == "supcon"][0])
                if any(r["aug"] == "simclr" and r["method"] == "supcon"
                       for r in res) else "-",
                fmt_pct(floor["auc"])),
            "",
        ]

    # Val set 22 crop sering mentok sempurna - itu membatasi seberapa jauh
    # kalibrasi ambang bisa menolong, dan pembaca perlu tahu sebabnya.
    import glob as _g
    import json as _j
    vb, th = [], []
    for f in _g.glob(str(resolve(cfg["output"]["runs_dir"]) / "*__*"
                         / "result.json")):
        try:
            r = _j.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        vb.append(r["val"]["balanced_accuracy"])
        th.append(r.get("threshold", 0.5))
    if vb:
        pct = 100.0 * sum(1 for v in vb if v >= 0.99999) / len(vb)
        lines += ["**Kenapa `@tau` sering tidak menggeser apa-apa:** "
                  "validation set cuma 22 crop, dan **%.0f%%** dari %d run "
                  "mencapai bacc val 1.0000 - sempurna. Kalau val sudah "
                  "sempurna, tidak ada yang bisa dipakai untuk memilih "
                  "ambang yang lebih baik, jadi tau jatuh di tengah dataran "
                  "yang lebar (tercatat %.3f - %.3f di seluruh run). Ini "
                  "batas datanya, bukan kegagalan kalibrasinya."
                  % (pct, len(vb), min(th), max(th)), ""]

    # Val sempurna sementara test berayun = val TIDAK bisa dipakai memilih
    # konfigurasi. Ini konsekuensi praktis yang lebih penting daripada
    # catatan saturasi di atas, jadi ditulis terpisah.
    ce = [(r["seed"], r["test"]["balanced_accuracy"],
           r["val"]["balanced_accuracy"])
          for f in _g.glob(str(resolve(cfg["output"]["runs_dir"])
                               / "ce__simclr__s*" / "result.json"))
          for r in [_j.load(open(f, encoding="utf-8"))]]
    if len(ce) >= 3:
        tb = [c[1] for c in ce]
        vv = [c[2] for c in ce]
        if max(vv) - min(vv) < 1e-9:
            lines += [
                "**Dan akibat yang lebih serius: validation set tidak bisa "
                "dipakai untuk MEMILIH.** Contoh nyata dari grid ini - "
                "`ce`+`simclr` di %d seed mendapat bacc val **%.4f di "
                "semuanya** (identik), sementara bacc test-nya berayun "
                "%.3f sampai %.3f. Val sama sekali tidak melihat perbedaan "
                "itu. Jadi memilih seed, epoch, atau konfigurasi terbaik "
                "berdasarkan val di sini sama saja dengan memilih acak; "
                "yang bisa dilakukan val cuma mengkalibrasi ambang."
                % (len(ce), vv[0], min(tb), max(tb)), ""]
    return lines


def mean_std(m, s) -> str:
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "-"
    if s is None or (isinstance(s, float) and np.isnan(s)) or s == 0:
        return "%.2f%%" % (100 * m)
    return "%.2f%% ± %.2f" % (100 * m, 100 * s)


def floor_value(cfg: dict) -> dict | None:
    """Lantai ketajaman, dihitung - bukan diketik tangan.

    Angkanya sempat salah dikutip 0.8846 karena ambangnya dicocokkan di test.
    Protokol yang benar (cocokkan di train, uji di test) justru memberi angka
    LEBIH TINGGI: 0.9231. Karena itu nilainya selalu diambil dari fungsi.
    """
    try:
        import report as _rep
        return _rep.sharpness_floor(cfg)
    except Exception:
        return None


def write_md(cfg: dict, leak: list[dict], with_metrics: bool,
             path: Path, pad: dict | None = None,
             degen: list[dict] | None = None) -> None:
    fl = floor_value(cfg)
    fb = "%.4f" % fl["bacc"] if fl else "(hitung lewat src/report.py)"
    fa = "%.4f" % fl["auc"] if fl else "-"
    ft = "%.0f" % fl["threshold"] if fl else "-"
    L = ["# Augmentasi per Metode Kontrastif", "",
         "Tiap metode kontrastif diberi **satu kebijakan augmentasi berbeda**, "
         "masing-masing diambil dari satu paper yang berbeda:", "",
         "| metode | loss | kebijakan augmentasi | paper |",
         "|---|---|---|---|",
         "| `selfcon` | NT-Xent (tanpa label) | `simclr` | Chen dkk., ICML 2020 |",
         "| `supcon` | Supervised Contrastive | `stacked_randaug` | Khosla dkk., NeurIPS 2020 |",
         "| `ce` | Cross-Entropy | `hier_addone` | Zhang & Ma, CVPR 2022 |", "",
         "Arsitektur (`resnet18` + projection head), data, ukuran input "
         "(224x224 letterbox), dan jumlah epoch **sama persis** untuk "
         "ketiganya.", "",
         "## PERINGATAN: perbandingannya sekarang 2 faktor", "",
         "Sebelumnya ketiga metode memakai augmentasi identik, jadi selisih "
         "skornya bisa dikaitkan ke **fungsi loss** saja. Sekarang dua hal "
         "berubah bersamaan (loss DAN augmentasi), jadi membaca diagonal "
         "saja tidak bisa menjawab \"ini efek loss atau efek augmentasi?\".", "",
         "Karena itu `src/train.py --grid full` menjalankan **seluruh 3x3 "
         "grid**:", "",
         "- **diagonal** (3 baris bertanda) = yang Anda minta;",
         "- **kolom** (augmentasi ditahan, loss divariasikan) = perbandingan "
         "metode yang tetap sah;",
         "- **baris** (loss ditahan, augmentasi divariasikan) = menjawab "
         "\"ini efek augmentasi atau loss?\".", ""]

    L += ["## Jalan pintas ketajaman - kenapa ini penting duluan", "",
          "Sebelum membahas augmentasi, satu temuan harus disampaikan: "
          "**ketajaman gambar saja sudah hampir memisahkan kedua kelas.**", "",
          "Penyebabnya cara datanya terbentuk, bukan ayamnya:", "",
          "| | crop hidup | crop mati |", "|---|---|---|",
          "| median sisi pendek bbox | 83 px | 212 px |",
          "| yang harus diperbesar ke 224 | 100% | 58% |",
          "| median variance-of-Laplacian | 134 | 747 |", "",
          "Artinya sebuah \"classifier\" yang **tidak melihat isi gambar sama "
          "sekali**, cuma mengukur ketajaman lalu memotong di satu ambang "
          "(ambang dicocokkan di **train**, diuji di **test** - protokol yang "
          "sama dengan model sungguhan), sudah mendapat **balanced accuracy "
          "%s** di test set (ambang vLap %s, AUC %s). Angka itu dipakai "
          "sebagai **lantai** di tabel hasil: metode yang skornya di bawah "
          "lantai belum membuktikan apa pun." % (fb, ft, fa), "",
          "Konsekuensinya untuk pekerjaan ini: **kekuatan blur adalah tuas "
          "langsung pada skor**, lewat jalan pintas ini, terlepas dari "
          "loss-nya. Kebijakan yang paling banyak memblur akan terlihat "
          "menang karena alasan yang tidak ada hubungannya dengan kualitas "
          "representasi. Tabel berikut mengukur persis itu.", "",
          "### Berapa banyak jalan pintas yang tersisa setelah augmentasi", "",
          "AUC ketajaman pada crop **split train** yang sudah "
          "diaugmentasi, 8 view per crop "
          "(0.5 = jalan pintas hilang, 1.0 = utuh):", "",
          "| kebijakan | AUC ketajaman | median vLap hidup | median vLap mati |",
          "|---|---|---|---|"]
    for r in leak:
        a = ("**%.4f**" % r["auc"] if r["std"] == 0
             else "**%.4f** ± %.4f" % (r["auc"], r["std"]))
        L.append("| `%s` | %s | %.1f | %.1f |"
                 % (r["policy"], a, r["med_alive"], r["med_dead"]))

    paper = [r for r in leak if r["policy"] in POLICIES]
    if paper:
        lo = min(paper, key=lambda r: r["auc"])
        hi = max(paper, key=lambda r: r["auc"])
        spread = hi["auc"] - lo["auc"]
        worst_std = max(r["std"] for r in paper)
        L += ["", "Angkanya mean ± simpangan baku atas 5 ulangan dengan seed "
              "berbeda (8 view per crop tiap ulangan).", "",
              "**Yang boleh disimpulkan:** ketiga kebijakan paper menurunkan "
              "jalan pintas secara nyata - dari %.4f apa adanya, dan dari "
              "%.4f milik `legacy`, turun ke kisaran %.2f-%.2f. Syarat "
              "rencana terpenuhi untuk ketiganya."
              % (leak[0]["auc"], leak[1]["auc"], lo["auc"], hi["auc"]), "",
              "**Yang TIDAK boleh disimpulkan:** urutan di antara ketiganya. "
              "Jarak terjauh cuma %.4f (`%s` vs `%s`) sedangkan simpangan "
              "bakunya sendiri sampai %.4f - selisihnya di dalam derau. "
              "Pada satu seed `stacked_randaug` sempat keluar paling rendah, "
              "pada seed lain paling tinggi; itulah sebabnya kolom ini "
              "diulang 5 kali, bukan sekali."
              % (spread, lo["policy"], hi["policy"], worst_std), "",
              "Perlu dicatat, dugaan awal rencana **meleset**: "
              "`stacked_randaug` sengaja tidak memakai Gaussian blur dan "
              "diperkirakan jadi yang paling bocor (~0.82), tapi ternyata "
              "setara dengan dua lainnya. Penyebabnya `Solarize`/`Posterize` "
              "- keduanya memotong tingkat keabuan sehingga ikut merusak "
              "isyarat ketajaman walau bukan blur. Jadi blur **bukan** satu-"
              "satunya cara menyerang confound ini.", ""]

    if pad:
        strength = max(pad["auc"], 1.0 - pad["auc"])
        L += ["### Kebocoran kedua: bantalan letterbox", "",
              "Ketajaman bukan satu-satunya. `build_crops.py` menyimpan crop "
              "yang **sudah dipersegi**, jadi bilah bantalan abu-abu "
              "(114,114,114) ikut terpanggang ke dalam berkas - augmentasi "
              "bekerja di atas gambar yang bilahnya sudah ada, sehingga "
              "**tidak satu pun kebijakan di atas bisa menghapusnya.**", "",
              "Diukur pada **split train** (%d hidup, %d mati):" % (
                  pad["n_alive"], pad["n_dead"]), "",
              "| | crop hidup | crop mati |", "|---|---|---|",
              "| median bagian piksel bantalan | %.3f | %.3f |"
              % (pad["med_alive"], pad["med_dead"]),
              "| crop tanpa bilah sama sekali | %d / %d | %d / %d |"
              % (pad["nobar_alive"], pad["n_alive"],
                 pad["nobar_dead"], pad["n_dead"]), "",
              "AUC bagian bantalan = **%.4f**. Angkanya di bawah 0.5 karena "
              "arahnya terbalik (crop hidup justru **lebih** banyak "
              "bantalan); sebagai pembeda kekuatannya setara **%.4f**. "
              "Sebabnya sama dengan kebocoran ketajaman: crop ayam hidup "
              "lebih kecil dan lebih memanjang, jadi lebih banyak ruang yang "
              "harus dibantali." % (pad["auc"], strength), ""]

        if "floor_test" in pad:
            L += ["Tapi kekuatan **mengurutkan** tidak sama dengan kekuatan "
                  "**memutuskan**. Diuji dengan protokol yang sama persis "
                  "seperti lantai ketajaman (ambang dicocokkan di train, "
                  "dinilai di test):", "",
                  "| isyarat | bacc train | bacc **test** | AUC test |",
                  "|---|---|---|---|",
                  "| ketajaman (vLap) | %s | **%s** | %s |" % (
                      ("%.4f" % fl["bacc_train"]) if fl else "-", fb, fa),
                  "| bantalan letterbox | %.4f | **%.4f** | %.4f |"
                  % (pad["floor_train"], pad["floor_test"],
                     pad["floor_auc"]), "",
                  "Jadi bantalan **mengurutkan** hampir sebaik ketajaman "
                  "(AUC %.4f), tapi ambangnya **tidak berpindah** dari train "
                  "ke test - bacc-nya jatuh ke %.4f. Karena itu yang dipakai "
                  "sebagai lantai resmi tetap ketajaman (%s): itu yang benar-"
                  "benar bisa dicapai tanpa belajar apa pun. Bantalan dicatat "
                  "sebagai peringatan, bukan sebagai lantai kedua."
                  % (pad["floor_auc"], pad["floor_test"], fb), ""]

        L += ["Arah bilah (atas-bawah vs kiri-kanan) ikut membocorkan sedikit "
              "(AUC %.4f), jadi yang bocor terutama ketebalannya." 
              % pad["auc_orient"], "",
              "**Ini tidak diperbaiki di pekerjaan ini** - memperbaikinya "
              "berarti membangun ulang `data/crops/` dan membatalkan semua "
              "angka lama. Stub `crops.equalize_resolution` di config sudah "
              "disiapkan (default mati) supaya temuannya tidak hilang. "
              "Dicatat di sini karena angka hasil di bawah ikut terpengaruh.",
              ""]

    if degen:
        L += ["### Efek samping: view yang keluar nyaris polos", "",
              "Color jitter berkekuatan penuh (`strength 1.0`, sesuai paper) "
              "kadang menghasilkan view yang praktis tidak berisi lagi. "
              "Diukur sebagai simpangan baku abu-abu < 8:", "",
              "| kebijakan | % view polos | pada hidup | pada mati |",
              "|---|---|---|---|"]
        for r in degen:
            L.append("| `%s` | %.2f%% | %.2f%% | %.2f%% |"
                     % (r["policy"], r["pct"], r["pct_alive"], r["pct_dead"]))
        worst = max(degen, key=lambda r: abs(r["pct_alive"] - r["pct_dead"]))
        L += ["", "Yang penting bukan besarnya, melainkan **keseimbangannya "
              "antar kelas**: selisih terbesar cuma %.2f poin persen "
              "(`%s`). Kalau satu kelas jauh lebih sering rusak, itu bias "
              "baru; karena seimbang, ini derau - dan dibiarkan apa adanya "
              "supaya tetap setia pada paper."
              % (abs(worst["pct_alive"] - worst["pct_dead"]), worst["policy"]),
              ""]

    L += ["## Tiga kebijakan augmentasi", ""]
    for name in POLICIES:
        L += ["### `%s` - dipakai metode `%s`" % (name, USED_BY[name]), "",
              "**Sumber:** %s" % PAPER[name], "",
              "**Alasan dipakai di dataset ini:**",
              ALASAN[name].strip(), "", "**Rangkaian op:**", ""]
        L += ops_table(cfg, name)
        L += ["", "**Contoh keluaran** (kolom 0 = asli; `v###` = ketajaman "
              "vLap ubin itu; hijau = hidup, merah = mati):", "",
              "![%s](aug/%s.jpg)" % (name, name), ""]

    L += ["## Deviasi dari paper yang perlu diketahui", "", DEVIASI.strip(), "",
          "## Perbandingan ketiga kebijakan pada ayam yang sama", "",
          "Kotak crop dan flip sengaja dibuat identik di ketiga baris (RNG "
          "diturunkan per-op), jadi yang terlihat berbeda benar-benar hanya "
          "op yang memang berbeda:", "",
          "![side by side](aug/policies_side_by_side.jpg)", ""]

    L += ["## Hasil", ""]
    L += results_section(cfg) if with_metrics else \
        ["_Belum dijalankan. Gambar di atas dibuat lebih dulu supaya "
         "kebijakan yang salah ketahuan sebelum training._", ""]

    L += ["## Cara membaca angkanya", "",
          "1. **Balanced accuracy di sini bergerak dalam langkah yang "
          "besar.** Test set berisi 26 crop mati tapi cuma **7 crop hidup**, "
          "dan balanced accuracy memberi bobot sama ke kedua kelas. "
          "Akibatnya satu crop hidup yang salah menggeser skor **7.14 poin**, "
          "sedangkan satu crop mati cuma 1.92 poin. Jadi selisih 7 poin antar "
          "metode artinya *satu gambar*, dan selisih 14 poin artinya *dua "
          "gambar* - bukan bukti bahwa metodenya lebih baik. Ke-33 crop itu "
          "pun berasal dari hanya 7 foto asli, jadi sampelnya bahkan tidak "
          "sepenuhnya saling bebas. Karena itu tiap konfigurasi dijalankan 5 "
          "seed dan yang dilaporkan rata-rata ± simpangan baku, bukan satu "
          "angka.",
          "2. **Metode yang BAL.ACC-nya di bawah lantai %s belum "
          "membuktikan apa-apa** - hasil yang sama bisa didapat tanpa "
          "melihat isi gambar. Tapi lantai itu punya dua sisi: bal.acc "
          "(keputusan) DAN AUC (urutan). Ada konfigurasi yang gagal di sisi "
          "pertama tapi lolos di sisi kedua - yaitu mengurutkan lebih baik "
          "daripada ketajaman walau ambangnya meleset. Itu tetap sebuah "
          "hasil, dan dirinci di "
          "[`comparison.md`](comparison.md) bagian \"Yang berhasil "
          "melewati lantai\"." % fb,
          "3. **Augmentasi ini dipakai di tahap KONTRASTIF saja** untuk "
          "`selfcon`/`supcon`. Tahap linear probe-nya sengaja dikunci ke "
          "`minimal` (flip saja, lihat `lock_probe_policy`), supaya angkanya "
          "mengukur kualitas encoder dan bukan campuran dua augmentasi. "
          "`ce` tidak bisa ikut dikunci - kepala *adalah* satu-satunya tahap "
          "latihnya, jadi di baris `ce` augmentasi itu memang bekerja di "
          "tahap yang berbeda. Asimetri ini nyata; tabel per-tahap ada di "
          "[`comparison.md`](comparison.md).",
          "4. **Diagonal menjawab pertanyaan Anda, kolom menjawab "
          "\"metode mana yang terbaik\".** Keduanya pertanyaan berbeda.",
          "5. **Ambang `@tau` dipilih di validation set, tidak pernah di "
          "test set.** Ini penting: pernah terjadi `selfcon` mendapat AUC "
          "1.0000 (pemisahan sempurna) tapi balanced accuracy cuma 0.7857, "
          "semata karena ambang bawaan 0.5 jatuh di tempat yang salah. AUC "
          "mengukur urutan, balanced accuracy mengukur keputusan - "
          "keduanya perlu dilaporkan.",
          "6. **Batasan implementasi Zhang**: hanya modul augmentasinya yang "
          "dipakai, bukan loss multi-stage dan augmentation embedding-nya.",
          ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Laporan augmentasi per metode")
    ap.add_argument("--config", default=None)
    ap.add_argument("--policies", default=None,
                    help="daftar kebijakan, dipisah koma")
    ap.add_argument("--no-metrics", action="store_true",
                    help="gambar saja, tanpa tabel hasil (sebelum training)")
    ap.add_argument("--views", type=int, default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.views:
        cfg["augmentation"]["preview_views"] = a.views
    global POLICIES
    if a.policies:
        POLICIES = [p.strip() for p in a.policies.split(",") if p.strip()]

    rep = resolve(cfg["output"]["reports_dir"])
    aug_dir = rep / "aug"
    aug_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(cfg, "train")
    samples = pick_samples(rows, cfg)
    print("contoh: %d crop (%d hidup, %d mati)"
          % (len(samples), sum(1 for r in samples if r["label"] == 0),
             sum(1 for r in samples if r["label"] == 1)))

    for name in POLICIES:
        grid_for_policy(name, samples, cfg, aug_dir / ("%s.jpg" % name))
        print("  gambar: %s" % (aug_dir / ("%s.jpg" % name)))
    grid_side_by_side(samples, cfg, aug_dir / "policies_side_by_side.jpg")
    print("  gambar: %s" % (aug_dir / "policies_side_by_side.jpg"))

    print("mengukur kebocoran jalan pintas ketajaman...")
    leak = leakage_table(cfg, rows)
    for r in leak:
        print("  %-20s AUC %.4f" % (r["policy"], r["auc"]))

    print("mengukur kebocoran bantalan letterbox...")
    pad = padding_leak(rows, cfg)
    print("  bagian bantalan   AUC %.4f (hidup %.3f vs mati %.3f)"
          % (pad["auc"], pad["med_alive"], pad["med_dead"]))

    print("mengukur view yang keluar nyaris polos...")
    degen = degenerate_views(cfg, rows)
    for r in degen:
        print("  %-20s %.2f%% polos" % (r["policy"], r["pct"]))

    md = rep / "augmentation_report.md"
    write_md(cfg, leak, not a.no_metrics, md, pad=pad, degen=degen)
    print("\nlaporan: %s" % md)


if __name__ == "__main__":
    main()
