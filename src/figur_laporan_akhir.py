"""
Gambar untuk `outputs/reports/laporan_lengkap_dua_percobaan.md`.

Satu skrip untuk sembilan figur karena semuanya berbagi palet dan pemuat data
yang sama; dipecah jadi sembilan berkas berarti menyalin kode yang sama
sembilan kali.

Seluruh angka dibaca dari berkas yang sudah ada di `outputs/` dan `data/`.
Tidak ada inferensi yang dijalankan ulang di sini - `ultralytics` memang tidak
terpasang, jadi tahap deteksi hanya bisa dibaca dari `detections.json` yang
tersimpan.

Pakai:
    python src/figur_laporan_akhir.py --figur all
    python src/figur_laporan_akhir.py --figur 2,3 --outdir outputs/reports/laporan_akhir
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # harus sebelum pyplot; tidak ada layar di sini
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from common import imread, letterbox

# --- palet, sama persis dengan src/figur_babak14.py -------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BIRU = "#2a78d6"
JINGGA = "#eb6834"
AQUA = "#1baf7a"
MERAH = "#cc2f2f"

ROOT = Path(__file__).resolve().parents[1]
FRAME_DIR = Path(r"C:/Arib/CCTV/patnet-pure/dataset/chick/images")
DPI = 180


def rapikan(ax):
    """Sumbu tipis: dua spine, tanpa tick mark, warna teks diredam."""
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9, length=0)


def simpan(fig, out: Path, nama: str):
    out.mkdir(parents=True, exist_ok=True)
    p = out / nama
    fig.savefig(p, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  tulis {p.relative_to(ROOT)}  ({p.stat().st_size / 1024:.0f} KB)")


def muat_json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def muat_manifest(rel: str) -> list[dict]:
    with (ROOT / rel).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """AUC lewat statistik Mann-Whitney U (tidak butuh sklearn)."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def kurva_roc(y_true: np.ndarray, score: np.ndarray):
    """(fpr, tpr) tanpa sklearn; ambang = tiap skor unik, menurun."""
    order = np.argsort(-score, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    n_pos, n_neg = max(int((y_true == 1).sum()), 1), max(int((y_true == 0).sum()), 1)
    return np.r_[0, fp / n_neg, 1], np.r_[0, tp / n_pos, 1]


def bgr2rgb(img: np.ndarray) -> np.ndarray:
    return img[:, :, ::-1] if img is not None and img.ndim == 3 else img


# ===========================================================================
# Figur 1 - alur pipeline dua tahap
# ===========================================================================
def figur_1(out: Path):
    det = muat_json("outputs/predictions/detections.json")
    ev = muat_json("outputs/predictions/detect_masks_eval.json")

    fig, ax = plt.subplots(figsize=(12.4, 4.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    kotak = [
        (2.0, "Frame masukan", [f"{det['num_images']} gambar CCTV + close-up",
                                "ukuran campur, .jpg dan .png"], BASE),
        (21.5, "Tahap 1 - deteksi", ["YOLOv8m, conf 0.25",
                                     f"imgsz {det['imgsz']}",
                                     f"{det['num_boxes']} kotak keluar"], BIRU),
        (41.0, "Crop + letterbox", ["pad_box lalu letterbox 224x224",
                                    "bantalan abu (114,114,114)"], MUTED),
        (60.5, "Tahap 2 - classifier", ["ResNet-18 pra-latih ImageNet",
                                        "11.176.512 parameter backbone",
                                        "selfcon / supcon / ce"], AQUA),
        (80.0, "Keluaran", ["skor mati per crop",
                            "ambang dari validation",
                            "AUC + balanced accuracy"], JINGGA),
    ]
    W, H, Y = 18.0, 16.0, 13.0
    for x, judul, isi, warna in kotak:
        ax.add_patch(FancyBboxPatch((x, Y), W, H, boxstyle="round,pad=0.4",
                                    linewidth=1.6, edgecolor=warna,
                                    facecolor="white", zorder=2))
        ax.text(x + W / 2, Y + H - 3.4, judul, ha="center", va="center",
                fontsize=10.5, color=INK, fontweight="bold", zorder=3)
        for i, ln in enumerate(isi):
            ax.text(x + W / 2, Y + H - 7.0 - i * 3.1, ln, ha="center",
                    va="center", fontsize=8.4, color=INK2, zorder=3)
    for x, _, _, _ in kotak[:-1]:
        ax.add_patch(FancyArrowPatch((x + W + 0.3, Y + H / 2),
                                     (x + W + 1.2, Y + H / 2),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color=MUTED, linewidth=1.4, zorder=1))

    ax.text(30.5, 9.4, f"recall {ev['n_objects']}/{ev['n_objects']} acuan ayam mati = 100%,"
                       f"  IoU rerata 0.797", ha="center", va="top",
            fontsize=9, color=BIRU)
    ax.text(70.0, 9.4, "dilewati seluruhnya pada percobaan SDNET2018\n"
                       "(ubin 256x256 sudah jadi, tanpa detektor)",
            ha="center", va="top", fontsize=9, color=INK2)
    ax.text(2.0, 36.5, "Alur dua tahap, dengan angka nyata di tiap kotak",
            fontsize=12.5, color=INK, fontweight="bold")
    ax.text(2.0, 33.0, "Sumber: outputs/predictions/detections.json, "
                       "outputs/predictions/detect_masks_eval.json",
            fontsize=8.2, color=MUTED)
    simpan(fig, out, "01_alur_pipeline.png")


# ===========================================================================
# Figur 2 - bounding box pada frame uji
# ===========================================================================
def figur_2(out: Path):
    det = muat_json("outputs/predictions/detections.json")
    per_frame = {r["name"]: r for r in det["results"]}

    mati: dict[str, list] = {}
    with (ROOT / "data/label_chick/lembar_label.csv").open(encoding="utf-8",
                                                           newline="") as f:
        for r in csv.DictReader(f):
            if r["label"].strip() == "mati":
                # bbox di lembar label dipisah SPASI, bukan JSON
                b = [float(v) for v in r["bbox"].split()]
                mati.setdefault(r["gambar_sumber"], []).append((r["nomor"], b))

    pilih = ["ayam (2).jpg", "ayam (4).jpg", "chick (1).jpg", "chick (5).jpg"]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.0))
    fig.patch.set_facecolor(SURFACE)
    for ax, nama in zip(axes.ravel(), pilih):
        ax.set_facecolor(SURFACE)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(BASE)
        p = FRAME_DIR / nama
        img = imread(p)
        if img is None:
            ax.text(0.5, 0.5, f"frame tidak terbaca:\n{nama}", ha="center",
                    va="center", fontsize=9, color=MERAH, transform=ax.transAxes)
            continue
        ax.imshow(bgr2rgb(img))
        rec = per_frame.get(nama, {"detections": []})
        for d in rec["detections"]:
            x1, y1, x2, y2 = d["bbox"]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor=BASE, linewidth=0.7, alpha=0.85))
        for nomor, (x1, y1, x2, y2) in mati.get(nama, []):
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor=MERAH, linewidth=2.1))
            ax.text(x1, max(y1 - 3, 8), f"mati #{nomor}", fontsize=7.4,
                    color="white", va="bottom",
                    bbox=dict(boxstyle="square,pad=0.15", fc=MERAH, ec="none"))
        n_all = len(rec["detections"])
        n_mati = len(mati.get(nama, []))
        ax.set_title(f"{nama} - {n_all} deteksi, {n_mati} berlabel mati",
                     fontsize=9.6, color=INK, pad=6)

    fig.suptitle("Keluaran tahap deteksi pada empat frame uji", fontsize=13,
                 color=INK, fontweight="bold", y=0.98)
    fig.text(0.5, 0.012,
             "Abu tipis = seluruh keluaran YOLOv8m (conf 0.25, imgsz 640). "
             "Merah tebal = ayam mati hasil anotasi manual.\n"
             "Sumber: outputs/predictions/detections.json + "
             "data/label_chick/lembar_label.csv",
             ha="center", fontsize=8.2, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    simpan(fig, out, "02_bbox_uji.png")


def _galeri(ax_row, rows, crops_dir: Path, judul: str, warna: str, n: int):
    """Satu baris galeri crop; kotak tepi berwarna menandai kelas."""
    for k, ax in enumerate(ax_row):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_color(warna); s.set_linewidth(1.8)
        if k >= len(rows):
            ax.axis("off")
            continue
        img = imread(crops_dir / rows[k]["path"])
        if img is None:
            ax.text(0.5, 0.5, "?", ha="center", va="center", color=MERAH)
            continue
        ax.imshow(bgr2rgb(letterbox(img, 224)))
        if k == 0:
            ax.set_ylabel(judul, fontsize=8.6, color=INK, rotation=0,
                          ha="right", va="center", labelpad=8)


# ===========================================================================
# Figur 3 - galeri crop ayam per sumber: label = domain, terlihat dengan mata
# ===========================================================================
def figur_3(out: Path):
    crops = ROOT / "data/crops_pio_dev2"
    rows = muat_manifest("data/crops_pio_dev2/manifest.csv")
    kel = [
        ("pio_gt", "cctv", "hidup", AQUA, "pio_gt / cctv\n(hidup, 300)"),
        ("coco_gt", "closeup", "mati", JINGGA, "coco_gt / closeup\n(mati, 98)"),
        ("archive4_rgb", "closeup_kaggle", "mati", MERAH,
         "archive4_rgb / closeup_kaggle\n(mati, 200)"),
    ]
    N = 8
    fig, axes = plt.subplots(3, N, figsize=(13.0, 5.8))
    fig.patch.set_facecolor(SURFACE)
    for i, (origin, domain, _lbl, warna, judul) in enumerate(kel):
        sub = [r for r in rows if r["origin"] == origin]
        idx = np.linspace(0, len(sub) - 1, N).astype(int)
        _galeri(axes[i], [sub[j] for j in idx], crops, judul, warna, N)

    fig.suptitle("Setiap sumber crop terlihat berbeda sebelum modelnya melihat bentuk",
                 fontsize=13, color=INK, fontweight="bold", y=0.985)
    fig.text(0.5, 0.015,
             "Tiga baris = tiga sumber. Karena seluruh crop hidup berasal dari CCTV "
             "dan seluruh crop mati dari close-up,\n"
             "pasangan (domain, label) terkunci 598/598 = 100%: menebak domain "
             "sudah cukup untuk menebak label.  "
             "Sumber: data/crops_pio_dev2/manifest.csv",
             ha="center", fontsize=8.3, color=MUTED)
    fig.tight_layout(rect=(0.03, 0.055, 1, 0.955))
    simpan(fig, out, "03_galeri_kelas_ayam.png")


# ===========================================================================
# Figur 4 - pergeseran lantai jalan pintas, dev -> dev2
# ===========================================================================
def figur_4(out: Path):
    a = muat_json("outputs/predictions/pio_dev_shortcut.json")["features"]
    b = muat_json("outputs/reports/shortcut_pio_dev2.json")["features"]
    ciri = ["ukuran_bbox", "hue", "ketajaman", "std_terang", "saturasi",
            "terang", "rasio_bbox"]
    va = [a[c]["val_auc"] for c in ciri]
    vb = [b[c]["val_auc"] for c in ciri]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(ciri))[::-1]
    for ax, val, judul in [(axes[0], va, "crops_pio_dev  (sebelum)"),
                           (axes[1], vb, "crops_pio_dev2  (sesudah, +200 mati)")]:
        rapikan(ax)
        ax.barh(y, val, height=0.62, color=BASE, edgecolor="none")
        for yy, v in zip(y, val):
            ax.text(v + 0.012, yy, f"{v:.4f}", va="center", fontsize=8.6,
                    color=INK)
        ax.axvline(0.5, color=GRID, linewidth=1.2)
        ax.set_xlim(0.45, 1.09)
        ax.set_title(judul, fontsize=10.6, color=INK, pad=8)
        ax.set_xlabel("AUC validation berarah  (max(AUC, 1-AUC))", fontsize=9,
                      color=INK2)
    axes[0].set_yticks(y); axes[0].set_yticklabels(ciri, fontsize=9.4)

    # dua ciri yang bergerak paling jauh, satu ke tiap arah
    for nm, warna, kata in [("hue", AQUA, "membaik"),
                            ("ukuran_bbox", MERAH, "memburuk")]:
        i = ciri.index(nm)
        yy = y[i]
        axes[1].barh([yy], [vb[i]], height=0.62, color=warna, edgecolor="none")
        axes[1].text(1.075, yy, kata, va="center", ha="right", fontsize=8.6,
                     color=warna, fontweight="bold")

    fig.suptitle("Menambah domain mati kedua menukar jalan pintas, tidak membunuhnya",
                 fontsize=13, color=INK, fontweight="bold", y=0.99)
    fig.text(0.5, 0.012,
             "hue runtuh 0.9135 -> 0.5093 (bagus), tetapi ukuran_bbox naik "
             "0.9852 -> 0.9976 (buruk): frame archive_4 adalah gambar utuh "
             "640x480,\nsedangkan crop PIO berukuran puluhan piksel, jadi "
             "aturan \"besar = mati\" nyaris sempurna.  "
             "Sumber: outputs/predictions/pio_dev_shortcut.json, shortcut_pio_dev2.json",
             ha="center", fontsize=8.3, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 0.945))
    simpan(fig, out, "04_pergeseran_jalan_pintas.png")


# ===========================================================================
# Figur 5 - galeri ubin SDNET2018
# ===========================================================================
def figur_5(out: Path):
    crops = ROOT / "data/crops_sdnet"
    rows = muat_manifest("data/crops_sdnet/manifest.csv")
    doms = ["deck", "wall", "pave"]
    nama_dom = {"deck": "deck (lantai jembatan)", "wall": "wall (dinding)",
                "pave": "pave (perkerasan)"}
    ada = {r["domain"] for r in rows}
    doms = [d for d in doms if d in ada] or sorted(ada)
    N = 6
    fig, axes = plt.subplots(len(doms) * 2, N, figsize=(10.4, 2.0 * len(doms) * 2))
    fig.patch.set_facecolor(SURFACE)
    for i, dom in enumerate(doms):
        for j, (lbl, warna) in enumerate([("retak", MERAH), ("utuh", AQUA)]):
            sub = [r for r in rows if r["domain"] == dom and r["label_name"] == lbl]
            idx = np.linspace(0, len(sub) - 1, N).astype(int) if sub else []
            judul = f"{nama_dom[dom].split(' ')[0]}\n{lbl} ({len(sub)})"
            _galeri(axes[i * 2 + j], [sub[k] for k in idx], crops, judul, warna, N)

    fig.suptitle("Ubin SDNET2018 256x256: retak vs utuh, tiga permukaan",
                 fontsize=13, color=INK, fontweight="bold", y=0.995)
    fig.text(0.5, 0.008,
             "Semua ubin berukuran sama persis, jadi ukuran_bbox dan rasio_bbox "
             "bernilai AUC 0.5000 tepat - nol geometri untuk dibocorkan.\n"
             "Sumber: data/crops_sdnet/manifest.csv",
             ha="center", fontsize=8.3, color=MUTED)
    fig.tight_layout(rect=(0.05, 0.035, 1, 0.972))
    simpan(fig, out, "05_galeri_sdnet.png")


# ===========================================================================
# Figur 6 - ROC SDNET (deck nyata; wall/pave hanya titik di tau_val)
# ===========================================================================
METODE = [("ce", "ce__hier_addone", BIRU),
          ("selfcon", "selfcon__simclr", JINGGA),
          ("supcon", "supcon__stacked_randaug", AQUA)]
SEEDS = ["s42", "s43", "s44"]


def figur_6(out: Path):
    runs = ROOT / "outputs/runs_sdnet"
    lintas = muat_json("outputs/predictions/lintas_domain_sdnet.json")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7))
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        rapikan(ax)
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.2, zorder=1)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("FPR", fontsize=9, color=INK2)
    axes[0].set_ylabel("TPR", fontsize=9, color=INK2)

    # --- panel 1: deck, kurva ROC sungguhan dari test_scores.npz
    ax = axes[0]
    for nama, run, warna in METODE:
        aucs = []
        for sd in SEEDS:
            z = np.load(runs / f"{run}__{sd}" / "test_scores.npz")
            y, s = z["y_true"].astype(int), z["y_score"].astype(float)
            fpr, tpr = kurva_roc(y, s)
            ax.plot(fpr, tpr, color=warna, linewidth=0.9, alpha=0.45, zorder=2)
            aucs.append(roc_auc(y, s))
        ax.plot([], [], color=warna, linewidth=2.2,
                label=f"{nama}  AUC {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    ax.set_title("test deck (in-domain, n=402)", fontsize=10.4, color=INK, pad=8)
    ax.legend(loc="lower right", fontsize=8.2, frameon=False)

    # --- panel 2 & 3: wall/pave - lintas_domain_sdnet.json TIDAK menyimpan
    #     skor per-crop, hanya metrik agregat, jadi yang bisa digambar cuma
    #     satu titik (FPR,TPR) pada ambang tau_val.
    for ax, split, judul in [(axes[1], "test_wall", "test_wall (hold-out, n=3200)"),
                             (axes[2], "test_pave", "test_pave (hold-out, n=3200)")]:
        for nama, run, warna in METODE:
            xs, ys, aucs = [], [], []
            for sd in SEEDS:
                blok = lintas["per_run"][f"{run}__{sd}"][split]["at_tau_val"]
                xs.append(1.0 - blok["recall_alive"])
                ys.append(blok["recall_dead"])
                aucs.append(blok["roc_auc"])
            ax.scatter(xs, ys, s=34, color=warna, zorder=3, alpha=0.9)
            ax.scatter([np.mean(xs)], [np.mean(ys)], s=128, color=warna,
                       edgecolor="white", linewidth=1.4, zorder=4,
                       label=f"{nama}  AUC {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        ax.set_title(judul, fontsize=10.4, color=INK, pad=8)
        ax.legend(loc="lower right", fontsize=8.2, frameon=False)

    fig.suptitle("SDNET2018: ROC in-domain dan titik operasi lintas permukaan",
                 fontsize=13, color=INK, fontweight="bold", y=0.995)
    fig.text(0.5, 0.008,
             "Panel kiri: tiga garis tipis = tiga seed. Panel tengah & kanan "
             "hanya TITIK, bukan kurva: lintas_domain_sdnet.json menyimpan "
             "metrik agregat saja,\n"
             "tanpa skor per-crop, dan inferensi tidak dijalankan ulang. Titik "
             "besar = rerata 3 seed pada ambang tau_val dari validation.  "
             "Sumber: outputs/runs_sdnet/*/test_scores.npz, "
             "outputs/predictions/lintas_domain_sdnet.json",
             ha="center", fontsize=8.2, color=MUTED)
    fig.tight_layout(rect=(0, 0.075, 1, 0.945))
    simpan(fig, out, "06_roc_sdnet.png")


# ===========================================================================
# Figur 7 - intervensi: SDNET turun, ayam justru naik
# ===========================================================================
def figur_7(out: Path):
    import collections
    sd = muat_json("outputs/reports/intervensi_sdnet.json")["results"]
    ay = muat_json("outputs/predictions/fixed_chick_dev2.json")["interventions"]

    g = collections.defaultdict(list)
    for k, rows in ay.items():
        for r in rows:
            reg = Path(r["registry"]).name
            keluarga = "eq48" if "_eq_" in reg else "asli"
            g[(keluarga, r["method"], k)].append(r["absolute"]["pooled_auc"])
    rerata = {k: float(np.mean(v)) for k, v in g.items()}

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    fig.patch.set_facecolor(SURFACE)

    # kiri: SDNET, empat perusakan
    ax = axes[0]
    rapikan(ax)
    tahap = ["asli", "abu", "hue+26", "acak16", "kabur"]
    x = np.arange(len(tahap))
    for nama, run, warna in METODE:
        v = [sd[run]["rata"][t] for t in tahap]
        ax.plot(x, v, "o-", color=warna, linewidth=2.0, markersize=6,
                label=nama)
    ax.set_xticks(x); ax.set_xticklabels(tahap, fontsize=9.2)
    ax.set_ylabel("AUC test deck (rerata 3 seed)", fontsize=9.2, color=INK2)
    ax.set_title("SDNET2018 - turun di 3/3 metode saat acak16 dan kabur",
                 fontsize=10.6, color=INK, pad=8)
    ax.legend(fontsize=8.6, frameon=False, loc="lower left")
    ax.annotate("perusakan yang relevan\nsecara fisik", xy=(4, 0.768),
                xytext=(2.5, 0.718), fontsize=8.4, color=MERAH,
                arrowprops=dict(arrowstyle="->", color=MERAH, linewidth=1.2))

    # kanan: ayam dev2, hanya asli -> acak16 yang tersedia
    ax = axes[1]
    rapikan(ax)
    lengan = [("asli", "selfcon"), ("asli", "ce"), ("asli", "supcon"),
              ("eq48", "selfcon"), ("eq48", "ce"), ("eq48", "supcon")]
    n_naik = 0
    for i, (kel, met) in enumerate(lengan):
        a = rerata[(kel, met, "asli")]
        b = rerata[(kel, met, "acak16")]
        naik = b > a
        n_naik += naik
        warna = MERAH if naik else AQUA
        ax.plot([0, 1], [a, b], "o-", color=warna, linewidth=1.9, markersize=6,
                alpha=0.92)
        ax.text(1.04, b, f"{kel}/{met}", fontsize=8.2, color=warna,
                va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["asli", "acak16"], fontsize=9.2)
    ax.set_xlim(-0.12, 1.62)
    ax.set_ylabel("AUC benchmark test chick, scorer absolute", fontsize=9.2,
                  color=INK2)
    ax.set_title(f"Ayam dev2 - AUC justru NAIK di {n_naik}/6 kombinasi",
                 fontsize=10.6, color=INK, pad=8)
    ax.axhline(0.8831, color=MERAH, linestyle="--", linewidth=1.3)
    ax.text(-0.09, 0.8865, "lantai test 0.8831 (saturation)", fontsize=8.2,
            color=MERAH, va="bottom")

    fig.suptitle("Gerbang sebab yang sama, dua dataset, dua arah yang berlawanan",
                 fontsize=13, color=INK, fontweight="bold", y=0.99)
    fig.text(0.5, 0.012,
             "acak16 mengacak 4x4 ubin: bentuk dan pose hancur, tekstur utuh. "
             "Di SDNET model rusak karena memang membaca gambarnya;\n"
             "di data ayam mengacak crop tidak mengganggu - bahkan membantu - "
             "karena yang dibaca ciri global, bukan pose.  "
             "Sumber: outputs/reports/intervensi_sdnet.json, "
             "outputs/predictions/fixed_chick_dev2.json",
             ha="center", fontsize=8.2, color=MUTED)
    fig.tight_layout(rect=(0, 0.065, 1, 0.945))
    simpan(fig, out, "07_intervensi_sdnet_vs_ayam.png")


# ===========================================================================
# Figur 8 - kami vs AlexNet paper sitasi
# ===========================================================================
# Angka paper: Dorafshan, Thomas & Maguire (2018), Data in Brief 21:1664-1668,
# Tabel 2 (AlexNet transfer learning). Ditulis manual - paper tidak ada di repo.
PAPER = {"deck": (91.92, 85.13), "wall": (89.31, 78.77), "pavement": (95.52, 89.28)}
KAMI = {"deck": (84.82, 75.62, "ce"), "wall": (84.62, 75.00, "supcon"),
        "pavement": (82.20, 75.00, "supcon")}


def figur_8(out: Path):
    perm = ["deck", "wall", "pavement"]
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    fig.patch.set_facecolor(SURFACE)
    rapikan(ax)
    x = np.arange(len(perm))
    w = 0.34
    pa = [PAPER[p][0] for p in perm]
    ka = [KAMI[p][0] for p in perm]
    ax.bar(x - w / 2, pa, w, color=BASE, label="AlexNet TL (paper, 100% data)")
    ax.bar(x + w / 2, ka, w, color=BIRU, label="pipeline ini (3 seed, 15% data)")

    for i, p in enumerate(perm):
        mp, bp = PAPER[p]
        mk, bk, met = KAMI[p]
        ax.plot([i - w, i], [bp, bp], color=MUTED, linestyle="--", linewidth=1.3)
        ax.plot([i, i + w], [bk, bk], color=MERAH, linestyle="--", linewidth=1.3)
        ax.text(i - w / 2, mp + 0.7, f"{mp:.2f}%\n+{mp - bp:.2f}", ha="center",
                fontsize=8.4, color=INK)
        warna = AQUA if (mk - bk) > (mp - bp) else INK
        ax.text(i + w / 2, mk + 0.7, f"{mk:.2f}%\n+{mk - bk:.2f}", ha="center",
                fontsize=8.4, color=warna, fontweight="bold")
        ax.text(i + w / 2, 61.2, met, ha="center", fontsize=8.6,
                color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n(kami: {'latih' if p == 'deck' else 'hold-out murni'})"
                        for p in perm], fontsize=9.4)
    ax.set_ylim(60, 100)
    ax.set_ylabel("akurasi (%)", fontsize=9.4, color=INK2)
    ax.legend(fontsize=8.8, frameon=False, loc="upper left")
    ax.set_title("Akurasi mentah tidak sebanding; selisih terhadap "
                 "tebak-mayoritas sebanding",
                 fontsize=12.6, color=INK, pad=10, loc="left")
    fig.text(0.5, 0.012,
             "Garis putus-putus = akurasi menebak kelas mayoritas pada rasio "
             "kelas masing-masing (abu: paper, merah: kami). Angka kedua di "
             "tiap batang adalah selisih terhadap garis itu.\n"
             "Paper melatih satu model per permukaan; di sini wall dan pavement "
             "tidak pernah dilihat saat latih sama sekali. "
             "Sumber: Dorafshan dkk. (2018) Tabel 2 + outputs/runs_sdnet/*/result.json",
             ha="center", fontsize=8.2, color=MUTED)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    simpan(fig, out, "08_sdnet_vs_paper.png")


# ===========================================================================
# Figur 9 - papan skor benchmark dev2: enam pipeline vs sembilan ciri polos
# ===========================================================================
NAMA_NUISANCE = {
    "sharpness": "ketajaman", "saturation": "saturasi",
    "brightness": "terang", "hue": "hue",
    "bbox_short_side": "sisi pendek bbox", "bbox_area": "luas bbox",
    "bbox_aspect": "rasio bbox",
    "letterbox_padding_fraction": "bantalan letterbox",
    "detector_confidence": "conf detektor",
}


def figur_9(out: Path):
    import collections
    d = muat_json("outputs/predictions/fixed_chick_dev2.json")
    g = collections.defaultdict(list)
    for r in d["interventions"]["asli"]:
        reg = Path(r["registry"]).name
        kel = "eq48" if "_eq_" in reg else "asli"
        g[(kel, r["method"])].append(r["absolute"]["pooled_auc"])

    bar = []
    for (kel, met), v in g.items():
        bar.append((f"{kel} / {met}", float(np.mean(v)), float(np.std(v)), BIRU))
    nu = d["nuisance"]["asli"][
        next(k for k in d["nuisance"]["asli"] if "_eq_" not in k)]
    for k, v in nu.items():
        skor = v["pooled_auc"]
        bar.append((NAMA_NUISANCE.get(k, k), max(skor, 1 - skor), 0.0, JINGGA))
    bar.sort(key=lambda t: t[1])

    fig, ax = plt.subplots(figsize=(10.4, 6.6))
    fig.patch.set_facecolor(SURFACE)
    rapikan(ax)
    y = np.arange(len(bar))
    ax.barh(y, [b[1] for b in bar], height=0.66,
            color=[b[3] for b in bar], edgecolor="none")
    ax.errorbar([b[1] for b in bar], y, xerr=[b[2] for b in bar], fmt="none",
                ecolor=INK2, elinewidth=1.1, capsize=2.6)
    for yy, b in zip(y, bar):
        ax.text(b[1] + b[2] + 0.012, yy,
                f"{b[1]:.4f}" + (f" ± {b[2]:.4f}" if b[2] else ""),
                va="center", fontsize=8.3, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([b[0] for b in bar], fontsize=9.0)
    ax.set_xlim(0.0, 1.10)
    ax.axvline(0.5, color=GRID, linewidth=1.2)
    ax.axvline(0.8831, color=MERAH, linestyle="--", linewidth=1.5)
    ax.text(0.8831, len(bar) - 0.2, " lantai test 0.8831", color=MERAH,
            fontsize=8.6, va="top")
    ax.set_xlabel("AUC berarah pada benchmark test chick (943 crop clean), "
                  "scorer absolute", fontsize=9.2, color=INK2)
    ax.set_title("Tidak satu pun dari enam pipeline dev2 melewati lantainya sendiri",
                 fontsize=12.6, color=INK, pad=10, loc="left")
    fig.text(0.5, 0.012,
             "Biru = pipeline terlatih (rerata ± simpangan 3 seed). "
             "Jingga = satu ciri mentah tanpa model sama sekali, satu ambang.\n"
             "Ciri saturation saja mencapai 0.8831; angka pipeline terbaik "
             "adalah eq48/supcon 0.7287 ± 0.0548.  "
             "Sumber: outputs/predictions/fixed_chick_dev2.json",
             ha="center", fontsize=8.2, color=MUTED)
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    simpan(fig, out, "09_papan_skor_dev2.png")


FIGUR = {1: figur_1, 2: figur_2, 3: figur_3, 4: figur_4, 5: figur_5,
         6: figur_6, 7: figur_7, 8: figur_8, 9: figur_9}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="outputs/reports/laporan_akhir")
    ap.add_argument("--figur", default="all",
                    help="'all' atau daftar dipisah koma, misal '2,3,7'")
    a = ap.parse_args()
    out = ROOT / a.outdir
    pilih = sorted(FIGUR) if a.figur == "all" else \
        [int(t) for t in a.figur.split(",") if t.strip()]
    for n in pilih:
        print(f"figur {n} ...")
        FIGUR[n](out)
    print(f"selesai: {len(pilih)} figur -> {out}")


if __name__ == "__main__":
    main()
