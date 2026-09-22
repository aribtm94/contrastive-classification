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


def _lengan_dari_entri(e: dict) -> str:
    """Nama lengan (asli / eq48) dibaca dari CONFIG, bukan dari nama berkas.

    Heuristik lama `"_eq_" in nama_registry` kebetulan benar untuk dev2 (sudah
    diperiksa: 18/18 entri sepakat dengan config), tapi bentuknya persis
    kegagalan senyap yang sudah tercatat - begitu ada lengan baru yang tidak
    memakai pola nama itu, ia jatuh ke "asli" tanpa error dan dua lengan
    tercampur dalam satu rerata. Sumber kebenarannya crops.equalize_resolution.
    """
    eq = ((e.get("config_snapshot", {}).get("crops", {})
           .get("equalize_resolution") or {}).get("enabled"))
    if eq is None:
        raise KeyError(
            "entri benchmark tanpa config_snapshot.crops.equalize_resolution."
            "enabled - lengan tidak bisa ditentukan tanpa menebak nama berkas")
    return "eq48" if eq else "asli"


def _peta_lengan_registry(d: dict) -> dict:
    """registry -> lengan, dirakit dari config tiap entri (blok nuisance hanya
    berkunci nama berkas registry, jadi petanya diambil dari interventions)."""
    peta = {}
    for entri in d["interventions"].values():
        for e in entri:
            nama = Path(e["registry"]).name
            lengan = _lengan_dari_entri(e)
            if peta.setdefault(nama, lengan) != lengan:
                raise ValueError(f"registry {nama} mengaku dua lengan")
    return peta


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
            keluarga = _lengan_dari_entri(r)
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
        kel = _lengan_dari_entri(r)
        g[(kel, r["method"])].append(r["absolute"]["pooled_auc"])

    bar = []
    for (kel, met), v in g.items():
        bar.append((f"{kel} / {met}", float(np.mean(v)), float(np.std(v)), BIRU))
    peta = _peta_lengan_registry(d)
    nu = d["nuisance"]["asli"][
        next(k for k in d["nuisance"]["asli"] if peta[k] == "asli")]
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




# ===========================================================================
# Figur 10-14 - kurva latih/validasi per augmentasi, galeri augmentasi,
#               dan evaluasi akhir kedua percobaan
# ===========================================================================
# Satu augmentasi terikat ke satu metode (rancangan 2 faktor yang sengaja
# dibiarkan terikat, lihat bagian 5.2), jadi "per augmentasi" dan "per metode"
# menamai panel yang sama. Judul panel menyebut KEDUANYA supaya tidak ada yang
# menyangka augmentasinya bisa ditukar antar metode.
AUG_METODE = [("simclr", "selfcon", "selfcon__simclr", JINGGA),
              ("stacked_randaug", "supcon", "supcon__stacked_randaug", AQUA),
              ("hier_addone", "ce", "ce__hier_addone", BIRU)]

def _baca_history(run_dir: Path) -> list[dict]:
    """history.csv -> list dict; sel kosong jadi None, bukan 0.0.

    Sel kosong berarti "tidak diukur pada epoch ini" (mis. val_bacc selama
    tahap contrastive). Memaksanya jadi 0.0 akan menggambar garis yang turun
    ke nol - kejadian yang tidak pernah ada.
    """
    f = run_dir / "history.csv"
    if not f.exists():
        return []
    out = []
    with f.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            d = {"stage": r["stage"]}
            for k, v in r.items():
                if k == "stage":
                    continue
                d[k] = float(v) if v not in ("", None) else None
            out.append(d)
    return out

def _tahapan(hist: list[dict]) -> list[tuple[str, list[dict]]]:
    """Pecah history per tahap berurutan, tanpa mengubah urutan epoch."""
    blok, kini = [], None
    for row in hist:
        if kini is None or row["stage"] != kini[0]:
            kini = (row["stage"], [])
            blok.append(kini)
        kini[1].append(row)
    return blok

def _kurva_loss(ax, runs: Path, run_nama: str, warna: str, judul: str):
    """Loss per epoch untuk 3 seed; batas tahap ditandai garis vertikal."""
    rapikan(ax)
    batas, label_tahap = [], []
    for i, sd in enumerate(SEEDS):
        hist = _baca_history(runs / f"{run_nama}__{sd}")
        if not hist:
            continue
        x0 = 0
        for j, (nama_tahap, rows) in enumerate(_tahapan(hist)):
            xs = list(range(x0 + 1, x0 + 1 + len(rows)))
            ax.plot(xs, [r.get("train_loss") for r in rows],
                    color=warna, linewidth=1.0, alpha=0.75, zorder=3)
            # val loss putus-putus: dua besaran di satu sumbu hanya boleh
            # kalau bisa dibedakan tanpa melihat legenda.
            ax.plot(xs, [r.get("val_loss") for r in rows],
                    color=INK2, linewidth=0.9, alpha=0.5,
                    linestyle=(0, (3, 2)), zorder=2)
            x0 += len(rows)
            if i == 0:
                if j > 0:
                    batas.append(xs[0] - 0.5)
                label_tahap.append((nama_tahap, xs[0], xs[-1]))
    for b in batas:
        ax.axvline(b, color=BASE, linewidth=1.0, linestyle=":", zorder=1)
    for nama_tahap, a, b in label_tahap:
        ax.annotate(nama_tahap, xy=((a + b) / 2, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=7.4, color=MUTED)
    ax.plot([], [], color=warna, linewidth=1.6, label="train loss")
    ax.plot([], [], color=INK2, linewidth=1.2, linestyle=(0, (3, 2)),
            label="val loss")
    ax.set_title(judul, fontsize=9.8, color=INK, pad=15)
    ax.set_xlabel("epoch (tahap disambung)", fontsize=8.4, color=INK2)
    ax.legend(loc="upper right", fontsize=7.6, frameon=False)

def _kurva_val(ax, runs: Path, run_nama: str, warna: str, judul: str):
    """val_bacc & val_auc per epoch, 3 seed. Epoch tanpa validasi dilewati,
    bukan diisi nol - tahap contrastive memang tidak mengukur keduanya."""
    rapikan(ax)
    ax.set_ylim(0.38, 1.03)
    ax.axhline(0.5, color=GRID, linewidth=1.0, zorder=1)
    for sd in SEEDS:
        xs, bacc, auc = [], [], []
        for i, r in enumerate(_baca_history(runs / f"{run_nama}__{sd}"), start=1):
            if r.get("val_bacc") is None:
                continue
            xs.append(i); bacc.append(r["val_bacc"]); auc.append(r.get("val_auc"))
        ax.plot(xs, bacc, color=warna, linewidth=1.0, alpha=0.8, zorder=3)
        ax.plot(xs, auc, color=INK2, linewidth=0.9, alpha=0.45,
                linestyle=(0, (3, 2)), zorder=2)
    ax.plot([], [], color=warna, linewidth=1.6, label="val bacc")
    ax.plot([], [], color=INK2, linewidth=1.2, linestyle=(0, (3, 2)),
            label="val AUC")
    ax.set_title(judul, fontsize=9.8, color=INK, pad=7)
    ax.set_xlabel("epoch", fontsize=8.4, color=INK2)
    ax.legend(loc="lower right", fontsize=7.6, frameon=False)

def _roc_panel(ax, runs: Path, berkas: str, judul: str):
    """ROC dari skor per-crop tersimpan; satu kurva tipis per seed."""
    rapikan(ax)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.2, zorder=1)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    for aug, metode, run, warna in AUG_METODE:
        aucs = []
        for sd in SEEDS:
            f = runs / f"{run}__{sd}" / berkas
            if not f.exists():
                continue
            z = np.load(f)
            y, s = z["y_true"].astype(int), z["y_score"].astype(float)
            fpr, tpr = kurva_roc(y, s)
            ax.plot(fpr, tpr, color=warna, linewidth=1.0, alpha=0.5, zorder=2)
            aucs.append(roc_auc(y, s))
        if aucs:
            ax.plot([], [], color=warna, linewidth=2.0,
                    label=f"{metode}  {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    ax.set_title(judul, fontsize=9.8, color=INK, pad=7)
    ax.set_xlabel("FPR", fontsize=8.4, color=INK2)
    ax.set_ylabel("TPR", fontsize=8.8, color=INK2)
    ax.legend(loc="lower right", fontsize=7.4, frameon=False)

def figur_10(out: Path):
    runs = ROOT / "outputs/runs_sdnet"
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6))
    fig.patch.set_facecolor(SURFACE)
    axes[0].set_ylabel("loss", fontsize=9, color=INK2)
    for ax, (aug, metode, run, warna) in zip(axes, AUG_METODE):
        _kurva_loss(ax, runs, run, warna, f"{aug}  ({metode})")

    fig.suptitle("Percobaan A (SDNET2018): loss per epoch, satu panel per augmentasi",
                 fontsize=13, color=INK, fontweight="bold", y=0.995)
    fig.text(0.5, 0.008,
             "Tiga garis per panel = tiga seed (42/43/44); garis putus-putus = "
             "val loss. Garis titik-titik tegak = pergantian TAHAP, dan skala "
             "loss di kiri serta kanan garis itu berasal\n"
             "dari fungsi objektif yang berbeda (NT-Xent/SupCon vs BCE probe), "
             "jadi lompatan di situ bukan kejadian latihan. ce__hier_addone "
             "hanya punya satu tahap karena memang tidak\n"
             "melewati pra-latih kontrastif. Panjang tahap berbeda antar seed "
             "karena early stopping.  Sumber: outputs/runs_sdnet/*/history.csv",
             ha="center", fontsize=8.1, color=MUTED)
    fig.tight_layout(rect=(0, 0.115, 1, 0.94))
    simpan(fig, out, "10_loss_per_augmentasi_sdnet.png")

def figur_11(out: Path):
    runs = ROOT / "outputs/runs_sdnet"
    fig = plt.figure(figsize=(13.6, 8.6))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.26)

    for k, (aug, metode, run, warna) in enumerate(AUG_METODE):
        ax = fig.add_subplot(gs[0, k])
        _kurva_val(ax, runs, run, warna, f"{aug}  ({metode})")
        if k == 0:
            ax.set_ylabel("validation", fontsize=9, color=INK2)

    _roc_panel(fig.add_subplot(gs[1, 0]), runs, "val_scores.npz",
               "ROC validation (n=408)")
    _roc_panel(fig.add_subplot(gs[1, 1]), runs, "test_scores.npz",
               "ROC test deck (hold-out, n=402)")

    # --- panel kanan bawah: val -> test untuk seed yang SAMA.
    #     Menghubungkan dua angka milik satu checkpoint; rerata saja akan
    #     menyembunyikan seed mana yang jatuh.
    ax = fig.add_subplot(gs[1, 2])
    rapikan(ax)
    for aug, metode, run, warna in AUG_METODE:
        for sd in SEEDS:
            fv = runs / f"{run}__{sd}" / "val_scores.npz"
            ft = runs / f"{run}__{sd}" / "test_scores.npz"
            if not (fv.exists() and ft.exists()):
                continue
            zv, zt = np.load(fv), np.load(ft)
            av = roc_auc(zv["y_true"].astype(int), zv["y_score"].astype(float))
            at = roc_auc(zt["y_true"].astype(int), zt["y_score"].astype(float))
            ax.plot([0, 1], [av, at], color=warna, linewidth=1.1, alpha=0.6,
                    marker="o", markersize=3.6, zorder=3)
    for aug, metode, run, warna in AUG_METODE:
        ax.plot([], [], color=warna, linewidth=1.8, marker="o", markersize=4,
                label=metode)
    ax.set_xlim(-0.14, 1.14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["val", "test deck"], fontsize=9)
    ax.set_ylabel("AUC", fontsize=8.8, color=INK2)
    ax.set_title("val -> test, per seed", fontsize=9.8, color=INK, pad=7)
    ax.legend(loc="lower left", fontsize=7.4, frameon=False)

    fig.suptitle("Percobaan A (SDNET2018): validasi per epoch, dan ROC validasi vs test",
                 fontsize=13, color=INK, fontweight="bold", y=0.985)
    fig.text(0.5, 0.008,
             "Baris atas: validasi di SDNET TIDAK jenuh - kurvanya masih "
             "bergerak sampai akhir, jadi pemilihan checkpoint lewat val "
             "memang berarti di sini. Bandingkan bagian 9d: di susunan\n"
             "ayam val jenuh di 1.0000 dan tidak bisa memilih apa pun. Baris "
             "bawah: tiap kurva tipis satu seed, angka di legenda rerata ± "
             "simpangan 3 seed. Panel kanan bawah menghubungkan\n"
             "val dan test untuk seed yang SAMA - garis yang menukik = "
             "checkpoint itu tidak membawa kemampuannya ke data baru.  "
             "Sumber: outputs/runs_sdnet/*/history.csv, *_scores.npz",
             ha="center", fontsize=8.1, color=MUTED)
    fig.tight_layout(rect=(0, 0.085, 1, 0.945))
    simpan(fig, out, "11_validasi_dan_roc_sdnet.png")

def figur_12(out: Path):
    """Galeri augmentasi: crop yang SAMA lewat tiga kebijakan, dua dataset.

    Memakai dataset.augment() apa adanya - kode yang sama dengan latihan - dan
    `return_params=True` mengembalikan op yang benar-benar menyala, jadi
    caption tiap kotak adalah catatan eksekusi, bukan salinan config. Pada
    hier_addone sebagian op sengaja tidak menyala karena level yang tertarik
    lebih rendah; itu memang isi metodenya (Zhang & Ma, CVPR 2022).
    """
    import random
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import yaml
    from dataset import augment, policy_by_name

    SUMBER = [("configs/config_sdnet.yaml", "data/crops_sdnet",
               "SDNET2018 - ubin retak (percobaan A)", "retak"),
              ("configs/config_pio_dev2.yaml", "data/crops_pio_dev2",
               "crop ayam - kelas dead (percobaan B)", "dead")]
    N_VAR = 5

    blok = []
    for cfg_rel, crops_rel, judul, kelas in SUMBER:
        cfg_f = ROOT / cfg_rel
        man = ROOT / crops_rel / "manifest.csv"
        if not (cfg_f.exists() and man.exists()):
            print(f"  figur 12: {crops_rel} tidak ada, dilewati")
            continue
        cfg = yaml.safe_load(cfg_f.read_text(encoding="utf-8"))
        rows = [r for r in muat_manifest(f"{crops_rel}/manifest.csv")
                if r["split"] == "train" and r["label_name"] == kelas]
        if not rows:
            print(f"  figur 12: {crops_rel} tidak punya baris train '{kelas}'")
            continue
        # Deterministik: urutkan nama lalu ambil yang tengah, supaya gambar
        # ini sama persis tiap kali dibuat ulang.
        rows.sort(key=lambda r: r["path"])
        r = rows[len(rows) // 2]
        img = imread(ROOT / crops_rel / r["path"])
        if img is None:
            continue
        size = int(cfg["classifier"]["image_size"])
        blok.append((cfg, letterbox(img, size), judul, r["path"]))

    if not blok:
        print("  figur 12 dilewati: tidak ada sumber crop yang bisa dibaca")
        return

    n_baris = len(blok) * len(AUG_METODE)
    fig, axes = plt.subplots(n_baris, N_VAR + 1,
                             figsize=(1.92 * (N_VAR + 1), 2.16 * n_baris))
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_2d(axes)

    baris = 0
    for cfg, asli, judul_blok, nama_crop in blok:
        for aug, metode, _run, warna in AUG_METODE:
            ax = axes[baris][0]
            ax.imshow(bgr2rgb(asli)); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(warna); s.set_linewidth(1.6)
            ax.set_ylabel(f"{aug}\n({metode})", fontsize=8.0, color=INK)
            if baris % len(AUG_METODE) == 0:
                ax.set_title(f"{judul_blok}\nasli (sebelum augmentasi)",
                             fontsize=7.6, color=INK, pad=5)
            else:
                ax.set_title("asli", fontsize=7.6, color=INK2, pad=5)

            pol = policy_by_name(cfg, aug)
            for v in range(N_VAR):
                ax = axes[baris][v + 1]
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color(BASE)
                rng = random.Random(1000 + 17 * v + 101 * baris)
                hasil, par = augment(asli.copy(), rng, pol, return_params=True)
                ax.imshow(bgr2rgb(hasil))
                ops = ", ".join(par["ops"]) if par["ops"] else "(tidak ada op menyala)"
                lv = f"level {par['level']}  " if par["level"] is not None else ""
                ax.set_title(f"{lv}{ops}", fontsize=5.9, color=MUTED, pad=4,
                             wrap=True)
            baris += 1

    fig.suptitle("Augmentasi tiap kebijakan, dijalankan pada crop yang sama",
                 fontsize=13, color=INK, fontweight="bold", y=0.998)
    fig.text(0.5, 0.004,
             "Kolom pertama = crop sebelum augmentasi (tepi berwarna menandai "
             "kebijakannya). Lima kolom sisanya = lima tarikan acak dari "
             "kebijakan yang SAMA. Judul tiap kotak mencantumkan op\n"
             "yang benar-benar menyala, dibaca dari "
             "dataset.augment(return_params=True) - bukan daftar op di config. "
             "Ketiga kebijakan identik di kedua config (diperiksa), jadi "
             "perbedaan antar blok murni\n"
             "datang dari isi gambarnya.  Sumber: data/crops_sdnet + "
             "data/crops_pio_dev2, lewat src/dataset.py",
             ha="center", fontsize=8.1, color=MUTED)
    fig.tight_layout(rect=(0.012, 0.042, 1, 0.962))
    simpan(fig, out, "12_galeri_augmentasi.png")

def figur_13(out: Path):
    LENGAN = [("outputs/runs_pio_dev2", "lengan asli"),
              ("outputs/runs_pio_dev2_eq", "lengan eq48")]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.2))
    fig.patch.set_facecolor(SURFACE)
    for i, (base, judul_lengan) in enumerate(LENGAN):
        runs = ROOT / base
        for k, (aug, metode, run, warna) in enumerate(AUG_METODE):
            ax = axes[i][k]
            _kurva_loss(ax, runs, run, warna,
                        f"{aug} ({metode}) - {judul_lengan}")
            if k == 0:
                ax.set_ylabel("loss", fontsize=9, color=INK2)

    fig.suptitle("Percobaan B (domain ayam mati kedua): loss per epoch, per augmentasi",
                 fontsize=13, color=INK, fontweight="bold", y=0.995)
    fig.text(0.5, 0.008,
             "Baris atas lengan asli, baris bawah lengan eq48. Tiga garis per "
             "panel = tiga seed; putus-putus = val loss. Tahap probe/ce di "
             "sini PENDEK (16-40 epoch) karena early stopping\n"
             "berhenti cepat - val sudah sempurna sejak epoch-epoch awal. "
             "Itu justru gejala pokok percobaan B: val jenuh, jadi tidak ada "
             "yang bisa dipilih dengannya (bagian 10.2), dan loss yang\n"
             "terlihat rapi di sini tidak menjamin apa pun di luar val.  "
             "Sumber: outputs/runs_pio_dev2{,_eq}/*/history.csv",
             ha="center", fontsize=8.1, color=MUTED)
    fig.tight_layout(rect=(0, 0.105, 1, 0.945))
    simpan(fig, out, "13_loss_per_augmentasi_dev2.png")

def _benchmark_per_lengan(d: dict, kunci_lengan: str) -> dict:
    """Kelompokkan entri benchmark per (metode, intervensi) untuk satu lengan.

    fixed_chick_dev2.json memuat KEDUA lengan dalam satu daftar 18 entri, dan
    run_id-nya identik antar lengan - yang membedakan hanya config crop-nya.
    Memisahkannya lewat run_id akan diam-diam mencampur dua lengan.
    """
    g = {}
    for iv, entri in d["interventions"].items():
        for e in entri:
            if _lengan_dari_entri(e) != kunci_lengan:
                continue
            g.setdefault((e["method"], iv), []).append(
                e["relative_clean"]["pooled_auc"])
    return g

def figur_14(out: Path):
    """Validasi dan evaluasi held-out percobaan B.

    Percobaan B memakai protocol.development_only: TIDAK ada split test, dan
    test_scores.npz memang tidak ada di 18 run itu (diperiksa langsung di
    direktori run). Jadi panel kanan bawah BUKAN split test melainkan
    benchmark uji chick - satu-satunya evaluasi held-out lengan ini - dan
    judulnya menyebut itu apa adanya.
    """
    fig = plt.figure(figsize=(13.6, 8.6))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.26)
    runs_asli = ROOT / "outputs/runs_pio_dev2"

    # --- baris atas: validasi per epoch (lengan asli) + dua ROC validasi
    for k, (aug, metode, run, warna) in enumerate(AUG_METODE):
        ax = fig.add_subplot(gs[0, k])
        _kurva_val(ax, runs_asli, run, warna,
                   f"{aug} ({metode}) - lengan asli")
        if k == 0:
            ax.set_ylabel("validation", fontsize=9, color=INK2)

    _roc_panel(fig.add_subplot(gs[1, 0]), runs_asli, "val_scores.npz",
               "ROC validation, lengan asli (n=222)")
    _roc_panel(fig.add_subplot(gs[1, 1]),
               ROOT / "outputs/runs_pio_dev2_eq", "val_scores.npz",
               "ROC validation, lengan eq48 (n=222)")

    # --- panel kanan bawah: benchmark uji chick, asli -> acak16, dua lengan.
    #     Lengan dibedakan lewat garis penuh vs putus-putus, bukan warna:
    #     warna sudah dipakai untuk metode di seluruh laporan.
    ax = fig.add_subplot(gs[1, 2])
    rapikan(ax)
    f = ROOT / "outputs/predictions/fixed_chick_dev2.json"
    if not f.exists():
        ax.text(0.5, 0.5, "fixed_chick_dev2.json belum ada", ha="center",
                va="center", fontsize=9, color=MUTED, transform=ax.transAxes)
    else:
        d = json.loads(f.read_text(encoding="utf-8"))
        for lengan, gaya in [("asli", "-"), ("eq48", (0, (4, 2)))]:
            g = _benchmark_per_lengan(d, lengan)
            for aug, metode, run, warna in AUG_METODE:
                m = []
                for xi, iv in enumerate(["asli", "acak16"]):
                    ys = g.get((metode, iv), [])
                    m.append(float(np.mean(ys)) if ys else float("nan"))
                    if ys:
                        ax.scatter([xi] * len(ys), ys, s=22, color=warna,
                                   alpha=0.5, zorder=3)
                ax.plot([0, 1], m, color=warna, linewidth=1.6, alpha=0.9,
                        linestyle=gaya, marker="o", markersize=5,
                        markeredgecolor="white", markeredgewidth=1.0, zorder=4,
                        label=f"{lengan}/{metode}  {m[0]:.4f} -> {m[1]:.4f}")
        ax.axhline(0.5, color=GRID, linewidth=1.2, zorder=1)
        ax.set_xlim(-0.35, 1.35); ax.set_xticks([0, 1])
        ax.set_xticklabels(["crop asli", "acak16"], fontsize=9)
        ax.set_ylabel("pooled AUC (relative_clean)", fontsize=8.8, color=INK2)
    ax.set_title("benchmark uji chick - BUKAN split test", fontsize=9.8,
                 color=INK, pad=7)
    ax.legend(loc="best", fontsize=6.4, frameon=False, ncol=1)

    fig.suptitle("Percobaan B: validasi per epoch, ROC validasi, dan evaluasi held-out",
                 fontsize=13, color=INK, fontweight="bold", y=0.985)
    fig.text(0.5, 0.008,
             "Percobaan B memakai protocol.development_only: manifestnya TIDAK "
             "punya split test dan test_scores.npz memang tidak ada di 18 run "
             "itu, jadi panel kanan bawah BUKAN split test - itu\n"
             "benchmark uji chick, satu-satunya evaluasi held-out lengan ini. "
             "Baris atas dan kedua panel ROC memperlihatkan masalahnya: "
             "validasi menempel di 1.0000 untuk ketiga metode di kedua\n"
             "lengan, jadi val tidak bisa membedakan apa pun. Garis penuh = "
             "lengan asli, putus-putus = eq48; titik kecil = seed, titik besar "
             "= rerata 3 seed.  Sumber: outputs/runs_pio_dev2{,_eq}/*/\n"
             "val_scores.npz dan history.csv, outputs/predictions/fixed_chick_dev2.json",
             ha="center", fontsize=8.1, color=MUTED)
    fig.tight_layout(rect=(0, 0.095, 1, 0.945))
    simpan(fig, out, "14_validasi_dan_benchmark_dev2.png")


FIGUR = {1: figur_1, 2: figur_2, 3: figur_3, 4: figur_4, 5: figur_5,
         6: figur_6, 7: figur_7, 8: figur_8, 9: figur_9,
         10: figur_10, 11: figur_11, 12: figur_12, 13: figur_13,
         14: figur_14}


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
