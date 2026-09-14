"""Tiga figur kesimpulan kronologi 2: papan skor, empat metrik, dan ide 2-input.

Palet dan permukaan mengikuti figur babak 14 supaya kedua dokumen konsisten.
Setiap figur satu sumbu saja, tiap mark diberi label langsung.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BIRU = "#2a78d6"
JINGGA = "#eb6834"

FAMILY = {"pio_dev_registry.json": "asli", "pio_eq_dev_registry.json": "eq48"}
ORDER = [("asli", "selfcon"), ("asli", "ce"), ("asli", "supcon"),
         ("eq48", "selfcon"), ("eq48", "ce"), ("eq48", "supcon")]
NAMA_CIRI = {
    "saturation": "saturasi warna", "bbox_short_side": "sisi pendek bbox",
    "bbox_area": "luas bbox", "detector_confidence": "kepercayaan detektor",
    "sharpness": "ketajaman", "bbox_aspect": "rasio sisi bbox",
    "letterbox_padding_fraction": "bantalan letterbox",
    "hue": "hue", "brightness": "kecerahan",
}

def rapikan(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9, length=0)

def baca(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    agg = {}
    for iv in ("asli", "acak16"):
        grup = defaultdict(list)
        for row in data["interventions"][iv]:
            grup[(FAMILY[Path(row["registry"]).name], row["method"])].append(row)
        for key, rows in grup.items():
            agg[(iv, *key)] = {
                sc: {m: st.mean(r[sc][m] for r in rows)
                     for m in ("pooled_auc", "pooled_ap", "recall_at_3", "mrr")}
                for sc in ("absolute", "relative_clean")
            }
    return agg, data["nuisance"]["asli"]["pio_dev_registry.json"]

def figur_papan_skor(path: Path, agg, nuis) -> None:
    """Semua peserta pada satu sumbu: 6 pipeline vs 9 ciri tanpa model."""
    baris = []
    for f, m in ORDER:
        baris.append((f"{f} / {m}", agg[("asli", f, m)]["absolute"]["pooled_auc"],
                      "model"))
    for nama, met in nuis.items():
        if met["pooled_auc"] >= 0.5:
            baris.append((NAMA_CIRI.get(nama, nama), met["pooled_auc"], "ciri"))
    baris.sort(key=lambda t: t[1])

    fig, ax = plt.subplots(figsize=(9.4, 5.4), facecolor=SURFACE)
    y = np.arange(len(baris))
    warna = [JINGGA if jenis == "ciri" else BIRU for _, _, jenis in baris]
    ax.barh(y, [v for _, v, _ in baris], height=0.6, color=warna)
    for yi, (_, v, _) in zip(y, baris):
        ax.text(v + 0.006, yi, f"{v:.3f}", va="center", ha="left",
                fontsize=9.5, color=INK)
    ax.axvline(0.5, color=BASE, linewidth=1.4, linestyle="--", zorder=1)

    ax.set_yticks(y, [n for n, _, _ in baris])
    for tick, (_, _, jenis) in zip(ax.get_yticklabels(), baris):
        tick.set_color(INK if jenis == "ciri" else INK2)
    ax.set_xlim(0.45, 0.98)
    ax.set_ylim(-1.5, len(baris) - 0.4)
    ax.text(0.5, -1.05, "menebak  0.500", ha="center", va="center",
            fontsize=9, color=MUTED)
    ax.set_xlabel("AUC pooled pada 943 ayam dari 18 frame chick",
                  color=INK2, fontsize=9.5)
    ax.set_title("Juara papan skor bukan model: saturasi warna, tanpa training",
                 color=INK, fontsize=12.5, loc="left", pad=30)
    ax.annotate("jingga = ciri gambar sederhana (1 angka, tanpa model)   ·   "
                "biru = classifier hasil 18 run",
                xy=(0, 1.015), xycoords="axes fraction",
                fontsize=9.5, color=INK2, ha="left", va="bottom")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    rapikan(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)

def figur_empat_metrik(path: Path, agg, nuis) -> None:
    """Di mana model menang dan di mana kalah - empat metrik berdampingan."""
    metrik = [("pooled_auc", "AUC\nurutan keseluruhan"),
              ("pooled_ap", "AP\nkualitas peringkat atas"),
              ("recall_at_3", "Recall@3\ntertangkap di 3 teratas"),
              ("mrr", "MRR\nseberapa tinggi mati naik")]
    best = agg[("asli", "eq48", "supcon")]["absolute"]
    sat = nuis["saturation"]

    fig, ax = plt.subplots(figsize=(8.8, 4.6), facecolor=SURFACE)
    x = np.arange(len(metrik))
    lebar = 0.34
    nilai_m = [best[k] for k, _ in metrik]
    nilai_s = [sat[k] for k, _ in metrik]
    ax.bar(x - lebar / 2 - 0.012, nilai_m, lebar, color=BIRU)
    ax.bar(x + lebar / 2 + 0.012, nilai_s, lebar, color=JINGGA)
    for xi, (vm, vs) in enumerate(zip(nilai_m, nilai_s)):
        menang = vm > vs
        ax.text(xi - lebar / 2 - 0.012, vm + 0.022, f"{vm:.3f}", ha="center",
                va="bottom", fontsize=9.5, color=INK,
                fontweight="bold" if menang else "normal")
        ax.text(xi + lebar / 2 + 0.012, vs + 0.022, f"{vs:.3f}", ha="center",
                va="bottom", fontsize=9.5, color=INK,
                fontweight="bold" if not menang else "normal")
        ax.text(xi, -0.135, "model menang" if menang else "saturasi menang",
                ha="center", va="center", fontsize=9,
                color=BIRU if menang else JINGGA)

    ax.set_xticks(x, [t for _, t in metrik])
    ax.set_ylim(-0.2, 1.0)
    ax.set_xlim(-0.6, len(metrik) - 0.4)
    ax.set_ylabel("nilai metrik", color=INK2, fontsize=9.5)
    ax.set_title("Model terbaik (eq48 / supcon) vs saturasi: menang 3, kalah 1",
                 color=INK, fontsize=12.5, loc="left", pad=30)
    ax.annotate("biru = classifier eq48 / supcon   ·   jingga = saturasi warna "
                "tanpa model", xy=(0, 1.015), xycoords="axes fraction",
                fontsize=9.5, color=INK2, ha="left", va="bottom")
    ax.axhline(0, color=BASE, linewidth=1.0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    rapikan(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)

def figur_dua_input(path: Path, agg) -> None:
    """Status ide 2-input: skor relatif sudah jalan, tapi baru sebagai scorer."""
    fig, ax = plt.subplots(figsize=(9.0, 4.8), facecolor=SURFACE)
    y = np.arange(len(ORDER))[::-1]
    lebar = 0.36
    absol = [agg[("asli", f, m)]["absolute"]["pooled_ap"] for f, m in ORDER]
    relat = [agg[("asli", f, m)]["relative_clean"]["pooled_ap"] for f, m in ORDER]
    ax.barh(y + lebar / 2 + 0.012, absol, lebar, color=BIRU)
    ax.barh(y - lebar / 2 - 0.012, relat, lebar, color=JINGGA)
    for yi, (va, vr) in zip(y, zip(absol, relat)):
        ax.text(va + 0.006, yi + lebar / 2 + 0.012, f"{va:.3f}", va="center",
                ha="left", fontsize=9.5, color=INK)
        ax.text(vr + 0.006, yi - lebar / 2 - 0.012, f"{vr:.3f}", va="center",
                ha="left", fontsize=9.5, color=INK)
    ax.set_yticks(y, [f"{f} / {m}" for f, m in ORDER])
    ax.set_xlim(0, 0.46)
    ax.set_xlabel("AP pooled - makin tinggi makin baik menaruh ayam mati di atas",
                  color=INK2, fontsize=9.5)
    ax.set_title("Ide 2-input sudah diuji sebagai cara menilai, belum sebagai "
                 "cara melatih", color=INK, fontsize=12.5, loc="left", pad=30)
    ax.annotate("biru = 1 crop dinilai sendirian   ·   jingga = 1 crop "
                "dibandingkan ke crop lain di frame yang sama",
                xy=(0, 1.015), xycoords="axes fraction",
                fontsize=9.5, color=INK2, ha="left", va="bottom")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    rapikan(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)

def figur_siapa_baca_bentuk(path: Path, agg) -> None:
    """Delta AUC saat petak 4x4 diacak - absolut vs relatif satu-frame."""
    d_abs, d_rel = [], []
    for f, m in ORDER:
        a = agg[("asli", f, m)]; b = agg[("acak16", f, m)]
        d_abs.append(b["absolute"]["pooled_auc"] - a["absolute"]["pooled_auc"])
        d_rel.append(b["relative_clean"]["pooled_auc"]
                     - a["relative_clean"]["pooled_auc"])

    fig, ax = plt.subplots(figsize=(9.2, 4.8), facecolor=SURFACE)
    y = np.arange(len(ORDER))[::-1]
    lebar = 0.36
    ax.barh(y + lebar / 2 + 0.012, d_abs, lebar, color=BIRU)
    ax.barh(y - lebar / 2 - 0.012, d_rel, lebar, color=JINGGA)
    for yi, (va, vr) in zip(y, zip(d_abs, d_rel)):
        ax.text(va + (0.004 if va >= 0 else -0.004), yi + lebar / 2 + 0.012,
                f"{va:+.3f}", va="center", ha="left" if va >= 0 else "right",
                fontsize=9.5, color=INK)
        ax.text(vr + (0.004 if vr >= 0 else -0.004), yi - lebar / 2 - 0.012,
                f"{vr:+.3f}", va="center", ha="left" if vr >= 0 else "right",
                fontsize=9.5, color=INK)
    ax.axvline(0, color=INK2, linewidth=1.4, zorder=5)
    ax.set_yticks(y, [f"{f} / {m}" for f, m in ORDER])
    ax.set_xlim(-0.175, 0.165)
    ax.set_ylim(-1.25, len(ORDER) - 0.4)
    ax.text(-0.088, -0.85, "< skor TURUN: bentuk memang dibaca",
            ha="center", va="center", fontsize=9, color=JINGGA)
    ax.text(0.075, -0.85, "skor NAIK: bentuk tidak dibaca >",
            ha="center", va="center", fontsize=9, color=BIRU)
    ax.set_xlabel("perubahan AUC setelah susunan tubuh dihancurkan",
                  color=INK2, fontsize=9.5)
    ax.set_title("Yang membaca bentuk ayam justru cara-menilai 2-input",
                 color=INK, fontsize=12.5, loc="left", pad=30)
    ax.annotate("biru = 1 crop sendirian (naik di 3 dari 6)   ·   jingga = "
                "dibandingkan ke frame yang sama (turun di 5 dari 6)",
                xy=(0, 1.015), xycoords="axes fraction",
                fontsize=9.5, color=INK2, ha="left", va="bottom")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    rapikan(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)

def main() -> None:
    p = argparse.ArgumentParser(description="Figur kesimpulan kronologi 2")
    p.add_argument("--json", default="outputs/predictions/fixed_chick.json")
    p.add_argument("--outdir", default="outputs/reports")
    args = p.parse_args()
    agg, nuis = baca(Path(args.json))
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    figur_papan_skor(out / "kesimpulan_papan_skor.png", agg, nuis)
    figur_empat_metrik(out / "kesimpulan_empat_metrik.png", agg, nuis)
    figur_dua_input(out / "kesimpulan_dua_input.png", agg)
    figur_siapa_baca_bentuk(out / "kesimpulan_siapa_baca_bentuk.png", agg)
    print("figur ditulis ke", out)

if __name__ == "__main__":
    main()
