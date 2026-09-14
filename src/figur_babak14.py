"""Tiga figur untuk Babak 14: model vs baseline, gerbang acak16, dan saturasi.

Palet mengikuti slot kategorikal yang sudah tervalidasi (biru/jingga/aqua),
permukaan terang #fcfcfb, tinta sekunder #52514e, garis kisi #e1e0d9.
Setiap figur memakai satu sumbu saja dan diberi label langsung.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
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
AQUA = "#1baf7a"

FAMILY = {"pio_dev_registry.json": "asli", "pio_eq_dev_registry.json": "eq48"}
ORDER = [("asli", "selfcon"), ("asli", "ce"), ("asli", "supcon"),
         ("eq48", "selfcon"), ("eq48", "ce"), ("eq48", "supcon")]

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
            grup[(FAMILY[os.path.basename(row["registry"])], row["method"])].append(row)
        for key, rows in grup.items():
            agg[(iv, *key)] = {
                "auc": st.mean(r["absolute"]["pooled_auc"] for r in rows),
                "auc_sd": st.pstdev([r["absolute"]["pooled_auc"] for r in rows]),
                "ap": st.mean(r["absolute"]["pooled_ap"] for r in rows),
            }
    sat = data["nuisance"]["asli"]["pio_dev_registry.json"]["saturation"]
    return agg, sat

def figur_baseline(path: Path, agg, sat) -> None:
    """Batang AUC tiap pipeline, dengan garis acuan saturasi tanpa model."""
    label = [f"{f} / {m}" for f, m in ORDER]
    nilai = [agg[("asli", f, m)]["auc"] for f, m in ORDER]
    galat = [agg[("asli", f, m)]["auc_sd"] for f, m in ORDER]
    fig, ax = plt.subplots(figsize=(9.2, 4.4), facecolor=SURFACE)
    y = np.arange(len(label))[::-1]
    warna = [AQUA if f == "eq48" else BIRU for f, _ in ORDER]
    ax.barh(y, nilai, height=0.58, color=warna, xerr=galat, capsize=3,
            error_kw={"ecolor": MUTED, "elinewidth": 1.2})
    for yi, v in zip(y, nilai):
        ax.text(v + 0.018, yi, f"{v:.3f}", va="center", ha="left",
                fontsize=9.5, color=INK)
    ax.axvline(sat["pooled_auc"], color=JINGGA, linewidth=2, zorder=5)
    ax.text(sat["pooled_auc"] - 0.012, y[0] + 0.62,
            f"saturasi saja, tanpa model  {sat['pooled_auc']:.3f}",
            ha="right", va="center", fontsize=9.5, color=JINGGA)
    ax.axvline(0.5, color=BASE, linewidth=1.4, linestyle="--")
    ax.text(0.5, -0.95, "menebak", ha="center", va="center",
            fontsize=9, color=MUTED)
    ax.set_yticks(y, label)
    ax.set_ylim(-1.35, len(label) - 0.35)
    ax.set_xlim(0.35, 1.0)
    ax.set_xlabel("AUC pooled pada 943 ayam dari 18 frame", color=INK2, fontsize=9.5)
    ax.set_title("Tidak satu pun dari 18 checkpoint melewati garis jingga",
                 color=INK, fontsize=12, loc="left", pad=12)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    rapikan(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)

def figur_acak16(path: Path, agg) -> None:
    """Kemiringan asli -> acak16: kalau naik, model tidak membaca susunan."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6), facecolor=SURFACE)
    titik = []
    for family, method in ORDER:
        a = agg[("asli", family, method)]["auc"]
        b = agg[("acak16", family, method)]["auc"]
        naik = b > a
        warna = JINGGA if naik else BIRU
        ax.plot([0, 1], [a, b], color=warna, linewidth=2,
                marker="o", markersize=8, markerfacecolor=warna,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        titik.append([b, f"{family} / {method}   {b - a:+.3f}", naik])
    titik.sort(key=lambda t: t[0])
    jarak = 0.018
    for i in range(1, len(titik)):
        if titik[i][0] - titik[i - 1][0] < jarak:
            titik[i][0] = titik[i - 1][0] + jarak
    for yt, teks, naik in titik:
        ax.text(1.045, yt, teks, va="center", ha="left", fontsize=9.5,
                color=INK if naik else INK2)
    ax.set_xlim(-0.12, 1.85)
    ax.set_xticks([0, 1], ["crop asli", "petak 4x4 diacak"])
    ax.set_ylabel("AUC pooled", color=INK2, fontsize=9.5)
    ax.set_title("Menghancurkan susunan tubuh tidak menurunkan skor",
                 color=INK, fontsize=12, loc="left", pad=26)
    ax.annotate("jingga = AUC justru NAIK saat bentuk dirusak",
                xy=(0, 1.012), xycoords="axes fraction",
                fontsize=9.5, color=JINGGA, ha="left", va="bottom")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    rapikan(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)

def figur_saturasi(path: Path, csv_path: Path) -> None:
    """Sebaran saturasi mati vs hidup - kenapa satu angka warna sudah cukup."""
    mati, hidup = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row["run_id"] != "ce__hier_addone__s42"
                    or row["intervention"] != "asli"
                    or not row["registry"].endswith("pio_dev_registry.json")):
                continue
            if row["label"] == "mati":
                mati.append(float(row["saturation"]))
            elif row["label"] == "hidup":
                hidup.append(float(row["saturation"]))
    fig, ax = plt.subplots(figsize=(8.6, 3.6), facecolor=SURFACE)
    rng = np.random.default_rng(7)
    ax.scatter(hidup, 1 + rng.normal(0, 0.055, len(hidup)), s=16,
               color=BIRU, alpha=0.28, linewidths=0, zorder=3)
    ax.scatter(mati, rng.normal(0, 0.055, len(mati)), s=52,
               color=JINGGA, alpha=0.95, linewidths=1.4,
               edgecolors=SURFACE, zorder=4)
    ax.set_yticks([0, 1], [f"mati  (n={len(mati)})", f"hidup  (n={len(hidup)})"])
    ax.set_ylim(-0.45, 1.45)
    ax.set_xlabel("saturasi rata-rata crop", color=INK2, fontsize=9.5)
    ax.set_title("Satu angka warna sudah memisahkan sebagian besar ayam mati",
                 color=INK, fontsize=12, loc="left", pad=12)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    rapikan(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)

def main() -> None:
    ap = argparse.ArgumentParser(description="Figur Babak 14")
    ap.add_argument("--json", default="outputs/predictions/fixed_chick.json")
    ap.add_argument("--crops", default="outputs/predictions/fixed_chick_crops.csv")
    ap.add_argument("--outdir", default="outputs/reports")
    args = ap.parse_args()
    agg, sat = baca(Path(args.json))
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    figur_baseline(out / "babak14_vs_baseline.png", agg, sat)
    figur_acak16(out / "babak14_acak16.png", agg)
    figur_saturasi(out / "babak14_saturasi.png", Path(args.crops))
    print("figur ditulis ke", out)

if __name__ == "__main__":
    main()
