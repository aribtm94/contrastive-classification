"""Buat laporan dan plot benchmark test ayam tetap dari JSON evaluator."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from common import resolve

COLORS = {"absolute": "#2a78d6", "relative_clean": "#eb6834",
          "relative_operational": "#1baf7a"}


def aggregate(records: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for row in records:
        family = "eq48" if "eq_dev" in row["registry"] else "asli"
        for scorer in COLORS:
            buckets[(family, row["method"], scorer)].append(row[scorer])
    out = []
    keys = ("pooled_ap", "pooled_auc", "macro_ap", "macro_auc",
            "recall_at_1", "recall_at_3", "recall_at_5", "mrr")
    for (family, method, scorer), rows in sorted(buckets.items()):
        out.append({"family": family, "method": method, "scorer": scorer,
                    **{key: {"mean": float(np.mean([r[key] for r in rows])),
                             "std": float(np.std([r[key] for r in rows], ddof=1))}
                       for key in keys}})
    return out


def fmt(metric: dict) -> str:
    return f"{metric['mean']:.3f} +/- {metric['std']:.3f}"


def write_report(path, data: dict, aggregates: list[dict]) -> None:
    lines = [
        "# Benchmark Test Ayam Tetap", "",
        "## Kontrak", "",
        "Train dan validation hanya memakai ayam hidup PIO + ayam mati Roboflow. "
        "Seluruh 18 frame ayam/chick dipakai sebagai test. Tidak ada angka test "
        "yang dipakai memilih epoch, threshold, seed, checkpoint, atau scorer.", "",
        "Benchmark ini **retrospektif**, karena dataset chick sudah pernah dibaca "
        "dalam eksperimen historis. Validation development juga memiliki "
        "`label = domain`; hasil tinggi belum otomatis berarti model mengenali ayam mati.", "",
        "Kohort clean berisi 943 ayam valid (22 mati, 921 hidup). Operational "
        "ranking memasukkan 272 crop `bukan` sebagai nuisance, bukan sebagai ayam hidup.", "",
        "## Hasil asli, rata-rata tiga seed", "",
        "| varian | metode | scorer | AP pooled | AUC pooled | AP macro | AUC macro | Recall@3 | MRR |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(f"| {row['family']} | {row['method']} | {row['scorer']} | "
                     f"{fmt(row['pooled_ap'])} | {fmt(row['pooled_auc'])} | "
                     f"{fmt(row['macro_ap'])} | {fmt(row['macro_auc'])} | "
                     f"{fmt(row['recall_at_3'])} | {fmt(row['mrr'])} |")
    lines += ["", "## Audit crop bukan", "",
              "Jumlah berikut dirata-ratakan atas tiga seed; tiap baris menunjukkan "
              "berapa dari 272 crop bukan ayam memicu alarm/ranking operational.", "",
              "| varian | metode | >0.5 | >tau validation | top-1 | top-3 | top-5 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    records = data["interventions"]["asli"]
    buckets = defaultdict(list)
    for row in records:
        family = "eq48" if "eq_dev" in row["registry"] else "asli"
        buckets[(family, row["method"])].append(row["bukan_audit"])
    for (family, method), rows in sorted(buckets.items()):
        mean = lambda key: np.mean([r[key] for r in rows])
        lines.append(f"| {family} | {method} | {mean('bukan_above_0_5'):.1f} | "
                     f"{mean('bukan_above_tau_val'):.1f} | {mean('bukan_in_top_1'):.1f} | "
                     f"{mean('bukan_in_top_3'):.1f} | {mean('bukan_in_top_5'):.1f} |")
    if "acak16" in data["interventions"]:
        lines += ["", "## Sensitivitas acak16", "",
                  "Acak16 versi baru deterministik, tanpa petak tetap, dan mencakup "
                  "seluruh piksel. Hasil ini hanya mengukur sensitivitas terhadap "
                  "susunan global; ia tidak membuktikan penurunan berasal eksklusif dari pose.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot(path, aggregates: list[dict]) -> None:
    methods = ["selfcon", "supcon", "ce"]
    scorers = list(COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, family in zip(axes, ("asli", "eq48")):
        x = np.arange(len(methods)); width = 0.24
        for offset, scorer in enumerate(scorers):
            rows = {(r["method"], r["scorer"]): r for r in aggregates
                    if r["family"] == family}
            means = [rows[(method, scorer)]["pooled_ap"]["mean"] for method in methods]
            stds = [rows[(method, scorer)]["pooled_ap"]["std"] for method in methods]
            ax.bar(x + (offset - 1) * width, means, width=width - 0.02,
                   yerr=stds, color=COLORS[scorer], label=scorer, capsize=3)
        ax.set_title(f"PIO {family}")
        ax.set_xticks(x, methods)
        ax.set_ylabel("Average Precision pooled")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    fig.suptitle("Classifier absolut vs anomali relatif — benchmark ayam")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Laporan benchmark ayam tetap")
    ap.add_argument("--json", default="outputs/predictions/fixed_chick.json")
    ap.add_argument("--report", default="outputs/reports/fixed_chick.md")
    ap.add_argument("--plot", default="outputs/reports/fixed_chick.png")
    args = ap.parse_args()
    with open(resolve(args.json), encoding="utf-8") as f:
        data = json.load(f)
    aggregates = aggregate(data["interventions"]["asli"])
    write_report(resolve(args.report), data, aggregates)
    plot(resolve(args.plot), aggregates)
    print(f"report: {resolve(args.report)}")
    print(f"plot  : {resolve(args.plot)}")


if __name__ == "__main__":
    main()
