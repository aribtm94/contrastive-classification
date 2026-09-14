"""Gabungkan history run development dan buat grafik loss per tahap."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import load_config, resolve

TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"


def read_history(runs_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(runs_dir.glob("*__s*/history.csv")):
        run_id = path.parent.name
        method, aug, seed_text = run_id.rsplit("__", 2)
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append({"run_id": run_id, "method": method, "aug": aug,
                             "seed": int(seed_text.removeprefix("s")), **row})
    if not rows:
        raise FileNotFoundError(f"history.csv tidak ditemukan di {runs_dir}")
    return rows


def write_combined(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, rows: list[dict], title: str) -> None:
    """Small multiples: tiap panel satu objective, satu sumbu, dua seri."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["stage"])].append(row)
    keys = sorted(grouped)
    ncols = 2
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        by_seed = defaultdict(list)
        for row in grouped[key]:
            by_seed[row["seed"]].append(row)
        for seed, values in sorted(by_seed.items()):
            values.sort(key=lambda r: int(r["epoch"]))
            epoch = [int(r["epoch"]) for r in values]
            ax.plot(epoch, [float(r["train_loss"]) for r in values],
                    color=TRAIN_COLOR, linewidth=2, alpha=0.32)
            ax.plot(epoch, [float(r["val_loss"]) for r in values],
                    color=VAL_COLOR, linewidth=2, alpha=0.32)
        common_epochs = sorted({int(r["epoch"]) for r in grouped[key]})
        for field, color, label in (("train_loss", TRAIN_COLOR, "Train"),
                                    ("val_loss", VAL_COLOR, "Validation")):
            means = []
            for epoch in common_epochs:
                vals = [float(r[field]) for r in grouped[key]
                        if int(r["epoch"]) == epoch]
                means.append(float(np.mean(vals)))
            ax.plot(common_epochs, means, color=color, linewidth=2.5,
                    marker="o", markersize=4, markevery=max(1, len(means) // 6),
                    label=label)
        method, stage = key
        objective = "NT-Xent" if method == "selfcon" and stage == "contrastive" \
            else ("SupCon" if method == "supcon" and stage == "contrastive"
                  else "Weighted CE")
        ax.set_title(f"{method} · {stage} · {objective}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Objective loss")
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, ncol=2)
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=14, fontweight="semibold")
    fig.text(0.5, 0.005,
             "Garis tebal = rata-rata seed; garis transparan = masing-masing seed.",
             ha="center", color="#52514e")
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Laporan loss development per epoch")
    ap.add_argument("--config", required=True)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rows = read_history(resolve(cfg["output"]["runs_dir"]))
    history = resolve(cfg["protocol"]["history_csv"])
    write_combined(history, rows)
    suffix = Path(args.config).stem.replace("config_", "")
    destination = resolve(args.plot or f"outputs/reports/{suffix}_loss.png")
    plot(destination, rows, f"Loss per epoch — {suffix}")
    print(f"history: {history}")
    print(f"plot   : {destination}")


if __name__ == "__main__":
    main()
