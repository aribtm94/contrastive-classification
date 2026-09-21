"""
Ringkas 9 run satu lengan dev2 dari result.json langsung.

Dipakai selagi sweep berjalan (development_comparison.json baru ditulis di
akhir, dan namanya dipakai bersama antar lengan - lihat domain_mati_kedua.md
bagian 4h). Skrip ini hanya MEMBACA.

    python outputs/tambalan/ringkas_lengan.py outputs/runs_pio_dev2_eq
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def main() -> None:
    akar = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/runs_pio_dev2_eq")
    per_metode: dict[str, list[dict]] = {}
    for hasil in sorted(akar.glob("*/result.json")):
        r = json.loads(hasil.read_text(encoding="utf-8"))
        v = r["validation"]
        per_metode.setdefault(r["method"], []).append({
            "seed": r["seed"], "auc": v["roc_auc"], "bacc": v["balanced_accuracy"],
            "epoch": v["epoch"], "salah": v["fp"] + v["fn"], "n": v["n"],
            "tau": r["threshold_from_validation"], "detik": r["seconds"]})

    print(f"\n=== {akar} ===")
    total = 0.0
    for metode in ("selfcon", "supcon", "ce"):
        runs = sorted(per_metode.get(metode, []), key=lambda x: x["seed"])
        if not runs:
            print(f"{metode:8s} belum ada")
            continue
        for x in runs:
            print(f"  {metode:8s} s{x['seed']}  auc={x['auc']:.4f} "
                  f"bacc={x['bacc']:.4f} ep={x['epoch']:2d} "
                  f"salah={x['salah']}/{x['n']} tau={x['tau']:.4f} "
                  f"{x['detik']:.0f}s")
            total += x["detik"]
        if len(runs) >= 2:
            sd = lambda k: statistics.stdev([x[k] for x in runs])
            mn = lambda k: statistics.mean([x[k] for x in runs])
            print(f"  {metode:8s} RATA auc={mn('auc'):.4f}+/-{sd('auc'):.4f} "
                  f"bacc={mn('bacc'):.4f}+/-{sd('bacc'):.4f} "
                  f"epoch={[x['epoch'] for x in runs]} "
                  f"salah={[x['salah'] for x in runs]}")
    n = sum(len(v) for v in per_metode.values())
    print(f"\n{n}/9 run, total {total:.0f} detik")


if __name__ == "__main__":
    main()
