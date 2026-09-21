"""
Evaluasi LINTAS SUB-DOMAIN: model dilatih di satu sub-domain, diuji di
sub-domain yang tidak pernah dilihatnya sama sekali.

Kenapa skrip ini ada: `train.py` hanya membaca satu split bernama `test`
(`src/train.py:650`). Pada susunan SDNET2018 ada dua split hold-out tambahan -
`test_wall` dan `test_pave` - yang isinya foto beton dari permukaan yang
berbeda. Itu persis pola generalisasi yang dipersoalkan di skripsi: latih di
satu domain, pakai di domain lain.

Aturan yang dipegang skrip ini:

  - Ambang DIAMBIL dari `result.json` -> `threshold_from_validation`, yaitu
    ambang yang sudah dikalibrasi di validation saat training. Tidak pernah
    dihitung ulang di data uji. Menghitung ulang ambang di data uji berarti
    memakai jawaban untuk menyetel soal.
  - Tidak ada pemilihan model, epoch, atau seed berdasarkan hasil di sini.
    Seluruh checkpoint yang ada di folder run dievaluasi, lalu dilaporkan
    apa adanya beserta simpangan baku antar-seed.
  - Metrik dihitung ulang memakai fungsi yang sama dengan training
    (`train.compute_metrics`, `train.evaluate`), bukan implementasi baru.

Jalankan:
    python src/eval_lintas_domain.py --config configs/config_sdnet.yaml \
        --runs outputs/runs_sdnet --splits test_wall,test_pave
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import get_device, load_config, resolve, save_json
from dataset import ClassificationDataset, read_manifest
from pipeline import load_classifier
# _loader dipakai ulang supaya batch size, num_workers, dan benih pengacakan
# persis sama dengan yang dipakai saat evaluasi di dalam training.
from train import _loader, evaluate, metrics_at


def daftar_run(runs_dir: Path) -> list:
    """
    Semua folder run yang punya checkpoint sekaligus hasil validationnya.

    Hanya folder bernama `<metode>__<aug>__s<seed>` yang diambil. train.py
    juga menulis cermin legacy tanpa akhiran seed (`outputs/runs_*/ce`,
    lihat `mirror_legacy` di src/train.py:698) yang isinya salinan run seed
    pertama. Kalau ikut dibaca, run yang sama terhitung dua kali dan
    simpangan baku antar-seed jadi mengecil palsu.
    """
    if not runs_dir.exists():
        raise SystemExit(f"folder run tidak ada: {runs_dir}")
    return sorted(p.name for p in runs_dir.iterdir()
                  if re.search(r"__s\d+$", p.name)
                  and (p / "model.pt").exists()
                  and (p / "result.json").exists())


def tanpa_seed(tag: str) -> str:
    """"supcon__stacked_randaug__s43" -> "supcon__stacked_randaug"."""
    return re.sub(r"__s\d+$", "", tag)


def evaluasi(model, cfg: dict, split: str, device, seed: int) -> tuple:
    rows = read_manifest(cfg, split)
    if not rows:
        raise SystemExit(f"split '{split}' kosong di manifest")
    ds = ClassificationDataset(rows, cfg, train=False, seed=seed)
    loader = _loader(ds, cfg, shuffle=False, seed=seed + 907)
    m, y_true, y_score = evaluate(model, loader, device)
    return len(rows), m, y_true, y_score


def rerata(nilai: list) -> tuple:
    a = np.asarray([v for v in nilai if v is not None and not np.isnan(v)],
                   dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std())


def main():
    ap = argparse.ArgumentParser(
        description="Uji checkpoint pada split sub-domain hold-out")
    ap.add_argument("--config", default="configs/config_sdnet.yaml")
    ap.add_argument("--runs", default=None,
                    help="folder checkpoint; default dari config")
    ap.add_argument("--splits", default="test_wall,test_pave",
                    help="daftar nama split hold-out, dipisah koma")
    ap.add_argument("--json", default="outputs/predictions/lintas_domain_sdnet.json")
    a = ap.parse_args()

    cfg = load_config(a.config)
    cfg["_config_path"] = a.config
    if a.runs:
        cfg["output"] = dict(cfg["output"])
        cfg["output"]["runs_dir"] = a.runs
    runs_dir = resolve(cfg["output"]["runs_dir"])
    device = get_device(cfg.get("device", "auto"))
    splits = [s.strip() for s in a.splits.split(",") if s.strip()]

    tags = daftar_run(runs_dir)
    if not tags:
        raise SystemExit(f"tidak ada checkpoint di {runs_dir}")

    print(f"[lintas] runs {runs_dir}")
    print(f"[lintas] {len(tags)} checkpoint, split: {', '.join(splits)}\n")

    hasil = {"config": a.config, "runs_dir": str(runs_dir),
             "splits": splits, "per_run": {}}

    for tag in tags:
        res = json.loads((runs_dir / tag / "result.json").read_text("utf-8"))
        thr = float(res["threshold_from_validation"])
        model = load_classifier(cfg, tag, device)
        catatan = {"threshold_from_validation": thr,
                   "val_bacc": res.get("validation", {}).get(
                       "balanced_accuracy")}
        for split in splits:
            n, m05, y_true, y_score = evaluasi(model, cfg, split, device,
                                               int(res.get("seed", 42)))
            catatan[split] = {"n": n, "at_0.5": m05,
                              "at_tau_val": metrics_at(y_true, y_score, thr)}
        hasil["per_run"][tag] = catatan
        ringkas = "  ".join(
            f"{s}: bacc@tau {catatan[s]['at_tau_val']['balanced_accuracy']:.4f}"
            f" auc {catatan[s]['at_0.5'].get('roc_auc', float('nan')):.4f}"
            for s in splits)
        print(f"  {tag:34s} {ringkas}")

    # --- rata-rata tiga seed; satu seed tidak boleh dijadikan kesimpulan ---
    kelompok = defaultdict(list)
    for tag, c in hasil["per_run"].items():
        kelompok[tanpa_seed(tag)].append(c)

    print("\nRata-rata antar-seed (+/- simpangan baku)")
    lebar = max(len(k) for k in kelompok)
    print(f"{'pipeline':<{lebar}}  {'split':<10} {'bacc@tau':>16} {'AUC':>16}")
    hasil["rata_rata"] = {}
    for nama in sorted(kelompok):
        hasil["rata_rata"][nama] = {"n_seed": len(kelompok[nama])}
        for split in splits:
            b, bs = rerata([c[split]["at_tau_val"]["balanced_accuracy"]
                            for c in kelompok[nama]])
            u, us = rerata([c[split]["at_0.5"].get("roc_auc")
                            for c in kelompok[nama]])
            hasil["rata_rata"][nama][split] = {
                "bacc_at_tau_mean": round(b, 4), "bacc_at_tau_std": round(bs, 4),
                "auc_mean": round(u, 4), "auc_std": round(us, 4)}
            print(f"{nama:<{lebar}}  {split:<10} "
                  f"{b:>9.4f} +/-{bs:5.4f} {u:>9.4f} +/-{us:5.4f}")

    print("\n  Ambang berasal dari validation domain latih, tidak pernah")
    print("  dihitung ulang di sini. Bandingkan dengan lantai jalan pintas")
    print("  di outputs/reports/shortcut_sdnet.json sebelum menyimpulkan.")

    if a.json:
        save_json(hasil, resolve(a.json))
        print(f"\n[lintas] JSON: {resolve(a.json)}")


if __name__ == "__main__":
    main()
