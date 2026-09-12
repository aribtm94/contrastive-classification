"""
Berapa skor yang bisa dicapai TANPA melihat isi gambar sama sekali?

Beberapa ciri gangguan (nuisance) diuji sebagai "model" satu-angka:

  ketajaman    varians Laplacian crop tersimpan - seberapa kabur
  rasio_bbox   sisi panjang/pendek bbox; terlihat lewat lebar bantalan letterbox
  saturasi     rata-rata S (HSV) - warna kandang vs latar studio
  terang       rata-rata V (HSV)
  hue          rata-rata H (HSV)
  std_terang   simpangan baku V - kontras dalam crop
  ukuran_bbox  sisi pendek bbox ASLI, dibaca dari manifest (BUKAN dari piksel)

Tidak satu pun dari ciri ini tahu apa pun tentang ayam. Kalau salah satunya
sudah mendekati skor classifier, berarti skor classifier itu sebagian besar
bukan hasil mengenali ayam mati, melainkan hasil menebak crop ini berasal dari
dataset mana.

DUA ARAH DIUJI. Ciri dengan AUC 0.12 sama kuatnya dengan AUC 0.88 - model tidak
peduli tandanya, ia tinggal membalik bobotnya. Karena itu ambang dicari untuk
kedua arah (>= t dan <= t) dan yang dilaporkan adalah AUC TERARAH,
yaitu max(AUC, 1-AUC). Versi pertama skrip ini hanya menguji satu arah dan
karena itu menyatakan rasio bbox "bukan jalan pintas" padahal terarahnya 0.885.

Ambang dipilih di TRAIN lalu dipakai APA ADANYA di val dan test - memilih
ambang di test akan membuat baseline ini terlihat lebih kuat dari seharusnya.

Hati-hati menafsirkan ukuran_bbox: ia dihitung dari metadata, bukan dari piksel
crop yang tersimpan (semua crop sudah 224x224). AUC-nya yang tinggi TIDAK
berarti model bisa memakainya - ia mengukur seberapa jauh label masih
berbarengan dengan asal data. Ciri lainnya benar-benar terlihat di piksel yang
masuk ke model.

Jalankan:
    python src/eval_shortcut_baseline.py --crops data/crops
    python src/eval_shortcut_baseline.py --crops data/crops_pio
    python src/eval_shortcut_baseline.py --crops data/crops_pio_eq
"""
from __future__ import annotations

import argparse
import csv
import json

import cv2
import numpy as np

from common import imread, resolve, save_json

# nama ciri -> (dari piksel crop?, cara menghitungnya)
#   True  = benar-benar terlihat oleh model
#   False = metadata, hanya mengukur collinearity label vs asal data
FEATURES: list[tuple[str, bool]] = [
    ("ketajaman", True), ("rasio_bbox", True), ("saturasi", True),
    ("terang", True), ("hue", True), ("std_terang", True),
    ("ukuran_bbox", False),
]


def features(crops_dir: str) -> dict:
    """Baca manifest, kembalikan (matriks ciri, label) per split."""
    root = resolve(crops_dir)
    acc: dict[str, list] = {}
    for r in csv.DictReader(open(root / "manifest.csv", encoding="utf-8")):
        img = imread(str(root / r["path"]))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        x1, y1, x2, y2 = json.loads(r["bbox"])
        w, h = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
        acc.setdefault(r["split"], []).append((
            [float(cv2.Laplacian(gray, cv2.CV_64F).var()),
             float(max(w, h) / min(w, h)),
             float(hsv[:, :, 1].mean()),
             float(hsv[:, :, 2].mean()),
             float(hsv[:, :, 0].mean()),
             float(hsv[:, :, 2].std()),
             float(min(w, h))],
            int(r["label"])))
    return {k: (np.array([a for a, _ in v]), np.array([b for _, b in v]))
            for k, v in acc.items()}


def pick_rule(x: np.ndarray, y: np.ndarray) -> tuple[float, int, float]:
    """Ambang + arah terbaik di train. sign +1 berarti 'mati jika x >= t'."""
    from sklearn.metrics import balanced_accuracy_score
    best = (float(x.min()), 1, -1.0)
    for t in np.unique(x):
        for sign in (1, -1):
            pred = (x >= t) if sign > 0 else (x <= t)
            b = float(balanced_accuracy_score(y, pred.astype(int)))
            if b > best[2]:
                best = (float(t), sign, b)
    return best


def main():
    ap = argparse.ArgumentParser(
        description="Baseline jalan pintas: skor tanpa model sama sekali")
    ap.add_argument("--crops", default="data/crops",
                    help="folder berisi manifest.csv")
    ap.add_argument("--json", default=None, help="simpan hasil ke berkas JSON")
    a = ap.parse_args()

    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    f = features(a.crops)
    if "train" not in f:
        raise SystemExit(f"Tidak ada split train di {a.crops}/manifest.csv")

    print(f"[baseline] {a.crops}\n")
    print("{:<13}{:>6}{:>12}{:>22}{:>22}".format(
        "ciri", "arah", "train bacc", "val bacc/AUC-terarah",
        "test bacc/AUC-terarah"))
    out = {"crops": a.crops, "features": {}}

    for idx, (name, visible) in enumerate(FEATURES):
        xtr, ytr = f["train"][0][:, idx], f["train"][1]
        t, sign, btr = pick_rule(xtr, ytr)
        rec = {"threshold": t, "sign": sign, "from_pixels": visible,
               "train_bacc": btr}
        cells = ""
        for sp in ("val", "test"):
            if sp not in f:
                cells += "{:>22}".format("-")
                continue
            x, y = f[sp][0][:, idx], f[sp][1]
            pred = (x >= t) if sign > 0 else (x <= t)
            b = float(balanced_accuracy_score(y, pred.astype(int)))
            auc = float(roc_auc_score(y, x)) if len(set(y.tolist())) > 1 else 0.5
            dir_auc = max(auc, 1.0 - auc)       # arah tidak penting bagi model
            rec[f"{sp}_bacc"] = b
            rec[f"{sp}_auc_raw"] = auc
            rec[f"{sp}_auc"] = dir_auc
            cells += "{:>22}".format(f"{b:.4f} / {dir_auc:.4f}")
        mark = " " if visible else "*"
        print("{:<13}{:>6}{:>12.4f}{}{}".format(
            name + mark, ">=" if sign > 0 else "<=", btr, cells, ""))
        out["features"][name] = rec

    sharp, side, y = (f["test"][0][:, 0], f["test"][0][:, 6], f["test"][1])
    print(f"\n  test n={len(y)}  "
          f"mati: sisi {np.median(side[y == 1]):>5.0f}px "
          f"tajam {np.median(sharp[y == 1]):>6.0f}   "
          f"hidup: sisi {np.median(side[y == 0]):>5.0f}px "
          f"tajam {np.median(sharp[y == 0]):>6.0f}")

    vis = [(n, out["features"][n]["test_auc"]) for n, v in FEATURES if v]
    top, top_auc = max(vis, key=lambda r: r[1])
    print(f"\n  AUC-terarah 0.5 = ciri ini tidak memberi informasi apa pun.")
    print(f"  Jalan pintas terkuat yang TERLIHAT di piksel: {top} "
          f"(test AUC-terarah {top_auc:.4f}).")
    print("  Classifier harus melewati angka itu dulu sebelum skornya boleh "
          "dibaca\n  sebagai kemampuan mengenali ayam mati.")
    print("  * = dihitung dari manifest, bukan dari piksel crop.")

    if a.json:
        save_json(out, resolve(a.json))
        print(f"\n[baseline] JSON: {resolve(a.json)}")


if __name__ == "__main__":
    main()
