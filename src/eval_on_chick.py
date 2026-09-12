"""
Uji classifier ujung-ke-ujung pada dataset chick (CCTV tampak atas).

Ini uji yang sesungguhnya: gambar utuh -> detektor -> crop -> classifier.
Crop ayam mati dikenali lewat mask segmentasi (IoU >= 0.5 dengan box acuan),
sisanya dianggap ayam lain. Yang diukur adalah apakah classifier memberi skor
'mati' lebih tinggi pada 22 crop ayam mati dibanding ~1193 crop lainnya.

AUC dipakai, bukan akurasi: kelasnya timpang 22 lawan 1193, sehingga menebak
'hidup' untuk semuanya sudah memberi akurasi 98%. AUC 0.5 berarti menebak.

Mask chick TIDAK PERNAH dipakai melatih apa pun - hanya untuk menandai crop
mana yang ayam mati saat mengukur.

Praproses crop uji mengikuti config yang diberikan, termasuk
crops.equalize_resolution - model yang dilatih pada crop resolusi-disamakan
harus diuji dengan praproses yang sama, kalau tidak angkanya tidak berarti.

Jalankan:
    python src/eval_on_chick.py --runs outputs/runs
    python src/eval_on_chick.py --runs outputs/runs_pio_alive
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np

from build_crops import equalize
from common import (crop_box, get_device, imread, load_config, resolve,
                    save_json, to_square)
from eval_detect_masks import iou, masks_to_boxes, pair_files
from pipeline import classify_crops, load_classifier


def score_run(cfg, method, runs_dir, pairs, detector, device):
    """Kembalikan skor p(mati) untuk crop ayam mati dan crop lainnya."""
    cfg = dict(cfg)
    cfg["output"] = dict(cfg["output"])
    cfg["output"]["runs_dir"] = str(runs_dir)
    model = load_classifier(cfg, method, device)

    dcfg = cfg["detection"]
    ccfg = cfg["crops"]
    pad = float(ccfg["bbox_padding"])
    size = int(cfg["classifier"]["image_size"])
    mode = cfg["classifier"]["resize_mode"]

    s_dead, s_other = [], []
    for _stem, img_path, seg_path in pairs:
        img = imread(img_path)
        gts = masks_to_boxes(seg_path)
        if img is None or not gts:
            continue

        res = detector.predict(img, imgsz=int(dcfg["imgsz"]),
                               conf=float(dcfg["conf"]),
                               iou=float(dcfg["iou"]),
                               max_det=int(dcfg["max_det"]),
                               verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy().tolist()

        crops, kept = [], []
        for i, b in enumerate(boxes):
            c = crop_box(img, b, pad)
            if c is None:
                continue
            # praproses harus SAMA seperti saat melatih: kalau crop
            # latih disamakan resolusinya, crop uji juga.
            crops.append(to_square(equalize(c, ccfg), size, mode))
            kept.append(i)
        if not crops:
            continue

        _labels, probs = classify_crops(model, crops, cfg, device)
        probs = np.asarray(probs)
        probs = probs[:, 1] if probs.ndim > 1 else probs

        dead_idx = set()
        for gt in gts:
            best_j, best_v = -1, 0.0
            for j, k in enumerate(kept):
                v = iou(gt, boxes[k])
                if v > best_v:
                    best_j, best_v = j, v
            if best_j >= 0 and best_v >= 0.5:
                dead_idx.add(best_j)

        for j in range(len(kept)):
            (s_dead if j in dead_idx else s_other).append(float(probs[j]))

    return np.array(s_dead), np.array(s_other)


def main():
    ap = argparse.ArgumentParser(
        description="Uji classifier ujung-ke-ujung pada dataset chick")
    ap.add_argument("--config", default=None)
    ap.add_argument("--runs", default="outputs/runs",
                    help="folder berisi checkpoint (mis. outputs/runs_pio_alive)")
    ap.add_argument("--root", default="C:/Arib/CCTV/patnet-pure/dataset/chick")
    ap.add_argument("--methods", default=None,
                    help="daftar dipisah koma; default: semua yang ada")
    a = ap.parse_args()

    from sklearn.metrics import roc_auc_score
    from ultralytics import YOLO

    cfg = load_config(a.config)
    runs_dir = resolve(a.runs)
    pairs = pair_files(Path(a.root))
    if not pairs:
        raise SystemExit(f"Tidak ada pasangan gambar+mask di {a.root}")

    if a.methods:
        methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    else:
        methods = sorted(
            os.path.basename(os.path.dirname(p))
            for p in glob.glob(str(runs_dir / "*" / "model.pt")))
    if not methods:
        raise SystemExit(f"Tidak ada model.pt di {runs_dir}")

    device = get_device(cfg.get("device", "auto"))
    detector = YOLO(str(resolve(cfg["detection"]["weights"])))

    print(f"[chick] checkpoint : {runs_dir}")
    print(f"[chick] data       : {len(pairs)} gambar CCTV")
    print(f"[chick] imgsz      : {cfg['detection']['imgsz']}\n")
    print("{:<34}{:>12}{:>12}{:>9}".format(
        "checkpoint", "p(mati)mati", "p(mati)lain", "AUC"))

    out = []
    for m in methods:
        try:
            sd, so = score_run(cfg, m, runs_dir, pairs, detector, device)
        except Exception as e:                       # checkpoint rusak/beda bentuk
            print(f"{m:<34}  ! {type(e).__name__}: {e}")
            continue
        if len(sd) == 0 or len(so) == 0:
            print(f"{m:<34}  ! tidak ada crop untuk dinilai")
            continue
        y = np.r_[np.ones(len(sd)), np.zeros(len(so))]
        auc = float(roc_auc_score(y, np.r_[sd, so]))
        print(f"{m:<34}{sd.mean():>12.3f}{so.mean():>12.3f}{auc:>9.3f}")
        out.append({"method": m, "n_dead": int(len(sd)), "n_other": int(len(so)),
                    "p_dead": float(sd.mean()), "p_other": float(so.mean()),
                    "auc": auc})

    if out:
        best = max(out, key=lambda r: r["auc"])
        print(f"\n[chick] terbaik: {best['method']} AUC {best['auc']:.3f} "
              f"(n={best['n_dead']} mati / {best['n_other']} lain)")
        print("[chick] AUC 0.5 = menebak; kelas timpang, jadi akurasi tidak "
              "berarti di sini.")
        # nama JSON ikut folder checkpoint DAN praproses, supaya dua
        # evaluasi yang berbeda tidak saling menimpa
        tag = runs_dir.name.replace("runs_", "").replace("runs", "closeup")
        eq = cfg["crops"].get("equalize_resolution") or {}
        if eq.get("enabled"):
            tag += f"_pra{int(eq.get('target_short_side', 96))}"
        dest = (resolve(cfg["output"]["predictions_dir"])
                / f"eval_on_chick_{tag}.json")
        save_json({"runs_dir": str(runs_dir), "n_images": len(pairs),
                   "equalize": cfg["crops"].get("equalize_resolution"),
                   "results": out}, dest)
        print(f"[chick] JSON    : {dest}")


if __name__ == "__main__":
    main()
