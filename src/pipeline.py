"""
PIPELINE UJUNG-KE-UJUNG: gambar -> deteksi ayam -> crop 224x224 -> klasifikasi.

Inilah alur utama yang diminta:
    1. YOLO mendeteksi setiap ayam di dalam gambar.
    2. Setiap bounding box dipotong.
    3. Setiap potongan dibatasi 224x224 (letterbox, rasio aspek dijaga).
    4. Classifier hasil contrastive learning menilai tiap potongan:
       ayam hidup atau ayam mati.
    5. Hasilnya digambar ulang ke gambar asli + disimpan ke JSON.

Jalankan (setelah build_crops.py dan train.py):
    python src/pipeline.py
    python src/pipeline.py --method supcon --source <folder> --conf 0.3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from common import (crop_box, get_device, imread, imwrite, load_config,
                    resolve, save_json, set_seed, to_square)
from dataset import to_tensor
from models import build_model

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# BGR: hijau untuk hidup, merah untuk mati
COLOR = {0: (0, 200, 0), 1: (0, 0, 255)}
NAME = {0: "hidup", 1: "MATI"}


def load_classifier(cfg: dict, method: str, device):
    """Muat classifier hasil pelatihan salah satu metode."""
    ckpt_path = resolve(cfg["output"]["runs_dir"]) / method / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Model '{method}' belum ada: {ckpt_path}\n"
            f"Latih dulu: python src/train.py --method {method}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def classify_crops(model, crops: list[np.ndarray], cfg: dict,
                   device) -> tuple[np.ndarray, np.ndarray]:
    """Klasifikasi sekumpulan crop. Return (label, peluang_mati)."""
    if not crops:
        return np.array([], dtype=int), np.array([], dtype=float)

    size = int(cfg["classifier"]["image_size"])
    mode = cfg["classifier"]["resize_mode"]
    bs = int(cfg["classifier"]["batch_size"])

    labels, probs = [], []
    for i in range(0, len(crops), bs):
        batch = torch.stack([to_tensor(c, size, mode)
                             for c in crops[i:i + bs]]).to(device)
        logit = model.logits(batch)
        p = torch.softmax(logit, dim=1)[:, 1]      # peluang kelas 'mati'
        labels.append(logit.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
    return np.concatenate(labels), np.concatenate(probs)


def run(cfg: dict, method: str, source: str | None = None,
        conf: float | None = None, save_crops: bool = False) -> dict:
    dcfg = cfg["detection"]
    src = Path(source) if source else resolve(dcfg["test_images"])
    conf = float(conf) if conf is not None else float(dcfg["conf"])
    size = int(cfg["classifier"]["image_size"])
    mode = cfg["classifier"]["resize_mode"]
    pad = float(cfg["crops"]["bbox_padding"])

    if not src.exists():
        raise FileNotFoundError(f"Sumber gambar tidak ada: {src}")
    images = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not images:
        raise RuntimeError(f"Tidak ada gambar di {src}")

    device = get_device(cfg.get("device", "auto"))
    from ultralytics import YOLO
    detector = YOLO(str(resolve(dcfg["weights"])))
    model = load_classifier(cfg, method, device)

    out_root = resolve(cfg["output"]["predictions_dir"]) / f"pipeline_{method}"
    print(f"[pipeline] deteksi    : {Path(dcfg['weights']).name}")
    print(f"[pipeline] classifier : {method}")
    print(f"[pipeline] sumber     : {src} ({len(images)} gambar)")
    print(f"[pipeline] device     : {device}  conf={conf}\n")

    records, n_box, n_dead = [], 0, 0

    for img_path in images:
        img = imread(img_path)
        if img is None:
            print(f"  ! gagal baca: {img_path.name}")
            continue

        # --- 1. deteksi ---
        res = detector.predict(source=img, imgsz=int(dcfg["imgsz"]), conf=conf,
                               iou=float(dcfg["iou"]),
                               max_det=int(dcfg["max_det"]),
                               device=0 if device.type == "cuda" else "cpu",
                               verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy().tolist()
        scores = res.boxes.conf.cpu().numpy().tolist()

        # --- 2 & 3. crop tiap bbox, batasi ke 224x224 ---
        crops, kept = [], []
        for i, b in enumerate(boxes):
            c = crop_box(img, b, pad)
            if c is None:
                continue
            crops.append(to_square(c, size, mode))
            kept.append(i)

        # --- 4. klasifikasi tiap crop ---
        labels, probs = classify_crops(model, crops, cfg, device)

        # --- 5. gambar ulang hasilnya ---
        vis = img.copy()
        dets = []
        for k, idx in enumerate(kept):
            b, lab, p = boxes[idx], int(labels[k]), float(probs[k])
            dets.append({"id": idx, "bbox": [round(v, 2) for v in b],
                         "det_conf": round(float(scores[idx]), 4),
                         "class": NAME[lab], "label": lab,
                         "prob_dead": round(p, 4)})

            x1, y1, x2, y2 = map(int, b)
            col = COLOR[lab]
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            txt = f"{NAME[lab]} {p:.2f}" if lab == 1 else NAME[lab]
            cv2.putText(vis, txt, (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

            if save_crops:
                imwrite(out_root / "crops" /
                        f"{img_path.stem}_{idx:03d}_{NAME[lab]}.jpg", crops[k])

        n_mati = int((labels == 1).sum()) if len(labels) else 0
        n_box += len(dets)
        n_dead += n_mati

        # Ringkasan di pojok gambar
        cv2.rectangle(vis, (0, 0), (250, 26), (0, 0, 0), -1)
        cv2.putText(vis, f"ayam:{len(dets)}  MATI:{n_mati}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        imwrite(out_root / "vis" / f"{img_path.stem}.jpg", vis)
        records.append({"image": str(img_path), "name": img_path.name,
                        "num_chickens": len(dets), "num_dead": n_mati,
                        "detections": dets})
        print(f"  {img_path.name:<28} ayam {len(dets):>3}  ->  mati {n_mati:>3}")

    out_json = out_root / "pipeline_results.json"
    save_json({"detector": str(resolve(dcfg["weights"])), "classifier": method,
               "source": str(src), "conf": conf,
               "num_images": len(records), "num_chickens": n_box,
               "num_dead": n_dead, "results": records}, out_json)

    print(f"\n[pipeline] {len(records)} gambar, {n_box} ayam terdeteksi, "
          f"{n_dead} diklasifikasi MATI")
    print(f"[pipeline] JSON   : {out_json}")
    print(f"[pipeline] visual : {out_root / 'vis'}")
    return {"num_images": len(records), "num_chickens": n_box,
            "num_dead": n_dead}


def main():
    ap = argparse.ArgumentParser(
        description="Pipeline ujung-ke-ujung: deteksi -> crop -> klasifikasi")
    ap.add_argument("--config", default=None)
    ap.add_argument("--method", default="supcon",
                    choices=["selfcon", "supcon", "ce"],
                    help="classifier hasil metode mana yang dipakai")
    ap.add_argument("--source", default=None, help="folder gambar")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--save-crops", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    run(cfg, a.method, a.source, a.conf, a.save_crops)


if __name__ == "__main__":
    main()
