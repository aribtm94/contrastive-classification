"""
TAHAP 1 - Deteksi objek ayam dengan YOLO.

Model yang dipakai adalah hasil terbaik dari eksperimen di
'generalisasi-ayam-skripsi' (yolov8m, mAP50-95 = 0.763).

Script ini menjalankan deteksi pada folder gambar uji, menyimpan
setiap bounding box ke JSON, memotong tiap box jadi crop 224x224,
dan menggambar visualisasi hasil deteksi.

Jalankan:
    python src/detect.py
    python src/detect.py --source <folder_gambar> --conf 0.3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from common import (ROOT, crop_box, get_device, imread, imwrite, load_config,
                    resolve, save_json, set_seed, to_square)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)


def run_detection(cfg: dict, source: str | None = None,
                  conf: float | None = None, save_crops: bool = True) -> dict:
    dcfg = cfg["detection"]
    weights = resolve(dcfg["weights"])
    src = Path(source) if source else resolve(dcfg["test_images"])
    conf = float(conf) if conf is not None else float(dcfg["conf"])

    if not weights.exists():
        raise FileNotFoundError(f"Bobot YOLO tidak ditemukan: {weights}")
    if not src.exists():
        raise FileNotFoundError(f"Folder gambar uji tidak ditemukan: {src}")

    images = list_images(src)
    if not images:
        raise RuntimeError(f"Tidak ada gambar di {src}")

    out_root = resolve(cfg["output"]["predictions_dir"])
    vis_dir = out_root / "detect_vis"
    crop_dir = out_root / "detect_crops"

    # Import di sini supaya modul lain tidak wajib punya ultralytics
    from ultralytics import YOLO

    device = get_device(cfg.get("device", "auto"))
    print(f"[detect] model   : {weights.name}")
    print(f"[detect] sumber  : {src}  ({len(images)} gambar)")
    print(f"[detect] device  : {device}  conf={conf}  imgsz={dcfg['imgsz']}")

    model = YOLO(str(weights))

    records: list[dict] = []
    n_boxes = 0

    for img_path in images:
        img = imread(img_path)
        if img is None:
            print(f"  ! gagal baca: {img_path.name}")
            continue

        res = model.predict(
            source=img,
            imgsz=int(dcfg["imgsz"]),
            conf=conf,
            iou=float(dcfg["iou"]),
            max_det=int(dcfg["max_det"]),
            device=0 if device.type == "cuda" else "cpu",
            verbose=False,
        )[0]

        boxes = res.boxes.xyxy.cpu().numpy().tolist()
        scores = res.boxes.conf.cpu().numpy().tolist()

        vis = img.copy()
        dets = []
        for i, (b, s) in enumerate(zip(boxes, scores)):
            dets.append({"id": i, "bbox": [round(v, 2) for v in b],
                         "conf": round(float(s), 4)})

            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(vis, f"{s:.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

            # Tiap bbox dipotong lalu dibatasi 224x224 -> siap masuk classifier
            if save_crops:
                c = crop_box(img, b, cfg["crops"]["bbox_padding"])
                if c is not None:
                    sq = to_square(c, cfg["classifier"]["image_size"],
                                   cfg["classifier"]["resize_mode"])
                    imwrite(crop_dir / f"{img_path.stem}_det{i:03d}.jpg", sq)

        imwrite(vis_dir / f"{img_path.stem}.jpg", vis)
        records.append({"image": str(img_path), "name": img_path.name,
                        "width": img.shape[1], "height": img.shape[0],
                        "detections": dets})
        n_boxes += len(dets)
        print(f"  {img_path.name:<28} -> {len(dets)} ayam")

    out_json = out_root / "detections.json"
    save_json({"weights": str(weights), "source": str(src), "conf": conf,
               "imgsz": dcfg["imgsz"], "num_images": len(records),
               "num_boxes": n_boxes, "results": records}, out_json)

    print(f"\n[detect] total {n_boxes} bbox dari {len(records)} gambar")
    print(f"[detect] JSON      : {out_json}")
    print(f"[detect] visual    : {vis_dir}")
    if save_crops:
        print(f"[detect] crop 224  : {crop_dir}")
    return {"num_images": len(records), "num_boxes": n_boxes}


def main():
    ap = argparse.ArgumentParser(description="Tahap 1: deteksi ayam (YOLO)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", default=None, help="folder gambar uji")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--no-crops", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    run_detection(cfg, a.source, a.conf, save_crops=not a.no_crops)


if __name__ == "__main__":
    main()
