"""
Evaluasi deteksi terhadap anotasi SEGMENTASI ayam mati (dataset chick).

Dataset chick punya mask biner: putih = ayam mati. Mask itu diubah jadi
bounding box acuan (connected component -> boundingRect), lalu dipakai untuk
mengukur apakah detektor YOLO benar-benar mengurung ayam mati.

PENTING - mask ini HANYA untuk evaluasi, tidak pernah untuk melatih apa pun.
Jumlahnya cuma 22 objek di 18 gambar; kalau ikut dilatih, angka di bawah
berhenti jadi ukuran dan berubah jadi cermin. Detektor tetap satu kelas
(ayam, tanpa tahu mati/hidup); pemisahan mati-vs-hidup adalah tugas classifier
kontrastif di tahap berikutnya.

Dua kriteria dilaporkan:
    IoU      >= 0.5  - box detektor dan box acuan berimpit
    cakupan  >= 0.9  - box detektor menutup >=90% badan ayam mati
Untuk memberi makan classifier, cakupan lebih relevan daripada IoU: yang
dibutuhkan adalah crop yang memuat ayamnya secara utuh.

Jalankan:
    python src/eval_detect_masks.py
    python src/eval_detect_masks.py --sweep          # coba banyak imgsz
    python src/eval_detect_masks.py --weights <pt>   # bandingkan bobot lain
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import load_config, resolve, imread, imwrite, save_json

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
MIN_AREA = 20          # komponen di bawah ini dianggap derau mask


def imread_gray(path):
    """imread() di common.py selalu 3 kanal; mask perlu 1 kanal."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def masks_to_boxes(mask_path: Path) -> list[list[int]]:
    """Mask biner -> satu bbox per komponen tersambung."""
    m = imread_gray(mask_path)
    if m is None:
        return []
    binary = (m > 127).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        out.append([x, y, x + w, y + h])
    return out


def pair_files(root: Path):
    """Pasangkan tiap mask dengan gambarnya (ekstensinya bisa beda)."""
    pairs = []
    for seg in sorted((root / "segmentations").glob("*.png")):
        for ext in IMG_EXT:
            img = root / "images" / (seg.stem + ext)
            if img.exists():
                pairs.append((seg.stem, img, seg))
                break
    return pairs


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def coverage(gt, det) -> float:
    """Bagian dari box acuan yang tertutup box detektor."""
    x1, y1 = max(gt[0], det[0]), max(gt[1], det[1])
    x2, y2 = min(gt[2], det[2]), min(gt[3], det[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = (gt[2] - gt[0]) * (gt[3] - gt[1])
    return inter / area if area > 0 else 0.0


def evaluate(model, pairs, imgsz, conf, iou_nms, max_det, vis_dir=None):
    per_obj, n_det = [], []
    for stem, img_path, seg_path in pairs:
        img = imread(img_path)
        gts = masks_to_boxes(seg_path)
        if img is None or not gts:
            continue

        res = model.predict(img, imgsz=imgsz, conf=conf, iou=iou_nms,
                            max_det=max_det, verbose=False)[0]
        dets = (res.boxes.xyxy.cpu().numpy() if res.boxes is not None
                else np.zeros((0, 4)))
        scores = (res.boxes.conf.cpu().numpy() if res.boxes is not None
                  else np.zeros((0,)))
        n_det.append(len(dets))

        vis = img.copy() if vis_dir is not None else None
        if vis is not None:
            for d in dets:
                cv2.rectangle(vis, (int(d[0]), int(d[1])),
                              (int(d[2]), int(d[3])), (90, 90, 90), 1)

        for gt in gts:
            best_i, best_c, best_s = 0.0, 0.0, 0.0
            for d, s in zip(dets, scores):
                v = iou(gt, d)
                if v > best_i:
                    best_i, best_c, best_s = v, coverage(gt, d), float(s)
            per_obj.append({"image": stem, "gt": gt,
                            "iou": round(best_i, 4),
                            "coverage": round(best_c, 4),
                            "conf": round(best_s, 4)})
            if vis is not None:
                col = (0, 200, 0) if best_i >= 0.5 else (0, 0, 255)
                cv2.rectangle(vis, (gt[0], gt[1]), (gt[2], gt[3]), col, 2)
                cv2.putText(vis, f"IoU {best_i:.2f}",
                            (gt[0], max(12, gt[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        if vis is not None and vis_dir is not None:
            imwrite(vis_dir / f"{stem}.jpg", vis)

    n = len(per_obj)
    ious = np.array([o["iou"] for o in per_obj])
    covs = np.array([o["coverage"] for o in per_obj])
    return {
        "imgsz": imgsz, "n_objects": n,
        "recall_iou50": float((ious >= 0.5).mean()) if n else 0.0,
        "recall_cov90": float((covs >= 0.9).mean()) if n else 0.0,
        "iou_mean": float(ious.mean()) if n else 0.0,
        "det_per_image": float(np.median(n_det)) if n_det else 0.0,
        "per_object": per_obj,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Ukur recall deteksi terhadap mask ayam mati")
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--root", default="C:/Arib/CCTV/patnet-pure/dataset/chick")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="coba beberapa imgsz sekaligus")
    ap.add_argument("--vis", action="store_true", help="simpan gambar hasil")
    a = ap.parse_args()

    cfg = load_config(a.config)
    dcfg = cfg["detection"]
    weights = Path(a.weights) if a.weights else resolve(dcfg["weights"])
    root = Path(a.root)

    pairs = pair_files(root)
    if not pairs:
        raise SystemExit(f"Tidak ada pasangan gambar+mask di {root}")

    from ultralytics import YOLO
    model = YOLO(str(weights))

    n_gt = sum(len(masks_to_boxes(s)) for _, _, s in pairs)
    print(f"[eval] bobot   : {weights.name}")
    print(f"[eval] data    : {len(pairs)} gambar, {n_gt} ayam mati berlabel")
    print(f"[eval] conf={dcfg['conf']}  iou={dcfg['iou']}  "
          f"max_det={dcfg['max_det']}\n")

    sizes = ([256, 320, 416, 512, 640, 768, 960, 1280] if a.sweep
             else [a.imgsz or int(dcfg["imgsz"])])

    out_dir = resolve(cfg["output"]["predictions_dir"])
    head = ("{:>6}{:>16}{:>16}{:>11}{:>12}"
            .format("imgsz", "recall IoU>=.5", "recall cov>=.9",
                    "IoU rata2", "det/gambar"))
    print(head)
    results = []
    for s in sizes:
        vis = (out_dir / f"detect_masks_vis_{s}") if a.vis else None
        r = evaluate(model, pairs, s, float(dcfg["conf"]),
                     float(dcfg["iou"]), int(dcfg["max_det"]), vis)
        results.append(r)
        print(f"{s:>6}{r['recall_iou50']*100:>15.0f}%"
              f"{r['recall_cov90']*100:>15.0f}%"
              f"{r['iou_mean']:>11.3f}{r['det_per_image']:>12.0f}")

    best = max(results, key=lambda r: (r["recall_iou50"], r["iou_mean"]))
    if len(results) > 1:
        print(f"\n[eval] terbaik: imgsz {best['imgsz']} "
              f"(recall {best['recall_iou50']*100:.0f}%, "
              f"IoU {best['iou_mean']:.3f})")
    worst = min(best["per_object"], key=lambda o: o["iou"])
    print(f"[eval] objek terlemah: {worst['image']} IoU {worst['iou']:.2f} "
          f"cakupan {worst['coverage']:.2f} conf {worst['conf']:.2f}")

    out_json = out_dir / "detect_masks_eval.json"
    save_json({"weights": str(weights), "root": str(root),
               "n_images": len(pairs), "n_objects": n_gt,
               "conf": float(dcfg["conf"]), "iou_nms": float(dcfg["iou"]),
               "results": results}, out_json)
    print(f"[eval] JSON    : {out_json}")


if __name__ == "__main__":
    main()
