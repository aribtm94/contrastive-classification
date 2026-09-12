"""
Jalankan: python src/siapkan_label_chick.py

Siapkan 1215 crop chick untuk dilabeli manual.

Keluaran: data/label_chick/crops/  (1215 jpg, nama berurutan)
          data/label_chick/lembar_label.csv  (daftar siap dicatat)

Crop yang menimpa mask ayam mati (IoU >= 0.5) ditandai di kolom
'dugaan_awal' sebagai 'mati' - itu 22 crop acuan yang sudah diketahui.
Sisanya kosong, untuk diisi manual.
"""

import csv, glob, io, json, os, sys
sys.path.insert(0, "src")
import cv2
from common import crop_box, imread, load_config, resolve
from eval_detect_masks import iou, masks_to_boxes, pair_files

KELUAR = "data/label_chick"
cfg = load_config(None)
pad = float(cfg["crops"]["bbox_padding"])

det = json.load(io.open("outputs/predictions/detections.json", encoding="utf-8"))
per_nama = {r["name"]: r for r in det["results"]}

pairs = pair_files(resolve("C:/Arib/CCTV/patnet-pure/dataset/chick"))
print("pasangan gambar+mask:", len(pairs))

os.makedirs(os.path.join(KELUAR, "crops"), exist_ok=True)
baris = []
n_mati = 0

for stem, img_path, seg_path in sorted(pairs):
    nama = os.path.basename(img_path)
    if nama not in per_nama:
        print("  ! tidak ada deteksi untuk", nama); continue
    img = imread(img_path)
    gts = masks_to_boxes(seg_path)
    dets = per_nama[nama]["detections"]
    boxes = [d["bbox"] for d in dets]

    # crop mana yang ayam mati: IoU >= 0.5 dengan mask acuan
    idx_mati = set()
    for gt in gts:
        bj, bv = -1, 0.0
        for j, b in enumerate(boxes):
            v = iou(gt, b)
            if v > bv:
                bj, bv = j, v
        if bj >= 0 and bv >= 0.5:
            idx_mati.add(bj)
    n_mati += len(idx_mati)

    for d in dets:
        j = int(d["id"])
        c = crop_box(img, d["bbox"], pad)
        if c is None:
            continue
        berkas = "%s_det%03d.jpg" % (stem, j)
        cv2.imwrite(os.path.join(KELUAR, "crops", berkas), c,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        x1, y1, x2, y2 = [round(float(v), 1) for v in d["bbox"]]
        baris.append({
            "berkas": berkas,
            "gambar_sumber": nama,
            "det_id": j,
            "dugaan_awal": "mati" if j in idx_mati else "",
            "label": "",
            "conf": d["conf"],
            "bbox": "%s %s %s %s" % (x1, y1, x2, y2),
            "lebar_px": int(round(x2 - x1)),
            "tinggi_px": int(round(y2 - y1)),
        })

with io.open(os.path.join(KELUAR, "lembar_label.csv"), "w",
             encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(baris[0]))
    w.writeheader(); w.writerows(baris)

print("crop ditulis :", len(baris))
print("ditandai mati:", n_mati)
print("lembar       :", os.path.join(KELUAR, "lembar_label.csv"))
