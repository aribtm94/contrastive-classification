"""
Jalankan: python src/siapkan_label_chick.py

Siapkan 1215 crop chick untuk dilabeli manual.

Keluaran: data/label_chick/crops/  (0001.jpg sampai 1215.jpg)
          data/label_chick/lembar_label.csv  (daftar siap dicatat)

Penomoran berjalan terus 1..1215 menembus batas gambar sumber, dengan
urutan alami: ayam (1), ayam (2), ... ayam (11), chick (1), ... chick (7).
Jadi nomor yang tercetak di lembar kontak sama dengan nama berkas crop-nya.
Kolom gambar_sumber dan det_id tetap disimpan supaya tiap crop masih bisa
dilacak balik ke detections.json dan ke bbox yang dinilai eval_on_chick.py.

Crop yang menimpa mask ayam mati (IoU >= 0.5) ditandai di kolom
'dugaan_awal' sebagai 'mati' - itu 22 crop acuan yang sudah diketahui.
Sisanya kosong, untuk diisi manual.
"""

import csv, glob, io, json, os, re, sys
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


def kunci_alami(s):
    """Urutan mata manusia: ayam (2) sebelum ayam (10), bukan sesudahnya."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


os.makedirs(os.path.join(KELUAR, "crops"), exist_ok=True)
# hapus crop lama: nama berkas berubah total, kalau tidak dibersihkan
# penomoran yang lama akan tertinggal dan ikut terbaca
lama = glob.glob(os.path.join(KELUAR, "crops", "*.jpg"))
for f in lama:
    os.remove(f)
print("crop lama dihapus   :", len(lama))

baris = []
n_mati = 0
nomor = 0
halaman = 0

for stem, img_path, seg_path in sorted(pairs, key=lambda p: kunci_alami(p[0])):
    nama = os.path.basename(img_path)
    if nama not in per_nama:
        print("  ! tidak ada deteksi untuk", nama); continue
    img = imread(img_path)
    gts = masks_to_boxes(seg_path)
    dets = sorted(per_nama[nama]["detections"], key=lambda d: int(d["id"]))
    boxes = [d["bbox"] for d in dets]
    halaman += 1

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
        # nomor maju hanya kalau crop benar-benar tertulis, supaya
        # 1..N tidak pernah bolong
        nomor += 1
        berkas = "%04d.jpg" % nomor
        cv2.imwrite(os.path.join(KELUAR, "crops", berkas), c,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        x1, y1, x2, y2 = [round(float(v), 1) for v in d["bbox"]]
        baris.append({
            "nomor": nomor,
            "berkas": berkas,
            "halaman": halaman,
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

print("crop ditulis :", len(baris), "(%04d - %04d)" % (1, len(baris)))
print("ditandai mati:", n_mati)
print("lembar       :", os.path.join(KELUAR, "lembar_label.csv"))
