"""
Ambil angka NYATA bagian I-II (pra-pemrosesan + forward YOLO) UNTUK REPO INI.

KENAPA BERKAS INI ADA. docs/hitung_manual_klasifikasi.md menyebut dirinya
"lanjutan dari bagian I dan II", tapi bagian I-II yang beredar berasal dari
proyek LAIN (hitung-ayam PIO + SAFECount): gambar 1080x1920, letterbox 960x960,
NMS IoU 0.45, conf floor 0.05. Repo ini memakai angka yang berbeda di setiap
titik itu. Menempelkan halaman lama apa adanya berarti menerbitkan konstanta
yang tidak berlaku di sini.

Berkas ini mengukur ulang bagian I-II pada pipeline repo ini sendiri, dari
config dan keluaran deteksi yang sudah ada, supaya sambungan ke bagian III
(crop -> letterbox 224 -> klasifikasi) memakai satu rangkaian angka.

Yang TIDAK diukur di sini: contoh hitung conv/BN/DFL per-elemen pada YOLOv8m.
Alasannya bukan kemalasan - angka itu hanya bisa diambil dengan menjalankan
forward pass ber-hook pada bobot deteksi, dan halaman gabungan menandainya
sebagai pinjaman dari proyek lain, bukan sebagai hasil repo ini.

Semua keluaran -> results/hitung_manual_deteksi.json

Jalankan:
  "C:/Arib/MASSA AYAM/generalisasi-ayam-skripsi/.venv-yolo/Scripts/python.exe" \
      scripts/extract_manual_deteksi.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import box_iou, imread, load_config, pad_box, resolve  # noqa: E402

OUT = ROOT / "results" / "hitung_manual_deteksi.json"
V: dict = {}


def p(k, v):
    V[k] = v
    s = json.dumps(v, ensure_ascii=False)
    print(f"  {k} = {s[:110]}")
    return v


# ------------------------------------------------------------------ #
print("\n[1] Konstanta pipeline dari config - BUKAN dari proyek lain")
cfg = load_config(ROOT / "configs" / "config.yaml")
det = cfg["detection"]
crp = cfg["crops"]

p("detect_imgsz", int(det["imgsz"]))
p("detect_conf", float(det["conf"]))
p("detect_iou_nms", float(det["iou"]))
p("detect_max_det", int(det["max_det"]))
p("classifier_image_size", int(cfg["classifier"]["image_size"]))
p("bbox_padding", float(crp["bbox_padding"]))
p("min_box_size", int(crp["min_box_size"]))
p("alive_min_conf", float(crp["alive_min_conf"]))
p("alive_min_box", int(crp["alive_min_box"]))
p("alive_max_iou_with_dead", float(crp["alive_max_iou_with_dead"]))
p("alive_detector_class", int(crp["alive_detector_class"]))
p("alive_detector_imgsz", int(crp["alive_detector_imgsz"]))

# Perbedaan terhadap halaman I-II proyek lain, ditulis sebagai data supaya
# tabel di dokumen tidak perlu diketik tangan.
p("beda_dengan_pio", {
    "imgsz":      {"pio": 960,  "repo_ini": int(det["imgsz"])},
    "nms_iou":    {"pio": 0.45, "repo_ini": float(det["iou"])},
    "conf_floor": {"pio": 0.05, "repo_ini": float(det["conf"])},
    "ukuran_gambar_asal": {"pio": "1080x1920", "repo_ini": "432x432 / 400x400"},
    "kelas_lewat_detektor": {"pio": "semua objek",
                             "repo_ini": "hanya kelas HIDUP; kelas MATI dari bbox anotasi"},
})

# ------------------------------------------------------------------ #
print("\n[2] Letterbox I: gambar penuh -> imgsz deteksi")
dets = json.loads((ROOT / "outputs" / "predictions" / "detections.json")
                  .read_text(encoding="utf-8"))
res = dets["results"]
p("deteksi_sumber", dets["source"])
p("deteksi_n_gambar", int(dets["num_images"]))
p("deteksi_n_box", int(dets["num_boxes"]))
p("deteksi_conf_dipakai", float(dets["conf"]))
p("deteksi_imgsz_dipakai", int(dets["imgsz"]))

ukuran = sorted({(r["width"], r["height"]) for r in res})
p("ukuran_gambar_unik", [list(u) for u in ukuran])

S = int(det["imgsz"])
lb = {}
for (W, H) in ukuran:
    r = min(S / H, S / W)
    nh, nw = max(1, int(round(H * r))), max(1, int(round(W * r)))
    lb[f"{W}x{H}"] = {
        "r": round(float(r), 6),
        "nh_nw": [nh, nw],
        "pad_atas_bawah": round(float((S - nh) / 2), 6),
        "pad_kiri_kanan": round(float((S - nw) / 2), 6),
        "interpolasi": "INTER_LINEAR" if r > 1 else "INTER_AREA",
    }
p("letterbox_deteksi", lb)

# ------------------------------------------------------------------ #
print("\n[3] Sebaran keluaran detektor")
confs = [b["conf"] for r in res for b in r["detections"]]
per_img = [len(r["detections"]) for r in res]
p("conf_min_max", [round(min(confs), 6), round(max(confs), 6)])
p("conf_median", round(float(statistics.median(confs)), 6))
p("box_per_gambar", {"min": min(per_img), "maks": max(per_img),
                     "median": float(statistics.median(per_img))})

sisi = [min(b["bbox"][2] - b["bbox"][0], b["bbox"][3] - b["bbox"][1])
        for r in res for b in r["detections"]]
p("sisi_pendek_box_ruang_asal", {
    "min": round(min(sisi), 4), "maks": round(max(sisi), 4),
    "median": round(float(statistics.median(sisi)), 4)})
# Sisi pendek yang benar-benar dilihat jaringan = setelah diskala ke imgsz.
skala_432 = S / 432.0
p("sisi_pendek_box_ruang_imgsz_median",
  round(float(statistics.median(sisi)) * skala_432, 4))

# Contoh satu box: dipakai sebagai contoh hitung di dokumen.
r0 = res[0]
b0 = r0["detections"][0]
p("contoh_gambar", {"nama": r0["name"], "W": r0["width"], "H": r0["height"]})
p("contoh_box_bbox", b0["bbox"])
p("contoh_box_conf", b0["conf"])
bw = b0["bbox"][2] - b0["bbox"][0]
bh = b0["bbox"][3] - b0["bbox"][1]
p("contoh_box_wh", [round(bw, 4), round(bh, 4)])
r_lb = S / float(r0["height"])
p("contoh_box_di_ruang_imgsz",
  [round(v * r_lb, 4) for v in b0["bbox"]])

# Contoh NMS: dua box paling tumpang tindih di gambar contoh, dengan IoU
# sungguhan dihitung dari keluaran - menunjukkan ambang 0.7 repo ini, bukan 0.45.
boxes = [b["bbox"] for b in r0["detections"]]
iou_max, ia, ib = 0.0, 0, 1
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        v = box_iou(boxes[i], boxes[j])
        if v > iou_max:
            iou_max, ia, ib = v, i, j
p("contoh_iou_tertinggi_antar_box_tersisa", {
    "iou": round(iou_max, 6),
    "box_a": boxes[ia], "box_b": boxes[ib],
    "catatan": ("keduanya SELAMAT dari NMS, jadi IoU-nya di bawah ambang "
                f"{det['iou']} - ini batas bawah, bukan contoh penekanan")})

# ------------------------------------------------------------------ #
print("\n[4] Jembatan ke bagian III: dari mana crop tiap kelas datang")
rows = list(csv.DictReader(
    open(ROOT / "data" / "crops" / "manifest.csv", encoding="utf-8")))
asal = {}
for r in rows:
    k = (r["origin"], r["label_name"])
    asal[f"{r['origin']}__{r['label_name']}"] = asal.get(
        f"{r['origin']}__{r['label_name']}", 0) + 1
p("crop_per_origin", asal)
p("crop_total", len(rows))

conf_mati = sorted({float(r["conf"]) for r in rows if r["label"] == "1"})
p("conf_kelas_mati_unik", conf_mati)

sisi_kelas = {}
for nama in ("alive", "dead"):
    v = []
    for r in rows:
        if r["label_name"] != nama:
            continue
        x1, y1, x2, y2 = json.loads(r["bbox"])
        v.append(min(x2 - x1, y2 - y1))
    sisi_kelas[nama] = {"n": len(v), "median_sisi_pendek": round(
        float(statistics.median(v)), 4)}
p("sisi_pendek_bbox_per_kelas", sisi_kelas)

ringkas = json.loads((ROOT / "data" / "crops" / "crops_summary.json")
                     .read_text(encoding="utf-8"))
p("crops_summary_skipped", ringkas["skipped"])
p("crops_summary_stats", ringkas["stats"])

# ------------------------------------------------------------------ #
print("\n[5] pad_box: pelebaran 10% diuji pada contoh nyata")
# Contoh yang dipakai bagian III dokumen klasifikasi, dihitung lewat fungsi
# repo-nya sendiri supaya dokumen tidak menyalin aritmetika tangan.
contoh = next(r for r in rows if r["label"] == "1" and r["split"] == "test")
bb = json.loads(contoh["bbox"])
p("padbox_contoh_crop", contoh["path"])
p("padbox_contoh_bbox", bb)
# Ukuran gambar dibaca dari berkas asalnya, bukan diasumsikan.
src = (resolve(crp["dead_root"]) / contoh["source_split"] / contoh["src_file"])
im = imread(src)
if im is None:
    raise SystemExit(f"gambar asal tidak terbaca: {src}")
H, W = im.shape[:2]
p("padbox_contoh_src", contoh["src_file"])
p("padbox_contoh_WH", [int(W), int(H)])
p("padbox_hasil", [int(v) for v in
                   pad_box(bb, float(crp["bbox_padding"]), W, H)])
# pad_box menjepit ke tepi gambar; tunjukkan sisi mana yang kena jepit.
dw = (bb[2] - bb[0]) * float(crp["bbox_padding"])
dh = (bb[3] - bb[1]) * float(crp["bbox_padding"])
p("padbox_delta_wh", [round(dw, 4), round(dh, 4)])
p("padbox_tanpa_jepit", [round(bb[0] - dw, 4), round(bb[1] - dh, 4),
                         round(bb[2] + dw, 4), round(bb[3] + dh, 4)])

# ------------------------------------------------------------------ #
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(V, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nTersimpan: {OUT}  ({len(V)} nilai)")
