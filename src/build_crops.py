"""
TAHAP 2 - Menyiapkan dataset crop (ayam mati vs ayam hidup) untuk classifier.

Sumber label ayam MATI
    Dataset COCO "dead-chikens": 98 bbox pada 86 gambar, dianotasi manusia.

Sumber label ayam HIDUP  (lihat crops.alive_source di config)
    same_image : diambil dari FOTO YANG SAMA dengan ayam mati.
    pio        : diambil dari dataset CCTV PIO.
    both       : gabungan keduanya.

Kenapa 'same_image' yang dipakai sebagai default?
    Kalau crop hidup diambil dari dataset lain (PIO: CCTV tampak-atas)
    sedangkan crop mati dari foto close-up, maka perbedaan kamera, resolusi,
    dan sudut pandang jauh lebih mencolok daripada perbedaan mati-vs-hidup.
    Classifier bisa mencapai akurasi ~100% hanya dengan menebak "ini foto
    dari dataset mana" - dan skor itu sama sekali tidak membuktikan model
    bisa mengenali ayam mati. Dengan mengambil kedua kelas dari frame yang
    sama, satu-satunya pembeda yang tersisa adalah kondisi ayamnya.

Kenapa detektor ayam hidup BUKAN model YOLO PIO?
    Model PIO dilatih pada CCTV tampak-atas. Diuji pada foto close-up
    dataset ayam mati, ia gagal total: dari 1889 deteksi, NOL yang mengenai
    ayam mati, dan mayoritas box hanya ~6 px (tekstur sekam, bukan ayam).
    Karena itu dipakai YOLOv8m COCO yang punya kelas 'bird' - model ini
    mengenai 50 dari 98 bbox ayam mati, bukti bahwa yang ia deteksi memang
    benar unggas, dengan ukuran crop yang wajar (median ~84 px).

Pembagian train/val/test dilakukan per GAMBAR ASAL, bukan per crop, karena
dataset ini hasil augmentasi Roboflow: satu foto asli bisa muncul 3x dengan
nama berbeda. Kalau displit per crop, versi augmentasi dari foto yang sama
akan bocor ke train dan test sekaligus -> skor jadi palsu.

Jalankan:
    python src/build_crops.py
    python src/build_crops.py --alive-source pio
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import (box_iou, crop_box, get_device, imread, imwrite,
                    load_config, resolve, save_json, set_seed, to_square)

# Nama file Roboflow: "image--10-_jpg.rf.068c1ec....jpg"
# Bagian sebelum "_jpg.rf." adalah identitas foto aslinya.
_RF_PATTERN = re.compile(r"_(jpg|jpeg|png)\.rf\..*$", re.IGNORECASE)


def base_image_id(filename: str) -> str:
    """Ambil identitas foto asli dari nama file hasil augmentasi Roboflow."""
    stem = _RF_PATTERN.sub("", filename)
    return Path(stem).stem if "." in stem else stem


def load_coco_split(split_dir: Path) -> tuple[dict, dict]:
    """Baca _annotations.coco.json -> (nama file per image_id, bbox xyxy)."""
    ann_file = split_dir / "_annotations.coco.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"Anotasi tidak ada: {ann_file}")

    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    names = {im["id"]: im["file_name"] for im in data["images"]}
    boxes: dict[int, list] = defaultdict(list)
    for a in data["annotations"]:
        x, y, w, h = a["bbox"]                              # COCO = xywh
        boxes[a["image_id"]].append([x, y, x + w, y + h])   # -> xyxy
    return names, boxes


# --------------------------------------------------------------------------- #
# Pengumpulan crop
# --------------------------------------------------------------------------- #
def equalize(c, ccfg: dict):
    """Samakan resolusi EFEKTIF sebelum letterbox (crops.equalize_resolution).

    Crop ayam mati berasal dari foto close-up (sisi pendek ~224 px), crop ayam
    hidup PIO dari CCTV (~39 px). Setelah keduanya diperbesar ke 224, yang
    close-up tetap tajam dan yang CCTV jadi kabur - dan KETAJAMAN saja sudah
    memisahkan kedua kelas dengan AUC 1.0000 tanpa melihat isi gambar.
    Menurunkan kedua kelas ke sisi pendek yang sama mematikan jalan pintas itu
    di sumbernya, sebelum crop disimpan.
    """
    eq = ccfg.get("equalize_resolution") or {}
    if not eq.get("enabled"):
        return c
    t = int(eq.get("target_short_side", 96))
    h, w = c.shape[:2]
    short = min(h, w)
    if short <= t:
        return c
    f = t / short
    nw, nh = max(1, int(round(w * f))), max(1, int(round(h * f)))
    return cv2.resize(c, (nw, nh), interpolation=cv2.INTER_AREA)


def collect_dead_and_same_image_alive(cfg: dict, want_alive: bool) -> tuple:
    """Ambil crop ayam mati (anotasi) + ayam hidup (deteksi) dari foto yg sama."""
    ccfg = cfg["crops"]
    dead_root = resolve(ccfg["dead_root"])
    out_dir = resolve(ccfg["out_dir"])
    size = int(cfg["classifier"]["image_size"])
    mode = cfg["classifier"]["resize_mode"]
    pad = float(ccfg["bbox_padding"])
    min_box = int(ccfg["min_box_size"])

    max_iou = float(ccfg.get("alive_max_iou_with_dead", 0.01))
    min_conf = float(ccfg.get("alive_min_conf", 0.25))
    min_alive_box = int(ccfg.get("alive_min_box", 48))
    det_imgsz = int(ccfg.get("alive_detector_imgsz", 640))
    bird_cls = int(ccfg.get("alive_detector_class", 14))

    model = None
    if want_alive:
        from ultralytics import YOLO
        w = resolve(ccfg["alive_detector"])
        print(f"[crops] detektor ayam hidup : {w.name} "
              f"(kelas COCO 'bird', {get_device(cfg.get('device', 'auto'))})")
        model = YOLO(str(w))

    rows: list[dict] = []
    skipped = Counter()

    for split in ccfg["splits"]:
        sdir = dead_root / split
        if not sdir.exists():
            print(f"[crops] lewati (folder tidak ada): {sdir}")
            continue

        names, dead_boxes = load_coco_split(sdir)
        print(f"[crops] {split:<6}: {len(names):>3} gambar, "
              f"{sum(len(v) for v in dead_boxes.values()):>3} bbox ayam mati")

        for img_id, fname in names.items():
            img = imread(sdir / fname)
            if img is None:
                skipped["gagal_baca"] += 1
                continue

            base = base_image_id(fname)
            gt_dead = dead_boxes.get(img_id, [])

            # ---------- ayam MATI (anotasi manusia) ----------
            for k, box in enumerate(gt_dead):
                if (box[2] - box[0]) < min_box or (box[3] - box[1]) < min_box:
                    skipped["mati_terlalu_kecil"] += 1
                    continue
                c = crop_box(img, box, pad)
                if c is None:
                    skipped["mati_crop_kosong"] += 1
                    continue
                name = f"dead_{split}_{base}_{img_id}_{k:02d}.jpg"
                imwrite(out_dir / "dead" / name,
                        to_square(equalize(c, ccfg), size, mode))
                rows.append({"path": f"dead/{name}", "label": 1,
                             "label_name": "dead", "source_split": split,
                             "base_image": base, "src_file": fname,
                             "bbox": [round(v, 1) for v in box],
                             "conf": 1.0, "origin": "coco_gt",
                             "domain": "closeup"})

            # ---------- ayam HIDUP (deteksi 'bird', bukan ayam mati) ----------
            if model is None:
                continue

            res = model.predict(source=img, imgsz=det_imgsz, conf=min_conf,
                                classes=[bird_cls], verbose=False)[0]

            for k, (box, sc) in enumerate(zip(
                    res.boxes.xyxy.cpu().numpy().tolist(),
                    res.boxes.conf.cpu().numpy().tolist())):
                # Menyentuh bbox ayam mati -> buang (status ambigu)
                if any(box_iou(box, g) >= max_iou for g in gt_dead):
                    skipped["hidup_kena_ayam_mati"] += 1
                    continue
                if (box[2] - box[0]) < min_alive_box or \
                   (box[3] - box[1]) < min_alive_box:
                    skipped["hidup_terlalu_kecil"] += 1
                    continue
                c = crop_box(img, box, pad)
                if c is None:
                    skipped["hidup_crop_kosong"] += 1
                    continue
                name = f"alive_{split}_{base}_{img_id}_{k:03d}.jpg"
                imwrite(out_dir / "alive" / name,
                        to_square(equalize(c, ccfg), size, mode))
                rows.append({"path": f"alive/{name}", "label": 0,
                             "label_name": "alive", "source_split": split,
                             "base_image": base, "src_file": fname,
                             "bbox": [round(v, 1) for v in box],
                             "conf": round(float(sc), 4), "origin": "bird_det",
                             "domain": "closeup"})

    return rows, skipped


def collect_pio_alive(cfg: dict) -> tuple:
    """
    Ambil crop ayam hidup dari dataset CCTV PIO (label YOLO txt).

    PERINGATAN: domainnya berbeda jauh dari foto ayam mati. Lihat catatan
    di config - skor tinggi dengan sumber ini tidak bisa dipercaya.
    """
    ccfg = cfg["crops"]
    img_dir, lbl_dir = resolve(ccfg["pio_images"]), resolve(ccfg["pio_labels"])
    out_dir = resolve(ccfg["out_dir"])
    size = int(cfg["classifier"]["image_size"])
    mode = cfg["classifier"]["resize_mode"]
    pad = float(ccfg["bbox_padding"])
    min_alive_box = int(ccfg.get("alive_min_box", 48))
    limit = int(ccfg.get("pio_max_crops", 300))

    if not img_dir.exists():
        print(f"[crops] PIO tidak ditemukan: {img_dir}")
        return [], Counter()

    print(f"[crops] mengambil ayam hidup dari PIO (maks {limit} crop)")
    print("        PERINGATAN: domain berbeda dari foto ayam mati -> "
          "hasilnya bias, pakai hanya sebagai ablasi.")

    rng = random.Random(cfg["seed"])
    files = sorted(p for p in img_dir.glob("*.jpg"))
    rng.shuffle(files)

    # Batasi crop per gambar. Tanpa ini satu gambar PIO (yang berisi ~436 box)
    # menguras seluruh kuota, sehingga 300 crop cuma berasal dari 2 gambar -
    # split per-base_image lalu menaruh semuanya di satu sisi dan val/test
    # kehilangan kelas 'hidup' sama sekali.
    per_img = max(1, int(ccfg.get("pio_max_per_image", 12)))

    rows, skipped, n = [], Counter(), 0
    for ip in files:
        if n >= limit:
            break
        lp = lbl_dir / f"{ip.stem}.txt"
        if not lp.exists():
            continue
        img = imread(ip)
        if img is None:
            continue
        H, W = img.shape[:2]
        n_this = 0

        for k, line in enumerate(lp.read_text().strip().splitlines()):
            if n >= limit or n_this >= per_img:
                break
            parts = line.split()
            if len(parts) < 5:
                continue
            xc, yc, bw, bh = (float(v) for v in parts[1:5])
            box = [(xc - bw / 2) * W, (yc - bh / 2) * H,
                   (xc + bw / 2) * W, (yc + bh / 2) * H]
            if (box[2] - box[0]) < min_alive_box or \
               (box[3] - box[1]) < min_alive_box:
                skipped["pio_terlalu_kecil"] += 1
                continue
            c = crop_box(img, box, pad)
            if c is None:
                continue
            name = f"alive_pio_{ip.stem}_{k:03d}.jpg"
            imwrite(out_dir / "alive" / name,
                    to_square(equalize(c, ccfg), size, mode))
            rows.append({"path": f"alive/{name}", "label": 0,
                         "label_name": "alive", "source_split": "pio",
                         "base_image": f"pio_{ip.stem}", "src_file": ip.name,
                         "bbox": [round(v, 1) for v in box],
                         "conf": 1.0, "origin": "pio_gt", "domain": "cctv"})
            n += 1
            n_this += 1
    n_base = len({r["base_image"] for r in rows})
    print(f"[crops] PIO -> {len(rows)} crop ayam hidup dari {n_base} gambar")
    return rows, skipped


def build(cfg: dict) -> dict:
    ccfg = cfg["crops"]
    src = str(ccfg.get("alive_source", "same_image")).lower()
    if src not in {"same_image", "pio", "both"}:
        raise ValueError(f"alive_source tidak dikenal: {src}")

    print(f"[crops] sumber ayam hidup   : {src}")

    rows, skipped = collect_dead_and_same_image_alive(
        cfg, want_alive=src in {"same_image", "both"})

    if src in {"pio", "both"}:
        pio_rows, pio_skip = collect_pio_alive(cfg)
        rows += pio_rows
        skipped.update(pio_skip)

    if not rows:
        raise RuntimeError("Tidak ada crop yang terbentuk.")
    if not any(r["label"] == 0 for r in rows):
        raise RuntimeError("Tidak ada crop ayam hidup - classifier butuh 2 kelas.")

    rows = balance_and_split(rows, cfg)
    write_manifest(rows, cfg)
    if cfg.get("protocol", {}).get("development_only"):
        write_or_validate_split_lock(rows, cfg)
    if ccfg.get("save_review_grid", True):
        save_review_grid(rows, cfg)
    return summarize(rows, skipped, resolve(ccfg["manifest"]))


# --------------------------------------------------------------------------- #
# Penyeimbangan kelas & pembagian split
# --------------------------------------------------------------------------- #
def balance_and_split(rows: list[dict], cfg: dict) -> list[dict]:
    ccfg = cfg["crops"]
    rng = random.Random(cfg["seed"])

    dead = [r for r in rows if r["label"] == 1]
    alive = [r for r in rows if r["label"] == 0]

    # --- 1. batasi jumlah ayam hidup ---
    cap = int(len(dead) * float(ccfg.get("max_alive_per_dead", 3.0)))
    if len(alive) > cap:
        # Ambil merata per gambar asal supaya tidak menumpuk di satu foto
        by_base = defaultdict(list)
        for r in alive:
            by_base[r["base_image"]].append(r)
        for v in by_base.values():
            rng.shuffle(v)

        bases, picked, idx = sorted(by_base), [], 0
        while len(picked) < cap:
            added = False
            for b in bases:
                if idx < len(by_base[b]):
                    picked.append(by_base[b][idx])
                    added = True
                    if len(picked) >= cap:
                        break
            if not added:
                break
            idx += 1

        print(f"[crops] ayam hidup dibatasi : {len(alive)} -> {len(picked)} "
              f"(maks {ccfg['max_alive_per_dead']}x jumlah ayam mati)")
        alive = picked

    rows = dead + alive

    # --- 2. split per gambar asal ---
    if not ccfg.get("split_by_base_image", True):
        for r in rows:
            r["split"] = "val" if r["source_split"] == "valid" else r["source_split"]
        return rows

    # Split DISTRATIFIKASI per kelas. Kalau base image diacak begitu saja,
    # sumber hidup dan mati yang terpisah total (PIO vs close-up) bisa jatuh
    # ke satu sisi. Tiap kelas karena itu dibagi sendiri lalu digabung.
    #
    # Konfigurasi historis memakai train/val/test. Protokol development baru
    # hanya memakai train/val karena satu-satunya test adalah dataset chick.
    split_names = list(ccfg.get("split_names", ["train", "val", "test"]))
    ratios = [float(x) for x in ccfg["split_ratio"]]
    if len(split_names) not in (2, 3) or len(ratios) != len(split_names):
        raise ValueError("split_names/split_ratio harus berisi 2 atau 3 elemen")
    if split_names[:2] != ["train", "val"]:
        raise ValueError("dua split pertama wajib train dan val")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("jumlah split_ratio harus 1.0")

    label_of = {}
    for r in rows:
        old_label = label_of.setdefault(r["base_image"], r["label"])
        if old_label != r["label"]:
            raise ValueError(f"satu base_image punya dua label: {r['base_image']}")

    assign = {}
    for lab in sorted({r["label"] for r in rows}):
        bl = sorted({b for b, v in label_of.items() if v == lab})
        rng.shuffle(bl)
        n = len(bl)
        if len(split_names) == 2:
            if n < 2:
                raise ValueError(f"kelas {lab} tidak cukup untuk train/val")
            n_tr = max(1, min(int(round(n * ratios[0])), n - 1))
            for i, b in enumerate(bl):
                assign[b] = "train" if i < n_tr else "val"
        else:
            if n < 3:
                for b in bl:
                    assign[b] = "train"
                continue
            n_tr = min(int(round(n * ratios[0])), n - 2)
            n_va = max(1, min(int(round(n * ratios[1])), n - n_tr - 1))
            for i, b in enumerate(bl):
                assign[b] = ("train" if i < n_tr
                             else ("val" if i < n_tr + n_va
                                   else split_names[2]))

    for r in rows:
        r["split"] = assign[r["base_image"]]

    from collections import Counter as _C
    cnt = _C((r["split"], r["label_name"]) for r in rows)
    print(f"[crops] {len(assign)} gambar asal dibagi per kelas -> " + ", ".join(
        f"{sp}: hidup {cnt[(sp, 'alive')]} / mati {cnt[(sp, 'dead')]}"
        for sp in split_names))
    return rows


def write_manifest(rows: list[dict], cfg: dict) -> None:
    path = resolve(cfg["crops"]["manifest"])
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["path", "label", "label_name", "split", "source_split",
            "base_image", "src_file", "bbox", "conf", "origin", "domain"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["split"], x["label"], x["path"])):
            w.writerow({c: (json.dumps(r[c]) if c == "bbox" else r.get(c, ""))
                        for c in cols})


def _sha256(path: Path) -> str:
    """SHA-256 berkas untuk mengikat manifest ke protokol."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_or_validate_split_lock(rows: list[dict], cfg: dict) -> None:
    """Bekukan pembagian base_image development PIO hidup + Roboflow mati."""
    protocol = cfg["protocol"]
    lock_path = resolve(protocol["split_lock"])
    manifest = resolve(cfg["crops"]["manifest"])
    split_names = list(cfg["crops"]["split_names"])

    mapping = {}
    for r in rows:
        old_split = mapping.setdefault(r["base_image"], r["split"])
        if old_split != r["split"]:
            raise ValueError(f"base_image bocor antar split: {r['base_image']}")

    counts = {}
    for sp in split_names:
        alive = sum(r["split"] == sp and r["label"] == 0 for r in rows)
        dead = sum(r["split"] == sp and r["label"] == 1 for r in rows)
        counts[sp] = {"alive": alive, "dead": dead, "total": alive + dead}
    expected = protocol.get("expected_counts") or {}
    if expected and counts != expected:
        raise ValueError(f"jumlah development berubah: {counts} != {expected}")

    try:
        manifest_name = str(manifest.relative_to(resolve("."))).replace("\\", "/")
    except ValueError:
        manifest_name = str(manifest)
    payload = {
        "schema_version": 1,
        "purpose": "development_only_pio_alive_roboflow_dead",
        "seed": int(cfg["seed"]),
        "manifest": manifest_name,
        "manifest_sha256": _sha256(manifest),
        "counts": counts,
        "n_base_images": len(mapping),
        "base_image_to_split": dict(sorted(mapping.items())),
        "test_dataset_excluded": "C:/Arib/CCTV/patnet-pure/dataset/chick",
    }

    if lock_path.exists():
        with open(lock_path, encoding="utf-8") as f:
            current = json.load(f)
        comparable = ("seed", "counts", "n_base_images", "base_image_to_split")
        diffs = [k for k in comparable if current.get(k) != payload.get(k)]
        if diffs:
            raise ValueError(f"split lock berubah pada {diffs}: {lock_path}")
        if current.get("manifest_sha256") != payload["manifest_sha256"]:
            raise ValueError(f"hash manifest tidak cocok dengan split lock: {lock_path}")
        print(f"[crops] split lock          : cocok ({lock_path})")
        return

    save_json(payload, lock_path)
    print(f"[crops] split lock          : dibuat ({lock_path})")


def save_review_grid(rows: list[dict], cfg: dict, per_class: int = 24) -> None:
    """Lembar kontak supaya kualitas crop bisa diperiksa dengan mata."""
    out_dir = resolve(cfg["crops"]["out_dir"])
    rep = resolve(cfg["output"]["reports_dir"])
    rng = random.Random(cfg["seed"])

    for label, tag in [(1, "dead"), (0, "alive")]:
        sel = [r for r in rows if r["label"] == label]
        rng.shuffle(sel)
        sel = sel[:per_class]
        if not sel:
            continue

        tiles = []
        for r in sel:
            im = imread(out_dir / r["path"])
            if im is None:
                continue
            im = cv2.resize(im, (112, 112))
            color = (0, 0, 255) if label == 1 else (0, 200, 0)
            cv2.rectangle(im, (0, 0), (111, 111), color, 2)
            tiles.append(im)

        if not tiles:
            continue
        cols = 6
        while len(tiles) % cols:
            tiles.append(np.full((112, 112, 3), 255, np.uint8))
        grid = np.vstack([np.hstack(tiles[i:i + cols])
                          for i in range(0, len(tiles), cols)])
        imwrite(rep / f"crops_review_{tag}.jpg", grid)

    print(f"[crops] lembar review       : {rep}/crops_review_*.jpg")


def summarize(rows: list[dict], skipped: Counter, manifest: Path) -> dict:
    print("\n" + "=" * 58)
    print("RINGKASAN DATASET CROP")
    print("=" * 58)
    print(f"{'split':<8}{'hidup':>8}{'mati':>8}{'total':>8}")

    stats = {}
    present = {r["split"] for r in rows}
    split_names = [sp for sp in ("train", "val", "test") if sp in present]
    split_names += sorted(present - set(split_names))
    for sp in split_names:
        a = sum(1 for r in rows if r["split"] == sp and r["label"] == 0)
        d = sum(1 for r in rows if r["split"] == sp and r["label"] == 1)
        stats[sp] = {"alive": a, "dead": d, "total": a + d}
        print(f"{sp:<8}{a:>8}{d:>8}{a + d:>8}")

    ta = sum(1 for r in rows if r["label"] == 0)
    td = sum(1 for r in rows if r["label"] == 1)
    print(f"{'TOTAL':<8}{ta:>8}{td:>8}{ta + td:>8}")

    if skipped:
        print("\nDilewati:")
        for k, v in skipped.most_common():
            print(f"  {k:<24} {v}")

    # Cek ulang: tidak boleh ada gambar asal yang muncul di 2 split
    seen = defaultdict(set)
    for r in rows:
        seen[r["base_image"]].add(r["split"])
    leak = [b for b, s in seen.items() if len(s) > 1]
    print("\nCek kebocoran antar split: " +
          (f"ADA MASALAH -> {leak[:5]}" if leak else "AMAN (0 gambar bocor)"))

    # Peringatan kalau ada split yang kehilangan salah satu kelas
    for sp, s in stats.items():
        if s["alive"] == 0 or s["dead"] == 0:
            print(f"PERINGATAN: split '{sp}' hanya punya satu kelas "
                  f"(hidup={s['alive']}, mati={s['dead']}).")

    print(f"Manifest: {manifest}")

    out = {"stats": stats, "total_alive": ta, "total_dead": td,
           "skipped": dict(skipped), "leakage": leak}
    save_json(out, manifest.parent / "crops_summary.json")
    return out


def main():
    ap = argparse.ArgumentParser(description="Tahap 2: bangun dataset crop")
    ap.add_argument("--config", default=None)
    ap.add_argument("--alive-source", default=None,
                    choices=["same_image", "pio", "both"],
                    help="dari mana crop ayam hidup diambil")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.alive_source:
        cfg["crops"]["alive_source"] = a.alive_source
    set_seed(cfg["seed"])
    build(cfg)


if __name__ == "__main__":
    main()
