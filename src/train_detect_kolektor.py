"""
Latih detektor cacat YOLOv8 di KolektorSDD2.

KODE LATIH YOLO PERTAMA DI REPO INI. Semua pemanggilan ultralytics sebelumnya
inference (src/detect.py, src/eval_detect_masks.py). Bobot ayam TIDAK dipakai
sebagai titik awal: permukaan logam tergores vs ayam di kandang tidak berbagi
apa pun yang berguna, jadi mulai dari bobot COCO.

IMGSZ 640, BUKAN 960 - INI BUKAN SELERA. configs/config.yaml:14-22 mencatat
kegagalan persis ini pada lengan ayam: menaikkan imgsz inferensi di atas skala
objek yang dilihat saat latih menurunkan recall 100% -> 36% -> 0%. Gambar
Kolektor sekitar 230x637 piksel; imgsz 960 memperbesar sisi pendeknya lebih
dari 4x. Tier 'm' di notebook saudara memakai imgsz=960, batch=2 - jangan
disalin ke sini. Bandingkan memori yolo-imgsz-skala-objek.

Hyperparameter lain mengikuti notebook saudara supaya perbandingannya jujur:
optimizer AdamW, lr0 0.02. `--seed` diteruskan ke model.train(seed=...);
tanpa itu ultralytics memakai 0 dan angka 'seed 42' di laporan jadi bohong.

`--project` BUKAN relatif ke direktori kerja. Terukur di jalan pertama: ia
diselesaikan terhadap `runs_dir` di settings ultralytics
(~/AppData/Roaming/Ultralytics/settings.json, di mesin ini menunjuk repo
*generalisasi-ayam-skripsi*), jadi `outputs/runs_detect_kolektor` mendarat di
    <runs_dir>/detect/outputs/runs_detect_kolektor/cacat_m
- di LUAR repo ini. Karena itu jalur bobot TIDAK disusun ulang dari `--project`
(itu mencetak jalur yang tidak ada) melainkan dibaca dari `save_dir` milik
trainer, lalu dicetak absolut berikut apakah berkasnya ADA di disk.

Jalankan:
    python src/train_detect_kolektor.py --data data/kolektor2_yolo/dataset.yaml \
        --model yolov8m.pt --epochs 100 --imgsz 640 --batch 8 \
        --project outputs/runs_detect_kolektor --name cacat_m --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="640: lihat catatan imgsz di kepala berkas ini")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project", default="outputs/runs_detect_kolektor")
    ap.add_argument("--name", default="cacat_m")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--patience", type=int, default=30)
    a = ap.parse_args()

    data = Path(a.data)
    if not data.exists():
        raise SystemExit(f"dataset.yaml tidak ada: {data}\n"
                         f"jalankan src/build_yolo_kolektor.py dulu")

    from ultralytics import YOLO

    if a.imgsz > 640:
        print(f"[latih-kol] PERINGATAN: imgsz {a.imgsz} > 640. Gambar Kolektor "
              f"~230x637; membesarkan sisi pendek >4x adalah kegagalan yang "
              f"sudah terdokumentasi (configs/config.yaml:14-22).")

    print(f"[latih-kol] {a.model} | data {data} | imgsz {a.imgsz} | "
          f"batch {a.batch} | epochs {a.epochs} | seed {a.seed}")

    model = YOLO(a.model)
    kw = dict(
        data=str(data.resolve()),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        project=a.project,
        name=a.name,
        seed=a.seed,              # WAJIB: tanpa ini ultralytics pakai 0
        optimizer="AdamW",
        lr0=0.02,
        patience=a.patience,
        exist_ok=False,
        plots=True,
        val=True,
    )
    if a.device is not None:
        kw["device"] = a.device
    hasil = model.train(**kw)

    # save_dir dibaca dari trainer, BUKAN disusun dari --project: lihat
    # catatan --project di kepala berkas. Menyusunnya ulang mencetak jalur
    # yang tidak ada, dan itu ketahuan hanya kalau ada yang mengeceknya.
    save_dir = None
    for sumber in (getattr(hasil, "save_dir", None),
                   getattr(getattr(model, "trainer", None), "save_dir", None)):
        if sumber:
            save_dir = Path(sumber)
            break
    if save_dir is None:
        save_dir = Path(a.project) / a.name
        print("[latih-kol] PERINGATAN: save_dir tak terbaca dari trainer; "
              "jalur di bawah disusun dari --project dan mungkin salah.")
    simpan = (save_dir / "weights" / "best.pt").resolve()
    print(f"\n[latih-kol] bobot terbaik: {simpan}")
    print(f"[latih-kol] ada di disk: {simpan.exists()}")
    print(f"[latih-kol] pin ke detection.weights di configs/config_kolektor.yaml")
    try:
        m = hasil.results_dict
        print("[latih-kol] metrik akhir (val split detektor, BUKAN test paper):")
        for k in ("metrics/precision(B)", "metrics/recall(B)",
                  "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
            if k in m:
                print(f"    {k:26s} {m[k]:.4f}")
    except Exception as e:
        print(f"[latih-kol] metrik tak terbaca dari hasil: {e}")
    print("[latih-kol] LANGKAH BERIKUT adalah GERBANG: eval_detect_masks.py "
          "--sweep. Kalau recall jelek di semua imgsz, lengan dua tahap "
          "berhenti di situ dan itu temuannya.")


if __name__ == "__main__":
    main()
