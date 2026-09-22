"""
KolektorSDD2: mask segmentasi -> label YOLO untuk MELATIH detektor cacat.

Ini bagian pertama dari lengan dua tahap yang diminta: GT mask diubah jadi
kotak, kotak itu melatih detektor, lalu detektor itu yang memberi crop ke
classifier. Sampai berkas ini, repo tidak punya satu pun kode latih YOLO -
semua pemanggilan ultralytics sebelumnya inference.

DEFINISI KOTAK DIIMPOR, TIDAK DISALIN. `masks_to_boxes` diambil apa adanya
dari src/eval_detect_masks.py. Kalau ia disalin, definisi kotak yang MELATIH
detektor bisa menyimpang dari definisi kotak yang MENGEVALUASI-nya tanpa ada
yang gagal, dan recall-nya berhenti berarti. Ambang area ikut terbawa:
eval_detect_masks.MIN_AREA = 20.

DUA AMBANG AREA YANG MUDAH TERTUKAR - jangan disatukan:
    eval_detect_masks.MIN_AREA = 20      kotak acuan (latih detektor + evaluasi)
    crops.min_area_komponen  = 50        crop untuk classifier (berkas lain)
Berkas ini HANYA memakai yang pertama. Sebutkan mana yang dipakai di mana
sebelum membandingkan recall deteksi dengan jumlah crop, kalau tidak dua
populasi berbeda diperlakukan seolah satu.

GAMBAR NORMAL: `--negatif <int>`, bawaan 0. Ultralytics menerima gambar tanpa
berkas label sebagai latar negatif. Pilihan itu dibuat EKSPLISIT lewat flag,
bukan disembunyikan, karena jumlah negatif mengubah presisi detektor secara
langsung dan harus bisa dilaporkan sebagai keputusan.

VAL DARI MANA. KolektorSDD2 hanya mengirim train/test. Split paper dijaga
utuh: test tidak disentuh berkas ini sama sekali; train (245 cacat menurut
MIN_AREA) dibagi train/val di level GAMBAR. Terukur di scripts/
siapkan_kolektor.py: 2331 gambar train = 2331 kelompok korelasi, korelasi
maks antar-gambar 0.9189 (di bawah ambang 0.97), jadi di dataset INI split
per gambar memang cukup - beda dari HEN v2, dan itu diperiksa, tidak
diasumsikan.

Jalankan:
    python src/build_yolo_kolektor.py --root data/kolektor2 \
        --out data/kolektor2_yolo --seed 42 [--negatif 200]
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from eval_detect_masks import MIN_AREA, masks_to_boxes, pair_files
from common import imread


def kotak_ke_yolo(box: list[int], W: int, H: int) -> tuple[float, ...]:
    """[x1,y1,x2,y2] piksel -> (xc, yc, w, h) ternormalisasi.

    Kebalikan persis src/build_crops.py:257-265, supaya bolak-balik antara
    dua representasi itu tidak menggeser kotak satu piksel pun.
    """
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)


def bagi_per_gambar(stems: list[str], rasio_val: float,
                    rng: random.Random) -> dict[str, str]:
    """Split di level GAMBAR. Satu gambar tak pernah ada di dua split."""
    urut = sorted(stems)
    rng.shuffle(urut)
    n_val = int(round(len(urut) * rasio_val))
    pilih = set(urut[:n_val])
    return {s: ("val" if s in pilih else "train") for s in urut}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="hasil scripts/siapkan_kolektor.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rasio-val", type=float, default=0.25)
    ap.add_argument("--negatif", type=int, default=0,
                    help="jumlah gambar NORMAL sebagai latar negatif")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    dir_train = root / "train"
    if not (dir_train / "images").is_dir():
        raise SystemExit(f"tata letak tidak ditemukan: {dir_train / 'images'}")

    pasang = pair_files(dir_train)          # hanya gambar yang PUNYA mask
    if not pasang:
        raise SystemExit(f"tidak ada pasangan mask di {dir_train}")
    print(f"[yolo-kol] gambar bercacat (punya mask): {len(pasang)}")

    rng = random.Random(a.seed)
    peta = bagi_per_gambar([s for s, _, _ in pasang], a.rasio_val, rng)

    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    n_kotak = {"train": 0, "val": 0}
    n_img = {"train": 0, "val": 0}
    for stem, p_img, p_mask in pasang:
        img = imread(p_img)
        if img is None:
            print(f"[yolo-kol] gambar tak terbaca, dilewati: {p_img.name}")
            continue
        H, W = img.shape[:2]
        kotak = masks_to_boxes(p_mask)
        if not kotak:
            # Tak boleh terjadi: pair_files hanya melihat mask yang ditulis
            # siapkan_kolektor.py, dan itu sudah disaring dengan fungsi yang
            # SAMA. Kalau tetap terjadi, tata letaknya dibangun oleh versi
            # lain - berhenti, jangan tulis label kosong diam-diam.
            raise SystemExit(
                f"mask ada tapi nol kotak pada MIN_AREA={MIN_AREA}: {p_mask}. "
                f"Tata letak dibangun ulang dengan versi kode yang beda?")
        sp = peta[stem]
        shutil.copyfile(p_img, out / "images" / sp / f"{stem}.png")
        baris = [f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                 for xc, yc, bw, bh in (kotak_ke_yolo(b, W, H) for b in kotak)]
        (out / "labels" / sp / f"{stem}.txt").write_text(
            "\n".join(baris) + "\n", encoding="utf-8")
        n_kotak[sp] += len(kotak)
        n_img[sp] += 1

    # Latar negatif: gambar NORMAL, tanpa berkas label.
    n_neg = {"train": 0, "val": 0}
    if a.negatif > 0:
        bercacat = {s for s, _, _ in pasang}
        normal = sorted(p for p in (dir_train / "images").glob("*.png")
                        if p.stem not in bercacat)
        rng.shuffle(normal)
        dipilih = normal[: a.negatif]
        peta_neg = bagi_per_gambar([p.stem for p in dipilih], a.rasio_val, rng)
        lookup = {p.stem: p for p in dipilih}
        for stem, sp in peta_neg.items():
            shutil.copyfile(lookup[stem], out / "images" / sp / f"{stem}.png")
            n_neg[sp] += 1
        print(f"[yolo-kol] latar negatif dipakai: {len(dipilih)} dari "
              f"{len(normal)} gambar normal yang tersedia")
    else:
        print("[yolo-kol] latar negatif: 0 (--negatif 0). Detektor hanya "
              "melihat gambar bercacat; presisinya akan optimistis.")

    yml = out / "dataset.yaml"
    yml.write_text(
        "# Dibuat src/build_yolo_kolektor.py - jangan disunting tangan.\n"
        f"# kotak dari mask GT, MIN_AREA={MIN_AREA} (eval_detect_masks)\n"
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: [cacat]\n", encoding="utf-8")

    print(f"[yolo-kol] seed {a.seed} | rasio val {a.rasio_val}")
    for sp in ("train", "val"):
        print(f"    {sp:5s} gambar cacat {n_img[sp]:4d} | kotak "
              f"{n_kotak[sp]:4d} | negatif {n_neg[sp]:4d}")
    print(f"[yolo-kol] total kotak {sum(n_kotak.values())} "
          f"(acuan: 270 komponen train pada MIN_AREA={MIN_AREA})")
    print(f"[yolo-kol] dataset.yaml -> {yml}")
    print("[yolo-kol] test split paper TIDAK disentuh berkas ini.")


if __name__ == "__main__":
    main()
