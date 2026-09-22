"""
Staging KolektorSDD2: dari zip datar ke tata letak yang dibaca repo ini.

KENAPA PERLU SKRIP SENDIRI. Zip aslinya datar:

    train/10301.png          gambar permukaan
    train/10301_GT.png       mask biner cacat (HITAM SEMUA kalau gambar normal)

Sementara `eval_detect_masks.pair_files()` (src/eval_detect_masks.py:64)
memasangkan `<root>/images/<stem>.<ext>` dengan `<root>/segmentations/<stem>.png`
dan hanya melihat stem yang PUNYA mask. Jadi:

  - semua gambar -> images/
  - HANYA gambar yang benar-benar bercacat -> segmentations/

Konsekuensinya `--sweep` di eval_detect_masks.py langsung jalan tanpa satu
baris ubahan di sana, dan gambar normal tidak ikut dihitung sebagai "gagal
mendeteksi apa pun" - karena memang tidak ada yang harus dideteksi di sana.

MASK KOSONG BUKAN "TIDAK ADA MASK". Tiap gambar train/test punya berkas _GT,
termasuk yang normal; yang normal isinya hitam semua. Kalau berkas hitam itu
ikut ditulis ke segmentations/, pair_files() akan memasangkannya dan
masks_to_boxes() memberi nol kotak -> gambar normal masuk populasi evaluasi
recall sebagai gambar tanpa objek. Itu menggeser recall tanpa ada yang gagal.
Karena itu keputusannya eksplisit: mask ditulis HANYA kalau
masks_to_boxes() memberi >= 1 kotak, memakai definisi kotak yang SAMA dengan
evaluasi (impor, bukan salinan).

BERKAS YATIM. train/10301 (copy).png dan 10301_GT (copy).png tidak punya
pasangan. Dibuang dengan hitungan tercetak - bukan dilewati senyap. Setelah
dibuang: test 110/894 PERSIS sesuai paper, train 2331 gambar sesuai paper.

SATU GAMBAR TIDAK SESUAI PAPER, DAN SEBABNYA DIKETAHUI. Paper mencatat train
246 cacat / 2085 normal; skrip ini memberi 245/2086. Selisihnya seluruhnya
train/10045, yang mask-nya berisi 23 piksel putih dengan contourArea 14.5 -
di bawah eval_detect_masks.MIN_AREA = 20. Jadi bukan berkas hilang dan bukan
galat pasangan: ia cacat yang terlalu kecil untuk ambang area yang dipakai
EVALUASI. Ambang itu tidak diturunkan hanya supaya angkanya cocok - kalau
diturunkan, definisi kotak acuan latih berhenti sama dengan definisi kotak
acuan evaluasi, dan itu harga yang jauh lebih mahal daripada satu gambar.
Gambar itu tetap masuk images/ sebagai NORMAL, dan namanya dicetak.

KELOMPOK FOTO. Pelajaran dari HEN v2 berlaku umum: id berkas yang unik TIDAK
membuktikan foto yang unik (di HEN, 3929 berkas = 1034 foto; 465 pasangan
kembar bahkan ber-stem beda). Skrip ini menjalankan union-find korelasi
piksel 32x32 yang sama atas gambar train dan MENCETAK jumlah kelompoknya,
supaya kalau Kolektor juga memuat duplikat, itu terlihat SEBELUM split dibuat,
bukan setelah skornya keluar.

Jalankan:
    python scripts/siapkan_kolektor.py --zip "data/non ayam/KolektorSDD2.zip" \
        --out data/kolektor2
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval_detect_masks import MIN_AREA, imread_gray, masks_to_boxes  # definisi kotak = evaluasi
from build_crops_hen import sidik_kecil, _Union       # kunci kelompok = HEN

SPLITS = ("train", "test")
SUFIKS_GT = "_GT"


def bongkar(zip_path: Path, tmp: Path) -> None:
    with zipfile.ZipFile(zip_path) as z:
        anggota = [n for n in z.namelist()
                   if n.lower().endswith(".png") and "/" in n
                   and n.split("/", 1)[0] in SPLITS]
        z.extractall(tmp, members=anggota)
    print(f"[kolektor] dibongkar: {len(anggota)} png")


def pasangkan(dir_split: Path) -> tuple[dict[str, tuple[Path, Path]], Counter]:
    """stem -> (gambar, mask). Yatim dibuang dengan hitungan."""
    gambar, mask = {}, {}
    for p in sorted(dir_split.glob("*.png")):
        if p.stem.endswith(SUFIKS_GT):
            mask[p.stem[: -len(SUFIKS_GT)]] = p
        else:
            gambar[p.stem] = p
    dibuang = Counter()
    pasang = {}
    for stem, p in gambar.items():
        if stem in mask:
            pasang[stem] = (p, mask[stem])
        else:
            dibuang["gambar_tanpa_GT"] += 1
            print(f"[kolektor] YATIM dibuang (tak ada _GT): {p.name}")
    for stem in mask:
        if stem not in gambar:
            dibuang["GT_tanpa_gambar"] += 1
            print(f"[kolektor] YATIM dibuang (tak ada gambar): {stem}{SUFIKS_GT}")
    return pasang, dibuang


def kelompokkan_gambar(jalur: list[Path], ambang: float = 0.97) -> int:
    """Union-find korelasi 32x32 atas gambar. Kembalikan jumlah kelompok.

    Hanya DIAGNOSTIK di sini: hasilnya dicetak, tidak dipakai membagi split,
    karena Kolektor tidak punya penamaan stem seperti Roboflow. Kalau
    jumlahnya jauh di bawah jumlah gambar, split per gambar TIDAK cukup dan
    itu harus terlihat sekarang.
    """
    sidik, indeks = [], []
    for i, p in enumerate(jalur):
        s = sidik_kecil(p.read_bytes())
        if s is None:
            continue
        sidik.append(s)
        indeks.append(i)
    if not sidik:
        return 0
    M = np.stack(sidik).astype(np.float32)
    u = _Union(len(jalur))
    # 2332 x 2332 float32 = 22 MB; muat sekaligus.
    R = M @ M.T
    np.fill_diagonal(R, -9.0)
    ia, ib = np.where(R >= ambang)
    n_pas = 0
    for a, b in zip(ia.tolist(), ib.tolist()):
        if a < b:
            n_pas += 1
            u.gabung(indeks[a], indeks[b])
    grup = len({u.cari(i) for i in range(len(jalur))})
    sah = R[R > -8.0]
    print(f"[kolektor] kelompok korelasi>={ambang}: {grup} kelompok dari "
          f"{len(jalur)} gambar ({n_pas} pasangan digabung)")
    if len(sah):
        print(f"[kolektor]   korelasi maks antar-gambar {float(sah.max()):.4f} "
              f"| persentil 99.9 {float(np.percentile(sah, 99.9)):.4f}")
    if grup < len(jalur):
        print(f"[kolektor]   -> PERINGATAN: {len(jalur) - grup} gambar punya "
              f"kembaran. Split per gambar TIDAK cukup; pakai kelompok.")
    return grup


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ambang-korelasi", type=float, default=0.97)
    ap.add_argument("--lewati-kelompok", action="store_true",
                    help="lewati diagnostik kelompok (lambat pada 2332 gambar)")
    a = ap.parse_args()

    zip_path, out = Path(a.zip), Path(a.out)
    if not zip_path.exists():
        raise SystemExit(f"zip tidak ada: {zip_path}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        bongkar(zip_path, tmp)

        ringkas = {}
        for split in SPLITS:
            dir_split = tmp / split
            if not dir_split.is_dir():
                raise SystemExit(f"split tidak ada di zip: {split}")
            pasang, dibuang = pasangkan(dir_split)

            d_img = out / split / "images"
            d_seg = out / split / "segmentations"
            d_img.mkdir(parents=True, exist_ok=True)
            d_seg.mkdir(parents=True, exist_ok=True)

            n_cacat = n_normal = 0
            n_kotak = 0
            bawah_ambang = []
            for stem, (p_img, p_mask) in sorted(pasang.items()):
                shutil.copyfile(p_img, d_img / f"{stem}.png")
                kotak = masks_to_boxes(p_mask)
                if kotak:
                    shutil.copyfile(p_mask, d_seg / f"{stem}.png")
                    n_cacat += 1
                    n_kotak += len(kotak)
                else:
                    n_normal += 1
                    # Bedakan "mask hitam total" (gambar memang normal) dari
                    # "mask berisi tapi komponennya lebih kecil dari MIN_AREA".
                    # Yang kedua adalah gambar CACAT menurut paper yang kita
                    # golongkan normal karena ambang area. Selisih 246 vs 245
                    # di train seluruhnya berasal dari sini, dan itu harus
                    # tercetak - bukan jadi off-by-one yang dikira wajar.
                    m = imread_gray(p_mask)
                    if m is not None and int((m > 127).sum()) > 0:
                        bawah_ambang.append((stem, int((m > 127).sum())))

            print(f"[kolektor] {split}: {len(pasang)} gambar | cacat {n_cacat} "
                  f"({n_kotak} komponen area>={MIN_AREA}) "
                  f"| normal {n_normal} | dibuang {dict(dibuang)}")
            for stem, px in bawah_ambang:
                print(f"[kolektor]   CACAT di bawah MIN_AREA={MIN_AREA} -> "
                      f"digolongkan normal: {stem} ({px} piksel putih)")
            if bawah_ambang:
                print(f"[kolektor]   jadi 'cacat menurut paper' = "
                      f"{n_cacat + len(bawah_ambang)}, "
                      f"'cacat menurut MIN_AREA={MIN_AREA}' = {n_cacat}")
            ringkas[split] = dict(gambar=len(pasang), cacat=n_cacat,
                                  normal=n_normal, komponen=n_kotak,
                                  dibuang=dict(dibuang),
                                  cacat_bawah_ambang=[s for s, _ in bawah_ambang])

        if not a.lewati_kelompok:
            jalur = sorted((out / "train" / "images").glob("*.png"))
            kelompokkan_gambar(jalur, a.ambang_korelasi)

    print(f"\n[kolektor] tata letak siap di {out.resolve()}")
    print("[kolektor] periksa terhadap paper: train 246/2085, test 110/894")
    for split, r in ringkas.items():
        print(f"    {split:6s} cacat {r['cacat']:4d} / normal {r['normal']:5d}")


if __name__ == "__main__":
    main()
