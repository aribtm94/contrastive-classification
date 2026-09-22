"""
Bangun crop + manifest dari HEN v2 (COCO) - ayam MATI vs ayam HIDUP.

Kenapa skrip terpisah dari build_crops.py
-----------------------------------------
Di build_crops.py kelas mati datang dari anotasi manusia dan kelas hidup dari
DETEKSI pada dataset lain (PIO, CCTV). Di sini kedua kelas dianotasi manusia,
dari zip yang sama, sering dari FRAME yang sama - jadi tidak ada tahap deteksi
sama sekali. Yang dipakai ulang adalah KONTRAK keluarannya (manifest 11 kolom,
split lock, split per foto), bukan jalur pengambilan datanya.

Alasan teknis yang mengikat: balance_and_split() di build_crops.py sengaja
melempar ValueError("satu base_image punya dua label") karena pada susunan lama
kondisi itu berarti data rusak. Di sini frame campur - frame yang memuat ayam
mati DAN ayam hidup sekaligus - adalah justru inti susunannya. Presedennya
sudah ada: build_manifest_sdnet.py lahir karena alasan yang persis sama.

Dua susunan (crops.hen_mode)
----------------------------
campuran : HANYA frame yang memuat kedua kelas. 40 foto unik -> 87 mati /
           146 hidup. Lantai jalan pintas terukur 0.6903 (saturasi) - satu-
           satunya susunan HEN dengan lantai di bawah 0.70.
semua    : semua frame beranotasi. 1399 foto unik -> 719 mati / 4378 hidup.
           Lantainya 0.8182 (sisi_pendek): luas relatif ayam mati median
           0.1823 vs hidup 0.0309, hampir 6x. Susunan ini BUKAN klaim, ia
           pembanding - selisih skornya terhadap 'campuran' yang mengukur
           berapa banyak skor berasal dari jalan pintas.

KENAPA domain SAMA UNTUK KEDUA KELAS
------------------------------------
Kolom domain di sini "hen_kandang" untuk mati maupun hidup. Itu bukan
kemalasan penamaan, itu inti susunannya: pada percobaan sebelumnya label
ternyata bisa dibaca dari domain (latih dengan PIO -> bacc 1.0 palsu karena
label = dataset asal). Dengan kedua kelas berasal dari satu kandang, satu
kamera, dan pada susunan 'campuran' bahkan satu frame, "label = domain"
menjadi mustahil secara konstruksi, bukan sekadar berharap.

coco_gt ADALAH HIMPUNAN BAGIAN HEN v2
-------------------------------------
Terukur: 36/36 foto sumber dataset "dead-chikens" (crops.dead_root pada config
lama) punya kembaran di antara 429 foto ber-Dead HEN v2, MAE thumbnail < 1.0
untuk semuanya, 13 pasang tepat 0.000. Namanya beda hanya pada tanda hubung
ganda ("image--26-" vs "image-26-"), itu sebabnya irisan stem naif memberi 0.
Konsekuensinya "latih dari HEN v2 + coco_gt" secara aritmetika = HEN v2 saja,
dan skrip ini memakai SATU origin "hen_gt". Yang hilang bukan data, hanya
nama yang menyesatkan - berikut konflik split yang menyertainya (tanpa
pelipatan ini: 6 foto coco_gt-train tapi HEN-valid, 12 coco_gt-val tapi
HEN-train, 3 lainnya menyeberang).

SPLIT BAWAAN ZIP TIDAK DIPAKAI
------------------------------
Terukur: 3929 frame beranotasi hanya berisi 1399 FOTO unik, dan 529 foto di
antaranya tersebar lintas split train/valid/test bawaan Roboflow. Mengikuti
split zip berarti crop dari foto yang sama muncul di train dan val sekaligus.
Karena itu split dibangun ulang di tingkat foto; source_split disimpan hanya
sebagai jejak.

Jalankan:
    python src/build_crops_hen.py --config configs/config_hen_campur.yaml
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import crop_box, imwrite, load_config, resolve, save_json, to_square

# --------------------------------------------------------------------------- #
# Kategori HEN v2. Zip ini memuat tiga ejaan "healthy" dan dua ejaan "dead",
# ditambah dua kategori sampah. Semua dipetakan eksplisit: id di luar daftar
# ini menghentikan skrip, TIDAK dilewati diam-diam, supaya penambahan kelas
# pada ekspor berikutnya tidak lolos tanpa ada yang tahu.
#   0 healthy-sickv2  -> sampah: NOL anotasi di seluruh zip
#   1 "0"             -> sampah: 5 anotasi di train, 5 di test, tanpa makna
# --------------------------------------------------------------------------- #
KATEGORI_MATI = {2, 3}          # Dead, Dead_chicken
KATEGORI_HIDUP = {4, 5, 6}      # Healthly, Healthy, healthy
KATEGORI_SAMPAH = {0, 1}        # healthy-sickv2, "0"
KATEGORI_DIKENAL = KATEGORI_MATI | KATEGORI_HIDUP | KATEGORI_SAMPAH

NAMA_LABEL = {0: "hidup", 1: "mati"}
ORIGIN = "hen_gt"
DOMAIN = "hen_kandang"
SPLIT_ZIP = ("train", "valid", "test")

# Nama berkas Roboflow: "image-26-_jpg.rf.d9c5b948....jpg"
_POLA_RF = re.compile(r"_(jpg|jpeg|png)\.rf\..*$", re.IGNORECASE)


def stem_roboflow(nama: str) -> str:
    """Identitas foto asal menurut NAMA berkas. Satu cabang, bukan kunci final.

    Regex identik build_crops.base_image_id - keduanya membaca konvensi nama
    Roboflow yang sama. Stem sendirian TIDAK cukup jadi kunci grup: stem
    generik "images-2-" terukur memuat foto tak berkaitan (korelasi -0.02)
    SEKALIGUS sepasang kembar (0.9992). Lihat kelompokkan_foto().
    """
    stem = _POLA_RF.sub("", Path(nama).name)
    return Path(stem).stem if "." in stem else stem


def sidik_kecil(blob: bytes) -> np.ndarray | None:
    """Sidik piksel 32x32 abu, ternormalisasi, untuk korelasi antar frame.

    Dipakai karena hash EKSAK tidak bisa mendeteksi kebocoran di sini:
    Roboflow mengekspor ulang foto yang sama pada 320/608/640 px, dan
    resampling yang berbeda menggeser byte walau fotonya identik. Hash apa pun
    - termasuk hash atas thumbnail - akan melaporkan "tidak ada duplikat"
    secara salah. Korelasi piksel tidak punya masalah itu.
    """
    img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    kecil = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    kecil -= kecil.mean()
    norm = float(np.linalg.norm(kecil))
    return None if norm < 1e-6 else (kecil / norm).ravel()


def baca_split_coco(z: zipfile.ZipFile, split: str) -> dict:
    """Baca <split>/_annotations.coco.json -> {image_id: {...}}.

    Beda mendasar dari build_crops.load_coco_split: fungsi itu MENGABAIKAN
    category_id sepenuhnya dan menganggap setiap anotasi sebagai ayam mati.
    Benar untuk dataset "dead-chikens" yang cuma punya satu kelas, fatal di
    sini - HEN v2 punya tujuh kategori dan dua di antaranya sampah.
    """
    nama_ann = f"{split}/_annotations.coco.json"
    if nama_ann not in z.namelist():
        raise SystemExit(f"anotasi tidak ada di dalam zip: {nama_ann}")
    data = json.loads(z.read(nama_ann).decode("utf-8"))

    tak_dikenal = {c["id"]: c.get("name", "?") for c in data.get("categories", [])
                   if c["id"] not in KATEGORI_DIKENAL}
    if tak_dikenal:
        raise SystemExit(
            f"kategori tak dikenal di {split}: {tak_dikenal}. "
            f"Daftarkan di KATEGORI_MATI/HIDUP/SAMPAH sebelum melanjutkan - "
            f"melewatinya diam-diam akan menghilangkan objek tanpa jejak.")

    per_gambar = {}
    for im in data["images"]:
        per_gambar[im["id"]] = {
            "nama": im["file_name"],
            "dalam_zip": f"{split}/{im['file_name']}",
            "split_zip": split,
            "w": int(im.get("width") or 0),
            "h": int(im.get("height") or 0),
            "box": {0: [], 1: []},
        }

    dibuang = Counter()
    for a in data["annotations"]:
        kat = a["category_id"]
        if kat in KATEGORI_SAMPAH:
            dibuang[str(kat)] += 1
            continue
        label = 1 if kat in KATEGORI_MATI else 0
        x, y, w, h = a["bbox"]                                 # COCO = xywh
        g = per_gambar.get(a["image_id"])
        if g is None:
            dibuang["image_id_hilang"] += 1
            continue
        g["box"][label].append([x, y, x + w, y + h])           # -> xyxy
    return {"gambar": per_gambar, "dibuang": dibuang}


# --------------------------------------------------------------------------- #
# Kunci grup foto
# --------------------------------------------------------------------------- #
class _Union:
    """Union-find sederhana atas indeks frame."""

    def __init__(self, n: int):
        self.p = list(range(n))

    def cari(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def gabung(self, a: int, b: int) -> None:
        ra, rb = self.cari(a), self.cari(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def kelompokkan_foto(frames: list[dict], ambang: float = 0.97) -> list[int]:
    """Kelompokkan frame yang sebenarnya satu FOTO. Kembalikan id foto per frame.

    Dua relasi, digabung union-find:

        satu_foto(a, b)  iff  stem_roboflow(a) == stem_roboflow(b)
                         OR   korelasi_piksel_32x32(a, b) >= ambang

    Kenapa keduanya, bukan salah satu:
      - stem sendirian UNDER-merge di sini. Terukur: pada ambang 0.97 ada 465
        pasangan kembar yang stem-nya BERBEDA, melibatkan 177 frame; memakai
        stem saja menyisakan 1070 kelompok, menambahkan cabang korelasi
        menurunkannya ke 1034. Jadi cabang korelasi bukan hiasan - ia yang
        menutup 36 kebocoran yang stem lewatkan.
      - korelasi sendirian juga under-merge kalau satu sisi dipotong/di-pad
        beda sehingga korelasinya jatuh di bawah ambang, sementara stem-nya
        masih menjadi bukti asal yang sama.
    Arah galat gabungan hanya bisa ke OVER-merge, yang biayanya granularitas
    split. Under-merge biayanya kebocoran train<->val - jauh lebih mahal, dan
    tidak terlihat dari hasil.

    SOAL AMBANG 0.97 - KOREKSI ATAS ALASAN YANG SEBELUMNYA DIPAKAI. Rencana
    membenarkan 0.97 dengan "celah terukur: kembar 0.947-0.9999, tak berkaitan
    -0.02, nol di antara 0.05 dan 0.94". Celah itu TIDAK ADA pada populasi yang
    sebenarnya dipakai; ia hanya muncul pada 76 pasangan ber-stem sama di frame
    campur. Diukur atas seluruh 6840 pasangan dalam-stem sebarannya MENERUS:
    468 pasangan di [0.96,0.97) dan 774 di [0.97,0.98). Pasangan LINTAS stem
    memang terpisah jelas (5.9 juta pasangan di bawah 0.5, lalu hanya ~100-140
    per bin di atas 0.96), dan di situlah cabang ini bekerja.

    Maka 0.97 dipertahankan BUKAN karena ada celah, melainkan karena arah
    galatnya aman dan hasilnya tak sensitif: 0.97 -> 1034 foto, 0.99 -> 1040,
    0.995 -> 1048. Selisih 14 kelompok dari 1034 atas rentang ambang selebar
    itu; kesimpulan apa pun yang berubah karenanya tidak akan kokoh. Skrip
    tetap mencetak pasangan termerge-terlemah dan tak-termerge-terkuat setiap
    kali jalan supaya keputusan ini bisa diaudit ulang, bukan dipercaya.

    Biaya hitung: 3929 frame x 1024 dimensi -> satu perkalian matriks 3929^2,
    sekitar 60 MB float32. Sengaja tidak dipotong per-stem lagi; pemotongan
    itulah yang membuat cabang korelasi jadi kode mati pada jalan pertama.
    """
    u = _Union(len(frames))
    per_stem: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(frames):
        per_stem[f["stem"]].append(i)
    for anggota in per_stem.values():
        for a in anggota[1:]:
            u.gabung(anggota[0], a)                  # cabang 1: stem sama

    # Cabang 2: korelasi piksel, SELURUH pasangan.
    punya = [i for i, f in enumerate(frames) if f.get("sidik") is not None]
    n_lintas = 0
    termerge_terlemah, tak_termerge_terkuat = 1.0, -1.0
    if punya:
        M = np.stack([frames[i]["sidik"] for i in punya]).astype(np.float32)
        R = M @ M.T
        np.fill_diagonal(R, -9.0)
        stem_arr = np.array([frames[i]["stem"] for i in punya])
        beda_stem = stem_arr[:, None] != stem_arr[None, :]

        sah = R > -8.0
        atas = sah & (R >= ambang)
        bawah = sah & (R < ambang)
        if atas.any():
            termerge_terlemah = float(R[atas].min())
        if bawah.any():
            tak_termerge_terkuat = float(R[bawah].max())

        ia, ib = np.where(atas & beda_stem)
        for a, b in zip(ia.tolist(), ib.tolist()):
            if a < b:
                n_lintas += 1
                u.gabung(punya[a], punya[b])

    akar_ke_id, id_foto = {}, []
    for i in range(len(frames)):
        akar = u.cari(i)
        id_foto.append(akar_ke_id.setdefault(akar, len(akar_ke_id)))

    split_per_foto = defaultdict(set)
    for f, fid in zip(frames, id_foto):
        split_per_foto[fid].add(f["split_zip"])
    tersebar = sum(len(v) > 1 for v in split_per_foto.values())

    print(f"[hen] frame beranotasi      : {len(frames)}")
    print(f"[hen] stem unik             : {len(per_stem)}  <- kalau ini saja "
          f"dipakai, {len(per_stem) - len(akar_ke_id)} kembar lolos")
    print(f"[hen] FOTO unik             : {len(akar_ke_id)}")
    print(f"[hen] pasangan lintas-stem digabung korelasi: {n_lintas}")
    print(f"[hen] korelasi: termerge terlemah {termerge_terlemah:.4f} | "
          f"tak-termerge terkuat {tak_termerge_terkuat:.4f} (ambang {ambang}) "
          f"-> MENERUS, bukan celah; lihat docstring")
    print(f"[hen] foto lintas split zip : {tersebar}"
          + ("  -> split zip BOCOR, dibangun ulang" if tersebar else ""))
    return id_foto


def pilih_frame(frames: list[dict], mode: str) -> list[dict]:
    """Saring frame menurut susunan: 'campuran' atau 'semua'."""
    if mode == "semua":
        return list(frames)
    if mode != "campuran":
        raise SystemExit(f"crops.hen_mode tidak dikenal: {mode} "
                         f"(pilih 'campuran' atau 'semua')")
    return [f for f in frames if f["box"][1] and f["box"][0]]


def wakil_foto(anggota: list[dict]) -> dict:
    """Satu berkas per foto: resolusi terbesar.

    De-dup ekspor ulang Roboflow. Resolusi terbesar dipilih karena versi
    terkecil sudah kehilangan detail yang tidak bisa dikembalikan; kalau seri,
    nama berkas dipakai supaya pilihannya deterministik antar mesin.
    """
    return sorted(anggota, key=lambda f: (-(f["w"] * f["h"]), f["nama"]))[0]


def bagi_per_foto(foto_ke_jumlah: dict, nama_split: list,
                  rasio: list, rng: random.Random) -> dict:
    """Petakan tiap foto ke satu split; seluruh crop foto itu ikut ke sana.

    Meniru aturan build_manifest_sdnet.bagi_per_foto (split di tingkat foto,
    tanpa stratifikasi per kelas - pada susunan 'campuran' setiap foto memberi
    kedua kelas sekaligus, jadi keseimbangan muncul sendiri).

    Satu tambahan yang tidak ada di sana: rasio dihitung atas jumlah CROP,
    bukan jumlah foto. Satu foto HEN menyumbang 1 sampai 79 box, jadi membagi
    60/40 per foto bisa menghasilkan split crop yang jauh dari 60/40.
    """
    if len(nama_split) != 2:
        raise SystemExit(f"susunan ini hanya mendukung train+val, "
                         f"dapat {nama_split}")
    foto = sorted(foto_ke_jumlah)
    if len(foto) < 4:
        raise SystemExit(f"butuh minimal 4 foto, hanya ada {len(foto)}")
    rng.shuffle(foto)

    target_val = sum(foto_ke_jumlah.values()) * float(rasio[1])
    petakan, di_val = {}, 0
    for f in foto:
        # Isi val dulu sampai porsi CROP-nya tercapai, sisanya train. Foto
        # terakhir tidak boleh menghabiskan daftar: tiap split wajib berisi.
        if di_val < target_val and len(petakan) < len(foto) - 1:
            petakan[f] = nama_split[1]
            di_val += foto_ke_jumlah[f]
        else:
            petakan[f] = nama_split[0]
    return petakan


# --------------------------------------------------------------------------- #
# Pengumpulan crop
# --------------------------------------------------------------------------- #
def equalize(c, ccfg: dict):
    """Samakan resolusi EFEKTIF sebelum letterbox (crops.equalize_resolution).

    Salinan build_crops.equalize - disalin, bukan diimpor, supaya skrip ini
    tidak menarik seluruh build_crops.py (yang membuka dataset PIO dan memuat
    YOLO saat diimpor). Perilakunya harus tetap identik: lengan asli dan lengan
    eq48 hanya boleh berbeda pada langkah ini.
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


def batasi_hidup(baris: list[dict], maks_rasio: float,
                 rng: random.Random) -> tuple[list[dict], int]:
    """Batasi crop hidup jadi maks_rasio x jumlah crop mati, merata antar foto.

    Round-robin antar foto meniru build_crops.balance_and_split: memotong dari
    daftar terurut akan menumpuk sisa crop di beberapa foto saja, dan foto
    ITU yang lalu menentukan skor.
    """
    mati = [r for r in baris if r["label"] == 1]
    hidup = [r for r in baris if r["label"] == 0]
    batas = int(round(len(mati) * maks_rasio))
    if maks_rasio <= 0 or len(hidup) <= batas:
        return baris, 0

    per_foto = defaultdict(list)
    for r in hidup:
        per_foto[r["base_image"]].append(r)
    for v in per_foto.values():
        rng.shuffle(v)

    foto, terpilih, i = sorted(per_foto), [], 0
    while len(terpilih) < batas:
        bertambah = False
        for f in foto:
            if i < len(per_foto[f]):
                terpilih.append(per_foto[f][i])
                bertambah = True
                if len(terpilih) >= batas:
                    break
        if not bertambah:
            break
        i += 1
    return mati + terpilih, len(hidup) - len(terpilih)


def kumpulkan_hen(cfg: dict) -> dict:
    ccfg = cfg["crops"]
    rng = random.Random(cfg["seed"])
    zip_path = resolve(ccfg["hen_zip"])
    out_dir = resolve(ccfg["out_dir"])
    size = int(cfg["classifier"]["image_size"])
    mode_kotak = cfg["classifier"]["resize_mode"]
    pad = float(ccfg.get("bbox_padding", 0.0))
    min_box = int(ccfg["min_box_size"])
    mode = str(ccfg["hen_mode"])
    ambang = float(ccfg.get("hen_ambang_korelasi", 0.97))
    dedup = bool(ccfg.get("hen_dedup_ekspor_ulang", True))

    if not zip_path.exists():
        raise SystemExit(f"zip HEN v2 tidak ada: {zip_path}")

    dibuang = Counter()
    with zipfile.ZipFile(zip_path) as z:
        frames = []
        for split in SPLIT_ZIP:
            hasil = baca_split_coco(z, split)
            dibuang.update(hasil["dibuang"])
            for g in hasil["gambar"].values():
                if not (g["box"][0] or g["box"][1]):
                    dibuang["frame_tanpa_anotasi"] += 1
                    continue
                g["stem"] = stem_roboflow(g["nama"])
                frames.append(g)

        print(f"[hen] kategori sampah dibuang: {dict(dibuang)}")
        for f in frames:
            f["sidik"] = sidik_kecil(z.read(f["dalam_zip"]))

        id_foto = kelompokkan_foto(frames, ambang)
        for f, fid in zip(frames, id_foto):
            f["id_foto"] = fid

        terpilih = pilih_frame(frames, mode)
        print(f"[hen] susunan '{mode}'      : {len(terpilih)} frame terpilih")

        per_foto = defaultdict(list)
        for f in terpilih:
            per_foto[f["id_foto"]].append(f)
        if dedup:
            per_foto = {k: [wakil_foto(v)] for k, v in per_foto.items()}
        print(f"[hen] foto terpakai         : {len(per_foto)}"
              f"  (dedup ekspor ulang: {dedup})")

        # Jumlah crop per foto dihitung dulu supaya split bisa menyeimbangkan
        # porsi CROP, bukan porsi foto.
        jumlah_per_foto = {}
        for fid, anggota in per_foto.items():
            n = 0
            for f in anggota:
                for label in (0, 1):
                    for b in f["box"][label]:
                        if min(b[2] - b[0], b[3] - b[1]) >= min_box:
                            n += 1
            jumlah_per_foto[fid] = n
        jumlah_per_foto = {k: v for k, v in jumlah_per_foto.items() if v > 0}

        petakan = bagi_per_foto(jumlah_per_foto, list(ccfg["split_names"]),
                                [float(x) for x in ccfg["split_ratio"]], rng)

        baris = []
        for fid in sorted(jumlah_per_foto):
            for f in per_foto[fid]:
                img = cv2.imdecode(np.frombuffer(z.read(f["dalam_zip"]), np.uint8),
                                   cv2.IMREAD_COLOR)
                if img is None:
                    dibuang["gagal_dibaca"] += 1
                    continue
                for label in (0, 1):
                    for j, b in enumerate(f["box"][label]):
                        if min(b[2] - b[0], b[3] - b[1]) < min_box:
                            dibuang[f"box_kecil_{NAMA_LABEL[label]}"] += 1
                            continue
                        c = crop_box(img, b, pad)
                        if c is None:
                            dibuang["crop_kosong"] += 1
                            continue
                        c = to_square(equalize(c, ccfg), size, mode_kotak)
                        rel = (f"{NAMA_LABEL[label]}/"
                               f"hen_f{fid:04d}_{Path(f['nama']).stem}_{label}{j}.jpg")
                        if not imwrite(out_dir / rel, c):
                            dibuang["gagal_ditulis"] += 1
                            continue
                        baris.append({
                            "path": rel,
                            "label": label,
                            "label_name": NAMA_LABEL[label],
                            "split": petakan[fid],
                            # Jejak split asal Roboflow. DISIMPAN SAJA - tidak
                            # dipakai membagi, karena 529 foto tersebar lintas
                            # split bawaan zip.
                            "source_split": f["split_zip"],
                            "base_image": f"hen_f{fid:04d}",
                            "src_file": f["dalam_zip"],
                            "bbox": [int(round(v)) for v in b],
                            "conf": 1.0,
                            "origin": ORIGIN,
                            # Sama untuk kedua kelas - lihat docstring modul.
                            "domain": DOMAIN,
                        })

    maks = float(ccfg.get("max_alive_per_dead", 0) or 0)
    baris, dipangkas = batasi_hidup(baris, maks, rng)
    if dipangkas:
        print(f"[hen] crop hidup dipangkas  : {dipangkas} "
              f"(max_alive_per_dead={maks})")
        dibuang["hidup_dipangkas_seimbang"] += dipangkas

    tulis_manifest(baris, cfg)
    tulis_atau_periksa_split_lock(baris, cfg)
    return ringkas(baris, dibuang, resolve(ccfg["manifest"]))


def tulis_manifest(rows: list[dict], cfg: dict) -> None:
    """Kolom dan urutannya identik build_crops.write_manifest - itu kontraknya."""
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


def tulis_atau_periksa_split_lock(rows: list[dict], cfg: dict) -> None:
    """Bekukan pembagian foto HEN v2. Meniru build_crops.write_or_validate_split_lock.

    Dua kunci tambahan: builder + builder_sha256. Sebabnya rantai kustodi.
    train.source_hashes() memeriksa daftar berkas yang DIPAKU di train.py, dan
    berkas ini tidak ada di sana; menambahkannya berarti menyunting train.py
    dan membuat hash train.py di empat registry beku tak tereproduksi. Karena
    split lock sendiri ikut ter-hash ke registry lewat split_lock_sha256,
    mencatat hash builder DI SINI menutup rantainya - registry -> split lock
    -> versi kode yang membuat crop - tanpa menyentuh berkas beku mana pun.
    """
    protocol = cfg["protocol"]
    lock_path = resolve(protocol["split_lock"])
    manifest = resolve(cfg["crops"]["manifest"])
    split_names = list(cfg["crops"]["split_names"])

    mapping = {}
    for r in rows:
        lama = mapping.setdefault(r["base_image"], r["split"])
        if lama != r["split"]:
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
        nama_manifest = str(manifest.relative_to(resolve("."))).replace("\\", "/")
    except ValueError:
        nama_manifest = str(manifest)
    payload = {
        "schema_version": 1,
        "purpose": f"development_only_hen_v2_{cfg['crops']['hen_mode']}",
        "seed": int(cfg["seed"]),
        "manifest": nama_manifest,
        "manifest_sha256": _sha256(manifest),
        "builder": "src/build_crops_hen.py",
        "builder_sha256": _sha256(Path(__file__)),
        "hen_mode": str(cfg["crops"]["hen_mode"]),
        "hen_ambang_korelasi": float(cfg["crops"].get("hen_ambang_korelasi", 0.97)),
        "counts": counts,
        "n_base_images": len(mapping),
        "base_image_to_split": dict(sorted(mapping.items())),
        "test_dataset_excluded": "C:/Arib/CCTV/patnet-pure/dataset/chick",
    }

    if lock_path.exists():
        with open(lock_path, encoding="utf-8") as f:
            current = json.load(f)
        pembanding = ("seed", "counts", "n_base_images", "base_image_to_split",
                      "hen_mode", "hen_ambang_korelasi")
        beda = [k for k in pembanding if current.get(k) != payload.get(k)]
        if beda:
            raise ValueError(f"split lock berubah pada {beda}: {lock_path}")
        if current.get("manifest_sha256") != payload["manifest_sha256"]:
            raise ValueError(f"hash manifest tidak cocok dengan split lock: {lock_path}")
        if current.get("builder_sha256") != payload["builder_sha256"]:
            # Bukan kegagalan: kode builder boleh berubah. Tapi harus TERLIHAT,
            # karena crop yang ada dibuat oleh versi yang lain.
            print(f"[hen] PERINGATAN: builder_sha256 berubah dari "
                  f"{current.get('builder_sha256', '?')[:12]} ke "
                  f"{payload['builder_sha256'][:12]} - crop di disk dibuat "
                  f"versi kode lain. Catat di laporan.")
        print(f"[hen] split lock            : cocok ({lock_path})")
        return

    save_json(payload, lock_path)
    print(f"[hen] split lock            : dibuat ({lock_path})")


def ringkas(rows: list[dict], dibuang: Counter, manifest: Path) -> dict:
    per_split = defaultdict(Counter)
    foto_per_split = defaultdict(set)
    for r in rows:
        per_split[r["split"]][r["label_name"]] += 1
        foto_per_split[r["split"]].add(r["base_image"])

    print(f"\n[hen] manifest -> {manifest}")
    print(f"[hen] total crop            : {len(rows)}")
    for sp in sorted(per_split):
        c = per_split[sp]
        print(f"   {sp:7s} mati {c['mati']:5d} / hidup {c['hidup']:5d}"
              f"   ({len(foto_per_split[sp])} foto)")
    if dibuang:
        print(f"[hen] dibuang: {dict(dibuang)}")

    milik = {}
    for r in rows:
        lama = milik.setdefault(r["base_image"], r["split"])
        if lama != r["split"]:
            raise SystemExit(
                f"BOCOR: foto {r['base_image']} ada di {lama} dan {r['split']}")
    print("[hen] periksa bocor: tidak ada foto yang lintas split")

    # expected_counts yang siap disalin ke config, supaya tidak dikarang.
    counts = {sp: {"alive": per_split[sp]["hidup"], "dead": per_split[sp]["mati"],
                   "total": per_split[sp]["hidup"] + per_split[sp]["mati"]}
              for sp in sorted(per_split)}
    print(f"[hen] expected_counts untuk config:\n      {json.dumps(counts)}")

    info = {
        "total": len(rows),
        "per_split": {k: dict(v) for k, v in per_split.items()},
        "foto_per_split": {k: len(v) for k, v in foto_per_split.items()},
        "expected_counts": counts,
        "dibuang": dict(dibuang),
    }
    save_json(info, manifest.parent / "crops_summary.json")
    return info


def main():
    ap = argparse.ArgumentParser(
        description="Bangun crop + manifest HEN v2 (mati vs hidup)")
    ap.add_argument("--config", default="configs/config_hen_campur.yaml")
    a = ap.parse_args()
    kumpulkan_hen(load_config(a.config))


if __name__ == "__main__":
    main()
