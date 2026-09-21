"""
Bangun crop + manifest SDNET2018 (retak beton) untuk UJI KEWARASAN pipeline.

Kenapa skrip terpisah dari build_crops.py: di sana sumber datanya adalah
bounding box hasil deteksi pada foto ayam. Di sini tidak ada deteksi sama
sekali - SDNET2018 sudah berupa ubin 256x256 yang labelnya melekat pada ubin
itu sendiri. Yang dipakai ulang adalah KONTRAK keluarannya (manifest 11 kolom)
dan aturan anti-bocornya, bukan jalur pengambilan datanya.

Keluaran mengikuti kontrak yang dibaca dataset.read_manifest:

    data/crops_sdnet/
        manifest.csv        path,label,label_name,split,source_split,
                            base_image,src_file,bbox,conf,origin,domain
        retak/*.jpg
        utuh/*.jpg

ATURAN ANTI-BOCOR YANG MENGIKAT SKRIP INI
-----------------------------------------
Nama berkas SDNET memuat id foto sumbernya: "7001-115.jpg" adalah ubin ke-115
dari foto "7001". Satu foto menyumbang sampai 249 ubin. Kalau displit per ubin,
potongan dari dinding beton yang sama akan muncul di train dan test sekaligus,
dan skornya jadi palsu - persis kesalahan yang dilarang aturan proyek. Karena
itu split dilakukan per base_image.

Satu perbedaan penting dari data ayam: di SDNET2018 satu foto sumber menyumbang
ubin retak SEKALIGUS ubin utuh (terukur: irisan 54/54 deck, 72/72 wall, 99/99
pavement). Jadi base_image TIDAK bisa dipetakan ke satu label - itu kondisi
normal di sini, bukan kerusakan data. Konsekuensinya balance_and_split() dari
build_crops.py tidak bisa dipakai apa adanya: fungsi itu sengaja melempar
ValueError "satu base_image punya dua label" karena pada data ayam kondisi itu
memang berarti ada kesalahan. Aturannya yang ditiru di sini (kelompokkan per
foto, ambil merata antar foto), bukan kodenya.

Justru karena satu foto memberi kedua label, split per foto di sini lebih kuat
daripada pada data ayam: domain foto (beton yang mana, pencahayaan apa) menjadi
sama sekali tidak informatif terhadap label. "label = domain" mustahil.

Pemakaian:
    python src/build_manifest_sdnet.py --config configs/config_sdnet.yaml
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, imwrite, resolve, save_json

# Pemetaan folder di dalam SDNET2018.zip -> (sub-domain, label).
# Huruf pertama nama folder daun adalah labelnya: C = cracked, U = uncracked.
LEAVES = {
    "D/CD": ("deck",     1), "D/UD": ("deck",     0),
    "W/CW": ("wall",     1), "W/UW": ("wall",     0),
    "P/CP": ("pavement", 1), "P/UP": ("pavement", 0),
}
LABEL_NAMES = {0: "utuh", 1: "retak"}

# Zip luar membungkus satu zip lagi; ini jalur zip dalamnya.
INNER_ZIP = "DATA_Maguire_20180517_ALL/SDNET2018.zip"


def buka_zip_dalam(zip_luar: Path) -> zipfile.ZipFile:
    """SDNET2018 dikemas sebagai zip di dalam zip. Buka yang dalam di memori."""
    luar = zipfile.ZipFile(zip_luar)
    if INNER_ZIP not in luar.namelist():
        raise SystemExit(f"{INNER_ZIP} tidak ada di dalam {zip_luar}")
    return zipfile.ZipFile(io.BytesIO(luar.read(INNER_ZIP)))


def id_foto(nama_berkas: str) -> str:
    """"7001-115.jpg" -> "7001". Inilah kunci anti-bocor antar split."""
    return re.split(r"[-_.]", os.path.basename(nama_berkas))[0]


def kumpulkan(inner: zipfile.ZipFile) -> dict:
    """Daftar nama berkas di dalam zip, dikelompokkan per (sub-domain, label)."""
    hasil: dict = defaultdict(list)
    for nama in inner.namelist():
        if not nama.lower().endswith(".jpg"):
            continue
        daun = "/".join(nama.split("/")[:2])
        if daun in LEAVES:
            hasil[LEAVES[daun]].append(nama)
    for v in hasil.values():
        v.sort()
    return hasil


def ambil_merata(nama_nama: list, jumlah: int, rng: random.Random) -> list:
    """
    Ambil sejumlah ubin merata di seluruh foto sumber, bukan menumpuk di
    beberapa foto saja. Pola round-robin ini disalin dari cara build_crops.py
    membatasi jumlah crop ayam hidup (fungsi balance_and_split).
    """
    if len(nama_nama) <= jumlah:
        return list(nama_nama)
    per_foto = defaultdict(list)
    for n in nama_nama:
        per_foto[id_foto(n)].append(n)
    for v in per_foto.values():
        rng.shuffle(v)

    foto_foto, terpilih, i = sorted(per_foto), [], 0
    while len(terpilih) < jumlah:
        bertambah = False
        for f in foto_foto:
            if i < len(per_foto[f]):
                terpilih.append(per_foto[f][i])
                bertambah = True
                if len(terpilih) >= jumlah:
                    break
        if not bertambah:
            break
        i += 1
    return terpilih


def bagi_per_foto(foto_foto: list, nama_split: list,
                  rasio: list, rng: random.Random) -> dict:
    """
    Petakan tiap foto sumber ke satu split. Meniru aturan
    build_crops.balance_and_split: pembagian dilakukan pada tingkat FOTO,
    lalu seluruh ubin dari foto itu ikut ke split yang sama.

    Di sini tidak distratifikasi per kelas - tidak perlu dan tidak bisa,
    karena setiap foto sumber SDNET menyumbang ubin retak dan utuh sekaligus.
    Keseimbangan kelas di tiap split muncul dengan sendirinya.
    """
    foto_foto = sorted(foto_foto)
    rng.shuffle(foto_foto)
    n = len(foto_foto)
    if n < 3:
        raise SystemExit(f"butuh minimal 3 foto sumber, hanya ada {n}")
    n_tr = min(int(round(n * rasio[0])), n - 2)
    n_va = max(1, min(int(round(n * rasio[1])), n - n_tr - 1))
    petakan = {}
    for i, f in enumerate(foto_foto):
        petakan[f] = (nama_split[0] if i < n_tr
                      else (nama_split[1] if i < n_tr + n_va
                            else nama_split[2]))
    return petakan


def tulis_ubin(inner: zipfile.ZipFile, nama_dalam_zip: str, tujuan: Path):
    """Tulis satu ubin ke disk apa adanya. Kembalikan (lebar, tinggi)."""
    img = cv2.imdecode(np.frombuffer(inner.read(nama_dalam_zip), np.uint8),
                       cv2.IMREAD_COLOR)
    if img is None:
        return None
    if not imwrite(tujuan, img):
        return None
    h, w = img.shape[:2]
    return w, h


def bangun(cfg: dict) -> dict:
    ccfg = cfg["crops"]
    rng = random.Random(cfg["seed"])
    out_dir = resolve(ccfg["out_dir"])
    inner = buka_zip_dalam(resolve(ccfg["sdnet_zip"]))
    per_kelompok = kumpulkan(inner)

    dom_latih = ccfg["train_domain"]
    dom_holdout = list(ccfg["holdout_domains"])
    baris, dilewati = [], Counter()

    # --- 1. domain latih: dibagi train / val / test per foto sumber ---
    terpilih_latih = {
        1: ambil_merata(per_kelompok[(dom_latih, 1)],
                        int(ccfg["n_retak_train_domain"]), rng),
        0: ambil_merata(per_kelompok[(dom_latih, 0)],
                        int(ccfg["n_utuh_train_domain"]), rng),
    }
    foto_latih = {id_foto(n) for lst in terpilih_latih.values() for n in lst}
    petakan = bagi_per_foto(sorted(foto_latih), list(ccfg["split_names"]),
                            [float(x) for x in ccfg["split_ratio"]], rng)

    for label, nama_nama in terpilih_latih.items():
        for nama in nama_nama:
            baris.append((nama, dom_latih, label, petakan[id_foto(nama)]))

    # --- 2. domain hold-out: seluruhnya jadi split test_<domain> ---
    # Tidak dibagi sama sekali; tidak pernah dilihat saat latih.
    for dom in dom_holdout:
        tag = f"test_{'pave' if dom == 'pavement' else dom}"
        for label, kunci in ((1, "n_retak_holdout"), (0, "n_utuh_holdout")):
            for nama in ambil_merata(per_kelompok[(dom, label)],
                                     int(ccfg[kunci]), rng):
                baris.append((nama, dom, label, tag))

    # --- 3. tulis ubin ke disk + susun manifest ---
    manifes = []
    for nama, dom, label, split in baris:
        nama_kelas = LABEL_NAMES[label]
        # Nama berkas diberi awalan domain: 7001-115.jpg ada di deck DAN wall.
        rel = f"{nama_kelas}/{dom}_{os.path.basename(nama)}"
        ukuran = tulis_ubin(inner, nama, out_dir / rel)
        if ukuran is None:
            dilewati["gagal_dibaca"] += 1
            continue
        w, h = ukuran
        manifes.append({
            "path": rel,
            "label": label,
            "label_name": nama_kelas,
            "split": split,
            "source_split": dom,
            # base_image diberi awalan domain juga: id foto yang sama bisa
            # muncul di lebih dari satu sub-domain dengan penomoran terpisah.
            "base_image": f"{dom}_{id_foto(nama)}",
            "src_file": nama,
            # Ubin utuh, bukan hasil deteksi. Dicatat apa adanya supaya
            # eval_shortcut_baseline tetap bisa membaca kolomnya; efeknya
            # ciri rasio_bbox dan ukuran_bbox jadi konstan - memang benar,
            # karena di sini tidak ada metadata bbox yang bisa bocor.
            "bbox": [0, 0, w, h],
            "conf": 1.0,
            "origin": "sdnet_tile",
            "domain": dom,
        })

    tulis_manifest(manifes, cfg)
    return ringkas(manifes, dilewati, resolve(ccfg["manifest"]))


def tulis_manifest(rows: list, cfg: dict) -> None:
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


def ringkas(rows: list, dilewati: Counter, manifest: Path) -> dict:
    per_split = defaultdict(Counter)
    foto_per_split = defaultdict(set)
    for r in rows:
        per_split[r["split"]][r["label_name"]] += 1
        foto_per_split[r["split"]].add(r["base_image"])

    print(f"\n[sdnet] manifest -> {manifest}")
    print(f"[sdnet] total ubin: {len(rows)}")
    for sp in sorted(per_split):
        c = per_split[sp]
        print(f"   {sp:11s} retak {c['retak']:5d} / utuh {c['utuh']:5d}"
              f"   ({len(foto_per_split[sp])} foto sumber)")
    if dilewati:
        print(f"[sdnet] dilewati: {dict(dilewati)}")

    # Penjaga anti-bocor: satu foto sumber tidak boleh muncul di dua split.
    milik = {}
    for r in rows:
        lama = milik.setdefault(r["base_image"], r["split"])
        if lama != r["split"]:
            raise SystemExit(
                f"BOCOR: foto {r['base_image']} ada di {lama} dan {r['split']}")
    print("[sdnet] periksa bocor: tidak ada foto sumber yang lintas split")

    info = {
        "total": len(rows),
        "per_split": {k: dict(v) for k, v in per_split.items()},
        "foto_per_split": {k: len(v) for k, v in foto_per_split.items()},
        "dilewati": dict(dilewati),
    }
    save_json(info, manifest.parent / "crops_summary.json")
    return info


def main():
    ap = argparse.ArgumentParser(
        description="Bangun crop + manifest SDNET2018 untuk uji kewarasan")
    ap.add_argument("--config", default="configs/config_sdnet.yaml")
    a = ap.parse_args()
    bangun(load_config(a.config))


if __name__ == "__main__":
    main()
