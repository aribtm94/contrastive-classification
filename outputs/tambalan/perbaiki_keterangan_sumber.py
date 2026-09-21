"""
Tambalan pasca-sweep: keterangan susunan data dibaca dari manifest, bukan paku.

MASALAH. Tiga tempat di kode memaku keterangan tentang susunan data sebagai
teks (lihat `domain_mati_kedua.md` bagian 4e, 4f, 4g). Selama hanya ada satu
susunan data, ketiganya kebetulan benar. Begitu lengan kedua ditambahkan
(archive_4 sebagai domain mati kedua), ketiganya salah sekaligus:

    (e) report_fixed_chick.keluarga()  -> label varian hanya menyebut "asli"/
        "eq48", tidak menyebut susunan datanya; dua susunan berbeda bisa jatuh
        ke baris tabel yang sama dan dirata-ratakan diam-diam.
    (f) report_fixed_chick.write_report() -> kalimat "ayam hidup PIO + ayam
        mati Roboflow" dicetak walau kelas matinya berisi dua sumber.
    (g) build_crops.write_or_validate_split_lock() -> field `purpose` berisi
        "development_only_pio_alive_roboflow_dead".

KENAPA TERPISAH DAN BARU DIJALANKAN SETELAH SWEEP. `src/build_crops.py` dan
`src/report_fixed_chick.py` keduanya ikut di-hash ke `registry.source_sha256`
lewat `train.source_hashes()`, yang membaca berkas itu DARI DISK pada langkah
pembekuan di akhir sweep. Mengubahnya selagi sweep berjalan membuat registry
mencatat hash kode yang BUKAN kode yang menghasilkan checkpoint-nya.

APA YANG DISENTUH SKRIP INI. Hanya satu berkas sumber dan dua split lock.
Registry TIDAK disentuh di sini - itu tugas `perbaiki_data_contract.py` (d).

Perhatikan: field `purpose` di split lock TIDAK ikut dibandingkan saat lock
diverifikasi ulang (`build_crops.py` hanya membandingkan seed, counts,
n_base_images, base_image_to_split, manifest_sha256), jadi memperbaikinya tidak
bisa membuat penjaga mana pun lolos atau gagal secara keliru. Tapi
`split_lock_sha256` yang sudah tercatat di registry akan berubah - itu
disengaja dan dicatat, sama seperti source_sha256 di butir (c).

Skrip ini tidak dijalankan otomatis dan tidak menulis apa pun tanpa --tulis.

Pemakaian:
    python outputs/tambalan/perbaiki_keterangan_sumber.py            # kering
    python outputs/tambalan/perbaiki_keterangan_sumber.py --tulis
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR / "src"))


# --------------------------------------------------------------------------
# (g) purpose di split lock
# --------------------------------------------------------------------------

def purpose_dari_manifest(manifest: Path) -> str:
    """
    Susun nilai `purpose` dari isi manifest.

    Bentuknya: development_only__hidup-<sumber>__mati-<sumber>[+<sumber>]
    Sumber diurutkan menurun berdasarkan jumlah crop, jadi yang terbesar dulu.
    """
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    per_kelas: dict[str, Counter] = {}
    for r in rows:
        per_kelas.setdefault(r["label_name"], Counter())[r["origin"]] += 1
    bagian = []
    for kelas in ("alive", "dead"):
        sumber = per_kelas.get(kelas, Counter())
        nama = "+".join(n for n, _ in sumber.most_common())
        bagian.append(f"{'hidup' if kelas == 'alive' else 'mati'}-{nama or 'kosong'}")
    return "development_only__" + "__".join(bagian)


def tambal_split_lock(lock: Path, tulis: bool) -> bool:
    if not lock.exists():
        print(f"  [lewat] tidak ada: {lock}")
        return False
    data = json.loads(lock.read_text(encoding="utf-8"))
    manifest = AKAR / data["manifest"]
    if not manifest.exists():
        print(f"  [lewat] manifest hilang: {manifest}")
        return False
    baru = purpose_dari_manifest(manifest)
    lama = data.get("purpose")
    if lama == baru:
        print(f"  [sama]  {lock.name}: {lama}")
        return False
    print(f"  {lock.name}")
    print(f"     LAMA: {lama}")
    print(f"     BARU: {baru}")
    if tulis:
        data["purpose"] = baru
        data["catatan_tambalan"] = (
            "purpose diperbaiki pasca-sweep dari manifest; nilai semula dipaku "
            "di build_crops.py dan menyebut satu sumber mati saja. Field lain "
            "tidak disentuh.")
        lock.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        print("     -> ditulis")
    return True


# --------------------------------------------------------------------------
# (e) + (f) teks yang dipaku di report_fixed_chick.py
# --------------------------------------------------------------------------

# (f) kalimat kontrak: dibaca dari data, bukan dipaku.
LAMA_F = (
    '        "Train dan validation hanya memakai ayam hidup PIO + ayam mati Roboflow. "\n'
    '        "Seluruh 18 frame ayam/chick dipakai sebagai test. Tidak ada angka test "\n'
    '        "yang dipakai memilih epoch, threshold, seed, checkpoint, atau scorer.", "",\n'
)

BARU_F = (
    '        kalimat_kontrak(data) + " "\n'
    '        "Seluruh 18 frame ayam/chick dipakai sebagai test. Tidak ada angka test "\n'
    '        "yang dipakai memilih epoch, threshold, seed, checkpoint, atau scorer.", "",\n'
)

FUNGSI_F = '''def kalimat_kontrak(data: dict) -> str:
    """
    Susun kalimat sumber train/validation dari `data_contract` registry.

    Semula kalimat ini dipaku: "ayam hidup PIO + ayam mati Roboflow". Itu benar
    selama hanya ada satu susunan data, tapi DIAM-DIAM salah untuk lengan yang
    kelas matinya berisi dua sumber (lihat domain_mati_kedua.md bagian 4f).
    Sekarang dibaca dari registry, yang sendiri dibaca dari manifest.
    """
    kontrak = (data.get("data_contract") or {})
    if not kontrak:
        # JSON lama tidak menyimpan data_contract; pertahankan teks semula
        # supaya laporan yang sudah ter-commit terbit byte-identik.
        return ("Train dan validation hanya memakai ayam hidup PIO + "
                "ayam mati Roboflow.")

    def daftar(kunci: str) -> str:
        nilai = kontrak.get(kunci)
        if isinstance(nilai, str):
            return nilai
        if isinstance(nilai, (list, tuple)) and nilai:
            return " + ".join(str(x) for x in nilai)
        return "?"

    return (f"Train dan validation hanya memakai ayam hidup {daftar('alive')} "
            f"dan ayam mati {daftar('dead')}.")


'''

# (e) kunci pengelompokan harus memuat identitas susunan data, bukan hanya
#     apakah crop di-equalize.
LAMA_E = (
    '    if not eq.get("enabled"):\n'
    '        return "asli"\n'
    "    return f\"eq{int(eq.get('target_short_side', 96))}\"\n"
)

BARU_E = (
    '    varian = ("asli" if not eq.get("enabled")\n'
    "              else f\"eq{int(eq.get('target_short_side', 96))}\")\n"
    '    nama_susunan = susunan(row)\n'
    '    return f"{nama_susunan}/{varian}" if nama_susunan else varian\n'
    '\n'
    '\n'
    'def susunan(row: dict) -> str:\n'
    '    """\n'
    '    Identitas SUSUNAN DATA sebuah run, terpisah dari varian crop-nya.\n'
    '\n'
    '    Tanpa ini `keluarga()` hanya menyebut apakah crop di-equalize, sehingga\n'
    '    dua susunan data yang berbeda (mis. 392 crop satu domain mati vs 598\n'
    '    crop dua domain mati) jatuh ke baris tabel yang sama dan dirata-ratakan\n'
    '    diam-diam kalau kedua registry pernah dilewatkan dalam satu perintah\n'
    '    (lihat domain_mati_kedua.md bagian 4e).\n'
    '\n'
    '    Dibaca dari nama folder crop di config snapshot - sumber yang sama yang\n'
    '    menentukan gambar mana yang benar-benar dilatih. Kembalikan "" untuk\n'
    '    rekaman lama yang tidak menyimpan snapshot, supaya laporan yang sudah\n'
    '    ter-commit terbit byte-identik.\n'
    '    """\n'
    '    crops = (row.get("config_snapshot") or {}).get("crops", {})\n'
    '    out_dir = str(crops.get("out_dir") or "")\n'
    '    if not out_dir:\n'
    '        return ""\n'
    '    nama = out_dir.replace("\\\\", "/").rstrip("/").split("/")[-1]\n'
    '    return nama[len("crops_"):] if nama.startswith("crops_") else nama\n'
)


def tambal_report(path: Path, tulis: bool) -> bool:
    s = path.read_text(encoding="utf-8")
    berubah = False

    if LAMA_E in s:
        s = s.replace(LAMA_E, BARU_E, 1)
        berubah = True
        print("  (e) keluarga(): kunci pengelompokan + susunan() -> siap")
    elif "def susunan(" in s:
        print("  (e) sudah tertambal")
    else:
        print("  (e) GAGAL: potongan kode tidak cocok, periksa manual")

    if LAMA_F in s:
        s = s.replace(LAMA_F, BARU_F, 1)
        sisip = s.index("def aggregate(")
        s = s[:sisip] + FUNGSI_F + s[sisip:]
        berubah = True
        print("  (f) kalimat kontrak dibaca dari data_contract -> siap")
    elif "def kalimat_kontrak(" in s:
        print("  (f) sudah tertambal")
    else:
        print("  (f) GAGAL: potongan kode tidak cocok, periksa manual")

    if berubah and tulis:
        path.write_text(s, encoding="utf-8")
        print(f"  -> ditulis {path}")
    return berubah


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tulis", action="store_true",
                    help="tanpa ini hanya mencetak, tidak menulis")
    a = ap.parse_args()

    print("\n=== (g) purpose di split lock ===")
    # Nama berkasnya sama di kedua lengan ("dev_split_lock.json"); yang
    # membedakan adalah folder crop-nya, sesuai config masing-masing lengan.
    for nama in ("crops_pio_dev2/dev_split_lock.json",
                 "crops_pio_dev2_eq/dev_split_lock.json"):
        tambal_split_lock(AKAR / "data" / nama, a.tulis)

    print("\n=== (e) + (f) src/report_fixed_chick.py ===")
    tambal_report(AKAR / "src" / "report_fixed_chick.py", a.tulis)

    if not a.tulis:
        print("\n(kering - tidak ada yang ditulis; tambahkan --tulis)")
    else:
        print("\nSESUDAH INI: jalankan ulang `python tests/test_protocol.py -v` "
              "dan pastikan fixed_chick.md lama masih terbit byte-identik.")


if __name__ == "__main__":
    main()
