"""
Catat tambalan pasca-pembekuan ke registry, TANPA memalsukan hash lama.

MASALAH. `registry.source_sha256` dan `registry.split_lock_sha256` adalah klaim
tentang MASA LALU: "inilah kode dan split lock saat kesembilan checkpoint ini
dibuat". Dua tambalan yang sengaja ditunda sampai sweep selesai
(`perbaiki_keterangan_sumber.py`, butir e/f/g di domain_mati_kedua.md) mengubah
dua berkas yang ikut di-hash:

    src/report_fixed_chick.py          (butir e + f)
    data/crops_*/dev_split_lock.json   (butir g)

Ada tiga cara menangani ini, dan dua di antaranya salah:

  (1) SALAH - timpa source_sha256 dengan hash baru. Registry lalu menyatakan
      checkpoint dibuat oleh kode yang belum ada saat itu. Itu pemalsuan
      catatan, persis hal yang hash ini seharusnya cegah.
  (2) SALAH - kembalikan tambalannya. Cacat e/f/g kembali, dan keduanya sudah
      terbukti nyata: plot melempar KeyError, kalimat kontrak menyebut satu
      sumber mati padahal ada dua.
  (3) BENAR - biarkan hash lama apa adanya sebagai catatan sejarah, lalu
      tambahkan blok terpisah yang menyatakan berkas mana yang ditambal
      sesudah pembekuan, hash lama -> hash baru, dan alasannya.

Skrip ini melakukan (3). Ia TIDAK pernah menyentuh source_sha256,
split_lock_sha256, config_sha256, manifest_sha256, checkpoint_sha256, atau
threshold_from_validation.

Yang perlu diingat saat membaca hasilnya: tambalan ini hanya menyentuh
PENYAJIAN (report_fixed_chick.py merender laporan; split lock `purpose` hanya
keterangan yang tidak ikut dibandingkan saat lock diverifikasi). Kode yang
menghasilkan angka - train.py, dataset.py, eval_fixed_chick.py, models.py -
tidak disentuh satu byte pun, dan hash keempatnya masih cocok.

Pemakaian:
    python outputs/tambalan/catat_tambalan_pasca_beku.py            # kering
    python outputs/tambalan/catat_tambalan_pasca_beku.py --tulis
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

AKAR = Path(__file__).resolve().parents[2]

ALASAN = {
    "src/report_fixed_chick.py":
        "butir e+f: keluarga() kini memuat identitas susunan data sehingga dua "
        "lengan tidak bisa dirata-ratakan diam-diam; kalimat kontrak dibaca "
        "dari data_contract registry, bukan dipaku ke satu sumber mati; plot() "
        "tidak lagi memaku nama keluarga ke ('asli','eq48'). Hanya penyajian.",
    "split_lock":
        "butir g: field `purpose` semula dipaku "
        "'development_only_pio_alive_roboflow_dead' padahal kelas matinya "
        "berisi dua sumber. `purpose` TIDAK ikut dibandingkan saat lock "
        "diverifikasi ulang, jadi tidak ada penjaga yang berubah hasilnya.",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def catat(registry: Path, lock_rel: str, tulis: bool) -> None:
    d = json.loads(registry.read_text(encoding="utf-8"))
    print(f"\n=== {registry.name} ===")
    if "tambalan_pasca_beku" in d:
        print("  sudah tercatat, dilewati")
        return

    entri = []
    for rel, lama in sorted(d.get("source_sha256", {}).items()):
        baru = sha(AKAR / rel)
        if baru != lama:
            entri.append({"berkas": rel, "sha256_saat_beku": lama,
                          "sha256_sesudah_tambalan": baru,
                          "alasan": ALASAN.get(rel, "?")})
            print(f"  source     {rel}")
            print(f"     {lama[:16]}... -> {baru[:16]}...")

    lama = d.get("split_lock_sha256")
    baru = sha(AKAR / lock_rel)
    if lama and baru != lama:
        entri.append({"berkas": lock_rel, "sha256_saat_beku": lama,
                      "sha256_sesudah_tambalan": baru,
                      "alasan": ALASAN["split_lock"]})
        print(f"  split_lock {lock_rel}")
        print(f"     {lama[:16]}... -> {baru[:16]}...")

    if not entri:
        print("  tidak ada selisih - tidak ada yang dicatat")
        return

    if tulis:
        d["tambalan_pasca_beku"] = {
            "keterangan":
                "Berkas di bawah ditambal SESUDAH kesembilan checkpoint beku. "
                "Hash di source_sha256 dan split_lock_sha256 SENGAJA dibiarkan "
                "pada nilai saat pembekuan - itu catatan sejarah, bukan "
                "kesalahan. Seluruhnya menyentuh penyajian saja; train.py, "
                "dataset.py, eval_fixed_chick.py, models.py, build_crops.py, "
                "dan seluruh checkpoint tidak berubah.",
            "lihat": "outputs/reports/domain_mati_kedua.md bagian 4e, 4f, 4g",
            "berkas": entri,
        }
        registry.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print(f"  -> dicatat {len(entri)} berkas")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tulis", action="store_true")
    a = ap.parse_args()
    for reg, lock in (
            ("pio_dev2_registry.json", "data/crops_pio_dev2/dev_split_lock.json"),
            ("pio_dev2_eq_registry.json",
             "data/crops_pio_dev2_eq/dev_split_lock.json")):
        catat(AKAR / "outputs" / "predictions" / reg, lock, a.tulis)
    if not a.tulis:
        print("\n(kering - tambahkan --tulis)")


if __name__ == "__main__":
    main()
