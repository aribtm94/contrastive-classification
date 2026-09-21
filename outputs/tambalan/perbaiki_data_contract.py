"""
Tambalan pasca-sweep: `data_contract` registry dibaca dari manifest.

MASALAH. `train.py` menulis blok `data_contract` dengan nilai yang dipaku di
kode (lihat laporan `domain_mati_kedua.md` bagian 4d):

    "alive": "PIO/pio_gt/cctv", "dead": "Roboflow/coco_gt/closeup",

Untuk lengan `pio_dev2` itu SALAH - kelas mati berisi 98 crop Roboflow DAN 200
crop archive_4. Seluruh hash di registry tetap benar dan terverifikasi; yang
salah hanya keterangan sumbernya.

KENAPA TAMBALAN INI TERPISAH DAN BARU DIJALANKAN SETELAH SWEEP. `train.py` ikut
di-hash ke `registry.source_sha256` lewat `source_hashes()`, yang membaca berkas
itu DARI DISK pada langkah pembekuan di akhir sweep. Mengubah `train.py` selagi
sweep berjalan membuat registry mencatat hash kode yang BUKAN kode yang
benar-benar dijalankan - kerusakan provenance yang lebih buruk daripada
keterangan yang salah tapi terdokumentasi.

Skrip ini tidak dijalankan otomatis. Ia mencetak apa yang akan diubah, dan
hanya menulis kalau diberi --tulis.

Pemakaian:
    python outputs/tambalan/perbaiki_data_contract.py \
        --registry outputs/predictions/pio_dev2_registry.json --tulis
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import resolve, save_json  # noqa: E402


def kontrak_dari_manifest(manifest: Path) -> dict:
    """Susun keterangan sumber per kelas dari manifest, bukan dari kode."""
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    per_kelas: dict[str, Counter] = {}
    per_domain: dict[str, Counter] = {}
    for r in rows:
        per_kelas.setdefault(r["label_name"], Counter())[
            f'{r["origin"]}/{r["domain"]}'] += 1
        per_domain.setdefault(r["domain"], Counter())[r["label_name"]] += 1

    # Tebak-label-dari-domain: kalau tiap domain hanya memuat satu label,
    # angkanya 1.0 dan batasan `label = domain` MASIH berlaku. Dihitung, tidak
    # diasumsikan - inilah yang semula dipaku `True` di kode.
    benar = sum(c.most_common(1)[0][1] for c in per_domain.values())
    akurasi = benar / len(rows) if rows else 0.0

    return {
        "alive": sorted(per_kelas.get("alive", {})),
        "dead": sorted(per_kelas.get("dead", {})),
        "counts_per_source": {k: dict(v) for k, v in sorted(per_kelas.items())},
        "train_validation_only": True,
        "label_equals_domain_limitation": akurasi >= 1.0,
        "label_from_domain_accuracy": round(akurasi, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--tulis", action="store_true",
                    help="tanpa ini hanya mencetak, tidak menulis")
    a = ap.parse_args()

    path = resolve(a.registry)
    registry = json.loads(path.read_text(encoding="utf-8"))
    lama = registry["data_contract"]
    manifest = Path(registry["manifest"])
    if not manifest.exists():
        raise SystemExit(f"manifest tidak ada: {manifest}")

    baru = dict(lama)
    baru.update(kontrak_dari_manifest(manifest))
    # Jangan sentuh keterangan test - itu milik benchmark chick, bukan manifest.
    baru["test_root"] = lama.get("test_root")
    baru["test_counts"] = lama.get("test_counts")
    baru["catatan"] = ("data_contract diperbaiki pasca-sweep dari manifest; "
                       "nilai semula dipaku di train.py dan menyebut satu "
                       "sumber mati saja. Hash lain tidak disentuh.")

    print(f"\n=== {path} ===")
    print("LAMA:", json.dumps({k: lama.get(k) for k in
                               ("alive", "dead", "label_equals_domain_limitation")},
                              ensure_ascii=False))
    print("BARU:", json.dumps({k: baru.get(k) for k in
                               ("alive", "dead", "label_equals_domain_limitation",
                                "label_from_domain_accuracy")},
                              ensure_ascii=False))
    if not a.tulis:
        print("\n(kering - tidak ada yang ditulis; tambahkan --tulis)")
        return
    registry["data_contract"] = baru
    save_json(registry, path)
    print(f"\nditulis -> {path}")
    print("CATATAN: checkpoint_sha256, manifest_sha256, config_sha256, dan "
          "source_sha256 TIDAK disentuh - hanya blok keterangan.")


if __name__ == "__main__":
    main()
