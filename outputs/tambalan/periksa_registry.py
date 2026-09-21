"""
Periksa satu registry beku: kebijakan uji, jumlah checkpoint, dan SELURUH hash
yang tercatat di dalamnya dibandingkan dengan berkas di disk saat ini.

Dipakai untuk memverifikasi kedua lengan dev2 dengan cara yang sama persis,
supaya lengan kedua tidak diperiksa lebih longgar daripada lengan pertama
hanya karena diperiksa manual pada waktu yang berbeda.

    python outputs/tambalan/periksa_registry.py outputs/predictions/pio_dev2_registry.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha(p: str | Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tertambal(d: dict, berkas: str) -> str | None:
    """
    Kembalikan hash pasca-tambalan bila `berkas` tercatat di blok
    `tambalan_pasca_beku`, selain itu None.

    Blok itu ditulis `catat_tambalan_pasca_beku.py` untuk berkas yang sengaja
    diubah SESUDAH pembekuan. Selisih hash yang SUDAH tercatat di sana bukan
    kerusakan - yang berbahaya justru selisih yang TIDAK tercatat, karena itu
    berarti ada yang berubah tanpa sepengetahuan siapa pun.
    """
    # Registry menyimpan sebagian jalur sebagai ABSOLUT (config, manifest,
    # split_lock) dan sebagian sebagai RELATIF (source_sha256), jadi
    # perbandingan apa adanya akan gagal senyap untuk yang pertama.
    # Bandingkan jalur yang sudah diselesaikan, bukan teksnya.
    try:
        sasaran = Path(berkas).resolve()
    except OSError:
        return None
    for e in (d.get("tambalan_pasca_beku") or {}).get("berkas", []):
        try:
            if Path(e.get("berkas", "")).resolve() == sasaran:
                return e.get("sha256_sesudah_tambalan")
        except OSError:
            continue
    return None


def periksa(path: Path) -> bool:
    d = json.loads(path.read_text(encoding="utf-8"))
    print(f"=== {path.name} ===")
    print(f"test_policy      : {d.get('test_policy')}")
    print(f"checkpoints      : {len(d.get('checkpoints', []))}")
    ok = (d.get("test_policy") == "fixed_retrospective_chick_test_only"
          and len(d.get("checkpoints", [])) == 9)

    # Hash berkas tunggal: config, manifest, split lock, gerbang jalan pintas.
    for nama, kunci_berkas, kunci_hash in (
            ("config", "config", "config_sha256"),
            ("manifest", "manifest", "manifest_sha256"),
            ("split_lock", "split_lock", "split_lock_sha256"),
            ("shortcut", "shortcut_baseline", "shortcut_baseline_sha256")):
        berkas = Path(d[kunci_berkas])
        if not berkas.exists():
            print(f"{nama:16s} : HILANG {berkas}")
            ok = False
            continue
        nyata = sha(berkas)
        cocok = nyata == d[kunci_hash]
        dicatat = tertambal(d, d[kunci_berkas])
        if not cocok and dicatat == nyata:
            print(f"{nama:16s} : TERTAMBAL (tercatat, hash beku dibiarkan)")
        else:
            print(f"{nama:16s} : {'COCOK' if cocok else 'BEDA'}")
            ok = ok and cocok

    # Kode sumber yang di-hash saat pembekuan.
    beda, dicatat_src = [], []
    for f, v in d.get("source_sha256", {}).items():
        if not Path(f).exists():
            beda.append(f)
            continue
        nyata = sha(f)
        if nyata == v:
            continue
        (dicatat_src if tertambal(d, f) == nyata else beda).append(f)
    ringkas = "COCOK" if not beda else "BEDA: " + ", ".join(Path(f).name for f in beda)
    if dicatat_src:
        ringkas += (" | TERTAMBAL (tercatat): "
                    + ", ".join(Path(f).name for f in dicatat_src))
    print(f"source ({len(d.get('source_sha256', {}))} berkas) : {ringkas}")
    ok = ok and not beda

    # Bobot tiap checkpoint.
    rusak = [c["run_id"] for c in d["checkpoints"]
             if not Path(c["checkpoint"]).exists()
             or sha(c["checkpoint"]) != c["checkpoint_sha256"]]
    print(f"checkpoint hash  : {'9/9 cocok' if not rusak else 'BEDA: ' + ', '.join(rusak)}")
    ok = ok and not rusak

    print(f"VONIS            : {'LOLOS' if ok else 'GAGAL'}\n")
    return ok


if __name__ == "__main__":
    sys.exit(0 if all(periksa(Path(p)) for p in sys.argv[1:]) else 1)
