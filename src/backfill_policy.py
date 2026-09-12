"""
Menambahkan `policy_contrastive` / `policy_head` ke result.json yang sudah ada.

Kenapa perlu: field itu baru ditambahkan ke src/train.py setelah sweep berjalan,
dan proses Python yang sedang jalan memakai salinan modul yang lama - jadi 45
berkas hasil sweep ini tidak memuatnya. Nilainya deterministik dari
(method, aug) + config, jadi bisa dihitung ulang persis seperti run_method:

    ce           -> con=None,           head=<aug>
    selfcon/supcon -> con=<aug>,        head=resolve_policy(..., "head")

Run lama berlabel `legacy` DILEWATI: kebijakannya bukan berasal dari config ini
(waktu itu augmentasi masih hardcoded), jadi menuliskan nama kebijakan di sana
justru akan berbohong.

Jalankan:
    python src/backfill_policy.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json

from common import load_config, resolve
from dataset import policy_by_name, resolve_policy


def main() -> None:
    ap = argparse.ArgumentParser(description="Isi ulang field kebijakan")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    runs = resolve(cfg["output"]["runs_dir"])
    lock = bool(cfg.get("augmentation", {}).get("lock_probe_policy", True))

    n_done = n_skip = 0
    for f in sorted(runs.glob("*/result.json")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                r = json.load(fh)
        except json.JSONDecodeError:
            # Sweep mungkin sedang menulis berkas ini detik itu juga.
            # Dilewati saja - skrip ini aman dijalankan ulang.
            print(f"  {f.parent.name:<34} DILEWATI (sedang ditulis?)")
            n_skip += 1
            continue
        aug, method = r.get("aug"), r.get("method")
        if aug is None or aug == "legacy" or method is None:
            n_skip += 1
            continue
        if "policy_contrastive" in r:
            n_skip += 1
            continue

        if method == "ce":
            con, head = None, policy_by_name(cfg, aug)
        else:
            con, head = policy_by_name(cfg, aug), resolve_policy(cfg, method, "head")

        # Sisipkan tepat setelah `seed`, supaya urutan kunci sama dengan
        # berkas yang ditulis src/train.py versi baru.
        out = {}
        for k, v in r.items():
            out[k] = v
            if k == "seed":
                out["policy_contrastive"] = (con or {}).get("name")
                out["policy_head"] = (head or {}).get("name")
                out["lock_probe_policy"] = lock
                out["_backfill"] = ("kebijakan dihitung ulang dari config; "
                                    "sweep berjalan sebelum field ini ada")
        print(f"  {f.parent.name:<34} con={str((con or {}).get('name')):<16}"
              f" head={(head or {}).get('name')}")
        if not a.dry_run:
            with open(f, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2, ensure_ascii=False)
        n_done += 1

    verb = "akan diisi" if a.dry_run else "diisi"
    print(f"\n[backfill] {n_done} berkas {verb}, {n_skip} dilewati "
          f"(legacy / sudah punya field).")


if __name__ == "__main__":
    main()
