"""
Model ini benar-benar melihat apa? Uji SEBAB, bukan korelasi.

`eval_shortcut_baseline.py` menjawab "jalan pintas apa yang TERSEDIA".
Skrip ini menjawab pertanyaan yang berbeda dan lebih penting:
**yang mana yang benar-benar DIPAKAI model.**

Dua hal itu sering tidak sama. Korelasi Spearman antara p(mati) dan sebuah ciri
TIDAK bisa membuktikan model memakai ciri itu: di data ini ciri tersebut memang
berbarengan dengan label, jadi rho bisa besar walaupun model sesungguhnya
melihat hal lain. Satu-satunya cara memastikan adalah **merusak satu ciri saja,
lalu melihat jawabannya berubah atau tidak.**

Perlakuannya (isi gambar diubah, labelnya tidak):

  asli      acuan, tidak diapa-apakan
  abu       warna dibuang (grayscale) - hue & saturasi mati sekaligus
  hue+26    hue diputar ke wilayah warna kelas lawan; bentuk & tekstur utuh
  kabur     Gaussian sigma 4 - ketajaman dimatikan
  acak16    crop dipecah 4x4 lalu petaknya diacak - BENTUK ayam hancur,
            tekstur dan warna lokal utuh

Cara membacanya:

  AUC tetap tinggi setelah `acak16`  ->  model tidak melihat ayam. Bentuk,
      pose, dan susunan badan sudah hancur tapi jawabannya tidak berubah,
      berarti yang dibacanya hanya tekstur/warna sepetak-sepetak.
  AUC jatuh setelah `abu`            ->  model memang bergantung pada warna.
  AUC jatuh setelah `kabur`          ->  model bergantung pada ketajaman.
  AUC tidak bergerak setelah sebuah perlakuan -> ciri itu bukan andalannya.

Catatan sejarah yang membuat skrip ini ada: dari rho(p_mati, hue) = -0.599
sempat disimpulkan "model menempel pada hue". Intervensi membantahnya -
memutar hue +26 hampir tidak menggeser AUC (1.000 -> 0.998 pada ce/eq48),
sedangkan mengacak 16 petak juga tidak menggesernya. Kesimpulan yang benar
bukan "model membaca hue", melainkan "model tidak membaca bentuk ayam sama
sekali". rho tinggi ternyata cuma penumpang.

Jalankan:
    python src/eval_intervensi.py --config configs/config.yaml
    python src/eval_intervensi.py --config configs/config_pio_eq.yaml \
        --runs outputs/runs_pio_eq
"""
from __future__ import annotations

import argparse
import csv

import cv2
import numpy as np

from common import (get_device, imread, load_config, resolve, save_json,
                    to_square)
from pipeline import classify_crops, load_classifier
from intervensi import acak_petak


def t_asli(im):
    return im


def t_abu(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def t_hue(im, d=26):
    """Putar hue saja. Bentuk, pose, tekstur, ketajaman tidak tersentuh."""
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + d) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def t_kabur(im):
    return cv2.GaussianBlur(im, (0, 0), 4.0)


def t_acak(im, sample_id=0, n=4):
    """Acak petak deterministik, tanpa strip sisa atau fixed point."""
    return acak_petak(im, sample_id, seed=20260913, n=n)


PERLAKUAN = [("asli", t_asli), ("abu", t_abu), ("hue+26", t_hue),
             ("kabur", t_kabur), ("acak16", t_acak)]

METODE = ("ce__hier_addone", "selfcon__simclr", "supcon__stacked_randaug")


def muat_test(crops_dir: str):
    """Crop split test apa adanya - crop tersimpan sudah dipraproses."""
    root = resolve(crops_dir)
    imgs, lab = [], []
    for r in csv.DictReader(open(root / "manifest.csv", encoding="utf-8")):
        if r["split"] != "test":
            continue
        im = imread(str(root / r["path"]))
        if im is not None:
            imgs.append(im)
            lab.append(int(r["label"]))
    return imgs, np.asarray(lab)


def main():
    ap = argparse.ArgumentParser(
        description="Uji sebab: rusak satu ciri, lihat AUC bergerak")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--runs", default=None,
                    help="folder checkpoint; default dari config")
    ap.add_argument("--crops", default=None,
                    help="folder crop; default dari config")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    from sklearn.metrics import roc_auc_score

    cfg = load_config(a.config)
    if a.runs:
        cfg["output"] = dict(cfg["output"])
        cfg["output"]["runs_dir"] = a.runs
    crops_dir = a.crops or cfg["crops"]["out_dir"]
    dev = get_device(cfg.get("device", "auto"))
    size = int(cfg["classifier"]["image_size"])
    mode = cfg["classifier"]["resize_mode"]

    imgs, y = muat_test(crops_dir)
    if len(set(y.tolist())) < 2:
        raise SystemExit("test split tidak punya dua kelas")

    print(f"[intervensi] crop {crops_dir}  runs {cfg['output']['runs_dir']}")
    print(f"             test n={len(y)} "
          f"({int((y == 1).sum())} mati, {int((y == 0).sum())} hidup)\n")
    print("{:<12}".format("model")
          + "".join(f"{n:>9}" for n, _ in PERLAKUAN))

    out = {"crops": crops_dir, "runs_dir": str(cfg["output"]["runs_dir"]),
           "n_test": len(y), "results": {}}

    for meth in METODE:
        model = None
        for s in ("__s42", "__s43", "__s44", ""):
            try:
                model = load_classifier(cfg, f"{meth}{s}", dev)
                break
            except Exception:
                continue
        if model is None:
            print("{:<12}{}".format(meth.split("__")[0],
                                    "  (checkpoint tidak ada)"))
            continue
        row, rec = "", {}
        for nama, fn in PERLAKUAN:
            crops = [to_square(t_acak(im, i) if nama == "acak16" else fn(im),
                               size, mode) for i, im in enumerate(imgs)]
            _l, p = classify_crops(model, crops, cfg, dev)
            p = np.asarray(p)
            p = p[:, 1] if p.ndim > 1 else p
            rec[nama] = float(roc_auc_score(y, p))
            row += f"{rec[nama]:>9.3f}"
        out["results"][meth] = rec
        print("{:<12}".format(meth.split("__")[0]) + row)

    print("\n  Angka = test AUC. 0.5 = sudah tidak bisa memisahkan.")
    print("  'acak16' bentuk ayam sudah hancur: kalau AUC-nya tetap tinggi,")
    print("  model TIDAK mengenali ayam - ia membaca tekstur/warna sepetak.")

    if a.json:
        save_json(out, resolve(a.json))
        print(f"\n[intervensi] JSON: {resolve(a.json)}")


if __name__ == "__main__":
    main()
