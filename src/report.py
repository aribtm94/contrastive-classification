"""
Membuat laporan perbandingan metode x augmentasi.

Menghasilkan:
    outputs/reports/comparison.csv   - tabel metrik (format panjang)
    outputs/reports/comparison.md    - ringkasan yang bisa dibaca
    outputs/reports/comparison.png   - diagram batang + garis lantai

Hasil dari beberapa seed DIRATA-RATAKAN dan dilaporkan sebagai mean +- std.
Pada test set yang efektif cuma 7 foto, satu angka tunggal tidak berarti apa-apa.

Jalankan (setelah src/train.py):
    python src/report.py
"""
from __future__ import annotations

import argparse
import csv
import json

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")            # tanpa jendela GUI
import matplotlib.pyplot as plt  # noqa: E402

from common import load_config, resolve  # noqa: E402
from dataset import read_manifest  # noqa: E402

ORDER = ["selfcon", "supcon", "ce"]
LABEL = {"selfcon": "A. Self-Contrastive\n(NT-Xent)",
         "supcon": "B. Supervised Contrastive\n(hidup+mati digabung)",
         "ce": "C. Cross-Entropy\n(baseline)"}


# --------------------------------------------------------------------------- #
# Lantai: seberapa jauh bisa dicapai TANPA melihat isi gambar
# --------------------------------------------------------------------------- #
def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """AUC lewat statistik Mann-Whitney U (tidak butuh sklearn)."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def _bacc(y_true: np.ndarray, pred: np.ndarray) -> float:
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return (tp / max(1, tp + fn) + tn / max(1, tn + fp)) / 2


def sharpness_floor(cfg: dict) -> dict:
    """
    "Classifier" yang TIDAK melihat isi gambar sama sekali: cuma mengukur
    ketajaman (variance of Laplacian) lalu memotong di satu ambang.

    Ambang dicocokkan di TRAIN, dievaluasi di TEST - persis seperti model
    sungguhan, supaya perbandingannya jujur.

    Ini artefak PELAPORAN, bukan model: gunanya menjawab "berapa skor yang
    bisa didapat tanpa belajar apa pun?". Metode yang skornya di bawah angka
    ini belum membuktikan apa pun.
    """
    def feats(split):
        rows = read_manifest(cfg, split)
        y, v = [], []
        for r in rows:
            buf = np.fromfile(r["abspath"], dtype=np.uint8)
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if im is None:
                continue
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            v.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
            y.append(int(r["label"]))
        return np.array(y), np.array(v)

    ytr, vtr = feats("train")
    yte, vte = feats("test")

    u = np.unique(vtr)
    cands = (u[:-1] + u[1:]) / 2.0 if len(u) > 1 else u
    best_t, best_b = 0.0, -1.0
    for t in cands:
        b = _bacc(ytr, (vtr >= t).astype(int))
        if b > best_b:
            best_t, best_b = float(t), b

    return {"threshold": best_t, "bacc_train": best_b,
            "bacc": _bacc(yte, (vte >= best_t).astype(int)),
            "auc": roc_auc(yte, vte), "n_test": int(len(yte))}


# --------------------------------------------------------------------------- #
# Pemuatan & agregasi atas seed
# --------------------------------------------------------------------------- #
def _agg(vals: list[float]) -> tuple[float, float]:
    a = np.array([v for v in vals if v is not None and not np.isnan(v)])
    if len(a) == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=0))


def load_results(cfg: dict) -> list[dict]:
    """
    Kumpulkan semua run, kelompokkan per (aug, method), rata-ratakan atas seed.

    Run lama `runs/<method>/result.json` yang tidak punya field `aug` tetap
    dibaca dan diberi label `legacy`, supaya tiga hasil yang sudah ada tidak
    hilang dari laporan.
    """
    runs = resolve(cfg["output"]["runs_dir"])
    groups: dict[tuple[str, str], list[dict]] = {}
    seen_dirs = set()

    # Hasil sebelum sweep disimpan terpisah di runs_prev_legacy/, karena
    # direktori `runs/<method>/` sekarang dipakai sebagai CERMIN run utama
    # (dibutuhkan src/pipeline.py) sehingga isinya bukan lagi hasil lama.
    prev = runs.parent / "runs_prev_legacy"
    files = sorted(prev.glob("*/result.json")) + sorted(runs.glob("*/result.json"))

    for f in files:
        is_prev = f.parent.parent.name == "runs_prev_legacy"
        d = f.parent.name
        with open(f, "r", encoding="utf-8") as fh:
            r = json.load(fh)
        if is_prev:
            # hasil sebelum sweep: augmentasi lama, apa pun isi field `aug`.
            # Ini SUMBER UTAMA baris legacy - `runs/<method>/` dengan nama
            # sama dilewati di bawah supaya satu metode tidak terhitung dua
            # kali (berkas lama di sana tidak punya field `aug`, jadi
            # pemeriksaan cermin saja tidak cukup).
            r = dict(r, aug="legacy", seed=int(cfg["seed"]))
            d = "prev/" + d
        elif "__" not in d:
            if ("legacy", r["method"]) in groups:
                continue          # sudah diambil dari runs_prev_legacy/
            # direktori lama = cermin dari run utama ATAU hasil versi lama
            if "aug" in r:
                continue          # cermin: sudah terhitung lewat direktori baru
            r = dict(r, aug="legacy", seed=int(cfg["seed"]))
        if d in seen_dirs:
            continue
        seen_dirs.add(d)
        groups.setdefault((r.get("aug", "legacy"), r["method"]), []).append(r)

    if not groups:
        raise SystemExit("Belum ada hasil. Jalankan dulu: python src/train.py")

    out = []
    for (aug, method), rs in groups.items():
        bm, bs = _agg([r["test"]["balanced_accuracy"] for r in rs])
        tm, ts = _agg([r.get("test_tuned", {}).get("balanced_accuracy")
                       for r in rs])
        am, asd = _agg([r["test"].get("roc_auc") for r in rs])
        rdm, _ = _agg([r["test"]["recall_dead"] for r in rs])
        ram, _ = _agg([r["test"]["recall_alive"] for r in rs])
        accm, _ = _agg([r["test"]["accuracy"] for r in rs])
        f1m, _ = _agg([r["test"]["f1_dead"] for r in rs])
        vm, _ = _agg([r["val"]["balanced_accuracy"] for r in rs])
        # Kebijakan yang benar-benar terpakai per tahap. `aug` saja tidak
        # cukup: untuk selfcon/supcon tahap probe dikunci ke 'minimal', jadi
        # dua tahap itu memakai augmentasi berbeda. Kalau antar-seed ternyata
        # tidak seragam, itu bug - ditandai '?' supaya kelihatan, bukan
        # dirata-ratakan diam-diam.
        def _one(key):
            v = {r.get(key) for r in rs}
            return v.pop() if len(v) == 1 else "?"
        out.append({
            "policy_contrastive": _one("policy_contrastive"),
            "policy_head": _one("policy_head"),
            "aug": aug, "method": method, "n_seeds": len(rs),
            "seeds": sorted(r.get("seed", -1) for r in rs),
            "bacc_mean": bm, "bacc_std": bs,
            "tuned_mean": tm, "tuned_std": ts,
            "auc_mean": am, "auc_std": asd,
            "recall_dead": rdm, "recall_alive": ram,
            "accuracy": accm, "f1_dead": f1m, "val_bacc": vm,
            "description": rs[0].get("description", ""),
            "n_train": rs[0]["n_train"], "n_val": rs[0]["n_val"],
            "n_test": rs[0]["n_test"],
        })

    order = {m: i for i, m in enumerate(ORDER)}
    out.sort(key=lambda r: (r["aug"], order.get(r["method"], 9)))
    return out


def diagonal_aug(cfg: dict, method: str) -> str:
    by = cfg["augmentation"]["by_method"][method]
    return by["head" if method == "ce" else "contrastive"]


def is_diagonal(cfg: dict, r: dict) -> bool:
    return r["aug"] == diagonal_aug(cfg, r["method"])


# --------------------------------------------------------------------------- #
# Keluaran
# --------------------------------------------------------------------------- #
def fmt(m: float, s: float) -> str:
    if m is None or np.isnan(m):
        return "-"
    if s is None or np.isnan(s) or s == 0:
        return f"{m:.4f}"
    return f"{m:.4f} ± {s:.4f}"


def write_csv(res: list[dict], cfg: dict, path) -> None:
    cols = ["aug", "method", "diagonal", "n_seeds", "seeds",
            "test_bacc_mean", "test_bacc_std", "test_tuned_mean",
            "test_tuned_std", "test_auc_mean", "test_auc_std",
            "test_accuracy", "test_recall_dead", "test_recall_alive",
            "test_f1_dead", "val_bacc_mean",
            "policy_contrastive", "policy_head"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in res:
            w.writerow([r["aug"], r["method"], int(is_diagonal(cfg, r)),
                        r["n_seeds"], " ".join(str(s) for s in r["seeds"]),
                        round(r["bacc_mean"], 4), round(r["bacc_std"], 4),
                        round(r["tuned_mean"], 4), round(r["tuned_std"], 4),
                        round(r["auc_mean"], 4), round(r["auc_std"], 4),
                        round(r["accuracy"], 4), round(r["recall_dead"], 4),
                        round(r["recall_alive"], 4), round(r["f1_dead"], 4),
                        round(r["val_bacc"], 4),
                        r.get("policy_contrastive") or "-",
                        r.get("policy_head") or "-"])


def write_md(res: list[dict], cfg: dict, floor: dict, path) -> None:
    n = res[0]
    diag = [r for r in res if is_diagonal(cfg, r)]
    pool = diag or res
    best = max(pool, key=lambda r: r["bacc_mean"])
    above = [r for r in res if r["bacc_mean"] > floor["bacc"]]
    # Dipisah: baris 1 seed yang kebetulan di atas lantai BUKAN bukti - tidak
    # ada simpangan baku yang bisa menyanggah keberuntungan satu undian. Dua
    # angka ini dipakai di kalimat bawah supaya tidak saling bertentangan
    # dengan bagian AUC (yang memakai syarat n_seeds >= 2 yang sama).
    above_multi = [r for r in above if r["n_seeds"] >= 2]
    multi_aug = len({r["aug"] for r in res}) > 1

    lines = [
        "# Perbandingan Metode x Augmentasi - Ayam Mati vs Hidup", "",
        f"Data: train {n['n_train']} / val {n['n_val']} / test {n['n_test']} "
        f"crop, ukuran input {cfg['classifier']['image_size']}x"
        f"{cfg['classifier']['image_size']} ({cfg['classifier']['resize_mode']}), "
        f"backbone {cfg['classifier']['backbone']}.", "",
        "Kelas positif = ayam MATI. Metrik utama = **balanced accuracy** "
        "(rata-rata recall kedua kelas), karena jumlah kelasnya timpang.", "",
        "Tiap konfigurasi dijalankan beberapa seed; yang dilaporkan "
        "**mean ± simpangan baku**. `@0.5` = ambang bawaan, `@tau` = ambang "
        "yang dikalibrasi di **validation set** (tidak pernah di test set).",
        "",
        "| Augmentasi | Metode | Bal.Acc @0.5 | Bal.Acc @tau | AUC | "
        "R.mati | R.hidup | n seed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in res:
        mark = " **←**" if is_diagonal(cfg, r) else ""
        lines.append(
            f"| `{r['aug']}` | {r['method']}{mark} | "
            f"**{fmt(r['bacc_mean'], r['bacc_std'])}** | "
            f"{fmt(r['tuned_mean'], r['tuned_std'])} | "
            f"{fmt(r['auc_mean'], r['auc_std'])} | "
            f"{r['recall_dead']:.4f} | {r['recall_alive']:.4f} | "
            f"{r['n_seeds']} |")
    lines.append(
        f"| _(lantai)_ | **KETAJAMAN SAJA** | **{floor['bacc']:.4f}** | - | "
        f"{floor['auc']:.4f} | - | - | - |")

    lines += [
        "", "Baris bertanda **←** adalah diagonal: pasangan metode-augmentasi "
        "yang menjadi rancangan utama (`selfcon`+`simclr`, "
        "`supcon`+`stacked_randaug`, `ce`+`hier_addone`).", "",
        "## Lantai ketajaman - baca ini sebelum memeringkat apa pun", "",
        "Baris terakhir tabel bukan sebuah model. Itu ketajaman gambar "
        "(variance of Laplacian) saja, dengan satu ambang yang dicocokkan di "
        "train dan diuji di test - **tanpa melihat isi gambar sama sekali**.",
        "",
        f"Angkanya **{floor['bacc']:.4f}** (AUC {floor['auc']:.4f}). "
        "Penyebabnya cara data terbentuk: crop ayam hidup median sisi pendek "
        "83 px (semuanya diperbesar ke 224), crop ayam mati 212 px. Jadi "
        "ketajaman ikut menandai kelas.", "",
        f"**Konsekuensinya: konfigurasi dengan mean di bawah "
        f"{floor['bacc']:.4f} belum membuktikan apa pun** - hasil yang sama "
        "bisa diperoleh tanpa belajar. Dari "
        f"{len(res)} konfigurasi, **{len(above)}** berada di atas lantai"
        + (f", dan **{len(above_multi)}** di antaranya dijalankan lebih dari "
           "satu seed." if above_multi else
           " - dan semuanya cuma satu seed, jadi tidak ada simpangan baku "
           "yang bisa menyanggah keberuntungan satu undian. **Di antara "
           "konfigurasi yang dijalankan 5 seed, tidak ada yang di atas "
           "lantai pada bal.acc.**"),
        "",
    ]

    # Lantai selama ini cuma dibandingkan lewat bal.acc. Itu melewatkan satu
    # hal: bal.acc adalah KEPUTUSAN (ambang), AUC adalah URUTAN. Sebuah model
    # bisa mengurutkan jauh lebih baik daripada ketajaman tapi tetap kalah di
    # bal.acc karena ambangnya meleset - dan di test 7 ayam hidup, ambang
    # meleset satu crop saja sudah -7.14 poin.
    # n_seeds < 2 DIKECUALIKAN: simpangan bakunya 0 semata karena cuma ada
    # satu angka, jadi uji "mean - std > lantai" otomatis lolos tanpa bukti
    # kestabilan apa pun. Baris legacy justru yang paling rawan di sini.
    auc_ok = [r for r in res
              if r["n_seeds"] >= 2 and not np.isnan(r["auc_mean"])
              and not np.isnan(r["auc_std"])
              and r["auc_mean"] - r["auc_std"] > floor["auc"]]
    if auc_ok:
        npair = 7 * 26          # (hidup x mati) di test set
        lines += [
            "## Yang berhasil melewati lantai - tapi pada URUTAN, bukan "
            "keputusan", "",
            (f"Tidak ada konfigurasi 5-seed yang melewati lantai "
             f"{floor['bacc']:.4f} jika diukur dengan bal.acc"
             + (f" (yang melewatinya hanya {len(above)} baris 1 seed, yang "
                "tidak membuktikan kestabilan apa pun)." if above else ".")) +
            " Tapi bal.acc mengukur **keputusan** "
            "(setelah ambang), sedangkan AUC mengukur **urutan**. Keduanya "
            "bisa berbeda jauh di sini: dengan cuma 7 ayam hidup di test, "
            "ambang yang meleset satu crop saja sudah memotong 7.14 poin "
            "bal.acc walau urutannya sempurna.", "",
            f"Lantai ketajaman punya AUC **{floor['auc']:.4f}** - setara "
            f"salah mengurutkan **{round((1-floor['auc'])*npair)} dari "
            f"{npair} pasangan** (mati x hidup). Konfigurasi berikut "
            "mengurutkan **lebih baik dari itu**, bahkan setelah dikurangi "
            "satu simpangan baku:", "",
            "| Augmentasi | Metode | AUC | pasangan salah urut | n seed |",
            "|---|---|---|---|---|",
        ]
        for r in sorted(auc_ok, key=lambda r: -r["auc_mean"]):
            lines.append(
                f"| `{r['aug']}` | {r['method']} | "
                f"**{fmt(r['auc_mean'], r['auc_std'])}** | "
                f"{(1 - r['auc_mean']) * npair:.1f} dari {npair} | "
                f"{r['n_seeds']} |")
        only = len(auc_ok) == 1
        lines += [
            "", "Syaratnya sengaja ketat: **minimal 2 seed** dan "
            "`mean - simpangan baku` masih di atas lantai. Konfigurasi 1 "
            "seed (baris `legacy`) tidak ikut walau AUC-nya tinggi - "
            "simpangan bakunya 0 cuma karena angkanya cuma satu, jadi tidak "
            "membuktikan kestabilan apa pun.", "",
            ("**Ini satu-satunya hasil di seluruh grid yang benar-benar "
             "mengalahkan 'tidak belajar apa pun'**" if only else
             "**Hasil di atas yang benar-benar mengalahkan 'tidak belajar "
             "apa pun'**") + ", dan hanya pada urutan. "
            "Artinya representasinya memang memisahkan kedua kelas lebih "
            "baik daripada ketajaman; yang belum beres adalah kalibrasi "
            "ambangnya - dan itu tidak bisa diperbaiki lewat validation set "
            "di sini, karena val-nya jenuh (lihat catatan di "
            "[`augmentation_report.md`](augmentation_report.md)).", "",
            "Tetap perlu hati-hati: 7 ayam hidup itu sedikit sekali, jadi "
            "AUC setinggi ini lebih mudah terjadi kebetulan daripada "
            "kelihatannya. Yang bisa diklaim: **pada test set ini**, "
            "urutannya mengalahkan lantai di seluruh 5 seed.", "",
        ]

    if multi_aug:
        lines += [
            "## Cara membaca grid ini", "",
            "Sejak tiap metode punya augmentasinya sendiri, **dua hal berubah "
            "bersamaan** di diagonal (loss DAN augmentasi). Karena itu:", "",
            "- **Diagonal** (baris ←) = rancangan yang diminta, tapi tidak "
            "bisa mengatribusikan sebab.",
            "- **Bandingkan dalam satu nilai `aug` yang sama** (augmentasi "
            "ditahan, loss divariasikan) = perbandingan metode yang sah.",
            "- **Bandingkan dalam satu `method` yang sama** (loss ditahan, "
            "augmentasi divariasikan) = menjawab \"ini efek augmentasi atau "
            "efek loss?\".", "",
        ]

    pol = [r for r in res if r.get("policy_head") not in (None, "-", "?")]
    if pol:
        lines += [
            "## Kebijakan augmentasi yang benar-benar dipakai tiap tahap", "",
            "Kolom `Augmentasi` di tabel atas adalah augmentasi TAHAP UTAMA. "
            "Itu bukan keseluruhan cerita: `selfcon`/`supcon` punya dua tahap "
            "latih (kontrastif, lalu linear probe di atas encoder beku), dan "
            "tahap probe **sengaja dikunci ke `minimal`** (flip saja).", "",
            "Alasannya: probe mengukur kualitas ENCODER. Kalau augmentasi "
            "probe ikut berbeda per metode, angkanya mencampur dua efek. "
            "`ce` tidak bisa ikut dikunci karena kepala *adalah* satu-satunya "
            "tahap latihnya - asimetri ini nyata dan tidak disembunyikan.", "",
            "| Augmentasi | Metode | Tahap kontrastif | Tahap kepala/probe |",
            "|---|---|---|---|",
        ]
        for r in pol:
            lines.append(
                f"| `{r['aug']}` | {r['method']} | "
                f"`{r.get('policy_contrastive') or '-'}` | "
                f"`{r.get('policy_head')}` |")
        lines += [
            "", "Nilai di atas dibaca dari `result.json` tiap run, bukan "
            "ditulis tangan - jadi tabel ini ikut berfungsi sebagai audit "
            "bahwa kunci probe benar-benar berlaku di seluruh grid.", "",
        ]

    lines += ["## Keterangan metode", ""]
    for m in ORDER:
        r = next((x for x in res if x["method"] == m), None)
        if r:
            lines.append(f"- **{m}** - {r['description']}")

    lines += [
        "", "## Catatan pembacaan hasil", "",
        f"- Terbaik pada diagonal: **{best['method']}** + `{best['aug']}` "
        f"(bal.acc {fmt(best['bacc_mean'], best['bacc_std'])})"
        + (f" - tapi masih **di bawah lantai {floor['bacc']:.4f}**, "
           "jadi belum bisa disebut berhasil pada bal.acc."
           if best["bacc_mean"] <= floor["bacc"] else ".")
        + ((" Pada **AUC** ceritanya berbeda: lihat bagian "
            "\"Yang berhasil melewati lantai\" di atas - ada konfigurasi "
            "yang mengurutkan lebih baik daripada lantai di seluruh 5 seed, "
            "dan yang belum beres di situ cuma ambangnya.") if auc_ok else ""),
        f"- Test set berisi {n['n_test']} crop yang berasal dari hanya "
        "**7 foto asli**. Selisih kecil antar metode belum tentu bermakna; "
        "itulah sebabnya simpangan baku antar-seed ikut dilaporkan dan "
        "harus dibaca bersama rata-ratanya.",
        "- Akurasi biasa menyesatkan di sini: menebak 'mati' untuk semua "
        "sampel sudah memberi akurasi ~79% tanpa model belajar apa pun. "
        "Karena itu balanced accuracy yang dipakai.",
        "- `@tau` bukan angka yang lebih baik, melainkan angka yang "
        "menjawab pertanyaan berbeda: `@0.5` mengukur model apa adanya, "
        "`@tau` mengukur model setelah ambangnya dikalibrasi di validation "
        "set. Keduanya dilaporkan supaya tidak ada yang dipilih belakangan.",
        "- `selfcon` tidak memakai label saat melatih encoder, jadi wajar "
        "kalau hasilnya paling lemah pada dataset sekecil ini - metode "
        "self-supervised umumnya baru unggul kalau data tak berlabelnya "
        "banyak (ribuan sampai jutaan).",
        "- `ce` melatih seluruh jaringan, sedangkan `selfcon`/`supcon` hanya "
        "melatih linear probe di atas encoder beku. Perbedaan ini disengaja: "
        "yang diukur adalah kualitas representasi hasil contrastive.",
        "- Alasan tiap augmentasi dipilih ada di "
        "[`augmentation_report.md`](augmentation_report.md).",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot(res: list[dict], cfg: dict, floor: dict, path) -> None:
    """
    Batang dikelompokkan per metode, satu seri per augmentasi, yerr = std.

    Garis putus-putus di lantai ketajaman adalah elemen paling informatif di
    gambar ini: tanpanya, batang-batang tersebut justru mengundang kesimpulan
    yang salah.
    """
    augs = sorted({r["aug"] for r in res})
    methods = [m for m in ORDER if any(r["method"] == m for r in res)]
    x = np.arange(len(methods), dtype=float)
    width = min(0.8 / max(1, len(augs)), 0.28)

    fig, ax = plt.subplots(figsize=(max(8, 3 * len(methods)), 5.5))
    for i, aug in enumerate(augs):
        vals, errs, hatch, nseed = [], [], [], []
        for m in methods:
            r = next((x_ for x_ in res
                      if x_["method"] == m and x_["aug"] == aug), None)
            vals.append(r["bacc_mean"] if r else np.nan)
            n = r["n_seeds"] if r else 0
            errs.append(r["bacc_std"] if (r and n > 1) else 0.0)
            hatch.append(bool(r and is_diagonal(cfg, r)))
            nseed.append(n)
        pos = x + (i - (len(augs) - 1) / 2) * width
        bars = ax.bar(pos, vals, width, yerr=errs, capsize=3,
                      label=aug, edgecolor="black", linewidth=0.6)
        for b, v, e, h, ns in zip(bars, vals, errs, hatch, nseed):
            if h:
                b.set_hatch("//")          # tandai pasangan diagonal
            if not np.isnan(v):
                # n=1 tidak punya simpangan baku - ditandai supaya batang
                # tanpa galat tidak terbaca seolah-olah hasilnya stabil
                ax.text(b.get_x() + b.get_width() / 2, v + e + 0.02,
                        f"{v:.2f}" + ("*" if ns < 2 else ""),
                        ha="center", va="bottom", fontsize=8)

    ax.axhline(floor["bacc"], ls="--", lw=1.8, color="crimson", zorder=5)
    # Label ditaruh di kanan-atas garis, bukan kiri-bawah: di kiri-bawah
    # kotaknya menimpa angka di atas batang pertama (terbukti menutupi dua
    # label sekaligus saat legacy/simclr sama-sama ~0.79).
    ax.text(x[-1] + 0.46, floor["bacc"] + 0.012,
            f"lantai ketajaman {floor['bacc']:.3f} (tanpa melihat isi gambar)",
            ha="right", va="bottom", fontsize=9, color="crimson", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="crimson",
                      lw=0.8, alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(m) or m for m in methods], fontsize=9)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("balanced accuracy (test set), mean ± std")
    ax.set_title("Metode x augmentasi - arsiran // = pasangan diagonal, "
                 "* = 1 seed (tanpa simpangan baku)", fontsize=10)
    ax.legend(loc="upper left", ncol=max(1, len(augs)), fontsize=9,
              title="augmentasi")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Laporan perbandingan metode")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    res = load_results(cfg)
    rep = resolve(cfg["output"]["reports_dir"])

    print("[report] menghitung lantai ketajaman...")
    floor = sharpness_floor(cfg)
    print(f"[report] lantai: bacc {floor['bacc']:.4f} "
          f"(ambang vLap {floor['threshold']:.1f}, AUC {floor['auc']:.4f})")

    write_csv(res, cfg, rep / "comparison.csv")
    write_md(res, cfg, floor, rep / "comparison.md")
    plot(res, cfg, floor, rep / "comparison.png")

    print(f"\n{'aug':<17}{'metode':<10}{'bacc':>17}{'@tau':>17}{'seed':>6}")
    for r in sorted(res, key=lambda x: -x["bacc_mean"]):
        flag = " *" if r["bacc_mean"] > floor["bacc"] else "  "
        print(f"{r['aug']:<17}{r['method']:<10}"
              f"{fmt(r['bacc_mean'], r['bacc_std']):>17}"
              f"{fmt(r['tuned_mean'], r['tuned_std']):>17}"
              f"{r['n_seeds']:>6}{flag}")
    print("  * = di atas lantai ketajaman")

    print(f"\n[report] {rep / 'comparison.csv'}")
    print(f"[report] {rep / 'comparison.md'}")
    print(f"[report] {rep / 'comparison.png'}")


if __name__ == "__main__":
    main()
