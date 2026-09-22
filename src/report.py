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


# Ciri gambar sederhana yang bisa dipakai sebagai lantai. Rumusnya identik
# dengan src/eval_shortcut_baseline.py supaya kedua laporan bicara soal angka
# yang sama.
CIRI_LANTAI = {
    "ketajaman": lambda g, hsv: float(cv2.Laplacian(g, cv2.CV_64F).var()),
    "saturasi":  lambda g, hsv: float(hsv[:, :, 1].mean()),
    "terang":    lambda g, hsv: float(hsv[:, :, 2].mean()),
    "hue":       lambda g, hsv: float(hsv[:, :, 0].mean()),
    "std_terang": lambda g, hsv: float(hsv[:, :, 2].std()),
}

def sharpness_floor(cfg: dict) -> dict:
    """
    "Classifier" yang TIDAK melihat isi gambar sama sekali: cuma mengukur
    satu ciri gambar sederhana lalu memotong di satu ambang.

    Ambang dicocokkan di TRAIN, dievaluasi di TEST - persis seperti model
    sungguhan, supaya perbandingannya jujur.

    Ini artefak PELAPORAN, bukan model: gunanya menjawab "berapa skor yang
    bisa didapat tanpa belajar apa pun?". Metode yang skornya di bawah angka
    ini belum membuktikan apa pun.

    Ciri mana yang dipakai diatur `report.lantai_ciri`. Bawaannya `None` =
    ketajaman saja, arah tetap ("mati jika nilai >= ambang") - itu perilaku
    lama yang dipakai seluruh laporan ayam ter-commit, jadi angkanya tidak
    boleh berubah. Kalau diisi daftar ciri, yang dipakai adalah ciri TERKUAT
    di train (arah ikut dicari, seperti eval_shortcut_baseline). Ini perlu
    untuk dataset yang lantainya bukan ketajaman: pada SDNET2018 ketajaman
    justru ciri paling lemah (0.5713) sementara std_terang mengikat di
    0.6430 - melaporkan ketajaman di sana akan menurunkan palangnya sendiri.
    """
    daftar = (cfg.get("report") or {}).get("lantai_ciri")
    nama_ciri = list(daftar) if daftar else ["ketajaman"]
    for n in nama_ciri:
        if n not in CIRI_LANTAI:
            raise SystemExit(f"report.lantai_ciri: ciri '{n}' tidak dikenal; "
                             f"pilihan: {sorted(CIRI_LANTAI)}")

    def feats(split):
        rows = read_manifest(cfg, split)
        y, v = [], []
        for r in rows:
            buf = np.fromfile(r["abspath"], dtype=np.uint8)
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if im is None:
                continue
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            v.append([CIRI_LANTAI[n](g, hsv) for n in nama_ciri])
            y.append(int(r["label"]))
        return np.array(y), np.array(v, dtype=float)

    ytr, vtr = feats("train")
    # Lantai WAJIB diukur di split yang sama dengan skor model yang akan
    # dibandingkan dengannya. Lantai dari test vs skor dari val = dua populasi,
    # dan perbandingannya tidak berarti apa pun.
    yte, vte = feats(split_eval(cfg))

    # Arah hanya dicari kalau ciri dipilih lewat config. Pada jalur bawaan
    # arahnya dipaku +1 supaya angka lantai laporan ayam tetap persis sama.
    # Jalur bawaan juga cuma punya SATU ciri, jadi tidak ada pemilihan yang
    # bisa berubah - laporan ayam lama aman tanpa syarat.
    arah_dicari = bool(daftar)
    terbaik = None
    for j, n in enumerate(nama_ciri):
        kol_tr, kol_te = vtr[:, j], vte[:, j]
        u = np.unique(kol_tr)
        cands = (u[:-1] + u[1:]) / 2.0 if len(u) > 1 else u
        best_t, best_s, best_b = 0.0, 1, -1.0
        for t in cands:
            for sign in ((1, -1) if arah_dicari else (1,)):
                pred = (kol_tr >= t) if sign > 0 else (kol_tr <= t)
                b = _bacc(ytr, pred.astype(int))
                if b > best_b:
                    best_t, best_s, best_b = float(t), sign, b
        pred_te = (kol_te >= best_t) if best_s > 0 else (kol_te <= best_t)
        skor_te = kol_te if best_s > 0 else -kol_te
        cand = {"ciri": n, "threshold": best_t, "arah": best_s,
                "bacc_train": best_b, "bacc": _bacc(yte, pred_te.astype(int)),
                "auc": roc_auc(yte, skor_te), "n_test": int(len(yte))}
        # Ciri pengikat dipilih dengan aturan yang SAMA seperti gerbang
        # (eval_shortcut_baseline.py:153): AUC-terarah tertinggi di split
        # evaluasi. Dua aturan berbeda untuk satu besaran = dua angka lantai
        # yang diam-diam beda; di lengan campur_eq selisih train_bacc 0.0012
        # membalik palang dari 0.7136 ke 0.5985. AMBANG tetap dicocokkan di
        # train (best_t di atas) - yang dipilih di sini cuma ciri mana yang
        # jadi palang, dan palang wajib = kebocoran TERBESAR yang ada.
        if terbaik is None or cand["auc"] > terbaik["auc"]:
            terbaik = cand
    if terbaik is None:
        raise SystemExit("report.lantai_ciri kosong: tidak ada ciri lantai "
                         "yang bisa dihitung")
    return terbaik


# --------------------------------------------------------------------------- #
# Pemuatan & agregasi atas seed
# --------------------------------------------------------------------------- #
def split_eval(cfg: dict) -> str:
    """Split yang dipakai sebagai EVALUASI AKHIR laporan ini.

    Lengan development_only (protocol.development_only: true) sengaja tidak
    punya split test - test ditahan supaya tidak terpakai selama pengembangan.
    Untuk lengan itu laporan dibuat di atas val, dan NAMA "val" ikut sampai ke
    kolom CSV, judul tabel, sumbu grafik dan prosa. Memetakannya diam-diam ke
    kolom test_* akan menerbitkan angka val dengan label test - salah lapor
    pada metrik utama.
    """
    return "val" if (cfg.get("protocol") or {}).get("development_only") else "test"


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
    # runs_prev_legacy/ HANYA milik eksperimen ayam bawaan (outputs/runs).
    # Untuk eksperimen lain - misalnya uji kewarasan SDNET di
    # outputs/runs_sdnet - folder itu tidak boleh ikut dibaca, karena
    # angkanya berasal dari dataset yang sama sekali berbeda dan akan
    # muncul sebagai baris "legacy" yang menyesatkan di laporan.
    prev = runs.parent / "runs_prev_legacy"
    files = []
    if runs.name == "runs":
        files += sorted(prev.glob("*/result.json"))
    files += sorted(runs.glob("*/result.json"))

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

    # train.py menulis blok "test" untuk lengan biasa dan "validation" untuk
    # lengan development_only (src/train.py:676 memakai kunci "validation";
    # run lama memakai "val"). Ketiganya dibaca lewat satu fungsi supaya tidak
    # ada cabang yang lupa diperbarui.
    ev = split_eval(cfg)

    def blok(r: dict) -> dict:
        if ev == "test":
            return r["test"]
        b = r.get("validation") or r.get("val")
        if b is None:
            raise SystemExit(
                f"run tanpa blok validation/val padahal config ini "
                f"development_only: {r.get('method')} seed {r.get('seed')}")
        return b

    out = []
    for (aug, method), rs in groups.items():
        bm, bs = _agg([blok(r)["balanced_accuracy"] for r in rs])
        # test_tuned tidak ada di lengan development_only: ambangnya justru
        # DIPILIH di val, jadi "val @tau" akan mengukur dirinya sendiri.
        # Dibiarkan NaN -> fmt() mencetak "-", bukan angka palsu.
        tm, ts = _agg([r.get("test_tuned", {}).get("balanced_accuracy")
                       for r in rs])
        am, asd = _agg([blok(r).get("roc_auc") for r in rs])
        rdm, _ = _agg([blok(r)["recall_dead"] for r in rs])
        ram, _ = _agg([blok(r)["recall_alive"] for r in rs])
        accm, _ = _agg([blok(r)["accuracy"] for r in rs])
        f1m, _ = _agg([blok(r)["f1_dead"] for r in rs])
        # Run lama memakai kunci "val", run baru "validation"
        # (src/train.py:676). Keduanya dibaca supaya hasil lama
        # tetap terbit dan hasil baru tidak menabrak KeyError.
        vm, _ = _agg([(r.get("validation") or r["val"])
                      ["balanced_accuracy"] for r in rs])
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
            # n_test tidak ditulis oleh run development_only; n_eval yang
            # dipakai laporan, dan namanya menyebut split mana.
            "n_test": rs[0].get("n_test"),
            "n_eval": rs[0].get("n_test") if ev == "test" else rs[0]["n_val"],
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
    # Awalan kolom = split yang benar-benar diukur. Untuk config biasa
    # hasilnya persis sama seperti sebelumnya (test_*); untuk lengan
    # development_only jadi val_* supaya tidak ada yang membaca angka val
    # sebagai angka test.
    ev = split_eval(cfg)
    # Kolom val_bacc_mean adalah bacc validation yang dipakai MEMILIH
    # checkpoint. Di config biasa ia berdampingan dengan kolom test_* dan
    # keduanya berbeda. Di lengan development_only evaluasinya sudah val,
    # jadi kolom itu akan mengulang val_bacc_mean dengan angka yang sama
    # persis - dua kolom bernama identik dalam satu CSV, yang bikin pembaca
    # csv mana pun mengambil salah satunya diam-diam. Karena itu dibuang
    # di lengan tanpa test, bukan diberi nama lain.
    kol_val = ["val_bacc_mean"] if ev == "test" else []
    cols = ["aug", "method", "diagonal", "n_seeds", "seeds",
            f"{ev}_bacc_mean", f"{ev}_bacc_std", f"{ev}_tuned_mean",
            f"{ev}_tuned_std", f"{ev}_auc_mean", f"{ev}_auc_std",
            f"{ev}_accuracy", f"{ev}_recall_dead", f"{ev}_recall_alive",
            f"{ev}_f1_dead", *kol_val,
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
                        *([round(r["val_bacc"], 4)] if ev == "test" else []),
                        r.get("policy_contrastive") or "-",
                        r.get("policy_head") or "-"])


# Istilah yang berbeda antar-dataset. Nilai bawaan = teks eksperimen ayam
# yang sudah dipakai di seluruh laporan ter-commit, jadi laporan lama terbit
# apa adanya. Dataset lain (mis. uji kewarasan SDNET2018) menimpanya lewat
# `report.istilah` di config, supaya laporannya tidak menyebut "ayam mati"
# untuk ubin beton retak.
ISTILAH = {
    "judul": "Ayam Mati vs Hidup",
    "positif": "ayam MATI",
    # Bentuk pendek untuk judul kolom tabel, bentuk panjang untuk kalimat.
    "pos": "mati",
    "neg": "hidup",
    "pos_panjang": "ayam mati",
    "neg_panjang": "ayam hidup",
    # Kenapa lantai ketajaman setinggi itu pada data ini. Kalimat ini khusus
    # data ayam; dataset lain wajib mengisinya sendiri atau mengosongkannya.
    "sebab_lantai": ("Penyebabnya cara data terbentuk: crop ayam hidup "
                     "median sisi pendek 83 px (semuanya diperbesar ke 224), "
                     "crop ayam mati 212 px. Jadi ketajaman ikut menandai "
                     "kelas."),
}

def istilah(cfg: dict) -> dict:
    """Istilah laporan: bawaan ayam, ditimpa oleh `report.istilah` di config."""
    d = dict(ISTILAH)
    d.update((cfg.get("report") or {}).get("istilah") or {})
    return d

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
    ist = istilah(cfg)
    # Ciri apa yang sebenarnya jadi lantai. Bawaannya ketajaman (perilaku
    # lama laporan ayam); kalau `report.lantai_ciri` diisi, yang dipakai
    # adalah ciri terkuat di train dan namanya ikut berubah di seluruh teks.
    nama_lantai = floor.get("ciri", "ketajaman")
    # Bentuk panjang untuk kalimat penjelas; "ketajaman gambar" adalah kata
    # yang dipakai laporan ayam ter-commit, jadi tidak boleh berubah.
    # Rumus ikut ciri yang terpilih; "(variance of Laplacian)" hanya benar
    # untuk ketajaman, dan pada SDNET yang terpilih std_terang.
    RUMUS_LANTAI = {
        "ketajaman": "ketajaman gambar (variance of Laplacian)",
        "std_terang": "simpangan baku terang (kanal V dari HSV)",
        "saturasi": "saturasi warna (rata-rata kanal S dari HSV)",
        "terang": "terang (rata-rata kanal V dari HSV)",
        "hue": "hue (rata-rata kanal H dari HSV)",
    }
    lantai_panjang = RUMUS_LANTAI.get(nama_lantai, nama_lantai)
    # Jumlah seed sebenarnya, bukan angka tetap. Laporan lama menyebut
    # "5 seed" karena grid ayam memang 5; SDNET memakai 3.
    n_seed_maks = max((r["n_seeds"] for r in res), default=0)
    # Nama split evaluasi dipakai APA ADANYA di seluruh prosa. Untuk config
    # biasa nilainya "test" sehingga tiap kalimat di bawah berbunyi persis
    # seperti versi sebelumnya; untuk lengan development_only jadi "val", dan
    # pembaca langsung melihat bahwa angkanya bukan test.
    ev = split_eval(cfg)
    EV = ev.upper()
    # Cacah kelas di split evaluasi, dibaca dari manifest, bukan ditulis tangan.
    y_te = np.array([int(r["label"]) for r in read_manifest(cfg, ev)])
    n_pos_te, n_neg_te = int((y_te == 1).sum()), int((y_te == 0).sum())
    # Peringatan yang HANYA muncul di lengan tanpa test. Ditaruh di paling
    # atas laporan, bukan di catatan kaki: kalau seseorang cuma membaca tabel
    # pertama, justru kalimat ini yang wajib sudah terbaca.
    catatan_dev = ([
        f"> **Angka di laporan ini diukur di split VAL, bukan test.** Lengan "
        f"ini memakai `protocol.development_only`, jadi manifestnya memang "
        f"tidak punya split test - test ditahan supaya tidak terpakai selama "
        f"pengembangan. Konsekuensinya mengikat cara membaca seluruh tabel: "
        f"val dipakai memilih checkpoint DAN ambang, jadi angka val di sini "
        f"optimistis dan **bukan** estimasi kemampuan pada data baru. Kolom "
        f"`@tau` sengaja kosong - ambangnya dipilih di val, jadi 'val @tau' "
        f"akan mengukur dirinya sendiri.", ""]
        if ev != "test" else [])

    lines = [
        f"# Perbandingan Metode x Augmentasi - {ist['judul']}", "",
    ] + catatan_dev + [
        (f"Data: train {n['n_train']} / val {n['n_val']} / test {n['n_test']} "
         if ev == "test" else
         f"Data: train {n['n_train']} / val {n['n_val']} (tanpa test) "
        ) + f"crop, ukuran input {cfg['classifier']['image_size']}x"
        f"{cfg['classifier']['image_size']} ({cfg['classifier']['resize_mode']}), "
        f"backbone {cfg['classifier']['backbone']}.", "",
        f"Kelas positif = {ist['positif']}. Metrik utama = **balanced accuracy** "
        "(rata-rata recall kedua kelas), karena jumlah kelasnya timpang.", "",
        "Tiap konfigurasi dijalankan beberapa seed; yang dilaporkan "
        "**mean ± simpangan baku**. `@0.5` = ambang bawaan, `@tau` = ambang "
        "yang dikalibrasi di **validation set** (tidak pernah di test set)."
        + ("" if ev == "test" else
           " Di lengan ini `@tau` kosong karena val-lah yang memilih ambang."),
        "",
        # HANYA untuk lengan tanpa test. Di config biasa baris ini tidak
        # ditulis sama sekali supaya comparison.md yang sudah ter-commit
        # tetap byte-identik - laporan lama yang berubah isinya, walau
        # kalimatnya benar, membuat artefak ter-commit tidak bisa
        # direproduksi oleh kode hari ini.
        *([] if ev == "test" else
          [f"Semua kolom di bawah diukur di split **{EV}**.", ""]),
        "| Augmentasi | Metode | Bal.Acc @0.5 | Bal.Acc @tau | AUC | "
        f"R.{ist['pos']} | R.{ist['neg']} | n seed |",
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
        f"| _(lantai)_ | **{nama_lantai.upper()} SAJA** | "
        f"**{floor['bacc']:.4f}** | - | "
        f"{floor['auc']:.4f} | - | - | - |")

    lines += [
        "", "Baris bertanda **←** adalah diagonal: pasangan metode-augmentasi "
        "yang menjadi rancangan utama (`selfcon`+`simclr`, "
        "`supcon`+`stacked_randaug`, `ce`+`hier_addone`).", "",
        f"## Lantai {nama_lantai} - baca ini sebelum memeringkat apa pun",
        "",
        f"Baris terakhir tabel bukan sebuah model. Itu {lantai_panjang} "
        "saja, dengan satu ambang yang dicocokkan di "
        f"train dan diuji di {ev} - **tanpa melihat isi gambar sama sekali**."
        + ("" if ev == "test" else
           f" Lantainya diukur di split yang SAMA dengan skor model di "
           f"atasnya ({ev}); lantai dari split lain tidak bisa dibandingkan "
           f"dengannya."),
        "",
        (f"Angkanya **{floor['bacc']:.4f}** (AUC {floor['auc']:.4f}). "
         + ist.get("sebab_lantai", "")).strip(), "",
        f"**Konsekuensinya: konfigurasi dengan mean di bawah "
        f"{floor['bacc']:.4f} belum membuktikan apa pun** - hasil yang sama "
        "bisa diperoleh tanpa belajar. Dari "
        f"{len(res)} konfigurasi, **{len(above)}** berada di atas lantai"
        + (f", dan **{len(above_multi)}** di antaranya dijalankan lebih dari "
           "satu seed." if above_multi else
           " - dan semuanya cuma satu seed, jadi tidak ada simpangan baku "
           "yang bisa menyanggah keberuntungan satu undian. **Di antara "
           f"konfigurasi yang dijalankan {n_seed_maks} seed, tidak ada yang "
           "di atas lantai pada bal.acc.**"),
        "",
    ]

    # Lantai selama ini cuma dibandingkan lewat bal.acc. Itu melewatkan satu
    # hal: bal.acc adalah KEPUTUSAN (ambang), AUC adalah URUTAN. Sebuah model
    # bisa mengurutkan jauh lebih baik daripada lantai tapi tetap kalah di
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
        # (neg x pos) di test set, dihitung dari manifest.
        npair = max(1, n_neg_te * n_pos_te)
        # Berapa poin bal.acc yang hilang kalau ambang meleset SATU crop
        # kelas negatif. Pada test ayam angkanya 7.14 (cuma 7 hidup);
        # pada test yang lebih besar angkanya kecil, dan kalimatnya harus
        # ikut berubah supaya tidak menyesatkan.
        poin_satu = 100 * 0.5 / max(1, n_neg_te)
        lines += [
            "## Yang berhasil melewati lantai - tapi pada URUTAN, bukan "
            "keputusan", "",
            (f"**{len(above_multi)} dari {len(res)}** konfigurasi "
             f"multi-seed melewati lantai {floor['bacc']:.4f} pada bal.acc."
             if above_multi else
             (f"Tidak ada konfigurasi {n_seed_maks}-seed yang melewati lantai "
              f"{floor['bacc']:.4f} jika diukur dengan bal.acc"
              + (f" (yang melewatinya hanya {len(above)} baris 1 seed, yang "
                 "tidak membuktikan kestabilan apa pun)." if above
                 else "."))) +
            " Tapi bal.acc mengukur **keputusan** "
            "(setelah ambang), sedangkan AUC mengukur **urutan**. Keduanya "
            f"bisa berbeda jauh di sini: dengan cuma {n_neg_te} "
            f"{ist['neg_panjang']} di "
            f"{ev}, ambang yang meleset satu crop saja sudah memotong "
            f"{poin_satu:.2f} poin bal.acc walau urutannya sempurna.", "",
            f"Lantai {nama_lantai} punya AUC **{floor['auc']:.4f}** - setara "
            f"salah mengurutkan **{round((1-floor['auc'])*npair)} dari "
            f"{npair} pasangan** ({ist['pos']} x {ist['neg']}). "
            "Konfigurasi berikut "
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
            f"baik daripada {nama_lantai}; yang belum beres adalah kalibrasi "
            "ambangnya"
            + (" - dan itu tidak bisa diperbaiki lewat validation set "
               "di sini, karena val-nya jenuh (lihat catatan di "
               "[`augmentation_report.md`](augmentation_report.md))."
               if ist.get("val_jenuh", True) else
               ". Validation di sini TIDAK jenuh, jadi ambangnya masih "
               "bisa dikalibrasi di sana.") , "",
            # Peringatan ini soal kelas yang paling SEDIKIT menopang AUC.
            # Pada test ayam itu sisi hidup (7 crop); kalau kedua kelas
            # sudah besar, kalimatnya dibuang supaya tidak mengada-ada.
            (f"Tetap perlu hati-hati: {n_neg_te} {ist['neg_panjang']} itu "
             "sedikit sekali, jadi AUC setinggi ini lebih mudah terjadi "
             "kebetulan daripada kelihatannya. "
             if min(n_pos_te, n_neg_te) < 40 else "")
            # Nama split ikut di sini juga. Untuk config biasa ev=="test"
            # sehingga kalimatnya berbunyi persis sama; untuk lengan
            # development_only kalimat ini akan menulis "pada val set ini",
            # yang memang benar - dan yang salah justru kalau dibiarkan
            # mengklaim test padahal test tidak pernah disentuh.
            + f"Yang bisa diklaim: **pada {ev} set ini**, urutannya "
            f"mengalahkan lantai di seluruh {n_seed_maks} seed.", "",
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
            f"yang mengurutkan lebih baik daripada lantai di seluruh "
            f"{n_seed_maks} seed, "
            "dan yang belum beres di situ cuma ambangnya.") if auc_ok else ""),
        (f"- Test set berisi {n['n_test']} crop yang berasal dari hanya "
         "**7 foto asli**. Selisih kecil antar metode belum tentu bermakna; "
         "itulah sebabnya simpangan baku antar-seed ikut dilaporkan dan "
         "harus dibaca bersama rata-ratanya."
         if ev == "test" else
         f"- Split {ev} berisi {n['n_eval']} crop ({n_pos_te} "
         f"{ist['pos_panjang']}, {n_neg_te} {ist['neg_panjang']}). Selisih "
         "kecil antar metode belum tentu bermakna; itulah sebabnya simpangan "
         "baku antar-seed ikut dilaporkan dan harus dibaca bersama "
         "rata-ratanya."),
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
            f"lantai {floor.get('ciri', 'ketajaman')} {floor['bacc']:.3f} "
            f"(tanpa melihat isi gambar)",
            ha="right", va="bottom", fontsize=9, color="crimson", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="crimson",
                      lw=0.8, alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(m) or m for m in methods], fontsize=9)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel(f"balanced accuracy ({split_eval(cfg)} set), mean ± std")
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

    nama_lantai = (cfg.get("report") or {}).get("lantai_ciri") or ["ketajaman"]
    print("[report] menghitung lantai "
          + ("/".join(nama_lantai) if len(nama_lantai) > 1 else nama_lantai[0])
          + "...")
    floor = sharpness_floor(cfg)
    print(f"[report] lantai: bacc {floor['bacc']:.4f} "
          f"(ciri {floor.get('ciri', 'ketajaman')}, "
          f"ambang {floor['threshold']:.1f}, AUC {floor['auc']:.4f})")

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
    print(f"  * = di atas lantai {floor.get('ciri', 'ketajaman')}")

    print(f"\n[report] {rep / 'comparison.csv'}")
    print(f"[report] {rep / 'comparison.md'}")
    print(f"[report] {rep / 'comparison.png'}")


if __name__ == "__main__":
    main()
