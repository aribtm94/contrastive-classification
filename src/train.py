"""
TAHAP 3 - Melatih classifier ayam mati vs ayam hidup dengan 3 metode.

  A. selfcon : Self-Contrastive (NT-Xent/SimCLR). Encoder dilatih TANPA
               label, lalu dibekukan, lalu dilatih linear probe pakai label.
  B. supcon  : Supervised Contrastive. Label hidup & mati digabung ke dalam
               satu loss, lalu linear probe dengan encoder yang dibekukan.
  C. ce      : Cross-Entropy biasa. Baseline, tanpa contrastive sama sekali.

Ketiganya memakai arsitektur, data, dan seed yang sama persis. Sejak tiap
metode diberi augmentasinya sendiri (lihat configs/config.yaml -> augmentation),
yang berbeda ada DUA: fungsi loss DAN kebijakan augmentasi. Karena itu
diagonal saja tidak bisa mengatribusikan sebab; jalankan --grid full untuk
mendapatkan kontrolnya.

Model dipilih berdasarkan skor validasi terbaik (bukan epoch terakhir), dan
metrik utamanya adalah BALANCED ACCURACY - bukan akurasi biasa - karena
kelasnya timpang: menebak "mati" untuk semua sampel sudah memberi akurasi
75%, angka yang terlihat bagus padahal modelnya tidak belajar apa pun.

Jalankan:
    python src/train.py --method supcon
    python src/train.py --grid diagonal --seeds 42,43,44,45,46
    python src/train.py --grid full --seeds 42,43,44,45,46
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import get_device, load_config, resolve, save_json, set_seed
from dataset import (ClassificationDataset, TwoViewDataset, class_weights,
                     policy_by_name, read_manifest, resolve_policy)
from models import build_model, ce_loss, nt_xent_loss, supcon_loss

METHODS = ["selfcon", "supcon", "ce"]


# --------------------------------------------------------------------------- #
# Metrik
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_score: np.ndarray | None = None) -> dict:
    """
    Hitung metrik untuk 2 kelas: 0 = hidup, 1 = mati.
    Kelas positif = ayam MATI (itu yang ingin dideteksi).
    """
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    acc = (tp + tn) / max(1, len(y_true))
    recall = tp / max(1, tp + fn)          # sensitivitas kelas mati
    spec = tn / max(1, tn + fp)            # sensitivitas kelas hidup
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0

    # Balanced accuracy = rata-rata recall kedua kelas.
    # Ini metrik utama karena tahan terhadap ketimpangan jumlah kelas.
    bacc = (recall + spec) / 2

    m = {"accuracy": round(acc, 4), "balanced_accuracy": round(bacc, 4),
         "precision_dead": round(prec, 4), "recall_dead": round(recall, 4),
         "recall_alive": round(spec, 4), "f1_dead": round(f1, 4),
         "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": int(len(y_true))}

    if y_score is not None and len(np.unique(y_true)) == 2:
        m["roc_auc"] = round(roc_auc(y_true, y_score), 4)
    return m


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """AUC lewat statistik Mann-Whitney U (tidak butuh sklearn)."""
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)

    # Rata-ratakan rank untuk skor yang sama persis (ties)
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[dict, np.ndarray, np.ndarray]:
    """Metrik @argmax (setara ambang 0.5), plus skor mentah untuk kalibrasi."""
    model.eval()
    ys, ps, ss = [], [], []
    for x, y, _ in loader:
        logit = model.logits(x.to(device))
        prob = torch.softmax(logit, dim=1)[:, 1]     # peluang kelas 'mati'
        ps.append(logit.argmax(1).cpu().numpy())
        ss.append(prob.cpu().numpy())
        ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    y_score = np.concatenate(ss)
    return compute_metrics(y_true, y_pred, y_score), y_true, y_score


def pick_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Pilih ambang yang memaksimalkan balanced accuracy DI VALIDATION SET.

    Ini perlu, bukan hiasan: pernah terjadi `selfcon` mendapat AUC test
    1.0000 (pemisahan sempurna) tapi balanced accuracy hanya 0.7857 - semata
    karena ambang bawaan 0.5 jatuh di tempat yang salah. AUC mengukur urutan,
    balanced accuracy mengukur keputusan.

    Kandidat = titik tengah antar skor unik yang berurutan. Ambang terbaik
    biasanya bukan satu titik melainkan satu DATARAN, dan dataran itu bisa
    TERPUTUS jadi beberapa kelompok terpisah (skor val sering bertumpuk).
    Mengambil median seluruh titik dataran kemudian jatuh di JURANG antara
    dua kelompok - sebuah ambang yang justru TIDAK optimal. Karena itu:
    ambil kelompok yang bersambung dan paling lebar, lalu tengah kelompok
    itu; ambang di tepi rapuh terhadap satu sampel saja.
    """
    u = np.unique(y_score)
    if len(u) < 2 or len(np.unique(y_true)) < 2:
        return 0.5          # tidak ada yang bisa dikalibrasi
    cands = np.concatenate([[u[0] - 1e-6], (u[:-1] + u[1:]) / 2.0,
                            [u[-1] + 1e-6]])
    scores = np.empty(len(cands), dtype=np.float64)
    for i, t in enumerate(cands):
        pred = (y_score >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        scores[i] = (tp / max(1, tp + fn) + tn / max(1, tn + fp)) / 2

    idx = np.flatnonzero(scores >= scores.max() - 1e-12)
    # pecah indeks dataran jadi kelompok-kelompok yang bersambung
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    widest = max(groups, key=lambda g: (cands[g[-1]] - cands[g[0]], len(g)))
    return float((cands[widest[0]] + cands[widest[-1]]) / 2.0)


def metrics_at(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> dict:
    """Metrik pada ambang tertentu. TIDAK menggantikan metrik @0.5."""
    return compute_metrics(y_true, (y_score >= thr).astype(int), y_score)

# --------------------------------------------------------------------------- #
# Tahap pelatihan
# --------------------------------------------------------------------------- #
def train_contrastive(model, rows, cfg, device, method: str, log: list,
                      seed: int, policy: dict | None) -> None:
    """
    Latih ENCODER dengan loss kontrastif.
      method 'selfcon' -> NT-Xent, label diabaikan
      method 'supcon'  -> SupCon, label dipakai

    `seed` dan `policy` WAJIB diteruskan dari pemanggil - jangan dibaca ulang
    dari cfg di sini. Kalau ada satu situs saja yang masih membaca cfg,
    simpangan baku antar-seed akan keluar 0 dan sweep-nya jadi bohong.
    """
    c = cfg["classifier"]
    ds = TwoViewDataset(rows, cfg, seed=seed, policy=policy)
    loader = DataLoader(ds, batch_size=int(c["batch_size"]), shuffle=True,
                        num_workers=int(c["num_workers"]), drop_last=False)

    # Hanya backbone + projector yang dilatih di tahap ini.
    # Classifier sengaja tidak disentuh - itu urusan linear probe nanti.
    params = list(model.backbone.parameters()) + list(model.projector.parameters())
    opt = torch.optim.AdamW(params, lr=float(c["lr_contrastive"]),
                            weight_decay=float(c["weight_decay"]))
    epochs = int(c["epochs_contrastive"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ce_w = float(c.get("supcon_ce_weight", 0.0))
    w = class_weights(rows).to(device) if c.get("class_weighting", True) else None
    temp = float(c["temperature"])

    print(f"  [{method}] tahap 1/2 - contrastive, {epochs} epoch, "
          f"{len(ds)} sampel")

    for ep in range(epochs):
        ds.set_epoch(ep)
        model.train()
        tot, nb = 0.0, 0

        for v1, v2, y, _ in loader:
            v1, v2, y = v1.to(device), v2.to(device), y.to(device)

            # BatchNorm di projection head butuh minimal 2 sampel
            if v1.shape[0] < 2:
                continue

            z1, z2 = model.project(v1), model.project(v2)

            if method == "selfcon":
                loss = nt_xent_loss(z1, z2, temp)          # label diabaikan
            else:
                loss = supcon_loss(z1, z2, y, temp)        # label dipakai
                if ce_w > 0:      # opsional: SupCon + CE dilatih bersamaan
                    logit = model.classifier(model.features(v1))
                    loss = loss + ce_w * ce_loss(logit, y, w)

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1

        sched.step()
        if nb and (ep % 10 == 0 or ep == epochs - 1):
            print(f"    epoch {ep + 1:>3}/{epochs}  loss {tot / nb:.4f}")
            log.append({"stage": "contrastive", "epoch": ep + 1,
                        "loss": round(tot / nb, 4)})


def train_head(model, tr_rows, va_rows, cfg, device, log: list,
               freeze: bool, epochs: int, lr: float, tag: str,
               seed: int, policy: dict | None) -> tuple[dict, np.ndarray,
                                                        np.ndarray]:
    """
    Latih kepala klasifikasi.
      freeze=True  -> linear probe (encoder beku), dipakai setelah kontrastif
      freeze=False -> latih seluruh jaringan, dipakai metode CE

    `policy` diterima sebagai parameter dan TIDAK diturunkan ulang dari
    `method` di dalam sini: jalur CE dan jalur linear probe memakai kebijakan
    yang berbeda, dan menurunkannya di dua tempat pasti melenceng.

    Mengembalikan (metrik val terbaik, y_true val, y_score val) - skor val
    dibutuhkan untuk mengkalibrasi ambang, dan harus berasal dari epoch yang
    bobotnya benar-benar dipakai, bukan epoch terakhir.
    """
    c = cfg["classifier"]
    tr_ds = ClassificationDataset(tr_rows, cfg, train=True, seed=seed,
                                  policy=policy)
    va_ds = ClassificationDataset(va_rows, cfg, train=False, seed=seed)
    tr = DataLoader(tr_ds, batch_size=int(c["batch_size"]), shuffle=True,
                    num_workers=int(c["num_workers"]))
    va = DataLoader(va_ds, batch_size=int(c["batch_size"]), shuffle=False,
                    num_workers=int(c["num_workers"]))

    if freeze:
        model.freeze_backbone()
        params = list(model.classifier.parameters())
    else:
        params = list(model.parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(c["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = class_weights(tr_rows).to(device) if c.get("class_weighting", True) else None

    patience = int(c.get("early_stop_patience", 15))
    best = {"balanced_accuracy": -1.0, "roc_auc": -1.0}
    best_state, bad = None, 0
    best_val: tuple[np.ndarray, np.ndarray] | None = None

    print(f"  [{tag}] {epochs} epoch, encoder "
          f"{'BEKU (linear probe)' if freeze else 'ikut dilatih'}")

    for ep in range(epochs):
        tr_ds.set_epoch(ep)
        model.train()
        if freeze:
            model.backbone.eval()     # jaga statistik BatchNorm tetap beku

        tot, nb = 0.0, 0
        for x, y, _ in tr:
            x, y = x.to(device), y.to(device)
            loss = ce_loss(model.logits(x), y, w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()

        m, vy, vs = evaluate(model, va, device)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"    epoch {ep + 1:>3}/{epochs}  loss {tot / max(1, nb):.4f}"
                  f"  val_bacc {m['balanced_accuracy']:.4f}"
                  f"  val_acc {m['accuracy']:.4f}")
        log.append({"stage": tag, "epoch": ep + 1,
                    "loss": round(tot / max(1, nb), 4),
                    "val_bacc": m["balanced_accuracy"]})

        # Simpan model dengan skor validasi terbaik.
        #
        # Tie-break LEKSIKOGRAFIK (bacc, lalu AUC). Alasannya konkret: val
        # bacc mentok di 1.0 untuk selfcon dan ce, dan supcon cuma punya dua
        # nilai berbeda sepanjang 16 epoch. Perbandingan '>' saja menyimpan
        # epoch PERTAMA yang menyentuh dataran - nyaris acak pada val 22 crop,
        # dan epoch mana yang duluan menyentuhnya bergeser karena alasan yang
        # tak ada hubungannya dengan mutu model. AUC memecah seri itu dengan
        # sesuatu yang benar-benar berbeda antar epoch.
        key = (m["balanced_accuracy"], m.get("roc_auc") or -1.0)
        best_key = (best["balanced_accuracy"], best.get("roc_auc") or -1.0)
        if key > best_key:
            best = m
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_val = (vy, vs)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"    early stop di epoch {ep + 1} "
                      f"(tidak membaik {patience} epoch)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_val is None:
        best_val = evaluate(model, va, device)[1:]
    return best, best_val[0], best_val[1]


def run_method(method: str, cfg: dict, device, seed: int,
               aug: str | None = None, save_model: bool = True,
               mirror_legacy: bool = False) -> dict:
    """
    Satu pelatihan penuh: (metode, augmentasi, seed).

    `aug` = nama kebijakan augmentasi untuk TAHAP UTAMA metode ini
    (kontrastif untuk selfcon/supcon, kepala untuk ce). None -> ambil dari
    augmentation.by_method, yaitu pasangan diagonal yang diminta.
    """
    diag = resolve_policy(cfg, method, "contrastive")
    head_default = resolve_policy(cfg, method, "head")

    if aug is None:
        aug = cfg["augmentation"]["by_method"][method][
            "head" if method == "ce" else "contrastive"]
        con_pol, head_pol = diag, head_default
    elif method == "ce":
        # CE hanya punya satu tahap latih, jadi augmentasinya di kepala.
        con_pol, head_pol = None, policy_by_name(cfg, aug)
    else:
        # Kontrastif: augmentasi diterapkan di tahap kontrastif. Kepala tetap
        # 'minimal' kalau lock_probe_policy aktif - lihat resolve_policy.
        con_pol, head_pol = policy_by_name(cfg, aug), head_default

    tag = f"{method}__{aug}__s{seed}"
    print("\n" + "=" * 62)
    print(f"METODE: {method}  |  aug: {aug}  |  seed: {seed}")
    print(f"  {cfg['methods'][method]}")
    print("=" * 62)

    set_seed(seed)
    t0 = time.time()

    tr_rows = read_manifest(cfg, "train")
    va_rows = read_manifest(cfg, "val")
    te_rows = read_manifest(cfg, "test")
    print(f"  data: train {len(tr_rows)}, val {len(va_rows)}, test {len(te_rows)}")

    model = build_model(cfg, device)
    c = cfg["classifier"]
    log: list = []

    if method == "ce":
        # Baseline: langsung latih semuanya dengan cross-entropy
        val_best, vy, vs = train_head(
            model, tr_rows, va_rows, cfg, device, log, freeze=False,
            epochs=int(c["epochs_ce"]), lr=float(c["lr_ce"]), tag="ce",
            seed=seed, policy=head_pol)
    else:
        # Tahap 1: encoder dilatih dengan loss kontrastif
        train_contrastive(model, tr_rows, cfg, device, method, log,
                          seed=seed, policy=con_pol)
        # Tahap 2: encoder dibekukan, hanya linear probe yang dilatih.
        # Encoder dibekukan supaya yang diukur benar-benar kualitas
        # representasi hasil contrastive, bukan hasil fine-tuning lanjutan.
        print(f"  [{method}] tahap 2/2 - linear probe")
        val_best, vy, vs = train_head(
            model, tr_rows, va_rows, cfg, device, log, freeze=True,
            epochs=int(c["epochs_probe"]), lr=float(c["lr_probe"]),
            tag=f"{method}_probe", seed=seed, policy=head_pol)

    # Ambang dikalibrasi DI VALIDATION SET, tidak pernah di test set.
    thr = pick_threshold(vy, vs)

    # Evaluasi akhir pada test set
    te_ds = ClassificationDataset(te_rows, cfg, train=False, seed=seed)
    te = DataLoader(te_ds, batch_size=int(c["batch_size"]), shuffle=False,
                    num_workers=int(c["num_workers"]))
    test_m, y_true, y_score = evaluate(model, te, device)
    test_tuned = metrics_at(y_true, y_score, thr)

    dur = time.time() - t0
    print(f"\n  VAL  bacc {val_best['balanced_accuracy']:.4f}"
          f"   ambang terpilih {thr:.4f}")
    print(f"  TEST bacc @0.5 {test_m['balanced_accuracy']:.4f}  "
          f"@tau {test_tuned['balanced_accuracy']:.4f}  "
          f"acc {test_m['accuracy']:.4f}  "
          f"recall_mati {test_m['recall_dead']:.4f}  "
          f"recall_hidup {test_m['recall_alive']:.4f}  "
          f"auc {test_m.get('roc_auc', float('nan')):.4f}")
    print(f"  waktu {dur:.1f} detik")

    runs = resolve(cfg["output"]["runs_dir"]) / tag
    runs.mkdir(parents=True, exist_ok=True)
    result = {"method": method, "aug": aug, "seed": seed,
              # Nama kebijakan yang BENAR-BENAR dipakai di tiap tahap. Tanpa
              # ini, `aug` saja tidak cukup: untuk selfcon/supcon tahap kepala
              # dikunci ke 'minimal' oleh lock_probe_policy, dan itu tidak bisa
              # diaudit balik dari berkas hasil.
              "policy_contrastive": (con_pol or {}).get("name"),
              "policy_head": (head_pol or {}).get("name"),
              "lock_probe_policy": bool(
                  cfg.get("augmentation", {}).get("lock_probe_policy", True)),
              "description": cfg["methods"][method],
              "val": val_best, "test": test_m, "threshold": round(thr, 6),
              "test_tuned": test_tuned, "seconds": round(dur, 1),
              "n_train": len(tr_rows), "n_val": len(va_rows),
              "n_test": len(te_rows)}
    save_json(result, runs / "result.json")
    save_json(log, runs / "log.json")
    np.savez(runs / "test_scores.npz", y_true=y_true, y_score=y_score)
    # Tanpa skor val tersimpan, ambangnya tidak bisa diaudit belakangan.
    np.savez(runs / "val_scores.npz", y_true=vy, y_score=vs)
    if save_model:
        torch.save({"state_dict": model.state_dict(), "method": method,
                    "aug": aug, "seed": seed, "config": cfg},
                   runs / "model.pt")

    if mirror_legacy:
        # src/pipeline.py:39 membaca runs_dir/<method>/model.pt. Run utama
        # (diagonal, seed pertama) dicerminkan ke path lama supaya pipeline
        # inferensi tetap jalan tanpa diubah.
        legacy = resolve(cfg["output"]["runs_dir"]) / method
        legacy.mkdir(parents=True, exist_ok=True)
        save_json(result, legacy / "result.json")
        save_json(log, legacy / "log.json")
        np.savez(legacy / "test_scores.npz", y_true=y_true, y_score=y_score)
        np.savez(legacy / "val_scores.npz", y_true=vy, y_score=vs)
        if save_model:
            torch.save({"state_dict": model.state_dict(), "method": method,
                        "aug": aug, "seed": seed, "config": cfg},
                       legacy / "model.pt")
    return result


def main():
    ap = argparse.ArgumentParser(description="Tahap 3: latih classifier")
    ap.add_argument("--config", default=None)
    ap.add_argument("--method", default="all", choices=METHODS + ["all"])
    ap.add_argument("--epochs", type=int, default=None,
                    help="timpa jumlah epoch (untuk uji cepat)")
    ap.add_argument("--seeds", default=None,
                    help="daftar seed dipisah koma, mis. 42,43,44,45,46")
    ap.add_argument("--grid", default="diagonal", choices=["diagonal", "full"],
                    help="diagonal = pasangan metode-augmentasi yang diminta; "
                         "full = seluruh 3x3, kontrol untuk memisahkan "
                         "efek loss dari efek augmentasi")
    ap.add_argument("--aug", default=None,
                    help="paksa satu kebijakan augmentasi untuk semua metode")
    ap.add_argument("--save-model", default="primary",
                    choices=["primary", "all", "none"],
                    help="primary = hanya run utama (hemat ~2 GB)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.epochs:
        cfg["classifier"]["epochs_contrastive"] = a.epochs
        cfg["classifier"]["epochs_probe"] = a.epochs
        cfg["classifier"]["epochs_ce"] = a.epochs

    seeds = ([int(x) for x in a.seeds.split(",") if x.strip()]
             if a.seeds else [int(cfg["seed"])])
    methods = METHODS if a.method == "all" else [a.method]

    if a.aug:
        augs = [a.aug]
    elif a.grid == "full":
        augs = [cfg["augmentation"]["by_method"][m][
            "head" if m == "ce" else "contrastive"] for m in METHODS]
    else:
        augs = [None]        # None -> pasangan diagonal per metode

    device = get_device(cfg.get("device", "auto"))
    print(f"device: {device}")
    print(f"grid: {a.grid} | metode {methods} | aug {augs} | seed {seeds}")

    results = []
    for aug in augs:
        for m in methods:
            for i, sd in enumerate(seeds):
                diag_aug = cfg["augmentation"]["by_method"][m][
                    "head" if m == "ce" else "contrastive"]
                is_primary = (aug in (None, diag_aug)) and i == 0
                results.append(run_method(
                    m, cfg, device, seed=sd, aug=aug,
                    save_model=(a.save_model == "all"
                                or (a.save_model == "primary" and is_primary)),
                    mirror_legacy=is_primary))

    if len(results) > 1:
        print("\n" + "=" * 62)
        print("PERBANDINGAN (test set)")
        print("=" * 62)
        print(f"{'metode':<9}{'aug':<17}{'seed':>5}{'bacc':>8}{'@tau':>8}"
              f"{'R.mati':>8}{'R.hidup':>9}{'AUC':>8}")
        for r in sorted(results, key=lambda x: -x["test"]["balanced_accuracy"]):
            t = r["test"]
            print(f"{r['method']:<9}{r['aug']:<17}{r['seed']:>5}"
                  f"{t['balanced_accuracy']:>8.4f}"
                  f"{r['test_tuned']['balanced_accuracy']:>8.4f}"
                  f"{t['recall_dead']:>8.4f}{t['recall_alive']:>9.4f}"
                  f"{t.get('roc_auc', float('nan')):>8.4f}")
        save_json(results,
                  resolve(cfg["output"]["reports_dir"]) / "comparison.json")


if __name__ == "__main__":
    main()
