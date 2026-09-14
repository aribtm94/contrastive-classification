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
import copy
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from common import get_device, load_config, resolve, save_json, set_seed
from dataset import (ClassificationDataset, TwoViewDataset, class_weights,
                     policy_by_name, read_manifest, resolve_policy,
                     validate_development_manifest)
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


def _loader(dataset, cfg: dict, shuffle: bool, seed: int) -> DataLoader:
    """DataLoader dengan urutan acak yang dapat diulang untuk satu tahap."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = (seed + worker_id) & 0xFFFFFFFF
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return DataLoader(
        dataset, batch_size=int(cfg["classifier"]["batch_size"]),
        shuffle=shuffle, num_workers=int(cfg["classifier"]["num_workers"]),
        drop_last=False, generator=generator, worker_init_fn=seed_worker)


def _ce_denominator(y: torch.Tensor, weight: torch.Tensor | None) -> float:
    """Penyebut yang dipakai reduction='mean' pada weighted CE PyTorch."""
    return float(len(y) if weight is None else weight[y].sum().item())


@torch.inference_mode()
def evaluate_with_ce(model, loader, device,
                     weight: torch.Tensor | None) -> tuple[dict, np.ndarray,
                                                            np.ndarray, float]:
    """Evaluasi klasifikasi sekaligus weighted CE yang sebanding antar-epoch."""
    model.eval()
    ys, ps, ss = [], [], []
    numerator, denominator = 0.0, 0.0
    for x, y, _ in loader:
        y_dev = y.to(device)
        logit = model.logits(x.to(device))
        loss = ce_loss(logit, y_dev, weight)
        den = _ce_denominator(y_dev, weight)
        numerator += float(loss.item()) * den
        denominator += den
        prob = torch.softmax(logit, dim=1)[:, 1]
        ps.append(logit.argmax(1).cpu().numpy())
        ss.append(prob.cpu().numpy())
        ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    y_score = np.concatenate(ss)
    return (compute_metrics(y_true, y_pred, y_score), y_true, y_score,
            numerator / max(denominator, 1e-12))


@torch.inference_mode()
def evaluate_contrastive_loss(model, loader, device, method: str,
                              temperature: float, ce_weight: float,
                              class_weight: torch.Tensor | None) -> dict:
    """Objective validation pada dua view tetap; tidak mengubah state model."""
    model.eval()
    total_num = base_num = aux_num = 0.0
    anchors = 0
    batches = 0
    for v1, v2, y, _ in loader:
        if v1.shape[0] < 2:
            continue
        v1, v2, y = v1.to(device), v2.to(device), y.to(device)
        z1, z2 = model.project(v1), model.project(v2)
        base = (nt_xent_loss(z1, z2, temperature) if method == "selfcon"
                else supcon_loss(z1, z2, y, temperature))
        aux = None
        total = base
        if method == "supcon" and ce_weight > 0:
            aux = ce_loss(model.classifier(model.features(v1)), y, class_weight)
            total = base + ce_weight * aux
        n_anchor = 2 * len(y)
        base_num += float(base.item()) * n_anchor
        total_num += float(total.item()) * n_anchor
        if aux is not None:
            aux_num += float(aux.item()) * n_anchor
        anchors += n_anchor
        batches += 1
    if not batches:
        raise RuntimeError("tidak ada batch contrastive validation yang valid")
    return {
        "total": total_num / anchors,
        "base": base_num / anchors,
        "aux_ce": aux_num / anchors if ce_weight > 0 else None,
        "anchors": anchors,
        "batches": batches,
    }


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def runtime_metadata(device: torch.device) -> dict:
    """Metadata minimum agar run development dapat diaudit ulang."""
    try:
        git_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        git_dirty = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": getattr(getattr(torch, "version", None), "cuda", None),
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device)
                        if device.type == "cuda" else platform.processor()),
        "git_commit": _git_commit(), "git_dirty": git_dirty,
    }


def source_hashes() -> dict[str, str]:
    """Hash source yang menentukan training dan evaluasi pra-test."""
    names = (
        "src/train.py", "src/dataset.py", "src/models.py", "src/common.py",
        "src/build_crops.py", "src/intervensi.py", "src/eval_same_frame.py",
        "src/eval_fixed_chick.py", "src/report_loss.py",
        "src/report_fixed_chick.py",
    )
    return {name: _sha256(resolve(name)) for name in names}


_HISTORY_FIELDS = [
    "stage", "epoch", "lr_used", "lr_next", "train_loss", "val_loss",
    "train_base_loss", "val_base_loss", "train_aux_ce", "val_aux_ce",
    "n_train_units", "n_val_units", "train_batches", "val_batches",
    "skipped_batches", "val_bacc", "val_auc", "val_recall_dead",
    "val_recall_alive", "val_precision_dead", "val_f1_dead",
]


def write_history_csv(path, log: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_HISTORY_FIELDS)
        writer.writeheader()
        for row in log:
            writer.writerow({k: row.get(k) for k in _HISTORY_FIELDS})


def write_development_bundle(cfg: dict, results: list[dict],
                             all_logs: dict[str, list[dict]],
                             device: torch.device) -> None:
    """Gabungkan history dan bekukan registry sembilan checkpoint pra-test."""
    protocol = cfg["protocol"]
    history_path = resolve(protocol["history_csv"])
    history_fields = ["run_id", "method", "aug", "seed"] + _HISTORY_FIELDS
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history_fields)
        writer.writeheader()
        for result in results:
            run_id = f"{result['method']}__{result['aug']}__s{result['seed']}"
            for row in all_logs[run_id]:
                writer.writerow({"run_id": run_id, "method": result["method"],
                                 "aug": result["aug"], "seed": result["seed"],
                                 **{k: row.get(k) for k in _HISTORY_FIELDS}})

    expected = {(m, s) for m in METHODS for s in (42, 43, 44)}
    actual = {(r["method"], int(r["seed"])) for r in results}
    if actual != expected or len(results) != 9:
        print("[development] smoke/partial run: registry final belum dibuat")
        return

    manifest = resolve(cfg["crops"]["manifest"])
    lock_path = resolve(protocol["split_lock"])
    config_path = resolve(cfg["_config_path"])
    entries = []
    for result in sorted(results, key=lambda r: (r["method"], r["seed"])):
        run_id = f"{result['method']}__{result['aug']}__s{result['seed']}"
        checkpoint = resolve(cfg["output"]["runs_dir"]) / run_id / "model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"registry final membutuhkan seluruh checkpoint: {checkpoint}")
        entries.append({
            "run_id": run_id, "method": result["method"],
            "augmentation": result["aug"], "seed": int(result["seed"]),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "threshold_from_validation": result["threshold_from_validation"],
            "validation": result["validation"],
        })
    shortcut_path = resolve(protocol["shortcut_json"])
    if not shortcut_path.exists():
        raise FileNotFoundError(f"baseline shortcut belum ada: {shortcut_path}")
    config_snapshot = copy.deepcopy(cfg)
    config_snapshot.pop("_config_path", None)
    registry = {
        "schema_version": 1,
        "experiment": "pio_roboflow_development",
        "test_policy": "fixed_retrospective_chick_test_only",
        "data_contract": {
            "alive": "PIO/pio_gt/cctv", "dead": "Roboflow/coco_gt/closeup",
            "train_validation_only": True,
            "label_equals_domain_limitation": True,
            "test_root": "C:/Arib/CCTV/patnet-pure/dataset/chick",
            "test_counts": {"total": 1215, "dead": 22,
                            "alive": 921, "bukan": 272, "frames": 18},
        },
        "config": str(config_path), "config_sha256": _sha256(config_path),
        "config_snapshot": config_snapshot,
        "manifest": str(manifest), "manifest_sha256": _sha256(manifest),
        "split_lock": str(lock_path), "split_lock_sha256": _sha256(lock_path),
        "shortcut_baseline": str(shortcut_path),
        "shortcut_baseline_sha256": _sha256(shortcut_path),
        "runtime": runtime_metadata(device), "source_sha256": source_hashes(),
        "seeds": [42, 43, 44],
        "primary_relative_scorer": "feature512_median_loo",
        "checkpoints": entries,
    }
    registry_path = resolve(protocol["registry"])
    save_json(registry, registry_path)
    print(f"[development] history : {history_path}")
    print(f"[development] registry: {registry_path}")


# --------------------------------------------------------------------------- #
# Tahap pelatihan
# --------------------------------------------------------------------------- #
def train_contrastive(model, tr_rows, va_rows, cfg, device, method: str,
                      log: list, seed: int, policy: dict | None) -> None:
    """Latih encoder dan catat objective train/validation pada setiap epoch."""
    c = cfg["classifier"]
    tr_ds = TwoViewDataset(tr_rows, cfg, seed=seed, policy=policy)
    val_offset = int(cfg.get("protocol", {}).get("validation_seed_offset", 900001))
    va_ds = TwoViewDataset(va_rows, cfg, seed=seed + val_offset, policy=policy)
    tr = _loader(tr_ds, cfg, shuffle=True, seed=seed + 101)
    # Manifest tersortir per label. SupCon loss bergantung komposisi batch;
    # satu permutasi validation tetap mencampur kelas tanpa berubah antar-epoch.
    val_rng = np.random.default_rng(seed + val_offset)
    va_order = val_rng.permutation(len(va_rows)).tolist()
    va = _loader(torch.utils.data.Subset(va_ds, va_order), cfg,
                 shuffle=False, seed=seed + 102)

    # Hanya backbone + projector dilatih. Auxiliary CE default-nya nonaktif.
    params = list(model.backbone.parameters()) + list(model.projector.parameters())
    opt = torch.optim.AdamW(params, lr=float(c["lr_contrastive"]),
                            weight_decay=float(c["weight_decay"]))
    epochs = int(c["epochs_contrastive"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce_w = float(c.get("supcon_ce_weight", 0.0))
    w = class_weights(tr_rows).to(device) if c.get("class_weighting", True) else None
    temp = float(c["temperature"])

    print(f"  [{method}] tahap 1/2 - contrastive, {epochs} epoch, "
          f"{len(tr_ds)} train / {len(va_ds)} val")
    for ep in range(epochs):
        tr_ds.set_epoch(ep)
        model.train()
        lr_used = float(opt.param_groups[0]["lr"])
        total_num = base_num = aux_num = 0.0
        anchors = batches = skipped = 0
        for v1, v2, y, _ in tr:
            v1, v2, y = v1.to(device), v2.to(device), y.to(device)
            if v1.shape[0] < 2:
                skipped += 1
                continue
            z1, z2 = model.project(v1), model.project(v2)
            base = (nt_xent_loss(z1, z2, temp) if method == "selfcon"
                    else supcon_loss(z1, z2, y, temp))
            aux = None
            loss = base
            if method == "supcon" and ce_w > 0:
                aux = ce_loss(model.classifier(model.features(v1)), y, w)
                loss = base + ce_w * aux
            opt.zero_grad()
            loss.backward()
            opt.step()
            n_anchor = 2 * len(y)
            total_num += float(loss.item()) * n_anchor
            base_num += float(base.item()) * n_anchor
            if aux is not None:
                aux_num += float(aux.item()) * n_anchor
            anchors += n_anchor
            batches += 1
        if not batches:
            raise RuntimeError("tidak ada batch contrastive train yang valid")
        val_obj = evaluate_contrastive_loss(
            model, va, device, method, temp, ce_w, w)
        sched.step()
        lr_next = float(opt.param_groups[0]["lr"])
        train_loss = total_num / anchors
        record = {
            "stage": "contrastive", "epoch": ep + 1,
            "lr_used": lr_used, "lr_next": lr_next,
            "train_loss": train_loss, "val_loss": val_obj["total"],
            "train_base_loss": base_num / anchors,
            "val_base_loss": val_obj["base"],
            "train_aux_ce": aux_num / anchors if ce_w > 0 else None,
            "val_aux_ce": val_obj["aux_ce"],
            "n_train_units": anchors, "n_val_units": val_obj["anchors"],
            "train_batches": batches, "val_batches": val_obj["batches"],
            "skipped_batches": skipped,
        }
        log.append(record)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"    epoch {ep + 1:>3}/{epochs}  "
                  f"train_loss {train_loss:.4f}  val_loss {val_obj['total']:.4f}")


def train_head(model, tr_rows, va_rows, cfg, device, log: list,
               freeze: bool, epochs: int, lr: float, tag: str,
               seed: int, policy: dict | None) -> tuple[dict, np.ndarray,
                                                        np.ndarray]:
    """Latih CE/probe; train dan validation loss dicatat setiap epoch."""
    c = cfg["classifier"]
    tr_ds = ClassificationDataset(tr_rows, cfg, train=True, seed=seed,
                                  policy=policy)
    va_ds = ClassificationDataset(va_rows, cfg, train=False, seed=seed)
    tr = _loader(tr_ds, cfg, shuffle=True, seed=seed + 201)
    va = _loader(va_ds, cfg, shuffle=False, seed=seed + 202)

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
            model.backbone.eval()
        lr_used = float(opt.param_groups[0]["lr"])
        numerator = denominator = 0.0
        batches = 0
        for x, y, _ in tr:
            x, y = x.to(device), y.to(device)
            loss = ce_loss(model.logits(x), y, w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            den = _ce_denominator(y, w)
            numerator += float(loss.item()) * den
            denominator += den
            batches += 1
        train_loss = numerator / max(denominator, 1e-12)
        m, vy, vs, val_loss = evaluate_with_ce(model, va, device, w)
        sched.step()
        lr_next = float(opt.param_groups[0]["lr"])
        log.append({
            "stage": tag, "epoch": ep + 1,
            "lr_used": lr_used, "lr_next": lr_next,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_base_loss": train_loss, "val_base_loss": val_loss,
            "train_aux_ce": None, "val_aux_ce": None,
            "n_train_units": len(tr_rows), "n_val_units": len(va_rows),
            "train_batches": batches, "val_batches": len(va),
            "skipped_batches": 0,
            "val_bacc": m["balanced_accuracy"], "val_auc": m.get("roc_auc"),
            "val_recall_dead": m["recall_dead"],
            "val_recall_alive": m["recall_alive"],
            "val_precision_dead": m["precision_dead"],
            "val_f1_dead": m["f1_dead"],
        })
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"    epoch {ep + 1:>3}/{epochs}  train_loss {train_loss:.4f}"
                  f"  val_loss {val_loss:.4f}  val_bacc {m['balanced_accuracy']:.4f}")

        # Aturan historis dipertahankan: validation bacc, lalu AUC.
        key = (m["balanced_accuracy"], m.get("roc_auc") or -1.0)
        best_key = (best["balanced_accuracy"], best.get("roc_auc") or -1.0)
        if key > best_key:
            best = dict(m, epoch=ep + 1, val_loss=val_loss)
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
        _, vy, vs, _ = evaluate_with_ce(model, va, device, w)
        best_val = (vy, vs)
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

    development_only = bool(cfg.get("protocol", {}).get("development_only"))
    if development_only:
        validate_development_manifest(cfg)
    tr_rows = read_manifest(cfg, "train")
    va_rows = read_manifest(cfg, "val")
    te_rows = [] if development_only else read_manifest(cfg, "test")
    print(f"  data: train {len(tr_rows)}, val {len(va_rows)}" +
          (" (development; chick tidak dibaca)" if development_only
           else f", test {len(te_rows)}"))

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
        train_contrastive(model, tr_rows, va_rows, cfg, device, method, log,
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

    test_m = test_tuned = None
    y_true = y_score = None
    if not development_only:
        te_ds = ClassificationDataset(te_rows, cfg, train=False, seed=seed)
        te = _loader(te_ds, cfg, shuffle=False, seed=seed + 203)
        test_m, y_true, y_score = evaluate(model, te, device)
        test_tuned = metrics_at(y_true, y_score, thr)

    dur = time.time() - t0
    print(f"\n  VAL  bacc {val_best['balanced_accuracy']:.4f}"
          f"   ambang terpilih {thr:.4f}")
    if test_m is not None and test_tuned is not None:
        print(f"  TEST bacc @0.5 {test_m['balanced_accuracy']:.4f}  "
              f"@tau {test_tuned['balanced_accuracy']:.4f}  "
              f"acc {test_m['accuracy']:.4f}  "
              f"recall_mati {test_m['recall_dead']:.4f}  "
              f"recall_hidup {test_m['recall_alive']:.4f}  "
              f"auc {test_m.get('roc_auc', float('nan')):.4f}")
    else:
        print("  TEST tidak dibaca; benchmark ayam dijalankan terpisah.")
    print(f"  waktu {dur:.1f} detik")

    runs = resolve(cfg["output"]["runs_dir"]) / tag
    runs.mkdir(parents=True, exist_ok=True)
    result = {
        "method": method, "aug": aug, "seed": seed,
        "policy_contrastive": (con_pol or {}).get("name"),
        "policy_head": (head_pol or {}).get("name"),
        "lock_probe_policy": bool(
            cfg.get("augmentation", {}).get("lock_probe_policy", True)),
        "description": cfg["methods"][method],
        "validation": val_best, "threshold_from_validation": thr,
        "seconds": round(dur, 1), "n_train": len(tr_rows),
        "n_val": len(va_rows), "development_only": development_only,
        "runtime": runtime_metadata(device),
    }
    if not development_only:
        result.update({"test": test_m, "test_tuned": test_tuned,
                       "n_test": len(te_rows)})
    save_json(result, runs / "result.json")
    save_json(log, runs / "log.json")
    write_history_csv(runs / "history.csv", log)
    np.savez(runs / "val_scores.npz", y_true=vy, y_score=vs)
    if not development_only and y_true is not None and y_score is not None:
        np.savez(runs / "test_scores.npz", y_true=y_true, y_score=y_score)
    if save_model:
        torch.save({"state_dict": model.state_dict(), "method": method,
                    "aug": aug, "seed": seed, "config": cfg,
                    "threshold_from_validation": thr,
                    "validation": val_best}, runs / "model.pt")

    if mirror_legacy and not development_only:
        if y_true is None or y_score is None:
            raise RuntimeError("skor test legacy tidak tersedia")
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
    cfg["_config_path"] = str(a.config or "configs/config.yaml")
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

    development_only = bool(cfg.get("protocol", {}).get("development_only"))
    if development_only:
        validate_development_manifest(cfg)
        if a.grid != "diagonal" or a.aug:
            raise SystemExit("protokol development hanya menerima grid diagonal")
        if a.seeds and seeds != [42, 43, 44]:
            print("PERINGATAN: seed override hanya untuk smoke test; registry final wajib 42,43,44")

    device = get_device(cfg.get("device", "auto"))
    print(f"device: {device}")
    print(f"grid: {a.grid} | metode {methods} | aug {augs} | seed {seeds}")

    results = []
    all_logs = {}
    for aug in augs:
        for m in methods:
            for i, sd in enumerate(seeds):
                diag_aug = cfg["augmentation"]["by_method"][m][
                    "head" if m == "ce" else "contrastive"]
                is_primary = (aug in (None, diag_aug)) and i == 0
                result = run_method(
                    m, cfg, device, seed=sd, aug=aug,
                    save_model=(a.save_model == "all"
                                or (a.save_model == "primary" and is_primary)),
                    mirror_legacy=is_primary)
                results.append(result)
                run_id = f"{result['method']}__{result['aug']}__s{result['seed']}"
                with open(resolve(cfg["output"]["runs_dir"]) / run_id / "log.json",
                          encoding="utf-8") as f:
                    all_logs[run_id] = json.load(f)

    if development_only:
        write_development_bundle(cfg, results, all_logs, device)

    if len(results) > 1:
        development_only = bool(cfg.get("protocol", {}).get("development_only"))
        print("\n" + "=" * 62)
        if development_only:
            print("RINGKASAN VALIDATION (bukan hasil test)")
            print("=" * 62)
            print(f"{'metode':<9}{'aug':<17}{'seed':>5}{'bacc':>9}"
                  f"{'AUC':>9}{'tau_val':>11}")
            for r in sorted(results, key=lambda x: (x["method"], x["seed"])):
                v = r["validation"]
                print(f"{r['method']:<9}{r['aug']:<17}{r['seed']:>5}"
                      f"{v['balanced_accuracy']:>9.4f}"
                      f"{v.get('roc_auc', float('nan')):>9.4f}"
                      f"{r['threshold_from_validation']:>11.4f}")
            dest = resolve(cfg["output"]["reports_dir"]) / "development_comparison.json"
        else:
            print("PERBANDINGAN (test set)")
            print("=" * 62)
            print(f"{'metode':<9}{'aug':<17}{'seed':>5}{'bacc':>8}{'@tau':>8}"
                  f"{'R.mati':>8}{'R.hidup':>9}{'AUC':>8}")
            for r in sorted(results,
                            key=lambda x: -x["test"]["balanced_accuracy"]):
                t = r["test"]
                print(f"{r['method']:<9}{r['aug']:<17}{r['seed']:>5}"
                      f"{t['balanced_accuracy']:>8.4f}"
                      f"{r['test_tuned']['balanced_accuracy']:>8.4f}"
                      f"{t['recall_dead']:>8.4f}{t['recall_alive']:>9.4f}"
                      f"{t.get('roc_auc', float('nan')):>8.4f}")
            dest = resolve(cfg["output"]["reports_dir"]) / "comparison.json"
        save_json(results, dest)


if __name__ == "__main__":
    main()
