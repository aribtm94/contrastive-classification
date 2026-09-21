"""Benchmark test ayam tetap dari registry checkpoint development.

Training dan validation hanya memakai PIO hidup + Roboflow mati. Skrip ini
memverifikasi hash registry, lalu mengukur classifier absolut dan anomali
relatif tanpa memilih checkpoint, ambang, atau scorer dari hasil chick.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from build_crops import equalize
from common import get_device, imread, resolve, save_json, to_square
from dataset import to_tensor
from eval_same_frame import (confusion, group_indices, ranks_within_frame,
                             relative_scores)
from intervensi import acak_petak
from models import build_model
from train import roc_auc


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_all_rows(labels: Path, crops: Path) -> list[dict]:
    with open(labels, newline="", encoding="utf-8-sig") as f:
        raw = list(csv.DictReader(f))
    if len(raw) != 1215:
        raise ValueError(f"benchmark wajib 1215 baris, ditemukan {len(raw)}")
    rows = []
    allowed = {"", "mati", "bukan"}
    for r in raw:
        label = r["label"].strip().lower()
        if label not in allowed:
            raise ValueError(f"label tidak dikenal: {r['nomor']}={label}")
        path = crops / r["berkas"]
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append({
            "nomor": int(r["nomor"]), "berkas": r["berkas"],
            "gambar_sumber": r["gambar_sumber"], "det_id": int(r["det_id"]),
            "label": "hidup" if not label else label,
            "y": 1 if label == "mati" else (0 if label == "" else -1),
            "conf": float(r["conf"]), "bbox": r["bbox"],
            "lebar_px": int(r["lebar_px"]),
            "tinggi_px": int(r["tinggi_px"]), "path": path,
        })
    counts = {lab: sum(r["label"] == lab for r in rows)
              for lab in ("mati", "hidup", "bukan")}
    frames = sorted({r["gambar_sumber"] for r in rows})
    if counts != {"mati": 22, "hidup": 921, "bukan": 272} or len(frames) != 18:
        raise ValueError(f"kontrak benchmark berubah: {counts}, {len(frames)} frame")
    n_ayam = sum(name.lower().startswith("ayam (") for name in frames)
    n_chick = sum(name.lower().startswith("chick (") for name in frames)
    if (n_ayam, n_chick) != (11, 7):
        raise ValueError("benchmark wajib 11 frame ayam + 7 frame chick")
    return rows


def prepare(rows: list[dict], cfg: dict, intervention: str) -> list[np.ndarray]:
    size = int(cfg["classifier"]["image_size"])
    mode = cfg["classifier"]["resize_mode"]
    out = []
    for row in rows:
        image = imread(row["path"])
        if image is None:
            raise ValueError(f"crop gagal dibaca: {row['path']}")
        image = equalize(image, cfg["crops"])
        if intervention == "acak16":
            sample_id = f"{row['gambar_sumber']}|{row['det_id']}"
            image = acak_petak(image, sample_id, seed=20260913)
        elif intervention != "asli":
            raise ValueError(f"intervensi tidak dikenal: {intervention}")
        out.append(to_square(image, size, mode))
    return out


@torch.inference_mode()
def extract(model, images: list[np.ndarray], cfg: dict, device,
            batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    size = int(cfg["classifier"]["image_size"])
    mode = cfg["classifier"]["resize_mode"]
    features, probabilities = [], []
    for start in range(0, len(images), batch_size):
        batch = torch.stack([to_tensor(x, size, mode)
                             for x in images[start:start + batch_size]]).to(device)
        feat = model.features(batch)
        features.append(feat.cpu().numpy())
        probabilities.append(torch.softmax(model.classifier(feat), dim=1)[:, 1]
                             .cpu().numpy())
    return np.concatenate(features), np.concatenate(probabilities)


def threshold_free(y, scores, groups, nomor) -> tuple[dict, np.ndarray,
                                                       np.ndarray]:
    ranks, percentiles, normalized = ranks_within_frame(scores, groups, nomor)
    aucs, aps, reciprocal, dead_pct = [], [], [], []
    recall = {k: [] for k in (1, 3, 5)}
    hit = {k: [] for k in (1, 3, 5)}
    for idx in group_indices(groups).values():
        aucs.append(roc_auc(y[idx], scores[idx]))
        aps.append(float(average_precision_score(y[idx], scores[idx])))
        dead_ranks = ranks[idx][y[idx] == 1]
        reciprocal.append(1.0 / dead_ranks.min())
        dead_pct.extend(percentiles[idx][y[idx] == 1])
        for k in recall:
            recall[k].append(float((dead_ranks <= k).sum() / len(dead_ranks)))
            hit[k].append(float((dead_ranks <= k).any()))
    metrics = {
        "pooled_auc": roc_auc(y, scores),
        "pooled_ap": float(average_precision_score(y, scores)),
        "macro_auc": float(np.mean(aucs)), "macro_ap": float(np.mean(aps)),
        **{f"recall_at_{k}": float(np.mean(recall[k])) for k in recall},
        **{f"hit_at_{k}": float(np.mean(hit[k])) for k in hit},
        "mrr": float(np.mean(reciprocal)),
        "mean_dead_rank_percentile": float(np.mean(dead_pct)),
    }
    return metrics, ranks, normalized


def absolute_metrics(y, score, groups, nomor, threshold: float) -> dict:
    metrics, _, _ = threshold_free(y, score, groups, nomor)
    pred05 = (score >= 0.5).astype(int)
    pred_val = (score >= threshold).astype(int)
    metrics["decision_at_0_5"] = confusion(y, pred05)
    metrics["decision_at_tau_val"] = confusion(y, pred_val)
    metrics["tau_val"] = float(threshold)
    metrics["fp_per_frame_at_0_5"] = int(((pred05 == 1) & (y == 0)).sum()) / 18
    metrics["fp_per_frame_at_tau_val"] = int(((pred_val == 1) & (y == 0)).sum()) / 18
    return metrics


def nuisance_features(image: np.ndarray, row: dict) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    w, h = row["lebar_px"], row["tinggi_px"]
    return {
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "saturation": float(hsv[:, :, 1].mean()),
        "brightness": float(hsv[:, :, 2].mean()),
        "hue": float(hsv[:, :, 0].mean()),
        "bbox_short_side": min(w, h), "bbox_area": w * h,
        "bbox_aspect": max(w, h) / max(1, min(w, h)),
        "letterbox_padding_fraction": 1.0 - min(w, h) / max(w, h),
        "detector_confidence": row["conf"],
    }


def nuisance_metrics(rows: list[dict], values: list[dict]) -> dict:
    """AUC signed; arah tidak pernah dibalik berdasarkan hasil test."""
    idx = np.asarray([i for i, row in enumerate(rows) if row["y"] >= 0])
    y = np.asarray([rows[i]["y"] for i in idx], dtype=int)
    groups = np.asarray([rows[i]["gambar_sumber"] for i in idx])
    numero = np.asarray([rows[i]["nomor"] for i in idx])
    out = {}
    for name in values[0]:
        score = np.asarray([values[i][name] for i in idx], dtype=float)
        metrics, _, _ = threshold_free(y, score, groups, numero)
        out[name] = metrics
    return out


def load_registries(paths: str) -> list[tuple[Path, dict]]:
    out = []
    for text in paths.split(","):
        path = resolve(text.strip())
        with open(path, encoding="utf-8") as f:
            registry = json.load(f)
        if registry.get("test_policy") != "fixed_retrospective_chick_test_only":
            raise ValueError(f"registry tidak mengunci chick: {path}")
        if len(registry.get("checkpoints", [])) != 9:
            raise ValueError(f"registry wajib sembilan checkpoint: {path}")
        # Registry harus tetap dapat dipakai setelah repo dipindah. Config asli
        # dicek hash-nya bila masih ada; snapshot menjadi sumber preprocessing.
        config_path = Path(registry["config"])
        if config_path.exists() and sha256(config_path) != registry["config_sha256"]:
            raise ValueError(f"config berubah sejak registry dibekukan: {config_path}")
        out.append((path, registry))
    return out


def main():
    ap = argparse.ArgumentParser(description="Benchmark test ayam tetap")
    ap.add_argument("--registries", required=True,
                    help="registry dipisah koma; dibuat sebelum test")
    ap.add_argument("--labels", default="data/label_chick/lembar_label.csv")
    ap.add_argument("--crops", default="data/label_chick/crops")
    ap.add_argument("--interventions", default="asli,acak16")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--output-prefix", default="fixed_chick")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    rows = load_all_rows(resolve(args.labels), resolve(args.crops))
    clean_idx = np.asarray([i for i, row in enumerate(rows) if row["y"] >= 0])
    clean_pos = {original: pos for pos, original in enumerate(clean_idx)}
    y = np.asarray([rows[i]["y"] for i in clean_idx], dtype=int)
    groups = np.asarray([rows[i]["gambar_sumber"] for i in clean_idx])
    numero = np.asarray([rows[i]["nomor"] for i in clean_idx])
    all_groups = np.asarray([r["gambar_sumber"] for r in rows])
    all_numero = np.asarray([r["nomor"] for r in rows])
    device = get_device(args.device)
    registries = load_registries(args.registries)
    interventions = [x.strip() for x in args.interventions.split(",") if x.strip()]

    output = {
        "schema_version": 1,
        "benchmark": "fixed_retrospective_chick_test_only",
        "selection_from_test": False,
        "data": {"total": 1215, "dead": 22, "alive": 921,
                 "bukan": 272, "valid": 943, "frames": 18},
        "interventions": {},
    }
    audit_rows = []
    for intervention in interventions:
        records = []
        for registry_path, registry in registries:
            for item in registry["checkpoints"]:
                checkpoint = Path(item["checkpoint"])
                if sha256(checkpoint) != item["checkpoint_sha256"]:
                    raise ValueError(f"hash checkpoint berubah: {checkpoint}")
                ckpt = torch.load(checkpoint, map_location=device)
                cfg = registry["config_snapshot"]
                model = build_model(cfg, device)
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                images = prepare(rows, cfg, intervention)
                feat_all, prob_all = extract(model, images, cfg, device,
                                             args.batch_size)
                feat, prob = feat_all[clean_idx], prob_all[clean_idx]
                rel_clean = relative_scores(feat, groups, "median")
                rel_operational_all = relative_scores(feat_all, all_groups, "median")
                rel_operational = rel_operational_all[clean_idx]
                abs_metric = absolute_metrics(
                    y, prob, groups, numero, item["threshold_from_validation"])
                rel_clean_metric, rel_clean_rank, _ = threshold_free(
                    y, rel_clean, groups, numero)
                rel_op_metric, _, _ = threshold_free(
                    y, rel_operational, groups, numero)
                rel_op_rank, _, rel_op_norm = ranks_within_frame(
                    rel_operational_all, all_groups, all_numero)
                nuisance = [nuisance_features(images[i], rows[i])
                            for i in range(len(rows))]
                bukan_idx = np.asarray([i for i, row in enumerate(rows)
                                        if row["label"] == "bukan"])
                tau = float(item["threshold_from_validation"])
                bukan_audit = {
                    "n_bukan": int(len(bukan_idx)),
                    "bukan_above_0_5": int((prob_all[bukan_idx] >= 0.5).sum()),
                    "bukan_above_tau_val": int((prob_all[bukan_idx] >= tau).sum()),
                    **{f"bukan_in_top_{k}": int((rel_op_rank[bukan_idx] <= k).sum())
                       for k in (1, 3, 5)},
                }
                # Varian crop (asli / eq48) dibaca hilir dari sini, bukan
                # ditebak dari nama berkas registry. Hanya blok crops yang
                # ikut supaya JSON tidak membengkak oleh config penuh.
                records.append({"registry": str(registry_path), **item,
                                "config_snapshot": {"crops": cfg["crops"]},
                                "intervention": intervention,
                                "absolute": abs_metric,
                                "relative_clean": rel_clean_metric,
                                "relative_operational": rel_op_metric,
                                "bukan_audit": bukan_audit})
                for i, row in enumerate(rows):
                    pos = clean_pos.get(i)
                    audit_rows.append({
                        "registry": registry_path.name, "run_id": item["run_id"],
                        "method": item["method"], "seed": item["seed"],
                        "intervention": intervention, "nomor": row["nomor"],
                        "gambar_sumber": row["gambar_sumber"],
                        "det_id": row["det_id"], "label": row["label"],
                        "prob_dead": float(prob_all[i]), "tau_val": tau,
                        "pred_0_5": int(prob_all[i] >= 0.5),
                        "pred_tau_val": int(prob_all[i] >= tau),
                        "relative_clean": (float(rel_clean[pos])
                                           if pos is not None else ""),
                        "rank_clean": (float(rel_clean_rank[pos])
                                       if pos is not None else ""),
                        "relative_operational": float(rel_operational_all[i]),
                        "rank_operational": float(rel_op_rank[i]),
                        "rank_operational_percentile": float(rel_op_norm[i]),
                        **nuisance[i],
                    })
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        output["interventions"][intervention] = records
        # Nilainya sama untuk semua checkpoint dengan preprocessing yang sama;
        # simpan signed AUC per registry supaya arah tidak dipilih dari test.
        output.setdefault("nuisance", {})[intervention] = {}
        for registry_path, registry in registries:
            cfg = registry["config_snapshot"]
            images = prepare(rows, cfg, intervention)
            values = [nuisance_features(images[i], rows[i])
                      for i in range(len(rows))]
            output["nuisance"][intervention][registry_path.name] = \
                nuisance_metrics(rows, values)

    json_path = resolve(f"outputs/predictions/{args.output_prefix}.json")
    csv_path = resolve(f"outputs/predictions/{args.output_prefix}_crops.csv")
    save_json(output, json_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    print("Benchmark retrospektif selesai tanpa pemilihan dari test.")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")


if __name__ == "__main__":
    main()
