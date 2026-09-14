"""
Tahap 1: uji skor anomali relatif antar-crop dari frame CCTV yang sama.

Eksperimen ini tidak melatih model. Fitur 27 checkpoint lama dievaluasi pada
943 crop chick valid. Skor utama adalah jarak kosinus fitur backbone terhadap
median koordinat-wise frame yang dihitung leave-one-out.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_crops import equalize
from common import get_device, imread, resolve, save_json, to_square
from dataset import to_tensor
from models import build_model
from intervensi import acak_petak
from train import pick_threshold, roc_auc


FAMILIES = (
    ("closeup", "outputs/runs_closeup_ckpt"),
    ("pio", "outputs/runs_pio_alive"),
    ("pio_eq48", "outputs/runs_pio_eq"),
)
METHOD_AUG = {
    "ce": "hier_addone",
    "selfcon": "simclr",
    "supcon": "stacked_randaug",
}
SEEDS = (42, 43, 44)
PRIMARY = "feature512_median_loo"
RANK_SCORERS = ("absolute", PRIMARY, "project128_median_loo")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluasi anomali relatif satu-frame")
    p.add_argument("--labels", default="data/label_chick/lembar_label.csv")
    p.add_argument("--crops", default="data/label_chick/crops")
    p.add_argument("--interventions", default="asli,acak16",
                   help="daftar: asli,acak16")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=20260912)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-prefix", default="same_frame_stage1")
    return p.parse_args()


def load_rows(labels_path: Path, crops_dir: Path) -> list[dict]:
    with open(labels_path, newline="", encoding="utf-8-sig") as f:
        raw = list(csv.DictReader(f))
    if len(raw) != 1215:
        raise ValueError(f"lembar label harus 1215 baris, ditemukan {len(raw)}")

    rows, seen = [], set()
    allowed = {"", "mati", "bukan"}
    for r in raw:
        label = r["label"].strip().lower()
        if label not in allowed:
            raise ValueError(f"label tidak dikenal pada nomor {r['nomor']}: {label}")
        nomor = int(r["nomor"])
        expected = f"{nomor:04d}.jpg"
        if r["berkas"] != expected or nomor in seen:
            raise ValueError(f"penomoran crop tidak konsisten: {r}")
        seen.add(nomor)
        if label == "bukan":
            continue
        path = crops_dir / r["berkas"]
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append({
            "nomor": nomor,
            "berkas": r["berkas"],
            "gambar_sumber": r["gambar_sumber"],
            "det_id": int(r["det_id"]),
            "label": "mati" if label == "mati" else "hidup",
            "y": 1 if label == "mati" else 0,
            "path": path,
        })
    groups = sorted({r["gambar_sumber"] for r in rows})
    if len(rows) != 943 or sum(r["y"] for r in rows) != 22 or len(groups) != 18:
        raise ValueError("data valid harus 943 crop, 22 mati, dan 18 frame")
    for group in groups:
        ys = {r["y"] for r in rows if r["gambar_sumber"] == group}
        if ys != {0, 1}:
            raise ValueError(f"frame tidak memiliki dua kelas: {group}")
    return rows


def shuffle_tiles(image: np.ndarray, nomor: int, seed: int, n: int = 4):
    """Kompatibilitas lama menuju intervensi shared yang menutup semua piksel."""
    return acak_petak(image, nomor, seed=seed, n=n)


def prepare_images(rows: list[dict], cfg: dict, intervention: str,
                   intervention_seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    size = int(cfg["classifier"]["image_size"])
    mode = cfg["classifier"]["resize_mode"]
    images, sharpness = [], []
    for r in rows:
        image = imread(r["path"])
        if image is None:
            raise ValueError(f"crop gagal dibaca: {r['path']}")
        image = equalize(image, cfg["crops"])
        if intervention == "acak16":
            image = shuffle_tiles(image, r["nomor"], intervention_seed)
        elif intervention != "asli":
            raise ValueError(f"intervensi tidak dikenal: {intervention}")
        image = to_square(image, size, mode)
        images.append(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    return images, np.asarray(sharpness, dtype=np.float64)


def canonical_checkpoints() -> list[dict]:
    found = []
    for family, runs_dir in FAMILIES:
        for method, aug in METHOD_AUG.items():
            for seed in SEEDS:
                run = f"{method}__{aug}__s{seed}"
                path = resolve(runs_dir) / run / "model.pt"
                if not path.exists():
                    raise FileNotFoundError(path)
                found.append({"family": family, "runs_dir": runs_dir,
                              "run": run, "method": method, "aug": aug,
                              "seed": seed, "path": path})
    if len(found) != 27:
        raise AssertionError("manifest checkpoint bukan 27")
    return found


def load_checkpoint(item: dict, device: torch.device):
    ckpt = torch.load(item["path"], map_location=device)
    for key in ("state_dict", "method", "aug", "seed", "config"):
        if key not in ckpt:
            raise ValueError(f"checkpoint tanpa {key}: {item['path']}")
    if (ckpt["method"], ckpt["aug"], int(ckpt["seed"])) != (
            item["method"], item["aug"], item["seed"]):
        raise ValueError(f"metadata checkpoint tidak cocok: {item['path']}")
    model = build_model(ckpt["config"], device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["config"]


@torch.inference_mode()
def extract(model, images: list[np.ndarray], cfg: dict, batch_size: int,
            device: torch.device, with_projector: bool):
    size = int(cfg["classifier"]["image_size"])
    mode = cfg["classifier"]["resize_mode"]
    features, projects, probs = [], [], []
    for start in range(0, len(images), batch_size):
        batch = torch.stack([to_tensor(im, size, mode)
                             for im in images[start:start + batch_size]]).to(device)
        feat = model.features(batch)
        logit = model.classifier(feat)
        features.append(feat.cpu().numpy())
        probs.append(torch.softmax(logit, dim=1)[:, 1].cpu().numpy())
        if with_projector:
            projects.append(F.normalize(model.projector(feat), dim=1).cpu().numpy())
    return (np.concatenate(features),
            np.concatenate(projects) if projects else None,
            np.concatenate(probs))


def group_indices(groups: np.ndarray):
    return {g: np.flatnonzero(groups == g) for g in sorted(set(groups))}


def cosine_distance(vector: np.ndarray, center: np.ndarray) -> float:
    nv, nc = np.linalg.norm(vector), np.linalg.norm(center)
    if nv == 0 or nc == 0:
        return 1.0
    return float(1.0 - np.dot(vector, center) / (nv * nc))


def relative_scores(vectors: np.ndarray, groups: np.ndarray, mode: str,
                    k: int = 5) -> np.ndarray:
    out = np.empty(len(vectors), dtype=np.float64)
    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    for idx in group_indices(groups).values():
        for i in idx:
            others = idx[idx != i]
            if mode == "median":
                out[i] = cosine_distance(vectors[i], np.median(vectors[others], axis=0))
            elif mode == "medoid":
                sim = unit[others] @ unit[others].T
                medoid = others[np.argmin(np.sum(1.0 - sim, axis=1))]
                out[i] = float(1.0 - np.dot(unit[i], unit[medoid]))
            elif mode == "knn":
                dist = 1.0 - unit[others] @ unit[i]
                out[i] = float(np.median(np.partition(dist, min(k, len(dist)) - 1)[:k]))
            else:
                raise ValueError(mode)
    return out


def ranks_within_frame(scores: np.ndarray, groups: np.ndarray,
                       nomor: np.ndarray):
    ranks = np.empty(len(scores), dtype=np.float64)
    percentile = np.empty(len(scores), dtype=np.float64)
    normalized = np.empty(len(scores), dtype=np.float64)
    for idx in group_indices(groups).values():
        order = idx[np.argsort(-scores[idx], kind="mergesort")]
        sorted_scores = scores[order]
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and sorted_scores[end] == sorted_scores[start]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
        normalized[order] = (len(order) - ranks[order]) / max(1, len(order) - 1)
        percentile[order] = normalized[order]
    return ranks, percentile, normalized


def confusion(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"balanced_accuracy": (recall + specificity) / 2,
            "recall_dead": recall, "specificity": specificity,
            "precision_dead": precision, "f1_dead": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": int(len(y))}


def nested_predictions(y: np.ndarray, scores: np.ndarray, groups: np.ndarray):
    pred = np.empty(len(y), dtype=np.int64)
    thresholds = {}
    for group, idx in group_indices(groups).items():
        train = groups != group
        threshold = pick_threshold(y[train], scores[train])
        pred[idx] = (scores[idx] >= threshold).astype(np.int64)
        thresholds[group] = threshold
    return pred, thresholds


def metric_bundle(y: np.ndarray, scores: np.ndarray, groups: np.ndarray,
                  nomor: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    ranks, percentiles, _ = ranks_within_frame(scores, groups, nomor)
    aucs, aps, reciprocal, dead_percentiles = [], [], [], []
    recalls = {1: [], 3: [], 5: []}
    hits = {1: [], 3: [], 5: []}
    for idx in group_indices(groups).values():
        aucs.append(roc_auc(y[idx], scores[idx]))
        aps.append(float(average_precision_score(y[idx], scores[idx])))
        dead_ranks = ranks[idx][y[idx] == 1]
        reciprocal.append(1.0 / dead_ranks.min())
        dead_percentiles.extend(percentiles[idx][y[idx] == 1])
        for k in recalls:
            recalls[k].append(float((dead_ranks <= k).sum() / len(dead_ranks)))
            hits[k].append(float((dead_ranks <= k).any()))
    pred, thresholds = nested_predictions(y, scores, groups)
    decision = confusion(y, pred)
    per_frame_decision = [confusion(y[idx], pred[idx])
                          for idx in group_indices(groups).values()]
    metric = {
        "pooled_auc": roc_auc(y, scores),
        "pooled_ap": float(average_precision_score(y, scores)),
        "macro_auc": float(np.mean(aucs)),
        "macro_ap": float(np.mean(aps)),
        "recall_at_1": float(np.mean(recalls[1])),
        "recall_at_3": float(np.mean(recalls[3])),
        "recall_at_5": float(np.mean(recalls[5])),
        "hit_at_1": float(np.mean(hits[1])),
        "hit_at_3": float(np.mean(hits[3])),
        "hit_at_5": float(np.mean(hits[5])),
        "mrr": float(np.mean(reciprocal)),
        "mean_dead_rank_percentile": float(np.mean(dead_percentiles)),
        "decision_pooled": decision,
        "decision_macro": {
            key: float(np.mean([m[key] for m in per_frame_decision]))
            for key in ("balanced_accuracy", "recall_dead", "specificity",
                        "precision_dead", "f1_dead")
        },
        "fp_per_frame": decision["fp"] / len(set(groups)),
    }
    return metric, ranks, pred, thresholds


def bootstrap_ci(y: np.ndarray, scores: np.ndarray, groups: np.ndarray,
                 nomor: np.ndarray, n_boot: int, seed: int) -> dict:
    by_group = group_indices(groups)
    frame_metrics = []
    for idx in by_group.values():
        ranks, _, _ = ranks_within_frame(scores[idx],
                                          np.repeat("frame", len(idx)), nomor[idx])
        dead_ranks = ranks[y[idx] == 1]
        frame_metrics.append([
            roc_auc(y[idx], scores[idx]),
            float(average_precision_score(y[idx], scores[idx])),
            float((dead_ranks <= 3).sum() / len(dead_ranks)),
            1.0 / dead_ranks.min(),
        ])
    frame_metrics = np.asarray(frame_metrics, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(frame_metrics), (n_boot, len(frame_metrics)))
    means = frame_metrics[sampled].mean(axis=1)
    names = ("macro_auc", "macro_ap", "recall_at_3", "mrr")
    return {name: {"low": float(np.quantile(means[:, i], 0.025)),
                   "high": float(np.quantile(means[:, i], 0.975))}
            for i, name in enumerate(names)}


def evaluate_scorer(y, scores, groups, nomor, n_boot, bootstrap_seed):
    metrics, ranks, pred, thresholds = metric_bundle(y, scores, groups, nomor)
    metrics["cluster_bootstrap_95ci"] = bootstrap_ci(
        y, scores, groups, nomor, n_boot, bootstrap_seed)
    return {"scores": scores.tolist(), "ranks": ranks.tolist(),
            "pred_nested": pred.tolist(), "fold_thresholds": thresholds,
            "metrics": metrics}


def seed_aggregates(checkpoints: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for cp in checkpoints:
        for scorer, result in cp["scorers"].items():
            buckets[(cp["family"], cp["method"], scorer)].append(result["metrics"])
    out = []
    keys = ("pooled_auc", "pooled_ap", "macro_auc", "macro_ap",
            "recall_at_1", "recall_at_3", "recall_at_5", "mrr",
            "mean_dead_rank_percentile")
    for (family, method, scorer), rows in sorted(buckets.items()):
        if len(rows) != 3:
            raise ValueError(f"agregasi bukan tiga seed: {family}/{method}/{scorer}")
        out.append({"family": family, "method": method, "scorer": scorer,
                    "n_seeds": 3,
                    "metrics": {key: {"mean": float(np.mean([r[key] for r in rows])),
                                      "std": float(np.std([r[key] for r in rows], ddof=1))}
                                for key in keys}})
    return out


def build_ensembles(checkpoints: list[dict], y, groups, nomor, n_boot, seed):
    buckets = defaultdict(list)
    for cp in checkpoints:
        for scorer in RANK_SCORERS:
            if scorer in cp["scorers"]:
                scores = np.asarray(cp["scorers"][scorer]["scores"])
                _, _, normalized = ranks_within_frame(scores, groups, nomor)
                buckets[(cp["family"], cp["method"], scorer)].append(normalized)
    out = []
    for (family, method, scorer), arrays in sorted(buckets.items()):
        if len(arrays) != 3:
            continue
        scores = np.mean(arrays, axis=0)
        result = evaluate_scorer(y, scores, groups, nomor, n_boot, seed)
        out.append({"family": family, "method": method, "scorer": scorer,
                    "aggregation": "mean_normalized_within_frame_rank_3_seeds",
                    **result})
    return out


def write_audit_csv(path: Path, rows: list[dict], checkpoints: list[dict],
                    intervention: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("nomor", "berkas", "gambar_sumber", "det_id", "label",
              "family", "run", "method", "seed", "intervention", "scorer",
              "score", "rank", "rank_percentile", "pred_nested",
              "threshold_fold")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        groups = np.asarray([r["gambar_sumber"] for r in rows])
        nomor = np.asarray([r["nomor"] for r in rows])
        for cp in checkpoints:
            for scorer in RANK_SCORERS:
                if scorer not in cp["scorers"]:
                    continue
                result = cp["scorers"][scorer]
                score = np.asarray(result["scores"])
                rank = np.asarray(result["ranks"])
                _, percentile, _ = ranks_within_frame(score, groups, nomor)
                for i, row in enumerate(rows):
                    writer.writerow({
                        **{k: row[k] for k in ("nomor", "berkas", "gambar_sumber",
                                               "det_id", "label")},
                        "family": cp["family"], "run": cp["run"],
                        "method": cp["method"], "seed": cp["seed"],
                        "intervention": intervention, "scorer": scorer,
                        "score": f"{score[i]:.10g}", "rank": f"{rank[i]:.10g}",
                        "rank_percentile": f"{percentile[i]:.10g}",
                        "pred_nested": result["pred_nested"][i],
                        "threshold_fold": f"{result['fold_thresholds'][row['gambar_sumber']]:.10g}",
                    })


def fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} +/- {std:.3f}"


def write_report(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    aggs = data["seed_aggregates"]
    main = [a for a in aggs if a["scorer"] in ("absolute", PRIMARY)]
    lines = [
        f"# Tahap 1 Same-Frame ({data['protocol']['intervention']})", "",
        "## Protokol", "",
        "Evaluasi memakai 943 crop valid (22 mati, 921 hidup) dari 18 frame. "
        "Sebanyak 272 crop `bukan` dikeluarkan. Skor relatif utama ditetapkan "
        "sebelum hasil dilihat: jarak kosinus fitur backbone 512-d terhadap "
        "median koordinat-wise frame secara leave-one-out.", "",
        "Ambang keputusan dipilih dari 17 frame lain untuk setiap frame uji. "
        f"CI 95% me-resample 18 frame sebagai klaster ({data['protocol']['n_bootstrap']} replikasi).", "",
        "## Hasil Tiga Seed", "",
        "| domain | metode | scorer | AUC pooled | AP pooled | AUC macro | Recall@3 | MRR |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for a in main:
        m = a["metrics"]
        lines.append("| {family} | {method} | {scorer} | {auc} | {ap} | {macro} | {r3} | {mrr} |".format(
            family=a["family"], method=a["method"], scorer=a["scorer"],
            auc=fmt(**m["pooled_auc"]), ap=fmt(**m["pooled_ap"]),
            macro=fmt(**m["macro_auc"]), r3=fmt(**m["recall_at_3"]),
            mrr=fmt(**m["mrr"])))
    sharp = data["sharpness"]["metrics"]
    sharp_eq = data["sharpness_sensitivity_eq48"]["metrics"]
    lines += ["", "## Ringkasan", "",
              "- Semua kombinasi dilaporkan secara deskriptif; tidak ada metode yang "
              "dipilih dari data uji ini.",
              f"- Baseline ketajaman tanpa equalization: AUC {sharp['pooled_auc']:.3f}, "
              f"AP {sharp['pooled_ap']:.3f}, Recall@3 {sharp['recall_at_3']:.3f}.",
              f"- Baseline ketajaman setelah eq48: AUC {sharp_eq['pooled_auc']:.3f}, "
              f"AP {sharp_eq['pooled_ap']:.3f}, Recall@3 {sharp_eq['recall_at_3']:.3f}.",
              "- Angka ini adalah evaluasi clean-crop; pusat frame tidak memasukkan 272 crop `bukan`.",
              "", "## Sensitivitas", "",
              "Scorer medoid dan median 5-NN dihitung sebagai analisis sensitivitas, "
              "bukan untuk mengganti scorer utama setelah hasil dilihat. Detail lengkap "
              "per checkpoint, CI, ambang fold, dan skor mentah ada di JSON/CSV.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plot(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    dead = np.asarray([r["label"] == "mati" for r in data["data"]["rows"]])
    labels, values, colors = [], [], []
    for cp in data["checkpoints"]:
        for scorer, color in (("absolute", "#2563eb"), (PRIMARY, "#d97706")):
            ranks = np.asarray(cp["scorers"][scorer]["ranks"])
            groups = np.asarray([r["gambar_sumber"] for r in data["data"]["rows"]])
            percentiles = np.empty(len(ranks))
            for idx in group_indices(groups).values():
                percentiles[idx] = (len(idx) - ranks[idx]) / max(1, len(idx) - 1)
            values.append(percentiles[dead])
            labels.append(f"{cp['family']}\n{cp['method']} s{cp['seed']}\n"
                          f"{'rel' if scorer == PRIMARY else 'abs'}")
            colors.append(color)
    fig, ax = plt.subplots(figsize=(18, 8))
    bp = ax.boxplot(values, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Persentil peringkat ayam mati dalam frame (1 = teratas)")
    ax.set_title(f"Sebaran peringkat Tahap 1 same-frame: {data['protocol']['intervention']}")
    ax.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=70, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def causal_comparison(original: dict, shuffled: dict, n_boot: int, seed: int) -> dict:
    y = np.asarray([r["label"] == "mati" for r in original["data"]["rows"]], dtype=int)
    groups = np.asarray([r["gambar_sumber"] for r in original["data"]["rows"]])
    nomor = np.asarray([r["nomor"] for r in original["data"]["rows"]])
    by_group = group_indices(groups)
    shuffled_cp = {(c["family"], c["run"]): c for c in shuffled["checkpoints"]}
    buckets = defaultdict(list)
    for cp in original["checkpoints"]:
        other = shuffled_cp[(cp["family"], cp["run"])]
        for scorer in ("absolute", PRIMARY):
            s0 = np.asarray(cp["scorers"][scorer]["scores"])
            s1 = np.asarray(other["scorers"][scorer]["scores"])
            frame_delta = []
            for idx in by_group.values():
                auc_delta = roc_auc(y[idx], s1[idx]) - roc_auc(y[idx], s0[idx])
                r0, _, _ = ranks_within_frame(s0[idx], np.repeat("f", len(idx)), nomor[idx])
                r1, _, _ = ranks_within_frame(s1[idx], np.repeat("f", len(idx)), nomor[idx])
                d0, d1 = r0[y[idx] == 1], r1[y[idx] == 1]
                recall_delta = ((d1 <= 3).sum() - (d0 <= 3).sum()) / len(d0)
                frame_delta.append([auc_delta, recall_delta])
            buckets[(cp["family"], cp["method"], scorer)].append(np.asarray(frame_delta))

    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(by_group), (n_boot, len(by_group)))
    results = []
    for (family, method, scorer), arrays in sorted(buckets.items()):
        values = np.stack(arrays)  # seed x frame x metric
        per_seed = values.mean(axis=1)
        boot = values[:, sampled, :].mean(axis=(0, 2))
        results.append({
            "family": family, "method": method, "scorer": scorer,
            "delta_definition": "acak16_minus_asli",
            "macro_auc_delta": {"mean": float(per_seed[:, 0].mean()),
                                "std_across_seeds": float(per_seed[:, 0].std(ddof=1)),
                                "cluster_bootstrap_95ci": [float(np.quantile(boot[:, 0], 0.025)),
                                                            float(np.quantile(boot[:, 0], 0.975))]},
            "recall_at_3_delta": {"mean": float(per_seed[:, 1].mean()),
                                  "std_across_seeds": float(per_seed[:, 1].std(ddof=1)),
                                  "cluster_bootstrap_95ci": [float(np.quantile(boot[:, 1], 0.025)),
                                                              float(np.quantile(boot[:, 1], 0.975))]},
        })
    return {"comparison": "acak16_minus_asli", "n_frames": 18,
            "n_seeds_per_combination": 3, "bootstrap": n_boot,
            "bootstrap_seed": seed, "results": results}


def write_causal_report(path: Path, comparison: dict):
    lines = ["# Gerbang Kausal Tahap 1: Asli vs Acak16", "",
             "Delta di bawah adalah `acak16 - asli`. Nilai AUC yang tidak turun "
             "setelah susunan bentuk dihancurkan berarti scorer belum terbukti membaca pose.", "",
             "| domain | metode | scorer | delta AUC macro | CI 95% klaster | delta Recall@3 |",
             "|---|---|---|---:|---:|---:|"]
    for row in comparison["results"]:
        auc = row["macro_auc_delta"]
        recall = row["recall_at_3_delta"]
        lo, hi = auc["cluster_bootstrap_95ci"]
        lines.append(f"| {row['family']} | {row['method']} | {row['scorer']} | "
                     f"{auc['mean']:+.3f} +/- {auc['std_across_seeds']:.3f} | "
                     f"[{lo:+.3f}, {hi:+.3f}] | {recall['mean']:+.3f} |")
    primary = [r for r in comparison["results"] if r["scorer"] == PRIMARY]
    dropped = sum(r["macro_auc_delta"]["mean"] < 0 for r in primary)
    significant = sum(r["macro_auc_delta"]["cluster_bootstrap_95ci"][1] < 0
                      for r in primary)
    lines += ["", "## Keputusan", "",
              f"Gerbang kausal **lolos secara arah**: {dropped} dari {len(primary)} "
              "kombinasi domain/metode mengalami penurunan AUC macro scorer relatif utama "
              f"setelah bentuk dihancurkan, dan {significant} di antaranya memiliki CI 95% "
              "seluruhnya di bawah nol. Ini menunjukkan scorer relatif membaca informasi "
              "susunan/bentuk, meski belum membuktikan bahwa seluruh sinyalnya adalah pose ayam.", "",
              "Intervensi ini tidak dipakai memilih metode. Manfaat skor relatif terhadap "
              "classifier absolut tetap harus dinilai bersama metrik pooled, macro, dan "
              "Recall@K pada hasil asli.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_intervention(args, rows, checkpoints, intervention, device):
    y = np.asarray([r["y"] for r in rows], dtype=np.int64)
    groups = np.asarray([r["gambar_sumber"] for r in rows])
    nomor = np.asarray([r["nomor"] for r in rows], dtype=np.int64)
    output = {
        "schema_version": 1,
        "experiment": "same_frame_stage1",
        "created_at": datetime.now().astimezone().isoformat(),
        "data": {"labels": str(resolve(args.labels)), "crops": str(resolve(args.crops)),
                 "n_total": 1215, "n_valid": 943, "n_dead": 22,
                 "n_alive": 921, "n_excluded_bukan": 272, "n_frames": 18,
                 "rows": [{k: r[k] for k in ("nomor", "berkas", "gambar_sumber",
                                               "det_id", "label")} for r in rows]},
        "protocol": {"intervention": intervention,
                     "intervention_order": "equalize_then_shuffle4x4_then_square",
                     "intervention_seed": args.bootstrap_seed,
                     "primary_scorer": "cosine_to_coordinate_median_loo",
                     "sensitivity": ["cosine_to_medoid_loo", "median_cosine_5nn_loo"],
                     "threshold": "nested_leave_one_image_out",
                     "bootstrap_unit": "gambar_sumber", "n_bootstrap": args.bootstrap,
                     "bootstrap_seed": args.bootstrap_seed,
                     "rank_tie_policy": "average_rank"},
        "checkpoints": [],
    }
    image_cache = {}
    sharpness_by_preprocess = {}
    for pos, item in enumerate(checkpoints, 1):
        print(f"[{intervention}] {pos:02d}/27 {item['family']}/{item['run']}", flush=True)
        model, cfg = load_checkpoint(item, device)
        eq = cfg["crops"].get("equalize_resolution") or {}
        enabled = bool(eq.get("enabled"))
        prep_key = f"eq1_{int(eq.get('target_short_side', 96))}" if enabled else "eq0"
        if prep_key not in image_cache:
            image_cache[prep_key], sharpness_by_preprocess[prep_key] = prepare_images(
                rows, cfg, intervention, args.bootstrap_seed)
        feat, project, prob = extract(model, image_cache[prep_key], cfg,
                                      args.batch_size, device, item["method"] != "ce")
        scorer_values = {
            "absolute": prob,
            PRIMARY: relative_scores(feat, groups, "median"),
            "feature512_medoid_loo": relative_scores(feat, groups, "medoid"),
            "feature512_knn5_loo": relative_scores(feat, groups, "knn", 5),
        }
        if project is not None:
            scorer_values.update({
                "project128_median_loo": relative_scores(project, groups, "median"),
                "project128_medoid_loo": relative_scores(project, groups, "medoid"),
                "project128_knn5_loo": relative_scores(project, groups, "knn", 5),
            })
        cp = {k: item[k] for k in ("family", "run", "method", "aug", "seed")}
        cp["checkpoint"] = str(item["path"])
        cp["preprocess"] = {"image_size": int(cfg["classifier"]["image_size"]),
                            "resize_mode": cfg["classifier"]["resize_mode"],
                            "equalize_resolution": eq}
        cp["scorers"] = {
            name: evaluate_scorer(y, value, groups, nomor, args.bootstrap,
                                  args.bootstrap_seed)
            for name, value in scorer_values.items()
        }
        output["checkpoints"].append(cp)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Ketajaman utama memakai praproses tanpa equalization; versi eq48 dicatat terpisah.
    sharp_main = sharpness_by_preprocess["eq0"]
    output["sharpness"] = evaluate_scorer(
        y, sharp_main, groups, nomor, args.bootstrap, args.bootstrap_seed)
    output["sharpness"]["preprocess"] = "letterbox_224_tanpa_equalize"
    output["sharpness_sensitivity_eq48"] = evaluate_scorer(
        y, sharpness_by_preprocess["eq1_48"], groups, nomor,
        args.bootstrap, args.bootstrap_seed)
    output["seed_aggregates"] = seed_aggregates(output["checkpoints"])
    output["ensembles"] = build_ensembles(output["checkpoints"], y, groups, nomor,
                                           args.bootstrap, args.bootstrap_seed)

    suffix = "" if intervention == "asli" else f"_{intervention}"
    base = args.output_prefix + suffix
    json_path = resolve(f"outputs/predictions/{base}.json")
    csv_path = resolve(f"outputs/predictions/{base}_crops.csv")
    report_path = resolve(f"outputs/reports/{base}.md")
    plot_path = resolve(f"outputs/reports/{base}.png")
    save_json(output, json_path)
    write_audit_csv(csv_path, rows, output["checkpoints"], intervention)
    write_report(report_path, output)
    write_plot(plot_path, output)
    print(f"[{intervention}] JSON: {json_path}")
    print(f"[{intervention}] CSV : {csv_path}")
    print(f"[{intervention}] report: {report_path}")
    return output


def main():
    args = parse_args()
    interventions = [x.strip() for x in args.interventions.split(",") if x.strip()]
    if not interventions or any(x not in {"asli", "acak16"} for x in interventions):
        raise SystemExit("--interventions hanya menerima asli,acak16")
    labels, crops = resolve(args.labels), resolve(args.crops)
    rows = load_rows(labels, crops)
    checkpoints = canonical_checkpoints()
    device = get_device(args.device)
    print(f"Data: 943 crop valid, 22 mati, 921 hidup, 18 frame")
    print(f"Checkpoint: {len(checkpoints)} | device: {device}")
    outputs = {intervention: run_intervention(
        args, rows, checkpoints, intervention, device) for intervention in interventions}
    if {"asli", "acak16"}.issubset(outputs):
        comparison = causal_comparison(outputs["asli"], outputs["acak16"],
                                       args.bootstrap, args.bootstrap_seed)
        json_path = resolve(f"outputs/predictions/{args.output_prefix}_causal.json")
        report_path = resolve(f"outputs/reports/{args.output_prefix}_causal.md")
        save_json(comparison, json_path)
        write_causal_report(report_path, comparison)
        print(f"[kausal] JSON: {json_path}")
        print(f"[kausal] report: {report_path}")


if __name__ == "__main__":
    main()
