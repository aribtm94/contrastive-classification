"""
Ambil angka NYATA dari data + checkpoint untuk docs/hitung_manual_klasifikasi.md

Lanjutan bagian I-II (pra-pemrosesan + forward YOLO). Berkas ini mengurus
bagian III ke atas: crop, letterbox 224, normalisasi, forward ResNet-18,
projection head, ketiga loss, linear probe, ambang, dan metrik.

Semua keluaran -> results/hitung_manual_klasifikasi.json

Jalankan:
  "C:/Arib/MASSA AYAM/generalisasi-ayam-skripsi/.venv-yolo/Scripts/python.exe" \
      scripts/extract_manual_klasifikasi.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import load_config, crop_box, to_square                    # noqa: E402
from dataset import (IMAGENET_MEAN, IMAGENET_STD, to_tensor, augment,  # noqa: E402
                     policy_by_name, class_weights)
from models import build_model, nt_xent_loss, supcon_loss              # noqa: E402
from train import roc_auc, compute_metrics, pick_threshold             # noqa: E402

OUT = ROOT / "results" / "hitung_manual_klasifikasi.json"
V: dict = {}


def p(k, v):
    V[k] = v
    s = json.dumps(v, ensure_ascii=False)
    print(f"  {k} = {s[:110]}")
    return v


def arr(a, n=6):
    return np.round(np.asarray(a, dtype=np.float64), n).tolist()


def vlap(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def imread(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


# ------------------------------------------------------------------ #
print("\n[1] Manifest & crop contoh")
cfg = load_config(ROOT / "configs" / "config.yaml")
rows = []
with open(ROOT / "data" / "crops" / "manifest.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        r["label"] = int(r["label"])
        rows.append(r)

n_split = {}
for sp in ("train", "val", "test"):
    a = sum(1 for r in rows if r["split"] == sp and r["label"] == 0)
    d = sum(1 for r in rows if r["split"] == sp and r["label"] == 1)
    n_split[sp] = {"hidup": a, "mati": d, "total": a + d}
p("n_split", n_split)
p("n_total", len(rows))

KOLOM = ("path", "label", "split", "base_image", "src_file", "bbox",
         "conf", "origin", "domain")
contoh_mati = next(r for r in rows if r["split"] == "test" and r["label"] == 1)
contoh_hidup = next(r for r in rows if r["split"] == "test" and r["label"] == 0)
p("contoh_mati", {k: contoh_mati[k] for k in KOLOM})
p("contoh_hidup", {k: contoh_hidup[k] for k in KOLOM})

# ------------------------------------------------------------------ #
print("\n[2] Bagian III - crop + padding bbox")
pad = float(cfg["crops"]["bbox_padding"])
p("pad_ratio", pad)
dead_root = Path(cfg["crops"]["dead_root"])
src_img = None
for sub in cfg["crops"]["splits"]:
    cand = dead_root / sub / contoh_mati["src_file"]
    if cand.exists():
        src_img = cand
        break
p("src_img_mati", str(src_img) if src_img else None)

sq = None
if src_img is not None:
    im0 = imread(src_img)
    H, W = im0.shape[:2]
    p("src_img_HW", [H, W])
    bbox = json.loads(contoh_mati["bbox"])
    p("bbox_asli", bbox)
    x1, y1, x2, y2 = bbox
    p("bbox_wh", [round(x2 - x1, 4), round(y2 - y1, 4)])
    dw, dh = (x2 - x1) * pad, (y2 - y1) * pad
    p("pad_dw_dh", [round(dw, 4), round(dh, 4)])
    bp = [max(0, x1 - dw), max(0, y1 - dh), min(W, x2 + dw), min(H, y2 + dh)]
    p("bbox_padded_float", arr(bp, 4))
    bpi = [int(round(v)) for v in bp]
    p("bbox_padded_int", bpi)
    p("bbox_padded_wh", [bpi[2] - bpi[0], bpi[3] - bpi[1]])
    crop = crop_box(im0, bbox, pad)
    p("crop_shape", list(crop.shape))

    # ---- Bagian IV: letterbox 224 ----
    h, w = crop.shape[:2]
    scale = min(224 / h, 224 / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    top, left = (224 - nh) // 2, (224 - nw) // 2
    p("lb_scale", round(float(scale), 6))
    p("lb_nh_nw", [nh, nw])
    p("lb_top_left", [top, left])
    p("lb_bottom_right_pad", [224 - nh - top, 224 - nw - left])
    p("lb_interp", "INTER_AREA" if scale < 1 else "INTER_LINEAR")
    p("lb_pad_fraction", round(1.0 - (nh * nw) / (224 * 224), 6))
    p("lb_pad_color", [114, 114, 114])
    sq = to_square(crop, 224, "letterbox")
    p("lb_out_shape", list(sq.shape))
    p("lb_cek_piksel_pojok", sq[0, 0].tolist())


# ------------------------------------------------------------------ #
print("\n[3] Bagian V - normalisasi ImageNet, satu piksel nyata")
cm = imread(ROOT / "data" / "crops" / contoh_mati["path"])
ch = imread(ROOT / "data" / "crops" / contoh_hidup["path"])
p("crop_disk_shape", list(cm.shape))
p("imagenet_mean", IMAGENET_MEAN.tolist())
p("imagenet_std", IMAGENET_STD.tolist())

yx = (112, 112)
bgr = cm[yx[0], yx[1]].tolist()
rgb = [bgr[2], bgr[1], bgr[0]]
p("piksel_yx", list(yx))
p("piksel_bgr_uint8", bgr)
p("piksel_rgb_uint8", rgb)
x01 = [v / 255.0 for v in rgb]
p("piksel_01", arr(x01))
p("piksel_imagenet", arr([(x01[c] - float(IMAGENET_MEAN[c]))
                          / float(IMAGENET_STD[c]) for c in range(3)]))

t_mati = to_tensor(cm, 224, "letterbox")
t_hidup = to_tensor(ch, 224, "letterbox")
p("tensor_shape", list(t_mati.shape))
p("tensor_cek_piksel", arr(t_mati[:, yx[0], yx[1]].tolist()))
p("tensor_min_max", [round(float(t_mati.min()), 6), round(float(t_mati.max()), 6)])
# nilai piksel bantalan abu 114 setelah normalisasi
g = 114 / 255.0
p("piksel_bantalan114_imagenet",
  arr([(g - float(IMAGENET_MEAN[c])) / float(IMAGENET_STD[c]) for c in range(3)]))

p("vlap_contoh_mati", round(vlap(cm), 4))
p("vlap_contoh_hidup", round(vlap(ch), 4))

# ------------------------------------------------------------------ #
print("\n[4] Bagian VI - augmentasi dua view")
pol = policy_by_name(cfg, "simclr")
p("policy_simclr", [dict(o) for o in pol["ops"]])
p("policy_by_method", cfg["augmentation"]["by_method"])

# TwoViewDataset: j1 = i + epoch*7919 ; j2 = j1 + 104729 ; seed(j) = (S*1000003 + j)
S, i_item, epoch = 42, 0, 0
j1 = i_item + epoch * 7919
j2 = j1 + 104_729
s1 = (S * 1_000_003 + j1) & 0xFFFFFFFF
s2 = (S * 1_000_003 + j2) & 0xFFFFFFFF
p("seed_dasar", S)
p("seed_j1_j2", [j1, j2])
p("seed_view1", s1)
p("seed_view2", s2)

v1, par1 = augment(cm, random.Random(s1), pol, seed=s1, return_params=True)
v2, par2 = augment(cm, random.Random(s2), pol, seed=s2, return_params=True)
p("aug_view1_ops", par1["ops"])
p("aug_view2_ops", par2["ops"])
p("vlap_view1", round(vlap(v1), 4))
p("vlap_view2", round(vlap(v2), 4))
k = int(0.10 * 224)
p("blur_kernel_frac_224", k + (1 - k % 2))


# ------------------------------------------------------------------ #
print("\n[5] Bagian VII - forward encoder (checkpoint nyata)")
device = torch.device("cpu")
RUN = ROOT / "outputs" / "runs" / "selfcon"      # diagonal: selfcon + simclr
ck = torch.load(RUN / "model.pt", map_location="cpu")
model = build_model(ck["config"], device)
model.load_state_dict(ck["state_dict"])
model.eval()
p("checkpoint", str(RUN.relative_to(ROOT)).replace("\\", "/") + "/model.pt")
p("checkpoint_meta", [ck["method"], ck["aug"], int(ck["seed"])])
p("arsitektur", {"backbone": cfg["classifier"]["backbone"],
                 "feat_dim": model.feat_dim,
                 "proj_hidden": int(cfg["classifier"]["proj_hidden"]),
                 "proj_dim": int(cfg["classifier"]["proj_dim"])})
n_par = sum(x.numel() for x in model.parameters())
p("jumlah_parameter", {"total": n_par,
                       "backbone": sum(x.numel() for x in model.backbone.parameters()),
                       "projector": sum(x.numel() for x in model.projector.parameters()),
                       "classifier": sum(x.numel() for x in model.classifier.parameters())})

with torch.no_grad():
    x = torch.stack([t_mati, t_hidup])
    feat = model.features(x)
    z = model.project(x)
    logit = model.logits(x)
    prob = torch.softmax(logit, dim=1)

p("feat_shape", list(feat.shape))
p("feat_mati_5", arr(feat[0, :5].tolist()))
p("feat_hidup_5", arr(feat[1, :5].tolist()))
p("feat_mati_norm", round(float(feat[0].norm()), 6))
p("feat_nol_persen", round(float((feat[0] == 0).float().mean()), 4))

# --- satu posisi conv1 dihitung tangan ---
conv1, bn1 = model.backbone.conv1, model.backbone.bn1
p("conv1_bentuk_bobot", list(conv1.weight.shape))
p("conv1_stride_padding", [list(conv1.stride), list(conv1.padding)])
oy, ox = 56, 56
sy, sx = oy * 2 - 3, ox * 2 - 3
patch = torch.zeros(3, 7, 7)
for a in range(7):
    for b in range(7):
        yy, xx = sy + a, sx + b
        if 0 <= yy < 224 and 0 <= xx < 224:
            patch[:, a, b] = t_mati[:, yy, xx]
k0 = conv1.weight[0]
z0 = float((patch * k0).sum())
p("conv1_posisi_keluaran", [oy, ox])
p("conv1_jendela_asal_yx", [sy, sx])
p("conv1_patch_R_baris0", arr(patch[0, 0].tolist(), 4))
p("conv1_kernel0_R_baris0", arr(k0[0, 0].tolist(), 4))
p("conv1_sum_per_kanal", arr([float((patch[c] * k0[c]).sum()) for c in range(3)]))
p("conv1_z_manual", round(z0, 6))
with torch.no_grad():
    p("conv1_z_pytorch", round(float(conv1(t_mati.unsqueeze(0))[0, 0, oy, ox]), 6))

mu, var = float(bn1.running_mean[0]), float(bn1.running_var[0])
gam, bet, eps = float(bn1.weight[0]), float(bn1.bias[0]), float(bn1.eps)
zh = (z0 - mu) / (var + eps) ** 0.5
yb = gam * zh + bet
p("bn1_param", {"mu": round(mu, 6), "var": round(var, 6),
                "gamma": round(gam, 6), "beta": round(bet, 6), "eps": eps})
p("bn1_zhat", round(zh, 6))
p("bn1_y", round(yb, 6))
p("relu_out", round(max(0.0, yb), 6))

# --- projection head, satu neuron ---
pj = model.projector
with torch.no_grad():
    h1 = pj[0](feat[0])
    h1n = pj[1](h1.unsqueeze(0))[0].clone()   # clone: ReLU di bawah inplace
    h1r = pj[2](h1n.clone())
    h2 = pj[3](h1r.unsqueeze(0))[0]
p("proj_W1_shape", list(pj[0].weight.shape))
p("proj_W2_shape", list(pj[3].weight.shape))
p("proj_h1_0", round(float(h1[0]), 6))
p("proj_h1_0_manual",
  round(float((pj[0].weight[0] * feat[0]).sum() + pj[0].bias[0]), 6))
bnp = pj[1]
p("proj_bn_param", {"mu": round(float(bnp.running_mean[0]), 6),
                    "var": round(float(bnp.running_var[0]), 6),
                    "gamma": round(float(bnp.weight[0]), 6),
                    "beta": round(float(bnp.bias[0]), 6),
                    "eps": float(bnp.eps)})
p("proj_h1_0_setelah_bn", round(float(h1n[0]), 6))
p("proj_h1_0_setelah_relu", round(float(h1r[0]), 6))
p("proj_h2_prenorm_5", arr(h2[:5].tolist()))
p("proj_h2_norm", round(float(h2.norm()), 6))
p("z_mati_5", arr(z[0, :5].tolist()))
p("z_hidup_5", arr(z[1, :5].tolist()))
p("z_mati_norm", round(float(z[0].norm()), 8))
p("z_hidup_norm", round(float(z[1].norm()), 8))
p("cos_mati_hidup", round(float((z[0] * z[1]).sum()), 6))


# ------------------------------------------------------------------ #
print("\n[6] Bagian VIII - ketiga loss pada batch kecil nyata")
tr = [r for r in rows if r["split"] == "train"]
sel = [r for r in tr if r["label"] == 1][:2] + [r for r in tr if r["label"] == 0][:2]
p("batch_contoh_path", [r["path"] for r in sel])
p("batch_contoh_label", [r["label"] for r in sel])

ims = [imread(ROOT / "data" / "crops" / r["path"]) for r in sel]
# Dua view dibuat SEPERTI SAAT LATIH: kebijakan simclr, benih turunan
# TwoViewDataset (j1 = i, j2 = i + 104729) supaya angkanya reproducible.
b1, b2, tag1, tag2 = [], [], [], []
for i_, im in enumerate(ims):
    sa = (S * 1_000_003 + i_) & 0xFFFFFFFF
    sb = (S * 1_000_003 + i_ + 104_729) & 0xFFFFFFFF
    a_, pa = augment(im, random.Random(sa), pol, seed=sa, return_params=True)
    c_, pc = augment(im, random.Random(sb), pol, seed=sb, return_params=True)
    b1.append(to_tensor(a_, 224, "letterbox"))
    b2.append(to_tensor(c_, 224, "letterbox"))
    tag1.append(pa["ops"]); tag2.append(pc["ops"])
xb1, xb2 = torch.stack(b1), torch.stack(b2)
p("batch_seed_view1", [(S * 1_000_003 + i_) & 0xFFFFFFFF for i_ in range(len(ims))])
p("batch_seed_view2", [(S * 1_000_003 + i_ + 104_729) & 0xFFFFFFFF
                       for i_ in range(len(ims))])
p("batch_ops_view1", tag1)
p("batch_ops_view2", tag2)
yb_lab = torch.tensor([r["label"] for r in sel])
with torch.no_grad():
    z1 = model.project(xb1)
    z2 = model.project(xb2)

T = float(cfg["classifier"]["temperature"])
N = int(z1.shape[0])
p("temperature", T)
p("N_batch_contoh", N)
p("batch_size_asli", int(cfg["classifier"]["batch_size"]))

zc = torch.cat([z1, z2], 0)
Smat = zc @ zc.t()
p("matriks_cos", arr(Smat.tolist(), 4))
p("matriks_sim_dibagi_T", arr((Smat / T).tolist(), 4))

with torch.no_grad():
    p("nt_xent_loss", round(float(nt_xent_loss(z1, z2, T)), 6))
    p("supcon_loss", round(float(supcon_loss(z1, z2, yb_lab, T)), 6))

# rincian baris 0 NT-Xent
s0 = (Smat[0] / T).clone()
s0[0] = float("-inf")
num = float(s0[N])
den = float(torch.logsumexp(s0, 0))
p("ntxent_b0_indeks_positif", N)
p("ntxent_b0_sim_positif", round(num, 6))
p("ntxent_b0_logsumexp", round(den, 6))
p("ntxent_b0_loss", round(den - num, 6))
p("ntxent_b0_negatif_count", 2 * N - 2)
p("ntxent_semua_baris_loss",
  arr([float(torch.logsumexp(
        (Smat[i] / T).clone().index_fill(0, torch.tensor([i]), float("-inf")), 0))
       - float((Smat[i] / T)[(i + N) % (2 * N)]) for i in range(2 * N)]))

# rincian baris 0 SupCon
lab2 = torch.cat([yb_lab, yb_lab])
sS = Smat / T
sS = sS - sS.max(dim=1, keepdim=True)[0]
self_mask = torch.eye(2 * N, dtype=torch.bool)
pos_mask = (lab2.unsqueeze(0) == lab2.unsqueeze(1)) & ~self_mask
exp_sim = torch.exp(sS).masked_fill(self_mask, 0)
log_prob = sS - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)
p("supcon_label_2N", lab2.tolist())
p("supcon_b0_n_positif", int(pos_mask[0].sum()))
p("supcon_b0_indeks_positif", torch.nonzero(pos_mask[0]).flatten().tolist())
p("supcon_b0_logprob_positif", arr(log_prob[0][pos_mask[0]].tolist()))
p("supcon_b0_loss", round(float(-(log_prob[0][pos_mask[0]]).mean()), 6))
p("supcon_semua_baris_loss",
  arr([float(-(log_prob[i][pos_mask[i]]).mean()) for i in range(2 * N)]))

# CE berbobot
w = class_weights(tr)
p("n_train_hidup_mati", [sum(1 for r in tr if r["label"] == 0),
                         sum(1 for r in tr if r["label"] == 1)])
p("class_weight", arr(w.tolist()))
xb0 = torch.stack([to_tensor(im, 224, "letterbox") for im in ims])
with torch.no_grad():
    lg = model.logits(xb0)
p("ce_logit", arr(lg.tolist()))
per = F.cross_entropy(lg, yb_lab, reduction="none")
p("ce_per_sampel", arr(per.tolist()))
p("ce_bobot_per_sampel", arr(w[yb_lab].tolist()))
p("ce_pembilang", round(float((per * w[yb_lab]).sum()), 6))
p("ce_penyebut", round(float(w[yb_lab].sum()), 6))
p("ce_loss_berbobot", round(float(F.cross_entropy(lg, yb_lab, weight=w)), 6))
p("ce_loss_tanpa_bobot", round(float(F.cross_entropy(lg, yb_lab)), 6))


# ------------------------------------------------------------------ #
print("\n[7] Bagian IX - logit -> softmax -> keputusan (dua crop uji)")
p("classifier_W_shape", list(model.classifier.weight.shape))
lm = logit[0].tolist()
lh = logit[1].tolist()
p("logit_mati", arr(lm))
p("logit_hidup", arr(lh))
p("logit_mati_kelas1_manual",
  round(float((model.classifier.weight[1] * feat[0]).sum()
              + model.classifier.bias[1]), 6))
p("softmax_mati", arr(prob[0].tolist()))
p("softmax_hidup", arr(prob[1].tolist()))
p("softmax_mati_manual",
  round(float(np.exp(lm[1]) / (np.exp(lm[0]) + np.exp(lm[1]))), 6))
p("selisih_logit_mati", round(lm[1] - lm[0], 6))
p("sigmoid_selisih_mati",
  round(float(1 / (1 + np.exp(-(lm[1] - lm[0])))), 6))

# ------------------------------------------------------------------ #
print("\n[8] Bagian X - ambang & metrik dari skor nyata")
vs = np.load(RUN / "val_scores.npz")
ts = np.load(RUN / "test_scores.npz")
vy, vsc = vs["y_true"], vs["y_score"]
ty, tsc = ts["y_true"], ts["y_score"]
p("val_n", int(len(vy)))
p("val_y_true", vy.tolist())
p("val_y_score", arr(vsc, 4))
p("test_n", int(len(ty)))
p("test_y_true", ty.tolist())
p("test_y_score", arr(tsc, 4))
p("test_n_mati_hidup", [int((ty == 1).sum()), int((ty == 0).sum())])

m05 = compute_metrics(ty, (tsc >= 0.5).astype(int), tsc)
p("test_metrik_05", m05)
thr = float(pick_threshold(vy, vsc))
p("ambang_dari_val", round(thr, 6))
p("test_metrik_ambang", compute_metrics(ty, (tsc >= thr).astype(int), tsc))

# kandidat ambang di val (tampilkan skor unik + bacc tiap kandidat)
u = np.unique(vsc)
cands = np.concatenate([[u[0] - 1e-6], (u[:-1] + u[1:]) / 2.0, [u[-1] + 1e-6]])
bl = []
for t in cands:
    pr = (vsc >= t).astype(int)
    tp = int(((pr == 1) & (vy == 1)).sum()); tn = int(((pr == 0) & (vy == 0)).sum())
    fp = int(((pr == 1) & (vy == 0)).sum()); fn = int(((pr == 0) & (vy == 1)).sum())
    bl.append((tp / max(1, tp + fn) + tn / max(1, tn + fp)) / 2)
bl = np.array(bl)
p("val_skor_unik_n", int(len(u)))
p("val_kandidat_n", int(len(cands)))
p("val_bacc_maks", round(float(bl.max()), 6))
idx = np.flatnonzero(bl >= bl.max() - 1e-12)
groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
p("val_dataran_n_kelompok", len(groups))
p("val_dataran_kelompok",
  [{"i0": int(g[0]), "i1": int(g[-1]), "n": int(len(g)),
    "t0": round(float(cands[g[0]]), 6), "t1": round(float(cands[g[-1]]), 6),
    "lebar": round(float(cands[g[-1]] - cands[g[0]]), 6)} for g in groups])
widest = max(groups, key=lambda g: (cands[g[-1]] - cands[g[0]], len(g)))
p("val_dataran_terlebar_tengah",
  round(float((cands[widest[0]] + cands[widest[-1]]) / 2), 6))

# AUC dihitung dua jalan
p("auc_fungsi", round(float(roc_auc(ty, tsc)), 6))
npos, nneg = int((ty == 1).sum()), int((ty == 0).sum())
salah = 0.0
for a in tsc[ty == 1]:
    for b in tsc[ty == 0]:
        salah += 1.0 if a < b else (0.5 if a == b else 0.0)
p("auc_n_pasangan", npos * nneg)
p("auc_pasangan_salah_urut", salah)
p("auc_dari_pasangan", round(1 - salah / (npos * nneg), 6))
p("skor_min_mati", round(float(tsc[ty == 1].min()), 6))
p("skor_maks_hidup", round(float(tsc[ty == 0].max()), 6))

# ------------------------------------------------------------------ #
print("\n[9] Hasil ketiga metode (diagonal) + lantai ketajaman")
hasil = {}
for m in ("selfcon", "supcon", "ce"):
    r = json.loads((ROOT / "outputs" / "runs" / m / "result.json")
                   .read_text(encoding="utf-8"))
    hasil[m] = {"aug": r["aug"],
                "val_bacc": r["val"]["balanced_accuracy"],
                "val_auc": r["val"]["roc_auc"],
                "test_bacc": r["test"]["balanced_accuracy"],
                "test_auc": r["test"]["roc_auc"],
                "test_tp_tn_fp_fn": [r["test"]["tp"], r["test"]["tn"],
                                     r["test"]["fp"], r["test"]["fn"]],
                "ambang": r["threshold"],
                "test_tuned_bacc": r["test_tuned"]["balanced_accuracy"],
                "detik": r["seconds"]}
p("hasil_diagonal", hasil)

# ------------------------------------------------------------------ #
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(V, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nTersimpan: {OUT}  ({len(V)} nilai)")
