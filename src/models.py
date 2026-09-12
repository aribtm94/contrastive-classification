r"""
Arsitektur model dan fungsi loss untuk ketiga metode yang dibandingkan.

Arsitektur (sama persis untuk ketiga metode, supaya perbandingannya adil):

    gambar 224x224
         |
    [ Encoder ResNet-18 ]  -> fitur 512-d
         |          \
         |           \--> [ Projection head ] -> embedding 128-d  (untuk loss kontrastif)
         |
         \--> [ Linear classifier ] -> 2 logit (hidup / mati)

Yang membedakan ketiga metode hanyalah CARA MELATIHNYA:

  A. selfcon (Self-Contrastive / SimCLR, NT-Xent)
        Label TIDAK dipakai saat melatih encoder. Yang dianggap "pasangan
        positif" hanyalah dua augmentasi dari gambar yang sama; semua
        gambar lain di dalam batch dianggap negatif.
        Konsekuensinya: dua ayam mati yang berbeda tetap DIDORONG SALING
        MENJAUH, walaupun kelasnya sama. Setelah encoder selesai, encoder
        dibekukan lalu dilatih linear probe memakai label.

  B. supcon (Supervised Contrastive)
        Label ikut masuk ke dalam loss. Semua gambar sekelas di dalam batch
        dianggap positif, sehingga seluruh ayam mati ditarik berkumpul dan
        dijauhkan dari kelompok ayam hidup. Inilah "loss ayam hidup dan
        ayam mati digabung" yang diminta. Opsional bisa ditambah CE
        bersamaan lewat classifier.supcon_ce_weight.

  C. ce (Cross-Entropy biasa)
        Baseline tanpa contrastive sama sekali - langsung latih encoder +
        classifier dengan cross-entropy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Encoder(nn.Module):
    """Backbone ResNet + projection head + kepala klasifikasi linear."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 proj_dim: int = 128, proj_hidden: int = 512,
                 num_classes: int = 2):
        super().__init__()

        weights = "IMAGENET1K_V1" if pretrained else None
        net = getattr(torchvision.models, backbone)(weights=weights)
        self.feat_dim = net.fc.in_features
        net.fc = nn.Identity()          # buang kepala ImageNet-nya
        self.backbone = net

        # Projection head: fitur 512-d -> embedding 128-d.
        # Loss kontrastif dihitung di ruang embedding ini, bukan di fitur,
        # supaya fitur backbone tidak ikut terdistorsi oleh loss kontrastif.
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_dim),
        )

        # Kepala klasifikasi dipakai oleh metode CE dan oleh linear probe
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Embedding kontrastif, sudah dinormalisasi ke bola satuan (L2)."""
        return F.normalize(self.projector(self.backbone(x)), dim=1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)

    def freeze_backbone(self) -> None:
        """Bekukan encoder - dipakai sebelum melatih linear probe."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()


# --------------------------------------------------------------------------- #
# METODE A - Self-Contrastive (NT-Xent / SimCLR), tanpa label
# --------------------------------------------------------------------------- #
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.07) -> torch.Tensor:
    """
    NT-Xent: Normalized Temperature-scaled Cross Entropy.

    z1, z2 : embedding dua view dari batch yang sama, masing-masing [N, D],
             sudah dinormalisasi L2.

    Untuk tiap sampel, HANYA view pasangannya sendiri yang dianggap positif.
    Sisanya (2N - 2 sampel) dianggap negatif - termasuk gambar lain yang
    sebenarnya sekelas. Di sinilah kelemahan metode ini pada dataset kecil:
    ayam mati yang satu didorong menjauh dari ayam mati yang lain.
    """
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                     # [2N, D]

    sim = z @ z.t() / temperature                      # kemiripan kosinus
    # Buang diagonal (kemiripan dengan diri sendiri) agar tidak jadi positif
    sim.fill_diagonal_(float("-inf"))

    # Pasangan positif: i <-> i+N
    target = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, target)


# --------------------------------------------------------------------------- #
# METODE B - Supervised Contrastive, label hidup & mati digabung
# --------------------------------------------------------------------------- #
def supcon_loss(z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor,
                temperature: float = 0.07) -> torch.Tensor:
    """
    Supervised Contrastive Loss (Khosla et al., 2020).

    Bedanya dengan NT-Xent: positif bukan cuma view pasangannya, tapi SEMUA
    sampel di dalam batch yang labelnya sama. Jadi seluruh crop ayam mati
    ditarik berkumpul jadi satu gugus, seluruh ayam hidup jadi gugus lain,
    lalu kedua gugus itu saling dijauhkan.

    Inilah bentuk "loss ayam hidup dan ayam mati digabung dalam satu loss".
    """
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                     # [2N, D]
    lab = torch.cat([labels, labels], dim=0)           # [2N]

    sim = z @ z.t() / temperature
    # Stabilitas numerik: kurangi nilai maksimum tiap baris sebelum exp()
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()

    self_mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    # positif = label sama, tapi bukan dirinya sendiri
    pos_mask = (lab.unsqueeze(0) == lab.unsqueeze(1)) & ~self_mask

    exp_sim = torch.exp(sim).masked_fill(self_mask, 0)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    n_pos = pos_mask.sum(dim=1)
    valid = n_pos > 0          # kalau satu batch isinya 1 kelas saja
    if not valid.any():
        return torch.zeros((), device=z.device, requires_grad=True)

    mean_log_prob = (pos_mask * log_prob).sum(dim=1)[valid] / n_pos[valid]
    return -mean_log_prob.mean()


# --------------------------------------------------------------------------- #
# METODE C - Cross-Entropy biasa
# --------------------------------------------------------------------------- #
def ce_loss(logits: torch.Tensor, labels: torch.Tensor,
            weight: torch.Tensor | None = None) -> torch.Tensor:
    """Cross-entropy standar, dengan bobot kelas opsional untuk imbalance."""
    return F.cross_entropy(logits, labels, weight=weight)


def build_model(cfg: dict, device: torch.device) -> Encoder:
    c = cfg["classifier"]
    model = Encoder(backbone=c["backbone"], pretrained=bool(c["pretrained"]),
                    proj_dim=int(c["proj_dim"]),
                    proj_hidden=int(c["proj_hidden"]), num_classes=2)
    return model.to(device)
