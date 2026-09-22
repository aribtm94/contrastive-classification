"""
KolektorSDD2: crop berlabel cacat/normal untuk classifier kontrastif.

DUA SUMBER KOTAK, DAN ITU INTINYA
---------------------------------
    --sumber gt        kotak dari mask GT      -> ACUAN (lantai atas)
    --sumber deteksi   kotak dari detektor     -> LENGAN DUA TAHAP

Yang diminta adalah lengan dua tahap: detektor memberi kotak, classifier
menilai kotak itu. Varian `gt` bukan pengganti - ia pembanding, supaya kalau
skor dua tahap jatuh, bisa dipisahkan mana yang salah: kotaknya (deteksi) atau
kelasnya (classifier).

KENAPA BUKAN src/detect.py. detect.py menulis crop TANPA manifest dan TANPA
label (detect.py:46 - hanya detections.json + detect_crops/). Classifier di
repo ini membaca manifest 11 kolom. Jadi berkas ini yang melabeli.

LABEL CROP DETEKSI - aturan eksplisit, sisanya dibuang dengan hitungan:
    IoU >= 0.5 dengan kotak GT mana pun          -> cacat
    IoU == 0 dan gambarnya NORMAL (tanpa mask)   -> normal
    sisanya (0 < IoU < 0.5, atau IoU 0 di gambar
    bercacat)                                    -> DIBUANG, ambigu
Cabang ketiga itu tidak boleh diam-diam masuk salah satu kelas: kotak yang
menyenggol cacat tapi tidak mengurungnya bukan contoh cacat maupun contoh
normal, dan memaksanya jadi salah satu akan mengajari classifier hal yang
salah sambil tetap memberi angka yang terlihat wajar.

DUA AMBANG AREA - jangan disatukan:
    eval_detect_masks.MIN_AREA = 20   kotak acuan (latih + evaluasi detektor)
    crops.min_area_komponen = 50      crop classifier (berkas INI)
Keduanya sengaja beda dan dicetak tiap jalan. Jangan bandingkan recall
deteksi dengan jumlah crop di sini seolah dari populasi yang sama.

LANTAI - ANGKANYA SUDAH DIKOREKSI, DAN KOREKSINYA ITU TEMUAN TERSENDIRI.
    gambar utuh (satu tahap)              : 0.612    bersih
    crop MENTAH (tanpa to_square)         : 0.9825   std_terang
    crop LETTERBOX 224 (dilihat model)    : 0.7300   saturasi
Angka 0.9815/0.9834 yang sempat ditulis di sini diukur pada crop MENTAH oleh
probe ad-hoc. Berkas ini menyimpan crop lewat to_square(c, 224, letterbox), dan
di situ std_terang runtuh ke 0.6214 - 36 poin - karena resize melicinkan
varians lokal dan bantalan abu 114 menambah piksel ber-varians nol. Lantai
WAJIB diukur pada crop yang TERSIMPAN, bukan pada kotak sebelum pra-proses;
mengukurnya lebih awal melaporkan lantai yang terlalu tinggi, dan lantai yang
terlalu tinggi menyembunyikan classifier yang tidak belajar apa pun.

Kesimpulan strukturalnya TETAP: 0.7300 masih jauh di atas 0.612, jadi pipeline
dua tahap tidak mewarisi lantai bersih versi satu tahapnya - hanya selisihnya
yang 0.12, bukan 0.37. Angka tambang-keras 0.9815 -> 0.9256 juga dari crop
mentah, jadi tidak berlaku di sini sampai diukur ulang.

BANTALAN LETTERBOX. common.letterbox() memasang bantalan abu 114 yang
lebarnya berkorelasi dengan bentuk kotak. Perlindungan struktural SDNET (ubin
persegi -> nol bantalan) TIDAK ADA di sini: kotak cacat w 1-231, h 2-423.
Untuk varian `gt` ukuran crop normal SENGAJA ditiru dari sebaran kotak cacat
sehingga `rasio` terukur 0.5083/0.4832; untuk varian `deteksi` sebaran itu
TIDAK dikendalikan. Itu SUDAH diukur ulang, dan hasilnya membenarkan
peringatan ini: rasio_bbox test AUC-terarah 0.9046 dan ukuran_bbox 0.9576 di
varian deteksi, lawan lantai 0.7300 di varian gt. Kotak cacat nyaris persegi
(rasio med 1.60), kotak normal serpih pipih (rasio med 4.81).

Sebabnya diuji lewat intervensi, bukan disimpulkan dari korelasi: memadankan
w DAN h sekaligus meruntuhkan seluruh ciri geometris (ukuran 0.9576 -> 0.6669,
saturasi 0.9150 -> 0.6843), tapi KETAJAMAN tetap 0.8708 pada ukuran setara -
jadi ada lantai kedua yang bukan artefak ukuran. Memadankan satu dimensi saja
hanya memindahkan kebocoran, tidak menghapusnya.

Tabel lengkap empat kondisi padanan ada di configs/config_kolektor.yaml
(kepala berkas, bagian "LANTAI VARIAN DETEKSI"). SATU sumber saja - jangan
disalin ke sini, dua salinan bisa berselisih. Yang mengikat kode ini: angka gt
tidak boleh dikutip untuk varian deteksi, dan sebaliknya.

JALUR DIPISAH PER SUMBER, dan itu koreksi cacat, bukan kerapian. Kedua varian
semula menulis ke crops.out_dir/crops.manifest yang sama, jadi jalan kedua
menimpa yang pertama tanpa ada yang gagal - dan karena gerbang
eval_shortcut_baseline.py berjalan SETELAH kedua build, "lantai varian gt"
akan terukur pada crop deteksi. Sekarang:
    --sumber gt       -> data/crops_kolektor_gt/manifest.csv
    --sumber deteksi  -> data/crops_kolektor_deteksi/manifest.csv
crops.manifest di config menunjuk varian yang dilatih train.py (train.py:348
membaca satu jalur literal); varian lainnya dilatih dengan menggesernya.

Jalankan:
    python src/build_crops_kolektor.py --config configs/config_kolektor.yaml --sumber gt
    python src/build_crops_kolektor.py --config configs/config_kolektor.yaml --sumber deteksi
    python src/build_crops_kolektor.py --config configs/config_kolektor.yaml --sumber gt --tambang-keras std
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import (box_iou, crop_box, imread, imwrite, load_config, resolve,
                    to_square)
from eval_detect_masks import MIN_AREA, imread_gray, masks_to_boxes

NAMA_LABEL = {0: "normal", 1: "cacat"}
DOMAIN = "kolektor_permukaan"          # SAMA untuk kedua kelas
SPLIT_PAPER = ("train", "test")


def kotak_gt(p_mask: Path, min_area: int) -> list[list[int]]:
    """Kotak GT dengan ambang area CLASSIFIER (bukan MIN_AREA evaluasi).

    masks_to_boxes() memaku MIN_AREA=20 sebagai konstanta modul, jadi untuk
    ambang classifier yang berbeda komponen harus dihitung ulang di sini.
    Definisi kotaknya tetap identik - boundingRect atas kontur RETR_EXTERNAL -
    yang berbeda HANYA ambang areanya, dan itu disebut di tiap keluaran.
    """
    m = imread_gray(p_mask)
    if m is None:
        return []
    b = (m > 127).astype(np.uint8)
    kontur, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in kontur:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        out.append([x, y, x + w, y + h])
    return out


def sebaran_kotak(semua: list[list[int]]) -> list[tuple[int, int]]:
    """(w, h) tiap kotak cacat, untuk ditiru saat menambang crop normal."""
    return [(b[2] - b[0], b[3] - b[1]) for b in semua]


def tambang_normal(img, n: int, sebaran: list[tuple[int, int]],
                   rng: random.Random, hindari: list[list[int]],
                   keras: str = "tidak") -> list[list[int]]:
    """Kotak di daerah normal, ukurannya DITIRU dari sebaran kotak cacat.

    Kenapa ditiru dan bukan diacak: kalau ukuran crop normal beda sebaran dari
    crop cacat, ukuran sendiri jadi jalan pintas dan classifier bisa menang
    tanpa melihat permukaannya. Bandingkan memori
    dataset-confounds-chicken-crops.

    keras='std': dari beberapa calon, pilih yang std_terang-nya TERTINGGI.
    PERHATIAN: std_terang dipilih sebagai kriteria ketika ia disangka lantainya
    (0.9815). Pada crop letterbox 224 yang benar-benar dilihat model ia hanya
    0.6659 dan yang mengikat adalah saturasi 0.7300, jadi mode ini menambang
    menurut ciri yang BUKAN lantai lagi. Angka 0.9815 -> 0.9256 tidak berlaku
    untuk crop letterbox dan tidak boleh dikutip tanpa diukur ulang.
    """
    H, W = img.shape[:2]
    keluar = []
    percobaan = 0
    n_calon = 8 if keras == "std" else 1
    while len(keluar) < n and percobaan < n * 200:
        percobaan += 1
        calon = []
        for _ in range(n_calon):
            w, h = sebaran[rng.randrange(len(sebaran))]
            w, h = min(w, W), min(h, H)
            if w < 1 or h < 1:
                continue
            x1 = rng.randint(0, max(0, W - w))
            y1 = rng.randint(0, max(0, H - h))
            b = [x1, y1, x1 + w, y1 + h]
            if any(box_iou(b, g) > 0.0 for g in hindari):
                continue
            calon.append(b)
        if not calon:
            continue
        if keras == "std":
            def skor(b):
                sub = img[b[1]:b[3], b[0]:b[2]]
                if sub.size == 0:
                    return -1.0
                return float(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).std())
            calon.sort(key=skor, reverse=True)
        keluar.append(calon[0])
        hindari = hindari + [calon[0]]
    return keluar


def muat_detektor(cfg: dict):
    dcfg = cfg["detection"]
    w = resolve(dcfg["weights"])
    if not w.exists():
        raise SystemExit(
            f"bobot detektor tidak ada: {w}\n"
            f"jalankan src/train_detect_kolektor.py dulu, lalu pin hasilnya "
            f"ke detection.weights")
    from ultralytics import YOLO
    print(f"[kol] detektor: {w.name} imgsz={dcfg['imgsz']} conf={dcfg['conf']}")
    return YOLO(str(w)), dcfg


def kotak_deteksi(model, dcfg: dict, p_img: Path) -> list[list[int]]:
    r = model.predict(str(p_img), imgsz=int(dcfg["imgsz"]),
                      conf=float(dcfg["conf"]),
                      iou=float(dcfg.get("iou", 0.5)),
                      max_det=int(dcfg.get("max_det", 50)),
                      verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    return [[int(v) for v in b] for b in r.boxes.xyxy.cpu().numpy().tolist()]


def bagi_per_gambar(stems: list[str], rasio_val: float,
                    rng: random.Random) -> dict[str, str]:
    """Split train/val di level GAMBAR.

    Satu gambar menyumbang crop cacat DAN normal, jadi
    build_crops.balance_and_split akan melempar ValueError("satu base_image
    punya dua label") - sama sebabnya src/build_manifest_sdnet.py lahir.
    Terukur di scripts/siapkan_kolektor.py: 2331 gambar train = 2331 kelompok
    korelasi (maks 0.9189 < 0.97), jadi di dataset INI gambar = kelompok, dan
    itu DIPERIKSA, bukan diasumsikan.
    """
    urut = sorted(stems)
    rng.shuffle(urut)
    n_val = int(round(len(urut) * rasio_val))
    pilih = set(urut[:n_val])
    return {s: ("val" if s in pilih else "train") for s in urut}


def jalur_per_sumber(cfg: dict, sumber: str) -> tuple[Path, Path]:
    """out_dir & manifest DIPISAH per --sumber. Bukan kerapian - koreksi cacat.

    Kedua varian menulis ke crops.out_dir dan crops.manifest yang SAMA, jadi
    jalan kedua menimpa jalan pertama dengan nama berkas yang identik: tepat
    kelas kegagalan senyap yang dicatat outputs/reports/domain_mati_kedua.md
    bagian 4h. Akibatnya konkret dan merusak: urutan verifikasi di rencana
    menjalankan eval_shortcut_baseline.py SETELAH kedua build, jadi "lantai
    varian gt" sebetulnya terukur di crop DETEKSI - padahal seluruh gunanya
    varian gt adalah jadi acuan yang terpisah dari deteksi.

    Sufiks dinormalkan (dibuang dulu, baru dipasang) supaya config boleh
    menunjuk data/crops_kolektor, _gt, atau _deteksi dan hasilnya tetap sama;
    tanpa itu --sumber gt atas config ber-_deteksi memberi _deteksi_gt.
    """
    base = resolve(cfg["crops"]["out_dir"])
    nama = base.name
    for suf in ("_gt", "_deteksi"):
        if nama.endswith(suf):
            nama = nama[: -len(suf)]
            break
    out_dir = base.with_name(f"{nama}_{sumber}")
    return out_dir, out_dir / "manifest.csv"


def tulis_manifest(rows: list, cfg: dict, path: Path | None = None) -> Path:
    """Kolom dan urutannya identik build_crops.write_manifest - itu kontraknya."""
    path = Path(path) if path is not None else resolve(cfg["crops"]["manifest"])
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["path", "label", "label_name", "split", "source_split",
            "base_image", "src_file", "bbox", "conf", "origin", "domain"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["split"], x["label"], x["path"])):
            w.writerow({c: (json.dumps(r[c]) if c == "bbox" else r.get(c, ""))
                        for c in cols})
    return path


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ringkas(rows: list, dilewati: Counter, manifest: Path,
            sumber: str, min_area: int) -> None:
    per_split = defaultdict(Counter)
    img_per_split = defaultdict(set)
    for r in rows:
        per_split[r["split"]][r["label_name"]] += 1
        img_per_split[r["split"]].add(r["base_image"])

    print(f"\n[kol] manifest -> {manifest}")
    print(f"[kol] sumber kotak: {sumber} | min_area_komponen {min_area} "
          f"(evaluasi detektor tetap MIN_AREA={MIN_AREA})")
    print(f"[kol] total crop: {len(rows)}")
    for sp in sorted(per_split):
        c = per_split[sp]
        print(f"   {sp:6s} cacat {c['cacat']:5d} / normal {c['normal']:5d}"
              f"   ({len(img_per_split[sp])} gambar)")
    if dilewati:
        print(f"[kol] dilewati: {dict(dilewati)}")

    # Penjaga anti-bocor: satu gambar tidak boleh muncul di dua split.
    dimana = defaultdict(set)
    for r in rows:
        dimana[r["base_image"]].add(r["split"])
    bocor = {k: sorted(v) for k, v in dimana.items() if len(v) > 1}
    if bocor:
        contoh = list(bocor.items())[:5]
        raise SystemExit(f"BOCOR: {len(bocor)} gambar ada di >1 split: {contoh}")
    print("[kol] periksa bocor: tidak ada gambar yang lintas split")

    # Sebaran bentuk kotak: bantalan letterbox berkorelasi dengan bentuk,
    # jadi angkanya dicetak per kelas, bukan disimpulkan dari varian lain.
    for nama in ("cacat", "normal"):
        r_ = [json.loads(x["bbox"]) if isinstance(x["bbox"], str) else x["bbox"]
              for x in rows if x["label_name"] == nama]
        if not r_:
            continue
        w = np.array([b[2] - b[0] for b in r_], dtype=float)
        h = np.array([b[3] - b[1] for b in r_], dtype=float)
        ras = np.maximum(w, h) / np.maximum(np.minimum(w, h), 1)
        print(f"[kol] bentuk {nama:6s}: w {w.min():.0f}-{w.max():.0f} "
              f"med {np.median(w):.0f} | h {h.min():.0f}-{h.max():.0f} "
              f"med {np.median(h):.0f} | rasio med {np.median(ras):.2f}")
    print("[kol] GERBANG berikutnya: eval_shortcut_baseline.py. Lantai varian "
          "gt TIDAK boleh dikutip untuk varian deteksi - sebaran ukurannya "
          "tidak dikendalikan di sana.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sumber", choices=("gt", "deteksi"), required=True)
    ap.add_argument("--tambang-keras", choices=("tidak", "std"), default=None,
                    help="bawaan dari crops.tambang_keras di config")
    a = ap.parse_args()

    cfg = load_config(a.config)
    ccfg = cfg["crops"]
    root = resolve(ccfg["kolektor_root"])
    size = int(ccfg["size"])
    pad = float(ccfg["bbox_padding"])
    mode_sq = ccfg.get("square_mode", "letterbox")
    min_area = int(ccfg["min_area_komponen"])
    min_box = int(ccfg.get("min_box_size", 8))
    rasio_norm = float(ccfg.get("normal_per_cacat", 1.0))
    rasio_val = float(ccfg.get("rasio_val", 0.25))
    keras = a.tambang_keras or str(ccfg.get("tambang_keras", "tidak"))

    out_dir, path_manifest = jalur_per_sumber(cfg, a.sumber)
    print(f"[kol] out_dir  : {out_dir}")
    print(f"[kol] manifest : {path_manifest}")
    for sp in ("train", "val", "test"):
        for nama in NAMA_LABEL.values():
            (out_dir / sp / nama).mkdir(parents=True, exist_ok=True)

    model = dcfg = None
    if a.sumber == "deteksi":
        model, dcfg = muat_detektor(cfg)

    rng = random.Random(int(cfg.get("seed", 42)))
    dilewati = Counter()
    rows = []

    for split_paper in SPLIT_PAPER:
        d = root / split_paper
        d_img = d / "images"
        if not d_img.is_dir():
            raise SystemExit(f"tata letak tidak ada: {d_img}")

        semua_img = sorted(d_img.glob("*.png"))
        peta_split = ({} if split_paper == "test"
                      else bagi_per_gambar([p.stem for p in semua_img],
                                           rasio_val, rng))

        # Sebaran bentuk kotak cacat, untuk ditiru saat menambang normal.
        gt_per_stem = {}
        for p in semua_img:
            p_mask = d / "segmentations" / f"{p.stem}.png"
            gt_per_stem[p.stem] = (kotak_gt(p_mask, min_area)
                                   if p_mask.exists() else [])
        sebaran = sebaran_kotak([b for v in gt_per_stem.values() for b in v])
        if not sebaran:
            raise SystemExit(f"nol kotak cacat di {split_paper} pada "
                             f"min_area_komponen={min_area}")
        print(f"[kol] {split_paper}: {len(semua_img)} gambar | "
              f"{sum(1 for v in gt_per_stem.values() if v)} bercacat | "
              f"{len(sebaran)} kotak cacat (min_area={min_area})")

        for p in semua_img:
            img = imread(p)
            if img is None:
                dilewati["gambar_tak_terbaca"] += 1
                continue
            gt = gt_per_stem[p.stem]
            sp = "test" if split_paper == "test" else peta_split[p.stem]

            if a.sumber == "gt":
                kotak_cacat = [(b, 1.0) for b in gt]
                kotak_normal_det = []
            else:
                kotak_cacat, kotak_normal_det = [], []
                for b in kotak_deteksi(model, dcfg, p):
                    iou_maks = max((box_iou(b, g) for g in gt), default=0.0)
                    if iou_maks >= 0.5:
                        kotak_cacat.append((b, iou_maks))
                    elif iou_maks == 0.0 and not gt:
                        # Deteksi di gambar NORMAL = positif palsu detektor,
                        # dan justru contoh normal yang paling informatif:
                        # kotak yang MENURUT detektor cacat. Inilah negatif
                        # yang benar untuk lengan dua tahap - bukan kotak acak.
                        kotak_normal_det.append(b)
                    else:
                        # 0 < IoU < 0.5, atau deteksi tanpa IoU di gambar yang
                        # PUNYA cacat. Bukan contoh cacat, bukan contoh normal.
                        dilewati["deteksi_ambigu"] += 1
                if gt and not kotak_cacat:
                    dilewati["cacat_tak_terdeteksi"] += 1

            for b, cf in kotak_cacat:
                rows.append(dict(_box=b, _label=1, _stem=p.stem, _src=p,
                                 _split=sp, _paper=split_paper, _conf=cf))
            for b in kotak_normal_det:
                rows.append(dict(_box=b, _label=0, _stem=p.stem, _src=p,
                                 _split=sp, _paper=split_paper, _conf=1.0))

            # Crop normal ditambang hanya untuk varian gt; untuk varian
            # deteksi, negatifnya adalah positif palsu detektor di atas -
            # kalau ditambah tambang acak, lengan dua tahap berhenti
            # mengukur detektornya dan berubah jadi campuran dua sumber.
            if a.sumber == "gt":
                n_norm = int(round(len(kotak_cacat) * rasio_norm)) if gt else 0
                if not gt:
                    # Gambar normal ikut menyumbang, proporsional.
                    n_norm = 1 if rng.random() < rasio_norm else 0
                for b in tambang_normal(img, n_norm, sebaran, rng, gt, keras):
                    rows.append(dict(_box=b, _label=0, _stem=p.stem, _src=p,
                                     _split=sp, _paper=split_paper, _conf=1.0))

    # Tulis crop + baris manifest.
    final = []
    for i, r in enumerate(rows):
        b = r["_box"]
        if b is None:
            continue
        if min(b[2] - b[0], b[3] - b[1]) < min_box:
            dilewati["box_terlalu_kecil"] += 1
            continue
        img = imread(r["_src"])
        if img is None:
            dilewati["gambar_tak_terbaca"] += 1
            continue
        c = crop_box(img, b, pad)
        if c is None or c.size == 0:
            dilewati["crop_kosong"] += 1
            continue
        nama = NAMA_LABEL[r["_label"]]
        rel = f"{r['_split']}/{nama}/{r['_stem']}_{i:05d}.png"
        imwrite(out_dir / rel, to_square(c, size, mode_sq))
        final.append(dict(
            path=(out_dir / rel).as_posix(),
            label=r["_label"], label_name=nama, split=r["_split"],
            source_split=r["_paper"], base_image=r["_stem"],
            src_file=r["_src"].name, bbox=[int(v) for v in b],
            conf=r.get("_conf", 1.0),
            origin=f"kolektor_{a.sumber}", domain=DOMAIN))

    manifest = tulis_manifest(final, cfg, path_manifest)
    ringkas(final, dilewati, manifest, a.sumber, min_area)
    print(f"[kol] builder_sha256 {_sha256(Path(__file__))[:16]}...")


if __name__ == "__main__":
    main()
