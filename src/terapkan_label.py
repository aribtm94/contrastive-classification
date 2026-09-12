"""
Jalankan: python src/terapkan_label.py

Terapkan catatan label manual ke lembar_label.csv, lalu pisahkan crop
'bukan ayam' ke folder tersendiri untuk diperiksa ulang.

Masukan  : data/label_chick/catatan_mati.txt
           data/label_chick/catatan_bukan.txt
           (isi bebas: angka dipisah koma/spasi/baris, sisanya diabaikan)

Keluaran : kolom 'label' di lembar_label.csv terisi
           data/label_chick/cek_ulang/bukan/       salinan crop 'bukan'
           data/label_chick/cek_ulang/lembar_bukan_NN.jpg  lembar kontaknya

Yang tidak tercatat di kedua berkas dianggap ayam hidup (label kosong),
sesuai keputusan pelabel.

Kolom 'dugaan_awal' tidak pernah ditimpa: itu catatan asal-usul 22 crop
yang cocok dengan mask acuan (IoU >= 0.5). Kalau label manual berbeda
dengan dugaan_awal, selisihnya dilaporkan - tidak didiamkan.
"""

import csv, io, math, os, re, shutil, sys
import cv2, numpy as np

AKAR = "data/label_chick"
CEK = os.path.join(AKAR, "cek_ulang")
PETAK, KOL, PAD = 128, 10, 26


def baca_nomor(p):
    """Baris berawalan # dibuang DULU, baru angkanya diambil - kalau tidak,
    angka yang kebetulan ada di dalam komentar ikut terbaca jadi label."""
    if not os.path.exists(p):
        print("  ! tidak ada:", p); return []
    isi = []
    for bar in io.open(p, encoding="utf-8"):
        bar = bar.split("#", 1)[0]
        isi.extend(int(t) for t in re.findall(r"\d+", bar))
    return isi


mati_n = baca_nomor(os.path.join(AKAR, "catatan_mati.txt"))
bukan_n = baca_nomor(os.path.join(AKAR, "catatan_bukan.txt"))

rows = list(csv.DictReader(io.open(os.path.join(AKAR, "lembar_label.csv"),
                                   encoding="utf-8")))
sah = set(int(r["nomor"]) for r in rows)

# nomor di luar 1..1215 dibuang, tapi disebut - jangan didiamkan
for nama, lst in (("mati", mati_n), ("bukan", bukan_n)):
    buang = [n for n in lst if n not in sah]
    if buang:
        print("  ! nomor %s di luar jangkauan, dilewati: %s" % (nama, buang))

mati = set(n for n in mati_n if n in sah)
bukan = set(n for n in bukan_n if n in sah)
bentrok = sorted(mati & bukan)
if bentrok:
    print("  ! nomor ada di dua daftar sekaligus: %s" % bentrok)
    bukan -= mati                      # 'mati' menang: lebih spesifik

for r in rows:
    n = int(r["nomor"])
    r["label"] = "mati" if n in mati else ("bukan" if n in bukan else "")

with io.open(os.path.join(AKAR, "lembar_label.csv"), "w",
             encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

n_hidup = sum(1 for r in rows if r["label"] == "")
print("mati       : %4d" % len(mati))
print("bukan ayam : %4d" % len(bukan))
print("hidup      : %4d" % n_hidup)
print("jumlah     : %4d" % len(rows))

# selisih terhadap 22 acuan mask
acuan = set(int(r["nomor"]) for r in rows if r["dugaan_awal"] == "mati")
print("\nacuan mask mati        : %d" % len(acuan))
print("  disetujui            : %d" % len(acuan & mati))
print("  acuan TIDAK dicatat  : %s" % (sorted(acuan - mati) or "-"))
print("  mati temuan baru     : %s" % (sorted(mati - acuan) or "-"))

# ---- pisahkan crop 'bukan' untuk diperiksa ulang ----
if os.path.isdir(CEK):
    shutil.rmtree(CEK)
os.makedirs(os.path.join(CEK, "bukan"))
item = [r for r in rows if r["label"] == "bukan"]
for r in item:
    shutil.copy2(os.path.join(AKAR, "crops", r["berkas"]),
                 os.path.join(CEK, "bukan", r["berkas"]))

# lembar kontak khusus 'bukan', 60 petak per halaman, nomor tetap global
PER_HAL = 60
for h in range(math.ceil(len(item) / PER_HAL)):
    blok = item[h * PER_HAL:(h + 1) * PER_HAL]
    bar = math.ceil(len(blok) / KOL)
    k = np.full((bar * (PETAK + PAD) + 46, KOL * PETAK, 3), 245, np.uint8)
    for i, r in enumerate(blok):
        y, x = divmod(i, KOL)
        oy, ox = 46 + y * (PETAK + PAD), x * PETAK
        k[oy + PAD - 2:oy + PAD + PETAK - 2, ox + 2:ox + PETAK - 2] = 205
        im = cv2.imread(os.path.join(AKAR, "crops", r["berkas"]))
        if im is not None:
            hh, ww = im.shape[:2]
            s = min(PETAK / ww, PETAK / hh)
            im = cv2.resize(im, (max(1, int(ww * s)), max(1, int(hh * s))))
            ph, pw = im.shape[:2]
            k[oy + PAD + (PETAK - ph) // 2:oy + PAD + (PETAK - ph) // 2 + ph,
              ox + (PETAK - pw) // 2:ox + (PETAK - pw) // 2 + pw] = im
        cv2.rectangle(k, (ox + 2, oy + PAD - 2),
                      (ox + PETAK - 2, oy + PAD + PETAK - 2), (90, 90, 90), 1)
        cv2.putText(k, r["berkas"][:4], (ox + 5, oy + PAD - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (90, 90, 90), 2, cv2.LINE_AA)
    cv2.putText(k, "BUKAN AYAM - cek ulang  (%d dari %d, hal %d)"
                % (len(blok), len(item), h + 1), (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    out = os.path.join(CEK, "lembar_bukan_%02d.jpg" % (h + 1))
    cv2.imwrite(out, k, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("  %s  %d petak" % (os.path.basename(out), len(blok)))

print("\ncek ulang:", CEK)
