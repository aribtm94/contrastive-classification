"""
Lembar kontak: 1215 crop jadi 18 halaman (satu per gambar sumber).
Tiap petak diberi nomor det_id besar supaya mudah dicatat.
Crop yang sudah diketahui mati diberi bingkai supaya tidak perlu dicek lagi.
"""
import csv, io, os, math, collections
import cv2, numpy as np

KELUAR = "data/label_chick/lembar_kontak"
PETAK = 128          # ukuran tiap petak
KOL = 10             # petak per baris
PAD = 26             # ruang tulisan nomor di atas petak

os.makedirs(KELUAR, exist_ok=True)
rows = list(csv.DictReader(io.open("data/label_chick/lembar_label.csv",
                                   encoding="utf-8")))
per = collections.defaultdict(list)
for r in rows:
    per[r["gambar_sumber"]].append(r)

for nama in sorted(per):
    item = sorted(per[nama], key=lambda r: int(r["det_id"]))
    n = len(item)
    baris = math.ceil(n / KOL)
    H = baris * (PETAK + PAD) + 46
    W = KOL * PETAK
    kanvas = np.full((H, W, 3), 245, np.uint8)
    # petak diberi dasar abu supaya batas crop terlihat (crop sempit
    # tidak memenuhi petak - itu petunjuk crop terpotong)
    for _i in range(n):
        _y, _x = divmod(_i, KOL)
        _oy = 46 + _y * (PETAK + PAD); _ox = _x * PETAK
        kanvas[_oy + PAD - 2:_oy + PAD + PETAK - 2, _ox + 2:_ox + PETAK - 2] = 205

    cv2.putText(kanvas, "%s  -  %d crop  (bingkai = sudah diketahui MATI)"
                % (nama, n), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (20, 20, 20), 2, cv2.LINE_AA)

    for i, r in enumerate(item):
        y, x = divmod(i, KOL)
        oy = 46 + y * (PETAK + PAD)
        ox = x * PETAK
        im = cv2.imread(os.path.join("data/label_chick/crops", r["berkas"]))
        if im is None:
            continue
        h, w = im.shape[:2]
        s = min(PETAK / w, PETAK / h)
        im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
        ph, pw = im.shape[:2]
        dy, dx = oy + PAD + (PETAK - ph) // 2, ox + (PETAK - pw) // 2
        kanvas[dy:dy + ph, dx:dx + pw] = im

        mati = r["dugaan_awal"] == "mati"
        warna = (0, 0, 220) if mati else (90, 90, 90)
        tebal = 3 if mati else 1
        cv2.rectangle(kanvas, (ox + 2, oy + PAD - 2),
                      (ox + PETAK - 2, oy + PAD + PETAK - 2), warna, tebal)
        tulis = "%s%s" % (r["det_id"], " MATI" if mati else "")
        cv2.putText(kanvas, tulis, (ox + 5, oy + PAD - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, warna, 2, cv2.LINE_AA)

    out = os.path.join(KELUAR, "%s.jpg" % os.path.splitext(nama)[0])
    cv2.imwrite(out, kanvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("  %-18s %3d crop -> %s" % (nama, n, os.path.basename(out)))

print("\nlembar kontak:", KELUAR)
