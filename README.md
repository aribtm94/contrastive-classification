# Deteksi Ayam + Klasifikasi Kontrastif (Mati vs Hidup)

Pipeline dua tahap: **YOLO mendeteksi ayam** → tiap bounding box **dipotong dan
dibatasi 224×224** → **classifier hasil contrastive learning** menilai tiap
potongan: ayam hidup atau ayam mati.

Tiga metode pelatihan classifier dibandingkan: Self-Contrastive, Supervised
Contrastive, dan Cross-Entropy biasa.

> **Baru membaca proyek ini?** [`outputs/reports/kronologi_lengkap.md`](outputs/reports/kronologi_lengkap.md)
> menceritakan seluruh eksperimen berurutan dari rancangan pipeline sampai
> temuan terakhir - lengkap dengan gambar hasil tiap percobaan, alasan di
> balik setiap perubahan, daftar masalah, dan naskah untuk menjelaskannya.

---

## Protokol eksperimen aktif

Sesuai arahan dosen, eksperimen lanjutan memakai **PIO + Roboflow hanya untuk
train dan validation**, sedangkan seluruh dataset ayam/chick tetap menjadi
test. Train/validation berisi ayam hidup PIO dan ayam mati Roboflow; akibatnya
`label = domain` masih menjadi keterbatasan yang wajib dibaca bersama hasil.
Data ayam/chick tidak dipakai memilih epoch, threshold, seed, atau checkpoint.

Training baru mencatat **train loss dan validation loss pada setiap epoch**
untuk NT-Xent, SupCon, dan weighted CE. Protokol lengkap ada di
[`outputs/reports/pio_development_protocol.md`](outputs/reports/pio_development_protocol.md).
Karena chick sudah dipakai pada eksperimen historis, hasilnya disebut benchmark
test tetap retrospektif, bukan holdout prospektif yang benar-benar buta.

---

## Alur pipeline

```
     gambar kandang
           |
   [1] YOLOv8m (deteksi ayam)          src/detect.py
           |  bounding box per ekor
           v
   [2] crop tiap bbox + letterbox      src/common.py: to_square()
           |  semua jadi 224 x 224
           v
   [3] classifier kontrastif           src/train.py
           |
           v
     hidup / MATI  + peluang
```

Untuk melatih classifier-nya, dibutuhkan crop berlabel. Itu disiapkan oleh
`src/build_crops.py` (tahap terpisah, sekali jalan).

---

## Cara menjalankan

Gunakan interpreter dari venv yang sudah ada (torch + CUDA + ultralytics):

```powershell
$PY = "C:\Arib\MASSA AYAM\generalisasi-ayam-skripsi\.venv-yolo\Scripts\python.exe"

& $PY src/detect.py        # Tahap 1: deteksi saja (demo di data uji)
& $PY src/build_crops.py   # Tahap 2: siapkan crop berlabel hidup/mati
& $PY src/train.py --method all   # Tahap 3: latih & bandingkan 3 metode
& $PY src/report.py        # Buat tabel + grafik perbandingan
& $PY src/pipeline.py --method supcon   # Pipeline ujung-ke-ujung
```

Semua pengaturan ada di `configs/config.yaml` — tidak ada path yang
di-hardcode di dalam kode.

---

## Tiga metode yang dibandingkan

| | Metode | Label saat melatih encoder | Yang dianggap "positif" |
|---|---|---|---|
| **A** | `selfcon` — Self-Contrastive (NT-Xent/SimCLR) | tidak dipakai | hanya augmentasi dari gambar yang sama |
| **B** | `supcon` — Supervised Contrastive | dipakai, digabung | semua gambar sekelas di dalam batch |
| **C** | `ce` — Cross-Entropy | dipakai | — (tanpa contrastive) |

Arsitektur, data, ukuran input, dan jumlah epoch ketiganya identik. Sejak tiap
metode diberi **kebijakan augmentasinya sendiri** — masing-masing diambil dari
satu paper yang berbeda — yang berbeda ada **dua**: fungsi loss **dan**
augmentasi.

| Metode | Augmentasi | Paper sumber |
|---|---|---|
| `selfcon` | `simclr` | Chen dkk., ICML 2020 |
| `supcon` | `stacked_randaug` | Khosla dkk., NeurIPS 2020 |
| `ce` | `hier_addone` | Zhang & Ma, CVPR 2022 |

Karena dua hal berubah bersamaan, **diagonal itu saja tidak bisa
mengatribusikan sebab**. Karena itu `--grid full` menjalankan seluruh 3×3:
membandingkan dalam satu augmentasi yang sama memulihkan perbandingan metode
yang sah, dan membandingkan dalam satu metode yang sama menjawab "ini efek
augmentasi atau efek loss?". Alasan tiap augmentasi dipilih, gambar contohnya,
dan deviasinya dari paper ada di
[`outputs/reports/augmentation_report.md`](outputs/reports/augmentation_report.md).

Pada A dan B, setelah encoder dilatih, encoder **dibekukan** lalu hanya
*linear probe* yang dilatih. Ini disengaja: yang diukur adalah kualitas
representasi hasil contrastive, bukan hasil fine-tuning lanjutan.

Metode B bisa digabung dengan cross-entropy sekaligus lewat
`classifier.supcon_ce_weight` (default 0.0 = SupCon murni).

---

## Hasil

Test set: 33 crop (26 mati, 7 hidup).

> **Tabel di bawah adalah hasil LAMA** (augmentasi `legacy` yang sama untuk
> ketiga metode, satu seed). Tabel itu tetap dipertahankan sebagai titik acuan.
> Hasil terkini - 3 augmentasi x 3 metode x 5 seed, dengan mean ± simpangan
> baku dan baris lantai - ada di
> [`outputs/reports/comparison.md`](outputs/reports/comparison.md).
>
> **Dua angka yang harus dibaca lebih dulu:** sebuah "classifier" yang cuma
> mengukur ketajaman gambar, tanpa melihat isinya sama sekali, sudah mendapat
> **bal.acc 0.9231** (AUC 0.9670) di test set ini. Artinya angka `ce` di bawah
> (0.9615) hampir seluruhnya bisa dijelaskan tanpa belajar apa pun, dan dua
> metode lainnya berada **di bawah** lantai itu. Sebabnya dua confound pada cara crop
> dibentuk - lihat
> [`outputs/reports/augmentation_report.md`](outputs/reports/augmentation_report.md).

| Metode | Bal. Acc | Akurasi | Recall mati | Recall hidup | AUC |
|---|---|---|---|---|---|
| C. `ce` | **0.9615** | 0.9394 | 0.9231 | 1.0000 | 0.9725 |
| B. `supcon` | 0.8516 | 0.8485 | 0.8462 | 0.8571 | 0.9176 |
| A. `selfcon` | 0.7857 | 0.9091 | 1.0000 | 0.5714 | 1.0000 |

Metrik utamanya **balanced accuracy**, bukan akurasi biasa: karena 79% test set
berisi ayam mati, model yang menebak "mati" untuk semuanya sudah mendapat
akurasi 79% tanpa belajar apa pun. Perhatikan `selfcon` — akurasinya 0.9091
(tertinggi kedua) padahal recall ayam hidupnya hanya 0.57.

Urutan hasil ini wajar: `selfcon` tidak memakai label saat melatih encoder,
dan metode self-supervised umumnya baru unggul kalau data tak berlabelnya
banyak (ribuan hingga jutaan). Dengan 75 crop latih, supervisi langsung
menang telak.

### Kenapa AUC `selfcon` 1.0000 padahal bal.acc-nya cuma 0.7857?

Ini bukan kesalahan hitung. AUC mengukur apakah **urutan** skornya benar,
sementara bal.acc mengukur hasil setelah dipotong di ambang 0.5. Skor test
`selfcon`:

```
ayam hidup : 0.041  0.330  0.348  0.463 | 0.501  0.559  0.586
ayam mati  :                                        0.644 ... 0.959
                                        ^                ^
                                     ambang 0.5     ambang 0.644
```

Semua ayam mati berskor lebih tinggi daripada semua ayam hidup — **tidak ada
satu pun yang tumpang tindih**, itulah arti AUC 1.0. Masalahnya hanya ambang:
di 0.5 ada 3 ayam hidup yang lolos ke sisi "mati". Geser ambang ke 0.644 dan
bal.acc-nya jadi **1.0000**.

Artinya representasi hasil self-contrastive di sini justru yang paling bersih;
yang kurang tepat adalah kalibrasi linear probe-nya. Ini efek yang wajar saat
melatih probe pada data timpang (55 mati : 20 hidup) — probe cenderung
menggeser ambangnya ke arah kelas mayoritas.

Tapi jangan buru-buru menyimpulkan `selfcon` yang terbaik: pemisahan sempurna
itu terjadi pada **7 ayam hidup saja**. Untuk 7 sampel, AUC 1.0 cukup mudah
terjadi secara kebetulan. Yang bisa disimpulkan hanyalah bahwa **membandingkan
ketiga metode lewat bal.acc di ambang tetap 0.5 itu menyesatkan** — sebaiknya
lihat AUC juga, atau tentukan ambang lewat validation set.

Sejak temuan ini, pemilihan ambang lewat validation set **sudah dikerjakan**:
`src/train.py` mencari ambang yang memaksimalkan bal.acc di val, lalu
melaporkan `test_tuned` berdampingan dengan `test` — bukan menggantikannya,
supaya tidak ada angka yang dipilih belakangan setelah melihat hasil test.
Skor val ikut disimpan (`val_scores.npz`) agar ambangnya bisa diaudit ulang.
Kolom `Bal.Acc @tau` di
[`comparison.md`](outputs/reports/comparison.md) adalah hasilnya.

### Satu konfigurasi akhirnya melewati lantai - tapi pada URUTAN

Sweep 3 augmentasi x 3 metode x 5 seed sudah selesai (45 run). Diukur dengan
bal.acc, **tidak satu pun konfigurasi 5-seed** berada di atas lantai 0.9231 -
termasuk ketiga baris diagonal yang diminta. (Yang melewatinya cuma baris
`legacy` `ce` 0.9615, dan itu satu seed - tidak ada simpangan baku yang bisa
menyanggah keberuntungan satu undian.) Kalau laporan berhenti di situ,
kesimpulannya "belum ada yang membuktikan apa-apa".

Tapi lantai itu punya dua sisi, dan sisi keduanya belum diuji: lantai ketajaman
juga punya **AUC 0.9670** - setara salah mengurutkan 6 dari 182 pasangan
(mati x hidup). Diuji terhadap angka itu, ada satu yang lolos:

| Augmentasi | Metode | AUC | pasangan salah urut | n seed |
|---|---|---|---|---|
| `simclr` | `supcon` | **0.9945 ± 0.0085** | 1.0 dari 182 | 5 |

`mean - simpangan baku` = 0.9860, masih di atas 0.9670, dan **kelima seed**-nya
di atas lantai (0.9780 / 1.0000 / 1.0000 / 1.0000 / 0.9945). Jadi
representasinya memang memisahkan kedua kelas lebih baik daripada sekadar
ketajaman - yang belum beres cuma kalibrasi ambangnya (bal.acc-nya 0.8066 ±
0.1585).

Dua catatan yang harus ikut dibaca:

1. **Ambangnya tidak bisa diperbaiki lewat validation set di sini.** Val cuma 22
   crop dan 47% run mencapai bal.acc val 1.0000 - sempurna. Buktinya `ce`+`simclr`:
   bal.acc val **1.0000 di kelima seed** (identik) sementara test-nya berayun
   0.585-0.857. Val tidak melihat perbedaan itu sama sekali, jadi ia hanya bisa
   mengkalibrasi ambang, tidak bisa **memilih** seed/epoch/konfigurasi.
2. **7 ayam hidup itu sedikit sekali**, jadi AUC setinggi ini lebih mudah terjadi
   kebetulan daripada kelihatannya. Yang boleh diklaim: *pada test set ini*,
   urutannya mengalahkan lantai di seluruh 5 seed.

Perlu ditegaskan juga bahwa ini **bukan** baris diagonal (`supcon` dipasangkan ke
`stacked_randaug` di rancangan). Ia muncul dari kolom grid penuh - yaitu justru
perbandingan yang sah, karena augmentasinya ditahan dan hanya loss-nya berubah.

### Diuji pada data CCTV yang sesungguhnya: AUC 0.44-0.64

Semua angka di atas diukur pada test set close-up. Diuji ujung-ke-ujung pada
dataset `chick` (CCTV tampak atas, 22 ayam mati vs 1193 crop lain), hasilnya
runtuh ke kisaran menebak - lihat
[`detection_chick.md`](outputs/reports/detection_chick.md). Deteksinya sendiri
sudah 100%; yang belum bisa adalah classifier-nya, karena seluruh crop latih
berdomain close-up.

Melatih dengan ayam hidup dari PIO sudah dicoba sebagai jalan keluar. Test
bacc-nya menembus 1.0000, tapi itu palsu: PIO satu kelas, jadi label jadi
identik dengan domain dan ketajaman gambar saja sudah memberi test bacc
0.8434. Buktinya paling jelas: model PIO menyebut ayam **hidup** close-up
sebagai mati dengan p 0.81. Duduk perkaranya di
[`pio_saja.md`](outputs/reports/pio_saja.md).

Di `chick` sendiri, ketiga susunan data dibandingkan berpasangan (metode dan
seed sama persis, 9 checkpoint tiap kelompok):

| susunan data latih | AUC chick, 3 seed | vs close-up | Wilcoxon |
|---|---|---|---|
| close-up (yang dipakai) | 0.651 +/- 0.062 | - | - |
| PIO saja | 0.600 +/- 0.087 | -0.051 | p = 0.301 |
| PIO + resolusi disamakan 48px | 0.674 +/- 0.080 | +0.023 | p = 0.570 |

**Tidak ada yang terbukti lebih baik.** Sebaran antar seed lebih besar daripada
jarak antar kelompok, jadi menata ulang sumber data belum menggerakkan apa pun.
Angka 1-seed yang sempat tercatat (close-up 0.522 / 0.569 / 0.606) ternyata
terlalu rendah - dengan 22 ayam mati di test, satu seed tidak cukup.

Karena itu setiap susunan data baru sekarang diperiksa dulu dengan
`python src/eval_shortcut_baseline.py --crops <folder>` - kalau ketajaman atau
ukuran saja sudah memberi skor tinggi, angka classifier-nya belum bisa dibaca
sebagai kemampuan mengenali ayam mati.

### Yang lebih mendasar: classifier tidak membaca bentuk ayam

Diuji dengan merusak satu ciri sekaligus (`src/eval_intervensi.py`). Crop
dipecah 4x4 lalu petaknya diacak - bentuk dan pose ayam hancur total:

| model | AUC asli | **AUC bentuk diacak** | AUC dikaburkan |
|---|---|---|---|
| ce + hier_addone | 0.934 | **0.940** | 0.396 |
| selfcon + simclr | 0.967 | **0.995** | 0.769 |
| supcon + stacked_randaug | 0.890 | **0.918** | 0.269 |

Delapan belas checkpoint diuji (3 metode x 3 seed x 2 susunan data; close-up,
PIO, PIO-disamakan) dan **tidak satu pun terganggu** - selisih AUC rata-rata
cuma -0.0100, dan cuma 1/18 yang turun lebih dari 0.05; sebagian malah naik.
Yang dipakai classifier adalah ketajaman dan statistik tekstur, bukan pose -
padahal mati-vs-hidup justru soal pose. Ini menjelaskan kenapa skor tinggi di
test set tidak pernah ikut pindah ke CCTV.

![bentuk diacak](outputs/reports/acak_bentuk.jpg)

Catatan cara mengukur: korelasi **tidak cukup** untuk menyimpulkan ini. rho
antara p(mati) dan hue sempat mencapai -0.599, tapi ketika hue benar-benar
diputar, AUC tidak bergerak - hue cuma penumpang. Di data yang label-nya =
domain, semua ciri berkorelasi dengan label, jadi yang berlaku hanya uji sebab:
rusak cirinya, lalu ukur akibatnya.

---

## Keputusan desain penting

### 1. Model deteksi: `cmp_yolov8m/weights/best.pt`

Dipilih dari 11 run di `generalisasi-ayam-skripsi` berdasarkan mAP50-95:

| Run | mAP50-95 | mAP50 |
|---|---|---|
| **cmp_yolov8m** | **0.7628** | 0.9353 |
| cmp_yolo11m | 0.7599 | 0.9344 |
| ft_radial_yolov8m | 0.7001 | 0.8993 |
| ft_rectified_yolov8m | 0.6821 | 0.9081 |
| ft_pad015_yolov8m | 0.6629 | 0.8984 |

Sisanya (`yolo12m`, `yolov9m`, `yolov10*`) berhenti di epoch 1–3, belum selesai
dilatih.

### 2. Crop ayam hidup diambil dari foto yang sama dengan ayam mati

Dataset ayam mati hanya berisi bbox ayam mati — tidak ada label ayam hidup.
Ayam hidup harus dicari sendiri. Ada dua pilihan, dan pilihan ini menentukan
apakah hasilnya bermakna:

- **Ambil dari dataset PIO** (CCTV tampak-atas): domainnya beda jauh dari foto
  ayam mati yang close-up. Classifier bisa mencapai akurasi ~100% hanya dengan
  menebak *"ini foto dari dataset mana"* — dan skor itu tidak membuktikan
  apa pun soal kemampuan mengenali ayam mati.
- **Ambil dari foto yang sama** ← dipakai. Kamera, kandang, dan pencahayaan
  identik, jadi satu-satunya pembeda yang tersisa adalah kondisi ayamnya.

Opsi PIO tetap tersedia lewat `--alive-source pio` untuk keperluan ablasi.

### 3. Detektor ayam hidup bukan YOLO PIO, tapi YOLOv8m COCO

YOLO PIO dilatih pada CCTV tampak-atas. Diuji pada foto close-up dataset ayam
mati, ia gagal total — terverifikasi dengan mengukurnya:

| Detektor | Deteksi | Yang mengenai ayam mati | Median ukuran crop |
|---|---|---|---|
| YOLO PIO | 1889 | **0** | ~6 px (tekstur sekam) |
| YOLOv8m COCO (kelas `bird`) | 93 | **50 dari 98** | ~84 px |

Tidak ada nilai ambang yang bisa menyelamatkan YOLO PIO di sini: crop terbesar
yang bisa dihasilkannya hanya ~42 px, sementara ayam mati median 238 px —
classifier akan belajar *"besar = mati, kecil = hidup"*. YOLOv8m COCO dipakai
karena terbukti benar-benar mendeteksi unggas, bukan tekstur lantai.

### 4. Split per gambar asal, bukan per crop

Dataset ini hasil augmentasi Roboflow — satu foto asli muncul sampai 3× dengan
nama file berbeda. Kalau displit per crop, versi augmentasi dari foto yang sama
bocor ke train dan test sekaligus, dan skornya jadi palsu. `build_crops.py`
memetakan nama file kembali ke foto aslinya, membagi split di level itu, lalu
memverifikasi ulang (`Cek kebocoran antar split: AMAN`).

### 5. Augmentasi tidak memakai rotasi

Orientasi justru sinyal utamanya: ayam mati tergeletak/terlentang, ayam hidup
berdiri tegak. Rotasi 90°/180° akan menghapus sinyal itu. Yang dipakai hanya
flip horizontal, variasi skala, warna, dan blur.

---

## Struktur

```
configs/
  config.yaml           semua pengaturan (pipeline utama)
  config_pio.yaml       ablasi: ayam hidup dari PIO          -> runs_pio_alive
  config_pio_eq.yaml    ablasi: PIO + resolusi disamakan 48px -> runs_pio_eq
  config_closeup_ckpt.yaml  acuan close-up 3 seed          -> runs_closeup_ckpt
src/
  common.py             config, seed, letterbox 224x224, utilitas bbox
  detect.py             [1] deteksi YOLO
  build_crops.py        [2] siapkan crop berlabel hidup/mati
  dataset.py            augmentasi, TwoViewDataset untuk contrastive
  models.py             ResNet18 + projection head, 3 fungsi loss
  train.py              [3] latih & evaluasi ketiga metode
  report.py             tabel + grafik perbandingan
  pipeline.py           inferensi ujung-ke-ujung
  --- alat ukur kejujuran (tidak ikut pipeline) ---
  eval_detect_masks.py      recall deteksi vs mask chick, + --sweep imgsz
  eval_on_chick.py          classifier ujung-ke-ujung di chick (AUC)
  eval_shortcut_baseline.py jalan pintas yang TERSEDIA (tanpa model)
  eval_intervensi.py        jalan pintas yang DIPAKAI (uji sebab)
data/
  crops/                crop hasil build_crops.py + manifest.csv
  crops_pio/            ablasi: ayam hidup dari PIO
  crops_pio_eq/         ablasi: sama, resolusi disamakan
outputs/
  runs/<metode>/        bobot model, metrik, log
  runs_closeup_ckpt/    9 run acuan close-up 3 seed
  runs_pio_alive/       9 run ablasi PIO
  runs_pio_eq/          9 run ablasi PIO + resolusi disamakan
  reports/              kronologi_lengkap.md  <- cerita utuh, mulai di sini
                        comparison.{csv,md,png}, lembar review crop,
                        detection_chick.md, pio_saja.md
  predictions/          hasil deteksi & pipeline (JSON + visual)
```

Tiga skrip `eval_*` itu tidak mengubah apa pun - tugasnya memeriksa apakah
angka yang keluar dari `train.py` benar-benar berasal dari mengenali ayam mati.
Jalankan `eval_shortcut_baseline.py` lebih dulu setiap kali susunan datanya
berubah; kalau baseline tanpa model sudah tinggi, skor classifier-nya belum
berarti apa pun.

---

## Batasan yang perlu diketahui

- **Dataset sangat kecil**: 130 crop (98 mati, 32 hidup), dari 36 foto asli.
  Test set 33 crop — satu crop salah menggeser metrik ~3%. Selisih antar metode
  di tabel di atas **belum tentu bermakna secara statistik**.
  Karena itu sweep terkini dijalankan dengan **5 seed** dan dilaporkan sebagai
  mean ± simpangan baku; simpangan bakunya keluar di kisaran 0.10–0.16 bal.acc,
  yang **lebih besar daripada hampir semua selisih antar metode**. Itu bukan
  cacat pengukuran — itu ukuran sebenarnya dari seberapa sedikit data ini.
  Cross-validation masih akan menolong lebih jauh.
- **Domain classifier ≠ domain data uji deteksi.** Classifier dilatih pada foto
  close-up; data uji deteksi (`patnet-pure`) adalah CCTV tampak-atas. Jadi label
  MATI yang muncul di `pipeline.py` **belum tervalidasi** untuk domain itu —
  pipeline-nya berjalan benar, tapi angkanya perlu dianotasi manual dulu
  sebelum dipercaya.
- **Anotasi ayam mati tidak lengkap**: banyak foto memuat beberapa ayam mati
  tapi hanya sebagian yang dilabeli. Karena itu `alive_max_iou_with_dead`
  disetel sangat ketat (0.01) — deteksi yang menyentuh bbox mati sedikit pun
  dibuang, agar ayam mati tak berlabel tidak salah masuk ke kelas "hidup".
  Sisa risiko tetap ada.
- Hanya ditemukan **satu** sumber data ayam mati di `MASSA AYAM`. Folder
  `broiler_healthy_sick` isinya hanya file JSON ringkasan (tanpa gambar), dan
  `repro/data/merged` adalah duplikat dari `dead-chikens`.

## Langkah lanjutan yang disarankan

1. Tambah data ayam mati, terutama dari sudut pandang CCTV tampak-atas.
2. Anotasi sebagian hasil `pipeline.py` di domain `patnet-pure` untuk mengukur
   akurasi sebenarnya di sana.
3. **Setarakan resolusi crop** (`crops.equalize_resolution`, kini masih mati).
   Ini satu-satunya cara benar-benar membuang jalan pintas ketajaman;
   konsekuensinya `data/crops/` harus dibangun ulang dan semua angka di atas
   jadi tidak sebanding lagi.
4. Perbaiki tahap deteksi — masih banyak ayam yang belum terdeteksi, dan itu
   masalah terpisah dari klasifikasi di halaman ini.
