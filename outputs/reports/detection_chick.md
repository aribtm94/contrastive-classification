# Deteksi ayam mati pada dataset `chick` - diagnosis dan saran

Ringkas: **ayam mati sudah terkena bounding box, 22 dari 22 (100%).** Yang
membuatnya terlihat "sama sekali tidak terdeteksi" adalah satu baris di
`configs/config.yaml`, bukan kemampuan detektornya. Masalah yang sebenarnya
ada di tahap sesudahnya - classifier - dan itu dijelaskan di bagian 4.

Semua angka di bawah bisa dihitung ulang:

```
python src/eval_detect_masks.py --sweep
```

---

## 1. Data acuan: 18 gambar, 22 ayam mati

`C:/Arib/CCTV/patnet-pure/dataset/chick` berisi 18 gambar dengan mask
segmentasi biner (putih = ayam mati). Mask diubah jadi bounding box acuan
lewat connected component.

| | |
|---|---|
| gambar | 18 (16x 432x432, 2x 400x400) |
| objek berlabel | 22 (1.2 per gambar) |
| luas komponen | 1357 - 2607 px |
| sisi pendek bbox | 46 - 66 px, median **57** |
| di bawah 32 px | **0%** |

Jadi ayam matinya **bukan objek kecil**. Dugaan "tidak terdeteksi karena
terlalu kecil" tidak didukung datanya.

---

## 2. Penyebabnya: `imgsz`, dan arahnya terbalik

`configs/config.yaml` memakai `imgsz: 960` dengan komentar "sama seperti saat
training". Niatnya masuk akal, tapi justru itulah yang mematikan deteksinya:

| imgsz | recall (IoU>=0.5) | IoU rata-rata | det/gambar |
|---|---|---|---|
| 256 | **100%** | 0.769 | 66 |
| 320 | **100%** | 0.777 | 69 |
| 416 | **100%** | 0.784 | 70 |
| 512 | **100%** | 0.790 | 68 |
| **640** | **100%** | **0.797** | 68 |
| 768 | 95% | 0.716 | 64 |
| **960** (config lama) | **36%** | 0.325 | 36 |
| 1280 | **0%** | 0.032 | 12 |

Kenapa memperbesar `imgsz` malah merusak:

- YOLO PIO dilatih pada gambar **1920x1080**, dengan sisi pendek objek median
  **33 px**. Pada `imgsz 960` gambar itu dikecilkan 2x, jadi jaringan belajar
  mengenali ayam berukuran **~17 px**.
- Gambar `chick` cuma **432x432**. Pada `imgsz 960` gambar ini
  **diperbesar 2.2x**, sehingga tiap ayam menjadi **~114 px** di input
  jaringan - sekitar **7x** lebih besar daripada yang pernah dilihat saat
  latihan.
- `args.yaml` mencatat `multi_scale: False` dan `scale: 0.5`. Model memang
  tidak pernah dilatih menghadapi objek sebesar itu, jadi objeknya jatuh di
  luar rentang skala yang dipelajari.

Dikonfirmasi lewat uji terpisah - gambar diperbesar dulu, lalu `imgsz`
disamakan dengan ukuran gambar supaya YOLO tidak menskala lagi:

| skala gambar | sisi objek di input | recall |
|---|---|---|
| 0.50 | 28 px | 100% |
| 0.75 | 43 px | 100% |
| 1.00 | 57 px | 100% |
| 1.50 | 85 px | 100% |
| 2.00 | 114 px | 41% |
| 3.00 | 171 px | **0%** |

Yang menentukan adalah **ukuran objek dalam piksel di input jaringan**, bukan
angka `imgsz`-nya sendiri. Runtuhnya mulai di sekitar 100 px - persis di mana
`imgsz 960` menempatkan ayam-ayam ini.

![960 vs 640](chick_imgsz_960_vs_640.jpg)

Kotak hijau = ayam mati tertangkap, merah = lolos, abu-abu = deteksi lain.

**Sudah diperbaiki:** `detection.imgsz` diubah `960 -> 640` di
`configs/config.yaml`, lengkap dengan tabel pengukurannya di komentar.

---

## 3. Mutu box-nya: cukup untuk classifier

Pada imgsz 640, 22/22 ayam mati tertangkap dengan:

| | |
|---|---|
| IoU | min 0.63, median 0.84, maks 0.95 |
| confidence | min 0.72, rata-rata **0.835** |
| piksel mask yang masuk crop (pad 0.10) | rata-rata **98.6%** |

Menariknya, confidence ayam mati (0.835) **lebih tinggi** daripada rata-rata
seluruh deteksi (0.737). Detektor sama sekali tidak ragu - ayam mati tetap
berbentuk ayam.

![crop hasil deteksi](chick_det_crops.jpg)

Soal padding crop: `bbox_padding: 0.10` sudah memasukkan 98.6% badan ayam.
Menaikkannya ke 0.25 memang membuat cakupan hampir 100%, tapi porsi ayam di
dalam crop turun dari 0.44 ke 0.29 - crop jadi lebih banyak berisi sekam dan
ayam tetangga. Nilainya dibiarkan 0.10 karena dipakai bersama oleh crop latih
dan crop inferensi; mengubahnya berarti membangun ulang `data/crops/`.

Sembilan bobot yang ada di `MASSA AYAM` diuji semua pada imgsz 640:

| bobot | recall | IoU | conf |
|---|---|---|---|
| **cmp_yolov8m** (dipakai) | **100%** | **0.797** | 0.83 |
| ft_pad015_yolov8m | 100% | 0.795 | 0.79 |
| ft_rectified_yolov8m | 100% | 0.786 | 0.81 |
| ft_radial_yolov8m | 100% | 0.782 | 0.81 |
| cmp_yolo11m | 100% | 0.781 | 0.83 |
| cmp_yolov9m | 100% | 0.710 | 0.75 |
| runs_pio/yolov10m | 95% | 0.697 | 0.69 |
| cmp_yolo12m | 95% | 0.639 | 0.72 |
| cmp_yolov10m | 82% | 0.597 | 0.47 |

Bobot yang sudah dipakai memang yang terbaik. **Tidak perlu melatih detektor
baru.**

---

## 4. Masalah yang sesungguhnya ada di classifier

Pipeline dijalankan utuh - deteksi, crop, lalu klasifikasi - dan diperiksa
apakah classifier memberi skor "mati" lebih tinggi pada 22 crop ayam mati
dibandingkan 1193 crop ayam lain dari gambar yang sama:

| checkpoint | p(mati) ayam mati | p(mati) ayam lain | AUC (seed 42) | **AUC 3 seed** |
|---|---|---|---|---|
| ce + hier_addone | 0.082 | 0.031 | 0.522 | **0.616 +/- 0.067** |
| selfcon + simclr | 0.333 | 0.265 | 0.569 | **0.675 +/- 0.061** |
| supcon + stacked_randaug | 0.262 | 0.201 | 0.606 | **0.662 +/- 0.038** |

AUC 0.5 berarti menebak. Jadi **classifier-nya yang belum bisa memisahkan**,
bukan detektornya.

Kolom 3-seed ditambahkan kemudian, dan perlu dibaca hati-hati: dengan 22 ayam
mati di test, satu seed bisa bergeser 0.14 poin (ce: 0.522 / 0.661 / 0.665).
Angka yang layak dikutip adalah kolom 3 seed. Nilainya tetap 0.62-0.68 -
masih jauh dari berguna - tapi tidak serendah yang sempat tercatat.

Ini konsisten dengan hasil klasifikasi sebelumnya: tidak ada konfigurasi
5-seed yang melewati lantai bacc 92.31%.

Penyebabnya kelihatan begitu crop latih dan crop uji disandingkan:

![jurang domain](domain_gap.jpg)

Crop latih semuanya **foto close-up** (`domain: closeup`, 130 crop dari 36
gambar). Crop uji adalah **CCTV tampak atas** dengan resolusi jauh lebih
rendah. Kamera, jarak, sudut, dan pencahayaannya berbeda total - classifier
dilatih di satu dunia lalu diminta bekerja di dunia lain.

Tambahan: hanya **130 crop latih** (98 mati / 32 hidup) dari 36 gambar. Itu
jauh terlalu sedikit untuk perbedaan sehalus mati-vs-hidup.

### Dan yang lebih mendasar: classifier tidak melihat bentuk ayam

Diuji dengan merusak satu ciri sekaligus lalu melihat AUC bergerak atau tidak
(`src/eval_intervensi.py`). Perlakuan `acak16` memecah crop jadi 4x4 petak lalu
mengacaknya - bentuk dan pose ayam hancur, tekstur serta warna lokal utuh:

| model (dilatih close-up) | AUC asli | **AUC bentuk diacak** | AUC dikaburkan |
|---|---|---|---|
| ce + hier_addone | 0.934 | **0.940** | 0.396 |
| selfcon + simclr | 0.967 | **0.995** | 0.769 |
| supcon + stacked_randaug | 0.890 | **0.918** | 0.269 |

![bentuk diacak](acak_bentuk.jpg)

Bentuk ayamnya dihancurkan, skornya tidak turun - bahkan naik. Sebaliknya,
dikaburkan sedikit saja skornya runtuh. Jadi yang dipakai classifier adalah
**ketajaman dan statistik tekstur**, bukan pose ayam - padahal mati-vs-hidup
justru soal pose.

Ini menjelaskan kenapa AUC di chick cuma 0.62-0.68 walau test bacc tinggi: di
chick semua crop berasal dari satu kamera, jadi ketajaman tidak lagi memisahkan
apa pun, dan tidak ada kemampuan lain yang tersisa. Rinciannya -- termasuk
kontrol sampai petak 14 px dan pembuktian bahwa warna/hue **bukan**
penyebabnya -- di `outputs/reports/pio_saja.md` bagian 5.

---

## 5. "Mendeteksi itu ayam tanpa menyentuhnya saat training"

Begini cara kerjanya sekarang, dan memang sudah sesuai:

1. **Detektor tidak pernah melihat mask `chick`.** Ia dilatih di dataset PIO
   dengan satu kelas saja (`Pollo` = ayam), tanpa label mati/hidup. Ia
   mengurung ayam mati **karena bentuknya ayam**, bukan karena pernah diajari
   bahwa itu ayam mati. Inilah yang terbukti di bagian 3.
2. **22 mask itu dipakai hanya untuk mengukur, tidak pernah untuk melatih.**
   Kalau ikut dilatih, angka 100% berhenti jadi ukuran dan berubah jadi
   cermin. `src/eval_detect_masks.py` sengaja dibuat hanya membaca.
3. **Pemisahan mati-vs-hidup adalah tugas classifier kontrastif**, bukan
   detektor. Detektor satu kelas secara struktural tidak bisa menjawab itu.

Jadi pembagian tugasnya sudah tepat. Yang timpang adalah data latih
classifier-nya.

---

## 6. Saran, diurutkan menurut dampaknya

### Sudah dikerjakan
- **`detection.imgsz` 960 -> 640.** Recall 36% -> 100%. Satu baris.
- **`src/eval_detect_masks.py`** - evaluasi berulang terhadap mask, dengan
  opsi `--sweep` dan `--vis`.

### Dampak terbesar berikutnya: perbaiki data latih classifier, bukan detektor

> **Catatan revisi.** Versi pertama bagian ini menyarankan melabeli 1215 crop
> hasil deteksi dari 18 gambar `chick`. Saran itu **dicabut** atas permintaan.
> Gantinya sudah diuji: melatih dengan ayam hidup dari **PIO** - hasilnya di
> `outputs/reports/pio_saja.md`, dan kesimpulannya angkanya menyesatkan.

1. **Ayam hidup dari PIO saja TIDAK cukup, dan angkanya menipu.** Sudah
   dikerjakan dan diukur: `data/crops_pio` (294 hidup PIO + 98 mati close-up),
   9 run. Test bacc sampai **1.0000** - tapi itu palsu. PIO satu kelas
   (`nc: 1, names: ['Pollo']`), jadi ia cuma bisa menyumbang ayam **hidup**;
   ayam mati tetap harus datang dari close-up. Akibatnya label = domain, dan
   **ketajaman gambar saja sudah memberi test bacc 0.8434 / AUC 0.8596**
   tanpa melihat isi gambar sama sekali.
   Buktinya: model PIO menyebut ayam **hidup** close-up sebagai mati dengan
   p 0.813 (yang benar: rendah). Rinciannya di `outputs/reports/pio_saja.md`.

2. **Yang bisa dipakai dari PIO: samakan resolusinya dulu - tapi itu cuma
   memulihkan, bukan memperbaiki.** `crops.equalize_resolution` sekarang
   benar-benar bekerja (sebelumnya cuma stub). Dengan `target_short_side: 48`,
   jalan pintas ketajaman runtuh dari AUC 0.8596 ke 0.4194. Di `chick`,
   dibandingkan berpasangan per metode-dan-seed terhadap acuan close-up yang
   juga dilatih 3 seed: PIO saja **-0.051** (p = 0.301), PIO+eq48 **+0.023**
   (p = 0.570). Artinya PIO menurunkan skor, dan menyamakan resolusi
   mengembalikannya ke titik awal - **tidak ada kemajuan atas model yang sudah
   ada**. Ini membuat angkanya jujur, bukan membuat modelnya pintar.

3. **Perbesar jumlah dan ragam crop latih - ini akar yang baru terbukti.**
   Dengan 130-392 crop, tekstur global sudah cukup memisahkan kedua kelas,
   sehingga model **tidak pernah punya alasan** untuk belajar pose (bukti:
   bentuk diacak, AUC tidak turun - bagian 4). Menutup satu jalan pintas hanya
   memindahkannya ke ciri global lain. Yang mengubah keadaan adalah data yang
   membuat tekstur **tidak lagi cukup**: banyak crop, banyak kandang, banyak
   kamera, dengan kedua kelas hadir di tiap kondisi.

4. **Sumber ayam mati adalah batasnya yang sebenarnya.** Tidak ada satu pun
   dataset di disk yang punya kelas ayam mati di domain CCTV: PIO,
   `broiler_instance_seg`, `chicken_detection_fum`, dan `rilis_rectified_pio`
   semuanya `nc: 1`. Selama ayam mati cuma ada dalam bentuk foto close-up,
   classifier akan selalu punya pilihan membedakan kamera, bukan kondisi ayam.

5. **Jadikan 18 gambar `chick` sebagai test set resmi** yang tidak pernah
   disentuh saat latihan. Dengan 22 objek, satu crop bernilai ~4.5 poin
   recall - sempit, tapi jujur, dan jauh lebih bermakna daripada val set lama
   yang jenuh di bacc 1.0. Inilah satu-satunya angka yang tidak bisa ditipu
   oleh jalan pintas domain.

### Kalau tetap ingin memperkuat detektor

Bukan prioritas (sudah 100%), tapi kalau tujuannya tahan banting terhadap
skala:

6. **Latih ulang dengan `multi_scale=True`.** Ini menghilangkan kerapuhan
   skala di bagian 2, sehingga `imgsz` tidak lagi jadi ranjau.
7. **Campur resolusi di data latih** - ikutkan versi 432x432 dari gambar PIO,
   bukan cuma 1920x1080.
8. **Dua dataset Roboflow yang sudah ada di disk** (`broiler_instance_seg`,
   `chicken_detection_fum`) bisa menambah ragam kandang dan kamera.

### Yang sebaiknya TIDAK dilakukan

- **Jangan melatih detektor memakai 22 mask `chick`.** Terlalu sedikit, dan
  itu memusnahkan satu-satunya alat ukur yang jujur.
- **Jangan membuat detektor 2 kelas (mati/hidup).** Dengan 22 contoh,
  detektor akan menghafal, bukan belajar. Pemisahan itu memang tugas
  classifier.
- **Jangan menaikkan `imgsz` lagi.** Sudah terbukti berbanding terbalik.
- **Jangan mempercayai skor tinggi dari data yang label-nya = domain.** Susunan
  "hidup dari PIO, mati dari close-up" memberi bacc 1.0000 yang kosong. Cek
  dulu dengan `python src/eval_shortcut_baseline.py --crops <folder>` sebelum
  menganggap angka itu nyata - pada `data/crops` sendiri ketajaman-saja sudah
  memberi test AUC 0.9670, jadi ini bukan cuma soal PIO.
- **Jangan menyimpulkan "model memakai ciri X" dari korelasi saja.** rho antara
  p(mati) dan hue mencapai -0.599, dan dari situ sempat disimpulkan model
  menempel pada warna. Ternyata salah: ketika hue diputar - bahkan ketika warna
  dibuang seluruhnya - AUC tidak bergerak. Di data yang label-nya = domain,
  SEMUA ciri berkorelasi dengan label, jadi rho tidak bisa membedakan sebab dan
  penumpang. Pakai `python src/eval_intervensi.py` yang merusak satu ciri lalu
  mengukur akibatnya.
- **Jangan menguji model dengan praproses yang berbeda dari saat melatih.**
  Model yang dilatih pada crop resolusi-disamakan harus diuji lewat config yang
  sama (`--config configs/config_pio_eq.yaml`); `src/eval_on_chick.py` sekarang
  mengikuti `crops.equalize_resolution` dari config yang diberikan.

---

## Berkas

| berkas | isi |
|---|---|
| `src/eval_detect_masks.py` | evaluasi deteksi terhadap mask (baru) |
| `configs/config.yaml` | `detection.imgsz` 960 -> 640 |
| `outputs/reports/chick_imgsz_960_vs_640.jpg` | perbandingan berdampingan |
| `outputs/reports/chick_det_crops.jpg` | 22 crop ayam mati hasil deteksi |
| `outputs/reports/domain_gap.jpg` | crop latih vs crop uji |
| `outputs/reports/acak_bentuk.jpg` | bentuk diacak, skor tidak turun |
| `outputs/predictions/detect_masks_eval.json` | angka mentah per objek |
| `outputs/reports/pio_saja.md` | percobaan latih dengan PIO saja + vonisnya |
| `src/eval_on_chick.py` | uji classifier ujung-ke-ujung pada chick (baru) |
| `src/eval_shortcut_baseline.py` | skor tanpa model: ketajaman/rasio/ukuran (baru) |
| `src/eval_intervensi.py` | uji sebab: bentuk diacak, warna dibuang (baru) |
| `configs/config_pio.yaml`, `config_pio_eq.yaml` | config ablasi PIO (baru) |

---

Cerita utuh seluruh eksperimen - berurutan, dengan gambar tiap
percobaan dan alasan di balik setiap perubahan - ada di
[`kronologi_lengkap.md`](kronologi_lengkap.md).
