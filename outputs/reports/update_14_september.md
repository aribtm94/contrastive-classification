# Update 14 September 2026

Ringkasan keadaan proyek per hari ini. Berbeda dengan draf pagi tadi, seluruh
pekerjaan yang tertunda **sudah selesai dijalankan**: 18 run training penuh,
registry dibekukan, dan benchmark test chick akhirnya dibuka untuk pertama
kalinya. Dokumen ini memuat angka test yang sesungguhnya.

Commit acuan: `e1ce0b9` (kode protokol) dan `a8994ed` (perbaikan audit
benchmark).

---

## 1. Satu kalimat

Seluruh 18 run selesai, registry dibekukan, dan test dibuka: hasil terbaik pada
benchmark chick adalah **SupCon pada varian eq48, AP 0.386 / AUC 0.792**, masih
kalah dari baseline saturasi tanpa model yang mencapai AUC 0.883 - jadi
pipeline ini belum terbukti mengenali kondisi ayam.

---

## 2. Perubahan arah: batas data dari dosen (13 Sept)

Rencana lama - melatih classifier langsung pada 943 crop chick dengan
leave-one-frame-out - **dibatalkan**. Batas data yang sekarang berlaku:

| peran | sumber | domain |
|---|---|---|
| train + validation, ayam hidup | PIO (`pio_gt`) | CCTV |
| train + validation, ayam mati | Roboflow `dead-chikens` (`coco_gt`) | close-up |
| test | seluruh 18 frame `dataset/chick` (11 `ayam (...)` + 7 `chick (...)`) | CCTV |

Crop, label manual, dan mask chick **tidak menyentuh** training, validation,
early stopping, threshold, atau pemilihan model. Split internal lama yang
bernama `test` dihapus perannya dan barisnya digabung ke validation.

Permintaan kedua dari dosen: **train loss dan validation loss dicatat pada
setiap epoch** - sudah berjalan, lihat bagian 4.

Perumpamaan untuk dosen: dulu kita ikut menyusun soal ujiannya sendiri, sekarang
soal ujian dikunci di laci; kunci lacinya baru dibuka hari ini, setelah semua
jawaban dikumpulkan.

---

## 3. Yang dijalankan hari ini

### 3.1 Delapan belas run training penuh

```
sh run_dev_full.sh > outputs/logs/dev_full.log 2>&1
```

| tahap | mulai | selesai | durasi |
|---|---|---|---|
| 9 run PIO asli | 11:52:19 | 12:25:22 | 33 menit (1972 detik komputasi) |
| 9 run PIO eq48 | 12:25:22 | 12:57:52 | 32 menit (1942 detik komputasi) |

2 varian x 3 pipeline diagonal x 3 seed (42, 43, 44) = 18 run, exit code 0,
tanpa satu pun error. Tahap contrastive selalu berjalan penuh 60 epoch; yang
berhenti lebih awal hanya linear probe dan CE (`early_stop_patience: 15`).

### 3.2 Registry dibekukan

`write_development_bundle` menolak membuat registry final sebelum ke-9 pasang
(metode, seed) lengkap. Karena sekarang lengkap, keduanya terbentuk:

- `outputs/predictions/pio_dev_registry.json` - 9 checkpoint
- `outputs/predictions/pio_eq_dev_registry.json` - 9 checkpoint

Isinya: `test_policy: fixed_retrospective_chick_test_only`, SHA-256 tiap
checkpoint, SHA-256 config/manifest/split-lock/shortcut, hash 10 berkas sumber,
threshold validation per run, plus runtime (Python 3.10.6, torch 2.6.0+cu124,
RTX 4060 Laptop) dan commit Git.

### 3.3 Test dibuka

```
python src/eval_fixed_chick.py --registries \
    outputs/predictions/pio_dev_registry.json,outputs/predictions/pio_eq_dev_registry.json
python src/report_fixed_chick.py
```

Evaluator memverifikasi ulang hash tiap checkpoint dan hash config sebelum
membaca satu pun crop chick. 18 checkpoint x 2 intervensi = 36 evaluasi,
1215 crop per evaluasi.

### 3.4 Dua cacat pelaporan yang ditemukan dan diperbaiki

Sebelum menulis angka ini, dua bug di lapisan pelaporan ditemukan dan
diperbaiki pada commit `a8994ed`:

1. `eval_fixed_chick.py` mengambil nilai **kedua** dari `ranks_within_frame`,
   yaitu persentil, lalu memakainya seolah peringkat. Karena persentil selalu
   di bawah 1, syarat `rank <= k` selalu benar dan audit crop `bukan`
   melaporkan **272/272/272** di seluruh baris - angka yang mustahil, sebab
   top-1 pada 18 frame paling banyak memuat 18 crop.
2. `report_fixed_chick.py` menulis judul "Sensitivitas acak16" beserta
   peringatannya, tetapi tidak pernah mengeluarkan tabelnya - gerbang kausal
   tidak menghasilkan satu angka pun yang terbaca.

Tabel metrik utama **tidak terpengaruh**, karena `threshold_free` menghitung
peringkatnya sendiri dengan benar. Ini dipastikan dengan menjalankan ulang
seluruh benchmark setelah perbaikan: tabel utama terbit identik angka demi
angka, hanya kolom audit yang berubah. Registry dibekukan ulang lebih dulu agar
`git_commit` dan `source_sha256` menunjuk ke kode yang diperbaiki.

### 3.5 Uji regresi

`python tests/test_protocol.py -v` - **7/7 lolos**, dijalankan ulang setelah
perbaikan.

---

## 4. Loss per epoch - permintaan dosen

Tercatat tiap epoch untuk setiap tahap: train loss, validation loss, loss dasar,
aux CE, learning rate yang dipakai, learning rate sesudah scheduler, jumlah
unit, jumlah batch, dan seluruh metrik validation. Total 563 baris untuk PIO
asli dan 559 baris untuk eq48
(`outputs/reports/pio_dev_history.csv`, `pio_eq_dev_history.csv`; grafik pada
`pio_dev_loss.png`, `pio_eq_dev_loss.png`).

Validation contrastive memakai **dua view deterministik yang tetap antar-epoch**
(`validation_seed_offset: 900001`), jadi kurva yang bergerak berasal dari
perubahan bobot, bukan dari augmentasi yang mengacak sendiri.

Ringkasan kurva, rata-rata tiga seed (awal -> akhir):

| varian | tahap | train loss | validation loss |
|---|---|---|---|
| asli | selfcon contrastive (60 ep) | 3.17 -> 0.24 | 3.19 -> 1.19 |
| asli | supcon contrastive (60 ep) | 4.09 -> 3.54 | 3.81 -> 3.58 |
| asli | CE (20-26 ep) | 0.44 -> 0.04 | 0.11 -> 0.01 |
| eq48 | selfcon contrastive (60 ep) | 3.64 -> 0.33 | 3.73 -> 1.34 |
| eq48 | supcon contrastive (60 ep) | 4.23 -> 3.55 | 4.05 -> 3.62 |
| eq48 | CE (20-22 ep) | 0.57 -> 0.12 | 0.38 -> 0.02 |

Dua hal yang perlu dibaca jujur di sini:

- **SelfCon turun bersama** - train dan validation NT-Xent sama-sama turun jauh,
  tidak ada divergensi liar. Kurvanya sehat.
- **SupCon nyaris tidak bergerak** (4.09 -> 3.54 dalam 60 epoch). Loss SupCon
  memang tidak menuju nol karena normalisasi atas banyak positif, tetapi
  datarnya kurva ini berarti tahap contrastive-nya hanya memberi sedikit
  tambahan; yang benar-benar memisahkan kelas adalah probe linier sesudahnya
  (loss probe turun ke 0.0005-0.0014). Ini perlu diperiksa lebih lanjut, bukan
  diklaim sebagai keberhasilan contrastive.

---

## 5. Validation: jenuh, seperti yang sudah diduga

Seluruh 9 run PIO asli mencapai **AUC validation 1.0000**, dengan balanced
accuracy 1.0000 pada 6 dari 9 run. Checkpoint terpilih sering pada epoch sangat
awal - SupCon pada epoch 1, 1, dan 2.

| varian | metode | epoch terpilih (s42/s43/s44) | val bacc | val AUC |
|---|---|---|---|---|
| asli | ce | 11 / 10 / 5 | 1.0000 semua | 1.0000 |
| asli | selfcon | 10 / 3 / 36 | 1.0000 / 1.0000 / 0.9913 | 1.0000 |
| asli | supcon | 1 / 1 / 2 | 1.0000 / 0.9957 / 0.9957 | 1.0000 |
| eq48 | ce | 5 / 7 / 7 | 1.0000 semua | 1.0000 |
| eq48 | selfcon | 24 / 8 / 6 | 0.9120 / 0.9245 / 0.8940 | 0.9930 / 0.9898 / 0.9674 |
| eq48 | supcon | 2 / 4 / 1 | 1.0000 / 1.0000 / 0.9625 | 1.0000 / 1.0000 / 0.9996 |

Satu-satunya baris yang **tidak** jenuh adalah selfcon pada eq48 (bacc
0.894-0.925). Itu konsisten: equalization memotong jalan pintas ketajaman
(0.878 -> 0.565), dan metode tanpa label paling terpukul, sementara SupCon yang
punya label - dan label itu identik dengan domain - tetap 1.0000.

### Jalan pintas tanpa model (baseline wajib)

Seberapa jauh orang bisa menebak label validation **tanpa model sama sekali**,
hanya dari satu angka statistik gambar:

| ciri | PIO asli - val AUC | PIO eq48 - val AUC |
|---|---:|---:|
| ukuran bbox (metadata) | **0.985** | **0.985** |
| hue | 0.913 | 0.922 |
| ketajaman | 0.878 | 0.565 |
| saturasi | 0.766 | 0.765 |
| std terang | 0.641 | 0.665 |
| rasio bbox | 0.594 | 0.594 |
| terang | 0.517 | 0.516 |

Artinya val AUC 1.0000 itu **bukan bukti** apa pun tentang mengenali ayam mati.
Perumpamaannya: nilai ujian 100, tetapi semua soal ganjil jawabannya A dan yang
genap B - siswa bisa dapat 100 tanpa membaca soalnya.

---

## 6. HASIL TEST - benchmark chick, pertama kali dibuka

Kohort: 943 ayam valid dari 18 frame (22 mati, 921 hidup), ditambah 272 crop
`bukan ayam` yang dipakai sebagai nuisance pada ranking operational. Rata-rata
tiga seed.

### 6.1 Scorer absolut (keluaran classifier)

| varian | metode | AP pooled | AUC pooled | Recall@3 | MRR |
|---|---|---:|---:|---:|---:|
| asli | selfcon | 0.024 +/- 0.006 | 0.493 +/- 0.101 | 0.037 | 0.080 |
| asli | ce | 0.039 +/- 0.005 | 0.591 +/- 0.037 | 0.093 | 0.149 |
| asli | supcon | 0.091 +/- 0.026 | 0.676 +/- 0.036 | 0.343 | 0.312 |
| eq48 | selfcon | 0.033 +/- 0.011 | 0.581 +/- 0.120 | 0.093 | 0.147 |
| eq48 | ce | 0.165 +/- 0.086 | 0.744 +/- 0.067 | 0.361 | 0.346 |
| **eq48** | **supcon** | **0.386 +/- 0.063** | **0.792 +/- 0.073** | **0.519** | **0.536** |

### 6.2 Skor relatif satu-frame (anomali terhadap tetangga sendiri)

| varian | metode | scorer | AP pooled | AUC pooled |
|---|---|---|---:|---:|
| asli | supcon | relative_operational | 0.050 | 0.658 |
| eq48 | ce | relative_operational | 0.107 | 0.684 |
| eq48 | supcon | relative_clean | 0.216 | 0.661 |
| eq48 | supcon | relative_operational | 0.223 | 0.687 |

Pada benchmark ini skor **absolut mengalahkan skor relatif** untuk kombinasi
terbaik (AP 0.386 vs 0.223). Ini kebalikan dari harapan awal bahwa perbandingan
antar-ayam dalam satu frame akan menetralkan perbedaan kamera.

### 6.3 Keputusan pada ambang dari validation

Ambang `tau_val` dihitung murni dari validation, tidak pernah dari chick:

| varian | metode | bacc | recall mati | spesifisitas | presisi mati | TP dari 22 |
|---|---|---:|---:|---:|---:|---:|
| asli | selfcon | 0.492 | 0.000 | 0.984 | 0.000 | 0.0 |
| asli | ce | 0.528 | 0.121 | 0.935 | 0.039 | 2.7 |
| asli | supcon | 0.533 | 0.091 | 0.975 | 0.032 | 2.0 |
| eq48 | selfcon | 0.527 | 0.212 | 0.841 | 0.032 | 4.7 |
| eq48 | ce | 0.648 | 0.333 | 0.963 | 0.203 | 7.3 |
| eq48 | supcon | 0.723 | 0.576 | 0.870 | 0.105 | 12.7 |

Yang terbaik menangkap 12.7 dari 22 ayam mati, tetapi menukarnya dengan 120
alarm palsu per rata-rata seed. Presisi 0.105 pada tabel adalah rata-rata
presisi tiga seed; dihitung dari rata-rata TP dan FP nilainya 0.096. Dua-duanya
berarti hal yang sama: **9 dari 10 alarm keliru**. Sebagai alat operasional,
ini belum layak pakai.

### 6.4 Audit crop bukan ayam

Berapa dari 272 crop bukan-ayam yang ikut memicu alarm atau masuk peringkat
teratas (angka top-k sesudah perbaikan bug di 3.4):

| varian | metode | >0.5 | >tau val | top-1 | top-3 | top-5 |
|---|---|---:|---:|---:|---:|---:|
| asli | ce | 11.0 | 8.3 | 17.3 | 47.3 | 74.0 |
| asli | selfcon | 13.0 | 2.3 | 16.0 | 46.0 | 73.3 |
| asli | supcon | 6.0 | 3.7 | 4.3 | 15.7 | 25.7 |
| eq48 | ce | 53.0 | 44.3 | 17.0 | 49.7 | 76.7 |
| eq48 | selfcon | 27.7 | 68.7 | 17.0 | 48.0 | 74.3 |
| eq48 | supcon | 22.7 | 49.0 | 5.7 | 23.3 | 38.3 |

Batas teoretis top-1 adalah 18 (satu per frame). Jadi pada `ce` dan `selfcon`,
**hampir setiap frame menempatkan objek bukan-ayam di peringkat 1** - ranking
relatifnya sebagian besar menunjuk bukan ke ayam. SupCon jauh lebih bersih
(4.3 dan 5.7 dari 18), dan itu satu-satunya sisi di mana ia unggul konsisten.

### 6.5 Gerbang kausal: asli vs acak16

`acak16` mengacak susunan 4x4 petak (derangement, tanpa petak tetap, tanpa
strip tersisa). Kalau skor bertahan, berarti model tidak membaca susunan global.

| varian | metode | AUC asli | AUC acak16 | delta |
|---|---|---:|---:|---:|
| asli | ce | 0.591 | 0.645 | **+0.054** |
| asli | selfcon | 0.493 | 0.598 | **+0.105** |
| asli | supcon | 0.676 | 0.661 | -0.014 |
| eq48 | ce | 0.744 | 0.721 | -0.024 |
| eq48 | selfcon | 0.581 | 0.673 | **+0.093** |
| eq48 | supcon | 0.792 | 0.769 | -0.022 |

Hasilnya sangat merugikan klaim "model membaca pose ayam". Merusak susunan
petak **tidak menurunkan** AUC; pada tiga dari enam kombinasi malah
**menaikkannya**. Model yang benar-benar membaca bentuk dan posisi tubuh ayam
seharusnya rusak. Ini mengkonfirmasi ulang temuan babak 10: classifier membaca
tekstur, bukan bentuk.

Satu pengecualian yang menarik: pada skor relatif eq48+supcon, AP jatuh dari
0.216 ke 0.046. Jadi komponen relatifnya lebih peka pada susunan daripada
komponen absolutnya - tetapi komponen relatif itu justru yang hasilnya lebih
rendah.

### 6.6 Baseline nuisance pada test

Diukur langsung pada crop chick, tanpa model sama sekali:

| ciri | AUC pooled | AP pooled | Recall@3 |
|---|---:|---:|---:|
| saturasi | **0.883** | 0.299 | 0.389 |
| bbox sisi pendek | 0.786 | 0.057 | 0.167 |
| luas bbox | 0.724 | 0.043 | 0.167 |
| kepercayaan detektor | 0.540 | 0.024 | 0.000 |
| ketajaman | 0.536 | 0.032 | 0.111 |

**Ini angka yang paling penting di seluruh dokumen.** Saturasi saja - satu angka
rata-rata warna, tanpa model, tanpa training, tanpa GPU - mencapai AUC 0.883
pada test. Model terbaik hasil 18 run mencapai 0.792. **Tidak satu pun dari 18
checkpoint mengalahkan baseline satu-angka itu pada AUC.**

Hanya pada AP dan Recall@3 model terbaik unggul (AP 0.386 vs 0.299; Recall@3
0.519 vs 0.389), artinya ia lebih baik menaruh ayam mati di peringkat paling
atas, walau urutan keseluruhannya lebih buruk.

---

## 7. Apa yang boleh dan tidak boleh disimpulkan

**Boleh:**

1. Pipeline lengkap berjalan end-to-end di bawah protokol yang ketat: batas data
   dosen dipatuhi, loss tercatat tiap epoch, 18 run reproducible, registry
   ter-hash, test dibuka sekali saja tanpa seleksi apa pun dari hasilnya.
2. Ada urutan yang konsisten antar-pipeline: supcon > ce > selfcon, dan eq48 >
   asli. Urutan ini sama pada AP, AUC, Recall@3, dan MRR.
3. Equalization 48 px membantu di test, bukan hanya di validation: AP eq48
   supcon 0.386 vs asli 0.091, naik lebih dari empat kali lipat.

**Tidak boleh:**

1. **Tidak boleh** diklaim model mengenali ayam mati. Baseline saturasi tanpa
   model mengalahkan seluruh 18 checkpoint pada AUC.
2. **Tidak boleh** dikreditkan ke fungsi loss. Pada tiap diagonal, loss dan
   augmentasi berubah bersamaan - ini perbandingan pipeline, bukan atribusi
   kausal.
3. **Tidak boleh** dibaca sebagai bukti membaca pose. Gerbang acak16 justru
   menaikkan AUC pada tiga kombinasi.
4. **Tidak boleh** disebut hasil konfirmatori. Chick adalah benchmark tetap
   **retrospektif** - dataset itu sudah pernah dibaca pada eksperimen historis
   proyek ini. Protokol sekarang mencegah kebocoran ke depan, tetapi tidak bisa
   menghapus ingatan eksperimen yang sudah terjadi.
5. **Tidak boleh** dianggap cukup datanya. Hanya 98 crop mati di development dan
   22 ayam mati di test. Satu ayam mati bernilai 4.5 poin recall.

---

## 8. Langkah berikutnya yang masuk akal

1. **Kejar baseline saturasi dulu.** Sebelum menambah metode, model harus bisa
   mengalahkan AUC 0.883 dari satu angka warna. Kalau tidak bisa, menambah
   pipeline hanya menambah biaya.
2. **Periksa kurva SupCon yang datar.** Loss contrastive nyaris tidak bergerak
   dalam 60 epoch; perlu dipastikan apakah temperatur, ukuran batch, atau jumlah
   positif per anchor membuat tahap itu praktis tidak belajar.
3. **Cari sumber ayam mati kedua** dengan domain berbeda dari Roboflow. Selama
   semua ayam mati berasal dari satu domain close-up, `label = domain` tidak
   akan pernah bisa dipatahkan dari sisi data.
4. **Benchmark prospektif** - frame CCTV baru yang belum pernah disentuh proyek
   ini - untuk klaim konfirmatori.

---

## 9. Berkas terkait

| berkas | isi |
|---|---|
| [`fixed_chick.md`](fixed_chick.md) | tabel lengkap benchmark test, 18 baris x 3 scorer |
| [`pio_development_protocol.md`](pio_development_protocol.md) | protokol development + aturan pembukaan test |
| [`kronologi_lengkap.md`](kronologi_lengkap.md) | riwayat seluruh eksperimen, babak 1-14 |
| [`same_frame_stage1.md`](same_frame_stage1.md) | Tahap 1 skor relatif satu-frame pada 27 checkpoint |
| [`pio_saja.md`](pio_saja.md) | percobaan PIO, uji intervensi, vonis akhir |
| [`comparison.md`](comparison.md) | grid 3x3 x 5 seed + lantai ketajaman |
