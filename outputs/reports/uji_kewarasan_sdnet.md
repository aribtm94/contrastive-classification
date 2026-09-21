# Uji Kewarasan Pipeline di SDNET2018 (Bagian A)

Dokumen ini menjawab satu pertanyaan yang tidak bisa dijawab oleh data ayam:

> **Yang rusak itu pipelinenya, atau datanya?**

Selama ini keduanya tidak bisa dibedakan, karena setiap susunan data ayam yang
pernah dipakai punya jalan pintas. Kalau pipeline yang **persis sama** diberi
data yang bebas jalan pintas dan tetap gagal, masalahnya di kode/metode. Kalau
berhasil, masalahnya di data - dan itu mengubah arah skripsi.

Jawabannya: **pipelinenya berfungsi.** Rinciannya di bawah.

---

## Daftar isi

| # | Bagian | Isi singkat |
|---|---|---|
| [1](#1-satu-kalimat) | Satu kalimat | vonisnya |
| [2](#2-kenapa-sdnet2018-dan-bukan-yang-lain) | Kenapa SDNET2018 | properti yang membuat `label = domain` mustahil |
| [3](#3-susunan-data) | Susunan data | 8400 ubin, split per foto sumber |
| [4](#4-lantai-jalan-pintas-gerbang-wajib) | Lantai jalan pintas | AUC 0.6430 / 0.6666 / 0.6341 · bacc 0.5947 / 0.6321 / 0.5300 |
| [5](#5-hasil-in-domain-deck) | Hasil in-domain | 3 dari 3 lolos lantai |
| [6](#6-hasil-lintas-sub-domain-wall--pavement) | Hasil lintas domain | tidak runtuh |
| [7](#7-gerbang-kausal-acak16) | Gerbang kausal | kontras paling tajam dengan data ayam |
| [8](#8-perbandingan-langsung-dengan-data-ayam) | Perbandingan | tabel berdampingan |
| [9](#9-yang-boleh-dan-tidak-boleh-disimpulkan) | Batasan | apa yang belum dibuktikan |
| [10](#10-dua-koreksi-atas-laporan-yang-dihasilkan-otomatis) | Koreksi | dua kalimat yang harus dibaca ulang |

---

## 1. Satu kalimat

Pipeline yang sama - ResNet-18, letterbox 224, batch 32, ketiga metode
(`selfcon`/`supcon`/`ce`) dengan augmentasi masing-masing, tiga seed - dilatih
pada retak beton SDNET2018 **melewati lantai jalan pintasnya di ketiga
konfigurasi, tetap bertahan pada dua sub-domain yang tidak pernah dilihat saat
latih, dan rusak seperti seharusnya saat susunan gambarnya dihancurkan.**
Ketiganya adalah hal yang **tidak** terjadi pada data ayam.

---

## 2. Kenapa SDNET2018, dan bukan yang lain

Tiga alasan, diukur bukan diasumsikan.

**a. Datanya bersih dari jalan pintas.** Ciri gambar terkuat tanpa model hanya
mencapai AUC 0.63-0.67, dibanding **0.883** pada data ayam (saturasi saja).

**b. `label = domain` mustahil secara struktur.** Nama berkas SDNET memuat id
foto sumbernya (`7001-115.jpg` = ubin ke-115 dari foto `7001`), dan **setiap
foto sumber menyumbang ubin retak sekaligus ubin utuh** - terukur: irisan 54/54
deck, 72/72 wall, 99/99 pavement. Jadi "ubin ini dari foto mana" tidak memberi
petunjuk apa pun tentang labelnya. Pada data ayam justru sebaliknya: seluruh
ayam mati dari satu domain close-up, seluruh ayam hidup dari CCTV.

**c. Ada tiga sub-domain bawaan** (deck, wall, pavement), jadi eksperimen
generalisasi lintas domain bisa dilakukan tanpa mencari dataset kedua.

Konsekuensi teknis dari (b): `build_crops.balance_and_split` **tidak bisa**
dipakai ulang di sini. Fungsi itu sengaja melempar
`ValueError("satu base_image punya dua label")`, karena pada data ayam kondisi
itu memang berarti data rusak. Yang ditiru di `src/build_manifest_sdnet.py`
adalah **aturannya** (kelompokkan per foto, ambil merata antar foto), bukan
kodenya.

---

## 3. Susunan data

Latih di **deck**, uji di **wall** + **pavement** yang tidak pernah disentuh
saat latih.

| split | retak | utuh | foto sumber | peran |
|---|---:|---:|---:|---|
| `train` | 299 | 891 | 32 | latih |
| `val` | 103 | 305 | 11 | pilih epoch + ambang |
| `test` | 98 | 304 | 11 | uji in-domain (deck) |
| `test_wall` | 800 | 2400 | 72 | **hold-out penuh** |
| `test_pave` | 800 | 2400 | 104 | **hold-out penuh** |
| **total** | **2100** | **6300** | **230** | 8400 ubin |

Split dilakukan **per foto sumber**, bukan per ubin. Penjaga di
`build_manifest_sdnet.ringkas` memverifikasi ini dan melaporkan:
**`base_image` lintas split: 0**.

Yang **tidak** diubah, dengan sengaja: arsitektur, ukuran input, batch size,
jumlah epoch, kebijakan augmentasi (termasuk pool RandAugment **tanpa rotasi**
dan `lock_probe_policy: true`). Untuk retak beton rotasi sebenarnya aman dan
kemungkinan membantu - tapi yang diuji adalah **pipeline yang sama**, bukan
pipeline yang dioptimalkan ulang. Ini batasan yang disengaja: angka SDNET di
bawah ini kemungkinan **lebih rendah** dari yang bisa dicapai kalau
augmentasinya disesuaikan.

Satu hal yang membuat susunan ini murah: tanpa blok `protocol:` di
`configs/config_sdnet.yaml`, `development_only` bernilai false dan `run_method`
otomatis memakai jalur train/val/test tiga arah biasa. Jadi **tidak satu pun
penjaga khusus-ayam perlu disentuh**, dan seluruh protokol beku tetap utuh.

---

## 4. Lantai jalan pintas (gerbang wajib)

Dijalankan **sebelum** training, sesuai aturan §6 `tujuan.md`. Aturan dicocokkan
di train deck, lalu dipakai apa adanya di ketiga split uji.

**AUC** (kemampuan mengurutkan):

| ciri | `test` (deck) | `test_wall` | `test_pave` |
|---|---:|---:|---:|
| ketajaman | 0.5713 | **0.6666** | 0.5867 |
| **std_terang** | **0.6430** | 0.6158 | **0.6341** |
| hue | 0.5983 | 0.5553 | 0.5035 |
| terang | 0.5629 | 0.5168 | 0.5105 |
| saturasi | 0.5530 | 0.5203 | 0.5131 |
| rasio_bbox | 0.5000 | 0.5000 | 0.5000 |
| ukuran_bbox | 0.5000 | 0.5000 | 0.5000 |

**Balanced accuracy** (kemampuan memutuskan, setelah ambang) - angka yang
**berbeda**, dan inilah yang harus dibandingkan dengan bacc model:

| ciri | `test` (deck) | `test_wall` | `test_pave` |
|---|---:|---:|---:|
| ketajaman | 0.5467 | **0.6321** | 0.5254 |
| **std_terang** | **0.5947** | 0.5883 | **0.5300** |
| hue | 0.5807 | 0.4733 | 0.4998 |
| terang | 0.5343 | 0.5077 | 0.5102 |
| saturasi | 0.5030 | 0.5010 | 0.5021 |
| rasio_bbox / ukuran_bbox | 0.5000 | 0.5000 | 0.5000 |

> **LANTAI bacc: 0.5947 (deck) · 0.6321 (wall) · 0.5300 (pavement)**
> **LANTAI AUC: 0.6430 (deck) · 0.6666 (wall) · 0.6341 (pavement)**

Dua metrik, dua lantai - jangan dicampur. Bacc mengukur **keputusan** (sesudah
ambang), AUC mengukur **urutan**. Membandingkan bacc model dengan lantai AUC
akan menaikkan palangnya secara keliru, dan sebaliknya.

Dua hal yang perlu dibaca:

**Kedua ciri bbox tepat 0.5000 - nol informasi.** Ini bukan kebetulan: ubin
SDNET berukuran seragam dan bukan hasil deteksi, jadi metadata kotaknya memang
tidak memuat apa pun. Bandingkan dengan validation ayam, di mana **ukuran bbox
saja mencapai AUC 0.985** - artinya di sana ukuran kotak nyaris sempurna
menandai label, tanpa melihat satu piksel pun isi gambar.

**Lantai di sini bukan ketajaman.** Pada data ayam ketajaman adalah ciri
terkuat (bacc 0.9231); pada SDNET ia justru **paling lemah** di deck (0.5713)
dan yang mengikat adalah `std_terang`. Karena `src/report.py` semula memaku
lantainya ke ketajaman, laporan otomatis SDNET akan melaporkan palang 0.5713 -
lebih rendah dari yang sebenarnya. Itu diperbaiki: ciri lantai sekarang dibaca
dari `report.lantai_ciri` dan **dipilih berdasarkan skor train, tidak pernah
skor test**. Bawaannya tetap ketajaman saja, sehingga seluruh laporan ayam yang
sudah ter-commit terbit byte-identik (diverifikasi).

---

## 5. Hasil in-domain (deck)

Rata-rata tiga seed (42/43/44), ambang dari validation.

| pipeline | val bacc | test bacc | test AUC | lolos lantai bacc 0.5947? |
|---|---:|---:|---:|:---:|
| `ce` + hier_addone | 0.8455 ± 0.0066 | **0.7994 ± 0.0120** | 0.8403 ± 0.0079 | ya |
| `supcon` + stacked_randaug | 0.8276 ± 0.0027 | 0.7882 ± 0.0071 | **0.8573 ± 0.0020** | ya |
| `selfcon` + simclr | 0.7811 ± 0.0194 | 0.7603 ± 0.0224 | 0.8035 ± 0.0051 | ya |

**3 dari 3 konfigurasi melewati lantai, dan ketiganya multi-seed.** Uji ketat
laporan (`mean − simpangan baku > lantai`, minimal 2 seed) lolos pada AUC untuk
ketiganya. Pada data ayam, tepat **satu** konfigurasi lolos, dan hanya pada
urutan - bukan pada keputusan.

### Validation di sini TIDAK jenuh

Ini sinyal kedua yang berdiri sendiri, dan penting:

| | data ayam (PIO dev) | SDNET |
|---|---|---|
| val AUC | **1.0000** di seluruh 9 run | 0.821-0.910 |
| epoch checkpoint terpilih | 1, 1, 2 (supcon) | 16-39 (probe supcon) |
| bisa dipakai kalibrasi ambang? | tidak | **ya** |

Pada data ayam val AUC 1.0000 dicapai di epoch 1-2, yang berarti validation
tidak bisa membedakan apa pun lagi - termasuk tidak bisa dipakai memilih epoch
atau ambang secara bermakna. Di SDNET validation bergerak dan berhenti di
0.82-0.91, jadi ia mengerjakan tugasnya.

### Kurva contrastive: satu hipotesis lama gugur

`kesimpulan_kronologi_2.md` §6 butir 3 mencurigai bahwa kurva SupCon yang datar
(4.09 → 3.54 dalam 60 epoch) berarti tahap contrastive-nya praktis tidak
belajar, sehingga keunggulan SupCon sebenarnya milik probe liniernya saja.

Data SDNET **membantah** ini:

| | data ayam | SDNET |
|---|---|---|
| supcon contrastive | 4.09 → 3.54 (turun 13%) | 4.26 → 4.06 (**turun 4.3-4.9%**) |
| selfcon contrastive | 3.17 → 0.24 (turun 92%) | 3.93 → 1.25 (**turun 67-70%**) |

Di SDNET kurva SupCon bergerak **lebih sedikit lagi** - tapi justru menghasilkan
representasi **terbaik** (AUC 0.8573 vs selfcon 0.8035), dengan probe yang lebih
pendek (16-39 epoch vs 24-40). Jadi besarnya penurunan loss SupCon **bukan alat
ukur yang sah** untuk menilai apakah tahap contrastive bekerja: loss SupCon
dinormalisasi atas banyak positif sehingga lantainya jauh di atas nol. Kurva
datar itu normal untuk loss tersebut.

**Konsekuensi:** butir 3 pada daftar langkah `kesimpulan_kronologi_2.md` dicoret.
Untuk benar-benar menguji sumbangan tahap contrastive, yang harus dibandingkan
adalah probe di atas encoder terlatih vs probe di atas encoder acak/beku - bukan
besarnya loss.

---

## 6. Hasil lintas sub-domain (wall + pavement)

Dilatih **hanya** di deck. Wall dan pavement tidak pernah dilihat - bukan hanya
foto sumbernya berbeda, tapi jenis permukaan betonnya berbeda. Ambang diambil
dari `result.json → threshold_from_validation`, **tidak pernah dihitung ulang di
data uji**.

| pipeline | `test_wall` bacc | `test_wall` AUC | `test_pave` bacc | `test_pave` AUC |
|---|---:|---:|---:|---:|
| `supcon` + stacked_randaug | **0.8230 ± 0.0132** | **0.8815 ± 0.0051** | **0.8017 ± 0.0083** | **0.8623 ± 0.0130** |
| `ce` + hier_addone | 0.8006 ± 0.0183 | 0.8506 ± 0.0169 | 0.6997 ± 0.0328 | 0.7397 ± 0.0376 |
| `selfcon` + simclr | 0.7427 ± 0.0048 | 0.8080 ± 0.0161 | 0.6990 ± 0.0233 | 0.8107 ± 0.0080 |
| _(lantai bacc / AUC)_ | _0.6321_ | _0.6666_ | _0.5300_ | _0.6341_ |

**Tidak runtuh.** Ketiga pipeline tetap di atas lantai pada kedua sub-domain
hold-out. Yang paling mencolok: `supcon` justru **naik** di wall (AUC 0.8815 vs
0.8573 in-domain) dan hampir tidak turun di pavement (0.8623).

Ada satu penurunan nyata yang layak dicatat: `ce` jatuh dari bacc 0.7994
(deck) ke 0.6997 (pavement), sementara `supcon` hanya turun ke 0.8017. Jadi
**urutan metode berubah saat diuji lintas domain** - `ce` unggul tipis
in-domain tapi `supcon` jauh lebih tahan banting. Ini persis jenis temuan yang
tidak bisa diperoleh dari data ayam, karena di sana seluruh angkanya sudah
tenggelam di bawah jalan pintas.

---

## 7. Gerbang kausal (`acak16`)

`acak16` mengacak susunan 4×4 petak (derangement: tidak ada petak yang tetap di
tempatnya). Ini menghancurkan **bentuk dan susunan global**, tapi mempertahankan
**tekstur lokal**. Tiga seed, AUC di `test` deck.

| pipeline | asli | abu | hue+26 | kabur | **acak16** |
|---|---:|---:|---:|---:|---:|
| `ce` | 0.8403 ± 0.0079 | 0.8393 | 0.8456 | 0.7819 | **0.8008** (−0.0395) |
| `selfcon` | 0.8035 ± 0.0050 | 0.8034 | 0.8037 | 0.7502 | **0.7686** (−0.0349) |
| `supcon` | 0.8573 ± 0.0020 | 0.8538 | 0.8626 | 0.7676 | **0.7767** (−0.0806) |

Cara membacanya di sini **berbeda** dari pada data ayam, dan justru itu yang
informatif:

- **Untuk retak beton, tekstur memang jawabannya.** Retak *adalah* pola tekstur
  lokal. Jadi AUC yang tetap tinggi setelah `acak16` di sini **wajar** dan bukan
  tuduhan.
- Tapi skornya tetap **turun konsisten di 3 dari 3** (−0.035 sampai −0.081),
  jauh melampaui simpangan antar-seed (±0.002-0.008). Artinya model tetap
  memakai sebagian informasi susunan, bukan hanya statistik sepetak.
- **`kabur` (Gaussian σ=4) merusak paling parah** (−0.053 sampai −0.090) pada
  ketiganya. Itu tepat seperti yang diharapkan kalau yang dibaca adalah retak:
  mengaburkan gambar menghapus tepi retaknya. Model ini bereaksi terhadap
  perusakan yang **relevan secara fisik**.
- **`abu` dan `hue+26` hampir tidak berpengaruh** (|Δ| ≤ 0.005). Model tidak
  bergantung pada warna - yang benar untuk retak beton.

Bandingkan dengan data ayam: di sana `acak16` **menaikkan** AUC pada 3 dari 6
kombinasi, dan 18/18 checkpoint tidak terganggu. Di sana bentuk tubuh
(berbaring vs berdiri) **adalah** sinyal yang sesungguhnya, jadi tidak
terganggunya skor adalah bukti model tidak membacanya.

> **Ringkasnya:** pada SDNET perusakan menurunkan skor sesuai dengan apa yang
> secara fisik penting (kabur > acak16 > warna). Pada data ayam perusakan yang
> menghancurkan sinyal sesungguhnya malah menaikkan skor. Pipeline yang sama,
> perilaku yang berlawanan - dan yang berbeda hanya datanya.

---

## 8. Perbandingan langsung dengan data ayam

| | data ayam | SDNET2018 |
|---|---|---|
| lantai jalan pintas (AUC) | **0.883** (saturasi) | 0.6430 (std_terang) |
| AUC ciri bbox saja | 0.985 (val) / 0.786 (test) | **0.5000** - nol informasi |
| model terbaik | AUC 0.792 (`eq48/supcon`) | AUC 0.8573 (`supcon`) |
| model mengalahkan lantai? | **tidak** (0.792 < 0.883) | **ya**, 3 dari 3 |
| val AUC | 1.0000, jenuh di epoch 1-2 | 0.82-0.91, tidak jenuh |
| sebaran antar-seed (AUC) | ±0.073 sampai ±0.101 | **±0.002 sampai ±0.005** |
| `acak16` | **naik** di 3/6, 18/18 tak terganggu | **turun** di 3/3 |
| lintas domain | AUC 0.65 (close-up → CCTV) | 0.86-0.88 (deck → wall/pave) |
| `label = domain`? | ya, sempurna | mustahil secara struktur |

Baris **sebaran antar-seed** perlu perhatian khusus. Pada data ayam simpangan
antar-seed (±0.073-0.101) **lebih besar daripada jarak antar-metode**, yang
berarti peringkat metode di sana sebagian besar adalah derau - kesimpulan
"supcon > ce > selfcon" pada data ayam tidak kokoh. Pada SDNET simpangannya
20-30× lebih rapat, jadi urutan yang sama (supcon > ce > selfcon pada AUC)
kali ini benar-benar terbaca.

---

## 9. Yang boleh dan tidak boleh disimpulkan

**Boleh:**

1. **Pipelinenya berfungsi.** Kode, arsitektur, ketiga loss, protokol split,
   kalibrasi ambang dari validation, dan evaluatornya menghasilkan model yang
   melewati lantainya sendiri, menggeneralisasi ke dua domain baru, dan rusak
   saat gambarnya dirusak. Hipotesis "ada cacat mendasar di implementasi
   classifier" **tidak didukung**.
2. **Ketiga metode jalan, termasuk `selfcon`.** Ini melemahkan dugaan bahwa
   implementasi NT-Xent-nya rusak. Ia memang paling lemah di sini juga
   (0.8035), tapi selisihnya wajar, bukan runtuh.
3. **Kurva SupCon datar bukan tanda kegagalan** (bagian 5). Satu butir dugaan
   dicoret dari daftar tersangka.
4. **Masalah skripsi ada di datanya.** Dengan pipeline yang sama terbukti
   berfungsi, kegagalan pada data ayam mengarah ke susunan datanya: jalan
   pintas saturasi, `label = domain`, dan 98 crop mati dari satu domain.

**Tidak boleh:**

1. **Tidak boleh** disimpulkan bahwa pipeline ini akan berhasil pada ayam kalau
   datanya diperbaiki. Yang dibuktikan cuma bahwa ia **bisa** bekerja pada
   tugas yang datanya bersih; retak beton adalah tugas tekstur, deteksi ayam
   mati adalah tugas pose. Keduanya tidak setara.
2. **Tidak boleh** dikreditkan ke fungsi loss tertentu. Tiap diagonal mengubah
   loss **dan** augmentasi bersamaan - ini perbandingan pipeline, bukan
   atribusi kausal. Batasan yang sama persis berlaku seperti pada data ayam.
3. **Tidak boleh** dianggap angka SDNET ini optimal. Augmentasi sengaja
   dibiarkan apa adanya (tanpa rotasi, padahal rotasi aman untuk beton), jadi
   angka sebenarnya kemungkinan lebih tinggi.
4. **Tidak boleh** disebut sudah menguji tahap deteksi. Tahap YOLO dilewati -
   tidak ada objek untuk dideteksi pada ubin SDNET. Yang diuji hanya tahap
   kedua.

---

## 10. Dua koreksi atas laporan yang dihasilkan otomatis

`outputs/reports_sdnet/comparison.md` dihasilkan oleh `src/report.py`, yang
teksnya ditulis untuk eksperimen ayam. Dua hal diperbaiki agar laporan SDNET
tidak menyatakan fakta ayam:

1. **Istilah dan angka.** Judul, nama kelas, kalimat sebab lantai, jumlah seed,
   jumlah pasangan, dan besar dampak satu crop salah - semuanya semula dipaku
   ke angka ayam ("7 ayam hidup", "7.14 poin", "5 seed", "crop ayam hidup
   median 83 px"). Sekarang dibaca dari data, dengan istilah dari
   `report.istilah` di config. Bawaannya teks ayam, sehingga laporan ayam
   ter-commit terbit **byte-identik** (diverifikasi pada `comparison.csv` dan
   `comparison.md`).
2. **Ciri lantai.** Semula dipaku ke ketajaman; sekarang dipilih dari
   `report.lantai_ciri` berdasarkan skor train. Tanpa ini laporan SDNET
   melaporkan palang 0.5713 padahal yang mengikat 0.6430 - yaitu menurunkan
   palang ujiannya sendiri.

Satu klaim lagi yang sudah diperbaiki di kode tapi perlu dicatat karena muncul
di seluruh laporan ayam: kalimat *"ambangnya tidak bisa diperbaiki lewat
validation set karena val-nya jenuh"* adalah temuan **data ayam**, bukan sifat
pipeline. Di SDNET validation tidak jenuh, dan kalimat itu tidak lagi tercetak
(`report.istilah.val_jenuh: false`).

---

## 11. Reproduksi

```bash
python src/build_manifest_sdnet.py --config configs/config_sdnet.yaml
python src/eval_shortcut_baseline.py --crops data/crops_sdnet \
    --json outputs/reports/shortcut_sdnet.json                    # gerbang wajib
python src/train.py --config configs/config_sdnet.yaml --method all \
    --seeds 42,43,44 --save-model all
python src/report.py --config configs/config_sdnet.yaml
python src/eval_lintas_domain.py --config configs/config_sdnet.yaml \
    --runs outputs/runs_sdnet --splits test_wall,test_pave \
    --json outputs/predictions/lintas_domain_sdnet.json
python src/eval_intervensi.py --config configs/config_sdnet.yaml \
    --runs outputs/runs_sdnet --seeds 42,43,44 \
    --json outputs/reports/intervensi_sdnet.json
```

Uji regresi: `python tests/test_protocol.py -v` → **10/10 lolos** pada saat
Bagian A selesai (7 uji lama tidak diubah, 3 uji baru untuk manifest SDNET).
Bagian B kemudian menambah 5 uji lagi, jadi jumlah sekarang **15/15**; ketiga
uji SDNET tetap termasuk dan tetap lolos.

Waktu: 9 run = 12005 detik GPU (~3.3 jam). Rata-rata per run: ce 429 detik,
selfcon 1674 detik, supcon 1898 detik.

## 12. Berkas terkait

| berkas | isi |
|---|---|
| [`../reports_sdnet/comparison.md`](../reports_sdnet/comparison.md) | tabel hasil otomatis + lantai |
| [`shortcut_sdnet_lintas.json`](shortcut_sdnet_lintas.json) | lantai jalan pintas ketiga split |
| [`intervensi_sdnet.json`](intervensi_sdnet.json) | gerbang kausal, per seed |
| [`../predictions/lintas_domain_sdnet.json`](../predictions/lintas_domain_sdnet.json) | hasil wall + pavement, per run |
| [`update_14_september.md`](update_14_september.md) | angka acuan data ayam |
| [`kesimpulan_kronologi_2.md`](kesimpulan_kronologi_2.md) | hipotesis kurva SupCon yang dibantah di §5 |
