# Domain Ayam Mati Kedua (Bagian B)

Dokumen ini melaporkan penambahan **sumber ayam mati kedua** (archive_4,
Kaggle "Multimodal Chicken Datasets") ke susunan data development, dan apa
yang berubah - serta apa yang **tidak** berubah - karenanya.

> **PERINGATAN PROTOKOL, dibaca sebelum satu pun angka di bawah.**
> Benchmark `chick` **sudah pernah dibuka** pada babak 14. Membukanya lagi
> dengan model dari lengan ini adalah pembukaan **KEDUA**. Hasilnya adalah
> lengan eksperimen terpisah dengan registry sendiri
> (`pio_dev2_registry.json`, `pio_dev2_eq_registry.json`) dan **tidak boleh
> disebut konfirmatori**. Tidak ada pemilihan metode, epoch, atau ambang yang
> dilakukan memakai hasil test - itu tetap dipatuhi - tapi kepercayaan
> statistik pada pembukaan kedua lebih rendah daripada pembukaan pertama, dan
> itu tidak bisa diperbaiki dengan cara apa pun selain data uji baru.

> **BATASAN SUMBER, juga dibaca lebih dulu.** archive_4 punya jalan pintasnya
> sendiri, terukur pada 200 RGB mati vs 200 RGB sehat **di dalam datasetnya
> sendiri**: terang AUC **0.835**, saturasi **0.874** (arah terbalik), ukuran
> berkas **0.703**. Angka apa pun yang naik setelah penambahan ini **tidak
> boleh** langsung dikreditkan ke "model jadi mengenali ayam mati" - bisa saja
> jalan pintas lama ditukar dengan yang baru.

---

## Daftar isi

| # | Bagian | Isi singkat |
|---|---|---|
| [1](#1-apa-yang-ditambahkan) | Apa yang ditambahkan | 98 mati jadi 298, rasio 1:3 jadi 1:1 |
| [2](#2-gerbang-jalan-pintas---hasilnya-bercabang) | Gerbang jalan pintas | warna membaik, ukuran memburuk |
| [3](#3-yang-tidak-diperbaiki-label--domain-belum-patah) | Yang tidak diperbaiki | tebak label dari domain masih 100% benar |
| [4](#4-delapan-cacat-yang-ditemukan-di-jalur-evaluasi) | Delapan cacat | varian ditebak dari nama berkas; benchmark beku bisa tertimpa; tiga keterangan susunan data dipaku di kode; label varian bertabrakan; ringkasan validation **sudah** tertimpa; tambalannya sendiri memaku nama keluarga; hash beku tidak boleh ditimpa |
| [5](#5-hasil-classifier) | Hasil classifier | kedua lengan 9/9; **0 dari 6** konfigurasi melewati lantai test 0.8831 AUC |
| [6](#6-reproduksi) | Reproduksi | perintah lengkap |
| [7](#7-berkas-terkait) | Berkas terkait | gerbang kedua lengan, laporan Bagian A |

---

## 1. Apa yang ditambahkan

| | susunan lama (`pio_dev`) | **susunan baru (`pio_dev2`)** |
|---|---|---|
| hidup | PIO CCTV, 294 crop, 26 foto | PIO CCTV, **300** crop, 26 foto |
| mati | Roboflow close-up, 98 crop, 36 foto | Roboflow **98** + archive_4 **200** = **298** crop |
| total | 392 | **598** |
| rasio mati:hidup | 1 : 3.0 | **1 : 1.01** |
| train | 179 hidup / 58 mati | 192 hidup / 184 mati |
| val | 115 hidup / 40 mati | 108 hidup / 114 mati |
| foto sumber | 62 | **262** |

archive_4 adalah **frame utuh 640x480 berisi satu ayam**, bukan keluaran
detektor. Itu dicatat apa adanya di manifest (`bbox [0,0,W,H]`, `conf 1.0`,
`origin archive4_rgb`, `domain closeup_kaggle`) supaya tidak ada yang mengira
baris ini hasil deteksi. Hanya `mati_rgb_*` yang dipakai; `mati_inframerah_*`
dilewati.

Satu koreksi atas rencana: rencana menyebut "buang 3 duplikat byte-identik".
Itu **salah** - SHA-256 atas seluruh 200 `mati_rgb_*` memberi 200 hash unik.
Duplikat yang ada berada di folder `healthy/` (`sehat_rgb_113` = `sehat_rgb_135`,
`sehat_rgb_136` = `sehat_rgb_141`), yang tidak dipakai di sini. Penjaga dedup
tetap dipasang di collector supaya rilis lain tidak lolos diam-diam.

**Kedua sumber mati tersebar di kedua split**, diperiksa dari manifest:

| split | `pio_gt` hidup | `coco_gt` mati | `archive4_rgb` mati |
|---|---:|---:|---:|
| `train` | 192 | 66 | 118 |
| `val` | 108 | 32 | 82 |

Ini bukan detail administratif. Kalau archive_4 hanya masuk train, validation
akan tetap menghadapi tugas satu-domain yang lama dan seluruh perbandingan
antar lengan jadi tidak bermakna. Pembagian per foto sumber di
`balance_and_split` membaginya sendiri; yang di atas adalah verifikasinya,
bukan pengaturan manual.

Penjaga protokol lama tidak dilumpuhkan untuk ini. `validate_development_manifest`
sekarang membaca daftar pasangan `(origin, domain)` yang diizinkan dari
`protocol.allowed_provenance`, dan **jatuh ke pasangan lama kalau kunci itu
tidak ada**. Jadi sumber yang tidak didaftarkan tetap ditolak, dan kedua config
beku lama tetap tervalidasi tepat pada 237/155 tanpa satu baris pun berubah.

---

## 2. Gerbang jalan pintas - hasilnya bercabang

Dijalankan **sebelum** training pada kedua susunan, sesuai aturan §6
`tujuan.md`. Ambang dicocokkan di train, dipakai apa adanya di val. Kolom test
kosong karena lengan development memang tidak punya split test - benchmark
`chick` dijalankan terpisah.

**val AUC-terarah:**

| ciri | `pio_dev` lama | **`pio_dev2`** | **`pio_dev2_eq`** |
|---|---:|---:|---:|
| ketajaman | 0.8778 | **0.8396** | 0.7915 |
| hue | **0.9135** | 0.5093 | 0.5247 |
| saturasi | 0.7659 | 0.7195 | 0.7180 |
| std_terang | 0.6413 | 0.7843 | **0.8034** |
| terang | 0.5167 | 0.6295 | 0.6314 |
| rasio_bbox | 0.5942 | 0.5868 | 0.5868 |
| ukuran_bbox\* | 0.9852 | **0.9976** | **0.9976** |

`*` dihitung dari manifest, bukan dari piksel crop.

**Balanced accuracy** - metrik yang **berbeda**, dan inilah yang harus
dibandingkan dengan bacc model, bukan dengan angka AUC di atas:

| ciri | `pio_dev2` bacc | `pio_dev2_eq` bacc |
|---|---:|---:|
| ukuran_bbox\* | **0.9518** | **0.9518** |
| saturasi | 0.7814 | 0.7814 |
| std_terang | 0.7761 | 0.7105 |
| rasio_bbox\* | 0.7132 | 0.7132 |
| ketajaman | 0.6981 | **0.7522** |
| terang | 0.6718 | 0.6888 |
| hue | 0.5210 | 0.5039 |

> **LANTAI AUC piksel: 0.8396 `pio_dev2` (ketajaman) · 0.8034 `pio_dev2_eq`
> (std_terang)**
> **LANTAI bacc piksel: 0.7814 `pio_dev2` (saturasi) · 0.7814 `pio_dev2_eq`
> (saturasi)**
>
> **LANTAI tertinggi kedua lengan: AUC 0.9976 · bacc 0.9518, keduanya
> `ukuran_bbox`**

Dua metrik, dua lantai - jangan dicampur. Perhatikan bahwa **ciri yang
mengikat pun berbeda** antar metrik: pada AUC lantai piksel `pio_dev2` adalah
ketajaman, pada bacc adalah saturasi. Dan pada bacc, eq48 justru **menaikkan**
ketajaman (0.6981 -> 0.7522) sementara pada AUC ia menurunkannya (0.8396 ->
0.7915) - konsisten dengan pembalikan arah yang dijelaskan di bawah: ambang
yang dibalik memutuskan lebih baik walau urutannya sedikit lebih buruk.

Dua palang, dan **yang menentukan adalah yang lebih tinggi.** Lantai piksel
(0.8396 / 0.8034) adalah palang untuk pertanyaan "apakah model membaca isi
gambar lebih baik daripada satu statistik piksel". Tapi untuk pertanyaan yang
sebenarnya - **"apakah model mengenali ayam mati"** - palangnya adalah
**0.9976**, karena menebak label dari lebar-tinggi kotak saja sudah mencapai
angka itu tanpa melihat satu piksel pun.

Sebuah checkpoint yang mencetak AUC 0.95 pada susunan ini karena itu **belum
melampaui apa pun**, meskipun ia jauh di atas lantai piksel. Membandingkannya
hanya dengan 0.8396 akan menurunkan palang ujiannya sendiri - kesalahan yang
sama persis dengan yang diperbaiki di laporan SDNET (§10 butir 2).

Yang bisa membela pemakaian lantai piksel: `ukuran_bbox` dihitung dari kolom
`bbox` manifest, dan crop yang benar-benar masuk ke model sudah di-letterbox
ke 224x224, jadi model **tidak menerima** angka itu secara langsung. Tapi
ukuran asli tetap terbaca dari crop lewat ketajaman dan tingkat detailnya,
jadi "tidak diberikan secara langsung" bukan berarti "tidak tersedia".

### Yang membaik: warna

`hue` runtuh dari **0.9135 ke 0.5093** - dari hampir sempurna menjadi tidak
informatif. Sebabnya terukur, dan ini inti dari kenapa penambahan sumber kedua
bekerja:

| sumber | n | tajam | **saturasi** | terang | sisi pendek |
|---|---:|---:|---:|---:|---:|
| archive_4 mati | 200 | 311 | **7.3** | 117.5 | **480 px** |
| Roboflow mati | 98 | 747 | **47.3** | 125.9 | 212 px |
| PIO hidup | 300 | 178 | **27.7** | 126.2 | **71 px** |

_(median per sumber, dihitung di crop 224x224 yang benar-benar dibaca model)_

Dua sumber mati mengapit kelas hidup dari **arah berlawanan**: saturasi mati
ada di 7.3 *dan* 47.3, sementara hidup duduk di 27.7 **di antara keduanya**.
Satu ambang warna tidak bisa lagi menangkap keduanya sekaligus. Ini persis
efek yang diharapkan dari menambah domain, dan ia terjadi.

Terlihat juga saat tiap sumber mati diadu sendiri-sendiri lawan kelas hidup:

| | ketajaman | saturasi |
|---|---:|---:|
| Roboflow mati vs PIO hidup | 0.9405 | 0.7683 |
| archive_4 mati vs PIO hidup | 0.7223 | **0.0359** |

Saturasi 0.0359 adalah arah **terbalik penuh** dari 0.7683. Gabungan keduanya
itulah yang menjatuhkan hue dan saturasi di tabel utama.

### Yang memburuk: ukuran

`ukuran_bbox` naik ke **0.9976**, hampir sempurna. Sebabnya ada di tabel di
atas: archive_4 adalah frame utuh bersisi pendek **480 px**, crop PIO dari
CCTV bersisi pendek **71 px**. Jadi "seberapa besar kotaknya" menandai label
nyaris tanpa salah, **tanpa melihat satu piksel pun isi gambar**.

Ini bukan cacat yang boleh diabaikan. Menambah domain mati kedua memperbaiki
warna dan **memperparah ukuran** - satu jalan pintas ditukar dengan yang lain,
persis yang diperingatkan di kepala dokumen ini.

Perlu satu catatan yang adil: classifier **tidak pernah membaca kolom bbox**.
Ia hanya menerima piksel crop 224x224. Jadi 0.9976 bukan angka yang bisa
langsung dipakai model ini - ia dilaporkan karena (a) ukuran ikut terbaca dari
piksel lewat tingkat kekaburan, dan (b) kalau nanti ada tahap yang memakai
metadata kotak, angkanya akan palsu dan ini peringatannya.

### Karena itu ada dua lengan, bukan satu

| lengan | crop | gunanya |
|---|---|---|
| `pio_dev2` | apa adanya | acuan; jalan pintas ukuran + ketajaman masih hidup |
| `pio_dev2_eq` | disamakan ke sisi pendek 48 px | menyerang jalan pintas ketajaman |

Angka 48 px bukan pilihan baru - itu angka yang sama dengan
`config_pio_eq_dev.yaml`, supaya lengan ini bisa dibandingkan langsung dengan
hasil eq48 satu-domain yang sudah ada.

Satu hal yang **tidak** terjadi seperti pada susunan lama: pada `pio_dev`,
eq48 menjatuhkan ketajaman dari 0.8778 ke **0.5646** - praktis tidak informatif
lagi. (AUC mentahnya 0.4354; angka yang dilaporkan di tabel ini selalu
AUC-terarah `max(AUC, 1-AUC)`, jadi 0.5646 yang berlaku. Draf dokumen ini
sempat menulis 0.4194 - itu keliru, tidak cocok dengan berkas gerbang mana pun,
dan sudah diperbaiki.) Di sini eq48 hanya **membalik arahnya** - dari `>=`
(mati lebih tajam) menjadi `<=` (mati lebih kabur), AUC 0.8396 -> 0.7915.
Sebabnya aritmetika sederhana: archive_4 480 px turun 10x lipat ke 48,
sementara PIO 71 px hanya turun 1.5x. Yang diperkecil
paling banyak kehilangan detail paling banyak. Jadi eq48 di sini **tidak**
membersihkan ketajaman, ia memindahkannya ke sisi seberang.

Itu sebabnya lantai AUC `pio_dev2_eq` pindah ke `std_terang` (0.8034) dan bukan
turun ke sekitar 0.56 seperti yang terjadi pada susunan satu-domain. Kedua
lengan tetap punya lantai tinggi, dan keduanya harus dilampaui sebelum angka
model boleh dibaca sebagai kemampuan mengenali ayam mati.

---

## 3. Yang TIDAK diperbaiki: `label = domain` belum patah

Ini harus ditulis apa adanya, karena mudah disalahbaca sebagai sudah selesai.

| domain | mati | hidup |
|---|---:|---:|
| `cctv` (PIO) | 0 | 300 |
| `closeup` (Roboflow) | 98 | 0 |
| `closeup_kaggle` (archive_4) | 200 | 0 |

Menebak label dari domain masih benar **598/598 = 100%**. Setiap domain masih
memuat tepat satu label.

Yang patah adalah **arah sebaliknya**: satu label tidak lagi berarti satu
domain. Kelas mati sekarang tersebar di dua domain yang sifat pikselnya
berlawanan, jadi model tidak bisa lagi belajar "mati = tampilan close-up
tertentu" dan lolos begitu saja. Itu perbaikan nyata - dan itulah yang
menurunkan `hue` dari 0.9135 ke 0.5093 - tapi ia **bukan** `label = domain`
yang patah.

Untuk benar-benar mematahkannya dibutuhkan **ayam mati dan ayam hidup dari
frame yang sama**. Itu persis yang dikerjakan ide 2-input (perbandingan
satu-frame), dan itu tetap Tahap 3 yang belum dijalankan.

---

## 4. Delapan cacat yang ditemukan di jalur evaluasi

Tujuh di antaranya ditemukan saat membaca jalur evaluasi dan berkas keluaran,
**sebelum** satu pun perintah benchmark dijalankan. Yang kedelapan, butir (i),
justru **dimunculkan oleh tambalan** untuk butir (e) dan baru terlihat saat
laporan dev2 pertama kali diterbitkan. Seluruhnya gagal diam-diam - kecuali
(i), yang melempar `KeyError` - jadi dicatat di sini. Butir (c) bukan cacat
melainkan konsekuensi yang harus diketahui pembaca, jadi ia tidak ikut
dihitung; yang dihitung adalah (a), (b), (d), (e), (f), (g), (h), (i).

Perbedaan penting di antara kedelapannya: (a), (b), (d), (e), (f), (g) masih
**hipotetis** - kerusakannya baru akan terjadi kalau perintah tertentu
dijalankan. Butir (h) **sudah terjadi**, dan ditemukan bukan dengan membaca
kode melainkan dengan membaca `git diff`. Butir (i) juga sudah terjadi, dan
ditemukan karena perintahnya gagal di depan mata.

Tiga di antaranya - (d), (f), (g) - adalah **satu jenis kesalahan yang sama**
di tiga tempat: keterangan tentang susunan data dipaku sebagai teks di kode,
bukan dibaca dari manifest. Selama hanya ada satu susunan data, teks paku itu
kebetulan benar dan tidak pernah ketahuan. Begitu lengan kedua ditambahkan,
ketiganya salah sekaligus.

### a. Varian crop ditebak dari nama berkas

`report_fixed_chick.py` menentukan varian sebuah run dengan
`"eq_dev" in nama_registry`. Tebakan itu benar untuk dua registry pertama:

| registry | aturan lama | benar? |
|---|---|---|
| `pio_dev_registry.json` | asli | ya |
| `pio_eq_dev_registry.json` | eq48 | ya |
| `pio_dev2_registry.json` | asli | ya |
| **`pio_dev2_eq_registry.json`** | **asli** | **TIDAK** |

Nama berkas lengan baru tidak memuat "eq_dev", jadi lengan eq48-nya akan dicap
`asli` dan **dirata-ratakan bersama lengan tanpa equalize** - dua susunan data
berbeda tercampur dalam satu baris tabel, tanpa peringatan apa pun.

Diperbaiki: varian sekarang dibaca dari
`config_snapshot.crops.equalize_resolution` di registry itu sendiri, yaitu
sumber yang sama yang dipakai saat crop dibuat. Nama berkas tidak lagi
menentukan apa pun. Rekaman lama yang tidak menyimpan `config_snapshot` jatuh
ke aturan nama berkas, sehingga `outputs/reports/fixed_chick.md` yang sudah
ter-commit terbit **byte-identik** (diverifikasi dengan `diff`).

### b. Benchmark beku bisa tertimpa tanpa disadari

`eval_fixed_chick.py` menulis ke `outputs/predictions/fixed_chick.json` dan
`report_fixed_chick.py` ke `outputs/reports/fixed_chick.md` - keduanya nilai
**bawaan**. Menjalankan keduanya apa adanya untuk lengan baru akan **menimpa
hasil babak 14** yang sudah ter-commit di `88a4463`.

Karena itu lengan ini wajib memakai prefix sendiri:

```bash
python src/eval_fixed_chick.py --output-prefix fixed_chick_dev2 ...
python src/report_fixed_chick.py --json outputs/predictions/fixed_chick_dev2.json     --report outputs/reports/fixed_chick_dev2.md     --plot outputs/reports/fixed_chick_dev2.png
```

Ini bukan kehati-hatian teoretis: kesalahan yang sama persis sudah terjadi
sekali pada sesi ini, saat `outputs/reports/comparison.json` tertimpa oleh
hasil SDNET dan harus dipulihkan dengan `git checkout`.

### c. Catatan provenance yang mengikuti dari (a)

Perbaikan (a) mengubah isi `src/eval_fixed_chick.py` dan
`src/report_fixed_chick.py`. Keduanya ikut di-hash ke dalam
`registry.source_sha256`, jadi registry lengan ini akan mencatat hash yang
**berbeda** dari registry babak 14 untuk dua berkas itu. Itu benar dan memang
diinginkan - yang dijalankan memang kode yang berbeda - tapi berarti kedua
registry tidak bisa dibandingkan lewat `source_sha256` begitu saja.

Hal yang sama akan terjadi lagi, dan lebih luas, sesudah tambalan (d)/(f)/(g)
dipasang. `source_hashes()` (`train.py:295`) meng-hash sepuluh berkas **dibaca
dari disk saat registry dibekukan** - termasuk `src/build_crops.py` dan
`src/train.py`, yaitu dua berkas yang akan diubah oleh tambalan itu. Jadi
begitu tambalan dipasang:

> `registry.source_sha256` untuk lengan ini mencatat kode **sebelum** tambalan,
> sementara berkas di disk sudah kode **sesudah** tambalan. Keduanya tidak akan
> cocok lagi, dan itu **benar** - registry memang harus mencatat kode yang
> betul-betul menghasilkan checkpoint-nya.

Inilah sebabnya keempat tambalan sengaja **tidak** dipasang di tengah sweep.
Kalau dipasang di tengah, sebagian dari 18 run dilatih oleh kode lama dan
sebagian oleh kode baru, sementara registry hanya bisa mencatat satu hash -
dan catatan provenance-nya jadi berbohong tentang run yang mana pun.

### d. `data_contract` di registry masih menyebut satu sumber mati

`train.py:370` menulis blok `data_contract` ke dalam setiap registry dengan
nilai yang **dipaku di kode**, bukan dibaca dari manifest:

```python
"alive": "PIO/pio_gt/cctv", "dead": "Roboflow/coco_gt/closeup",
"label_equals_domain_limitation": True,
```

Untuk lengan ini `dead` **tidak lagi** hanya Roboflow - ada 200 crop archive_4
di dalamnya. Jadi registry `pio_dev2` dan `pio_dev2_eq` akan memuat keterangan
sumber yang **salah**, meskipun seluruh hash (manifest, split lock, checkpoint)
benar dan tetap bisa diverifikasi.

Ini **tidak diperbaiki sekarang, dengan sengaja.** Sweep 18 run sedang berjalan;
Python sudah memuat `train.py` ke memori, tapi `source_hashes()` membaca berkas
itu **dari disk** pada langkah pembekuan di akhir. Mengubahnya sekarang berarti
registry mencatat hash kode yang **bukan** kode yang benar-benar dijalankan -
kerusakan provenance yang lebih buruk daripada keterangan yang salah dan sudah
terdokumentasi. Perbaikannya (baca `data_contract` dari manifest) dikerjakan
setelah sweep selesai, dan registry yang terbit dari sweep ini harus dibaca
bersama catatan ini.

Kebalikannya juga perlu dicatat supaya tidak salah baca:
`label_equals_domain_limitation: true` di registry lengan ini **kebetulan tetap
benar** - lihat bagian 3, tebak-label-dari-domain masih 100%.

Tambalannya sudah ditulis dan menunggu: `outputs/tambalan/perbaiki_data_contract.py`
menyusun ulang blok itu **dari manifest**, dan menghitung
`label_from_domain_accuracy` alih-alih memakukannya. Ia hanya menyentuh blok
keterangan - `checkpoint_sha256`, `manifest_sha256`, `config_sha256`, dan
`source_sha256` tidak disentuh, sehingga registry tetap bisa diverifikasi.
Tanpa `--tulis` ia hanya mencetak.

Diuji lebih dulu pada registry **lama yang beku** dalam mode kering (tidak
menulis): keluarannya mereproduksi kontrak lama persis - `pio_gt/cctv`,
`coco_gt/closeup`, `label_equals_domain_limitation: true`, akurasi 1.0. Jadi
tambalannya benar sebelum ia menyentuh registry mana pun.

Dijalankan kering pada registry `pio_dev2` yang baru, ia mengusulkan:

```
LAMA: {"alive": "PIO/pio_gt/cctv", "dead": "Roboflow/coco_gt/closeup", ...}
BARU: {"alive": ["pio_gt/cctv"],
       "dead": ["archive4_rgb/closeup_kaggle", "coco_gt/closeup"],
       "label_equals_domain_limitation": true,
       "label_from_domain_accuracy": 1.0}
```

Perhatikan baris terakhir: angka 1.0 itu **dihitung dari manifest**, bukan
disalin dari kode. Di susunan lama nilai `true` yang dipaku kebetulan benar;
di sini kebenarannya terbukti, dan kalau suatu saat susunan datanya benar-benar
mematahkan `label = domain`, angka itulah yang akan turun sendiri tanpa ada
yang perlu ingat mengubah teksnya.

### e. Label varian bertabrakan antar susunan data

Perbaikan (a) menghapus tebakan nama berkas, tapi **tidak** menghapus masalah
yang lebih dalam: label yang dihasilkan hanya menggambarkan *apakah crop
di-equalize*, bukan *susunan data mana*. Diuji langsung:

| registry | `keluarga()` |
|---|---|
| `pio_dev_registry.json` | `asli` |
| **`pio_dev2_registry.json`** | **`asli`** |
| `pio_eq_dev_registry.json` | `eq48` |
| **`pio_dev2_eq_registry.json`** | **`eq48`** |

Jadi kalau registry lama dan baru pernah dilewatkan **dalam satu perintah**
`--registries a,b`, hasil dua susunan data yang berbeda (392 crop satu domain
mati vs 598 crop dua domain mati) akan masuk ke baris tabel yang sama dan
dirata-ratakan. Angkanya akan terbit tanpa peringatan apa pun.

Ini **tidak terjadi** pada laporan ini: kedua lengan dev2 dijalankan bersama
satu sama lain dan **tidak pernah** bersama registry babak 14. Perintah di
bagian 6 sengaja hanya memuat dua registry dev2. Tapi tidak ada apa pun di kode
yang **mencegah** pencampuran itu - keamanannya bergantung pada perintah yang
diketik benar, dan itu bukan penjaga.

Tidak ditambal sekarang, alasannya sama dengan (d): `report_fixed_chick.py` ikut
di-hash ke `source_sha256` dan sweep sedang berjalan. Penambalan yang benar
adalah memasukkan identitas susunan data (mis. `manifest_sha256` pendek atau
nama registry) ke dalam kunci pengelompokan, supaya dua susunan berbeda mustahil
jatuh ke baris yang sama.

Tambalannya sudah ditulis dan **sudah diuji di luar berkas aslinya**: salinan
`report_fixed_chick.py` yang sudah tertambal dijalankan atas
`fixed_chick.json` babak 14, dan laporannya terbit **byte-identik** dengan
`fixed_chick.md` yang sudah ter-commit (sha256 `6c5a353844ad4736` pada
keduanya). Itu penting karena kedua tambalan sengaja memuat jalur mundur:
rekaman babak 14 memang **tidak** menyimpan `config_snapshot` (diperiksa: tiga
baris pertama semuanya `None`), jadi jalur mundur itu benar-benar terpakai dan
benar-benar teruji, bukan sekadar ada di kode.

Kunci pengelompokan yang baru memisahkan keempat susunan seperti ini:

| registry | kunci lama | kunci baru |
|---|---|---|
| `pio_dev_registry.json` | `asli` | `asli` (rekaman lama, tak berubah) |
| `pio_eq_dev_registry.json` | `eq48` | `eq48` (rekaman lama, tak berubah) |
| `pio_dev2_registry.json` | `asli` | **`pio_dev2/asli`** |
| `pio_dev2_eq_registry.json` | `asli` (salah) | **`pio_dev2_eq/eq48`** |

### f. Teks kontrak di laporan otomatis masih menyebut satu sumber mati

Cacat yang sejenis dengan (d), tapi di tempat lain. `report_fixed_chick.py`
mencetak kalimat kontrak yang dipaku di kode:

> "Train dan validation hanya memakai ayam hidup PIO + ayam mati Roboflow."

Untuk lengan ini kalimat itu **salah** - ada 200 crop archive_4 di kelas mati.
Jadi `fixed_chick_dev2.md` akan terbit dengan keterangan sumber yang keliru di
bagian Kontrak, sementara angkanya benar. Pembaca laporan itu harus membaca
bagian 1 dokumen ini untuk komposisi yang sebenarnya. Ditambal bersama (d) dan
(e) setelah sweep.

### g. `purpose` di split lock juga dipaku ke susunan lama

Ditemukan saat memeriksa keutuhan protokol selama sweep berjalan, bukan lewat
error. `build_crops.py:551` menulis:

```python
"purpose": "development_only_pio_alive_roboflow_dead",
```

Kedua split lock lengan ini - `data/crops_pio_dev2/dev_split_lock.json` dan
`data/crops_pio_dev2_eq/dev_split_lock.json` (nama berkasnya **sama**, yang
membedakan hanya foldernya) - karena itu menyatakan dirinya berisi "ayam mati
Roboflow" saja, padahal 200 dari 298 crop matinya berasal dari archive_4. Sama seperti
(d) dan (f): tidak melempar error, cuma berbohong pelan di berkas yang justru
dibuat untuk dipercaya sebagai catatan resmi susunan data.

Dampaknya paling kecil di antara ketiganya - `purpose` tidak ikut dibandingkan
saat lock diverifikasi ulang (yang dibandingkan hanya `seed`, `counts`,
`n_base_images`, `base_image_to_split`, dan `manifest_sha256`), jadi ia tidak
bisa membuat penjaga lolos atau gagal secara keliru. Tapi ia berada di berkas
yang paling mungkin dibaca orang lain untuk tahu lengan ini berisi apa.

Ditambal bersama (d), (e), (f) setelah sweep, dengan sumber yang sama:
manifest, bukan teks paku.

### h. `development_comparison.json` ditimpa diam-diam - dan sudah terjadi

Enam cacat di atas ditemukan dengan membaca kode. Yang ini ditemukan dengan
membaca `git diff`, dan bedanya penting: **yang ini sudah terjadi**, bukan
baru mungkin terjadi.

`train.py:800` menulis ringkasan validation ke:

```python
dest = resolve(cfg["output"]["reports_dir"]) / "development_comparison.json"
```

Nama berkasnya tetap; yang membedakan hanya `reports_dir`. Keempat config
development mengarah ke folder yang **sama**:

| config | `reports_dir` | berkas tujuan |
|---|---|---|
| `config_pio_dev.yaml` | `outputs/reports` | **sama** |
| `config_pio_eq_dev.yaml` | `outputs/reports` | **sama** |
| `config_pio_dev2.yaml` | `outputs/reports` | **sama** |
| `config_pio_dev2_eq.yaml` | `outputs/reports` | **sama** |
| `config_sdnet.yaml` | `outputs/reports_sdnet` | terpisah (kebetulan selamat) |

Hanya SDNET yang lolos, dan itu kebetulan - folder terpisahnya dipilih karena
alasan lain. Akibatnya terbaca langsung di `git diff`: berkas yang ter-commit
pada babak 14 (`n_val` 155, susunan lama) sekarang berisi angka `pio_dev2`
(`n_val` 222), dan ketika lengan eq48 selesai ia akan tertimpa **lagi**.

Tidak ada yang hilang - versi babak 14 masih ada di git (`88a4463`), dan
salinan lengan asli sudah disimpan ke
`development_comparison_pio_dev2.json` sebelum lengan eq48 menimpanya. Tapi
kalau berkas ini pernah ter-commit ulang tanpa diperiksa, angka babak 14 akan
tergantikan angka lengan lain **dengan nama berkas yang tidak berubah** -
persis jenis kerusakan yang dikhawatirkan di butir (b), hanya saja butir (b)
masih hipotetis sementara yang ini sudah nyata.

Cara membedakannya kalau ragu: `validation.n` bernilai **155** untuk susunan
lama dan **222** untuk susunan dev2. Jumlah crop validation-lah yang
menandai susunan datanya, bukan nama berkasnya.

Perbaikan yang benar sama dengan (b): nama keluaran harus memuat identitas
susunan data. Tidak ditambal karena `train.py` ikut di-hash
`source_hashes()` - menyentuhnya sesudah sweep akan memaksa jalur bagian 4j
untuk berkas yang **menghasilkan angka**, dan jawabannya di sana adalah
mengulang sweep, bukan mencatat tambalan. Yang dikerjakan sebagai gantinya:

1. Kedua lengan disalin ke nama sendiri sebelum tertimpa -
   `development_comparison_pio_dev2.json` (lengan asli) dan
   `development_comparison_pio_dev2_eq.json` (lengan eq48).
2. Berkas bersamanya **dikembalikan ke versi ter-commit** dengan
   `git checkout -- outputs/reports/development_comparison.json` sebelum
   commit, jadi angka babak 14 (`validation.n` 155) tetap yang tercatat di
   git. Diverifikasi sesudahnya: `validation.n` kembali 155, sha256
   `6b936aaede3c...`.

Butir 2 penting dan mudah terlewat: tanpa itu commit lengan dev2 akan
**menghapus angka babak 14 dari riwayat aktif** tanpa satu baris pun yang
menyebutkannya.

### i. Tambalan (e) memunculkan cacat baru: `plot()` memaku nama keluarga

Ini cacat yang **dibuat oleh tambalannya sendiri**, dan layak ditulis justru
karena itu.

Setelah `keluarga()` diperbaiki supaya memuat identitas susunan data (butir e),
lengan dev2 tidak lagi bernama `asli`/`eq48` melainkan `pio_dev2/asli` dan
`pio_dev2_eq/eq48`. Tapi `report_fixed_chick.plot()` memaku nama yang dicari:

```python
for ax, family in zip(axes, ("asli", "eq48")):      # dipaku
    ...
    means = [rows[(method, scorer)]["pooled_ap"]["mean"] for method in methods]
```

Hasilnya `KeyError: ('selfcon', 'absolute')` - `rows` kosong karena tidak ada
keluarga bernama persis `asli`. Sudah diperbaiki: daftar keluarga dibaca dari
data (`sorted({r["family"] for r in aggregates})`), jumlah panel mengikuti
jumlah keluarga, dan metode yang tidak ada di sebuah keluarga digambar sebagai
batang kosong, bukan melempar.

Dua hal yang perlu dicatat dari kejadian ini:

1. **Tambalan butir (f) juga tidak bekerja seperti dirancang.** Skrip tambalan
   membaca `data_contract` dari JSON benchmark - padahal `eval_fixed_chick.py`
   tidak menyalinnya ke sana; blok itu hanya ada di registry. Jadi kalimat
   kontrak tetap mencetak teks lama. Diperbaiki dengan membaca `data_contract`
   dari registry yang namanya sudah tercatat di tiap rekaman, jadi
   `eval_fixed_chick.py` - berkas yang ikut di-hash registry - tidak perlu
   disentuh sama sekali.
2. **Verifikasi byte-identik sempat gagal, dan itu benar.** Versi pertama
   tambalan (f) mengubah kalimat laporan babak 14 dari "ayam hidup PIO + ayam
   mati Roboflow" menjadi "ayam hidup PIO/pio_gt/cctv dan ayam mati
   Roboflow/coco_gt/closeup" - benar isinya, tapi melanggar janji bahwa
   laporan ter-commit terbit byte-identik. Sekarang susunan lama (tepat satu
   sumber mati, pasangan yang sama persis dengan babak 14) sengaja
   dikembalikan ke teks lama, dan hanya susunan dengan **lebih dari satu**
   sumber mati yang memakai kalimat baru. Diverifikasi: `fixed_chick.md`
   terbit ulang dengan sha256 `6c5a353844ad4736...`, identik dengan yang ada
   di disk dan di commit.

> **Pelajaran yang sama dengan butir (a):** memaku daftar nilai yang sah -
> nama keluarga, nama sumber, pasangan provenance - gagal senyap saat lengan
> baru ditambahkan. Bedanya di sini kegagalannya **berisik**, dan justru itu
> yang membuatnya ketahuan dalam hitungan detik.

### j. Hash beku tidak boleh ditimpa oleh tambalan pasca-sweep

Tambalan (e), (f), dan (g) menyentuh dua berkas yang **ikut di-hash** ke dalam
registry beku: `src/report_fixed_chick.py` (lewat `source_sha256`) dan kedua
`dev_split_lock.json` (lewat `split_lock_sha256`). Sesudah ditambal,
`periksa_registry.py` melaporkan `GAGAL` pada keduanya - dan itu **benar**,
bukan kesalahan alat.

Ada tiga cara menanganinya, dan dua di antaranya salah:

| cara | akibat |
|---|---|
| timpa `source_sha256` dengan hash baru | registry menyatakan checkpoint dibuat oleh kode yang **belum ada** saat itu - pemalsuan catatan, persis hal yang hash ini seharusnya cegah |
| batalkan tambalannya | cacat (e), (f), (g), (i) kembali, padahal keempatnya sudah terbukti nyata |
| **biarkan hash beku, catat tambalannya terpisah** | dipakai |

Yang dipakai: `outputs/tambalan/catat_tambalan_pasca_beku.py` menambahkan blok
`tambalan_pasca_beku` ke kedua registry, berisi nama berkas, hash **saat
pembekuan**, hash **sesudah tambalan**, dan alasannya. Tidak satu pun hash lama
disentuh - diverifikasi dengan membandingkan registry terhadap salinan
pra-tambalan di `outputs/tambalan/_acuan/`: satu-satunya field yang berubah
adalah `data_contract` dan `tambalan_pasca_beku`; `config_sha256`,
`manifest_sha256`, `split_lock_sha256`, `shortcut_baseline_sha256`,
`source_sha256`, dan seluruh blok `checkpoints` **byte-identik**.

`periksa_registry.py` lalu diajari membedakan keduanya, dan hasilnya justru
**lebih ketat** daripada sebelumnya:

| keadaan | vonis |
|---|---|
| hash cocok | `COCOK`, lolos |
| beda, dan selisihnya persis yang tercatat di `tambalan_pasca_beku` | `TERTAMBAL (tercatat)`, lolos |
| beda, tidak tercatat | `BEDA`, **gagal** |
| berkas yang sudah ditambal lalu **diubah lagi** | `BEDA`, **gagal** |

Baris ketiga dan keempat diuji langsung: menambahkan satu baris komentar ke
`src/train.py` membuat vonisnya `GAGAL` (exit 1), dan menambahkan satu baris
lagi ke `report_fixed_chick.py` yang sudah tercatat **juga** `GAGAL`, karena
hash barunya tidak lagi sama dengan yang tercatat. Keduanya dipulihkan setelah
diuji.

Satu perbaikan ikutan: versi lama `periksa_registry.py` mencetak selisih
`source_sha256` tapi **tidak** memasukkannya ke vonis - jadi kode yang berubah
tanpa dicatat akan lolos dengan tulisan `BEDA` di layar. Sekarang ikut
menentukan vonis.

**Yang membuat semua ini bisa diterima** adalah sifat berkas yang ditambal:
`report_fixed_chick.py` hanya **merender laporan**, dan field `purpose` di
split lock hanya keterangan yang tidak ikut dibandingkan saat lock
diverifikasi ulang. Kode yang menghasilkan angka - `train.py`, `dataset.py`,
`eval_fixed_chick.py`, `models.py`, `build_crops.py` - tidak berubah satu byte
pun, dan hash kelimanya masih cocok. Kalau yang perlu ditambal adalah salah
satu dari kelima berkas itu, jawabannya bukan mencatat tambalan melainkan
**mengulang sweep**.

### Satu hal yang justru benar: kedua lengan berbagi `manifest_sha256`

Diperiksa di kesempatan yang sama, dan hasilnya menenangkan:

| | `pio_dev2` | `pio_dev2_eq` |
|---|---|---|
| `manifest_sha256` | `c1df24cb...5b755b` | `c1df24cb...5b755b` (sama) |
| `n_base_images` | 262 | 262 |
| baris manifest | identik baris demi baris | identik |
| median ketajaman crop (598 crop) | 276.1 | **58.9** |

Ini **bukan** tanda satu lengan menimpa yang lain. Manifest hanya mencatat
baris mana yang terpilih dan masuk split mana; `equalize_resolution` bekerja di
tingkat piksel, sesudah pemilihan baris. Jadi dua lengan yang berbagi
`manifest_sha256` persis seperti yang dirancang, dan ketajaman yang turun
hampir lima kali lipat (276.1 -> 58.9, dihitung atas seluruh 598 crop)
membuktikan pikselnya memang berbeda.

Artinya perbandingan asli vs eq48 di bagian 5 adalah perbandingan yang bersih:
**satu-satunya** yang berubah adalah resolusi piksel. Crop yang sama, split
yang sama, seed yang sama.


---

## 5. Hasil classifier

Kedua lengan sudah lengkap: masing-masing 9 run, registry beku dan
terverifikasi (9/9 `checkpoint_sha256` cocok pada keduanya, begitu juga hash
config, manifest, split lock, dan gerbang jalan pintas). Tabel benchmark test
sengaja baru ditulis setelah keduanya selesai, karena keduanya harus dibaca
berdampingan.

Angka di bawah ini adalah angka **validation**, bukan test. Ia tidak boleh
dibaca sebagai kemampuan mengenali ayam mati - lihat lantai jalan pintas di
bagian 2 (lantai **AUC** piksel `pio_dev2` 0.8396 ketajaman, `pio_dev2_eq`
0.8034 std_terang; lantai AUC tertinggi kedua lengan **0.9976**
`ukuran_bbox`. Untuk bacc angkanya lain: 0.7814 piksel, 0.9518 tertinggi) dan
`label = domain` 100% di bagian 3.

### Saturasi validation: satu metode membaik, dua tidak

Lengan `pio_dev2` sudah lengkap 9 run (3 metode x 3 seed), jadi bagian ini
ditulis dari data penuh. Hasilnya **membelah menurut metode**, dan itu
membatalkan cara saya membingkainya sebelum lengan ini lengkap.

| metode | epoch terpilih | val AUC | val bacc | crop salah dari 222 |
|---|---|---:|---:|---|
| `selfcon` | 22, 7, 10 | 0.9993 +/- 0.0007 | 0.9895 +/- 0.0051 | 3, 3, 1 |
| `supcon` | **1, 2, 1** | **1.0000 +/- 0.0000** | 0.9985 +/- 0.0027 | 1, 0, 0 |
| `ce` | **1, 2, 7** | **1.0000 +/- 0.0000** | **1.0000 +/- 0.0000** | **0, 0, 0** |

Dibandingkan susunan lama (`pio_dev`: val AUC 1.0 persis di kesembilan run,
epoch supcon 1, 1, 2):

- **`selfcon` benar-benar berubah.** Val AUC tidak lagi persis 1.0, epoch
  terpilih bergerak ke 22/7/10, dan selalu ada 1-3 crop yang salah. Kurva
  validation-nya sekarang bergerak sebelum berhenti.
- **`supcon` dan `ce` tidak.** Keduanya tetap val AUC **1.0000 persis di
  ketiga seed**, dan `ce` bahkan bacc 1.0 tanpa satu pun crop salah. Epoch
  terpilih tetap menempel di 1-2 untuk supcon.

**Koreksi.** Sebelum ketiga metode selesai, dokumen ini menulis bahwa "epoch
terpilih tidak lagi menempel di epoch 1-2" seolah itu berlaku umum. Itu hanya
benar untuk `selfcon`. Untuk `supcon` dan `ce` validation masih jenuh
sepenuhnya, jadi untuk dua metode itu validation **tetap tidak bisa dipakai
memilih epoch secara bermakna** - keterbatasan yang sama persis dengan susunan
lama.

Kenapa ini penting dan bukan detail: kalau validation tidak bisa membedakan
apa pun, ambang yang dikalibrasi darinya pun ditentukan oleh crop-crop yang
kebetulan ada, bukan oleh batas keputusan yang sesungguhnya. Terlihat langsung
pada `tau` ketiga seed `ce`: 0.5804, 0.4864, 0.5952 - berbeda jauh padahal
ketiganya sama-sama "sempurna" di validation. Yang sempurna bukan modelnya,
melainkan mudahnya tugas validation itu.

Menambah domain mati kedua karena itu **melonggarkan** saturasi pada satu
metode, tidak mematahkannya. Ini konsisten dengan bagian 3: `label = domain`
masih 100%, jadi masih ada jalan pintas yang cukup untuk memisahkan validation
dengan sempurna.

### eq48 melonggarkan `selfcon` lebih jauh lagi

`selfcon` sudah lengkap tiga seed di **kedua** lengan, jadi perbandingan ini
berdiri sendiri dan tidak menunggu sisa sweep:

| lengan | val AUC | val bacc | epoch terpilih | crop salah dari 222 |
|---|---:|---:|---|---|
| `pio_dev2` (asli) | 0.9993 +/- 0.0007 | 0.9895 +/- 0.0051 | 22, 7, 10 | 3, 3, 1 |
| `pio_dev2_eq` (eq48) | **0.9968 +/- 0.0008** | **0.9622 +/- 0.0073** | **32, 33, 23** | **7, 10, 8** |

Menyamakan resolusi ke sisi pendek 48 px membuat validation **lebih sulit
lagi**: AUC turun, bacc turun hampir 3 poin, epoch terpilih bergerak lebih
jauh dari 1, dan jumlah crop salah naik di ketiga seed tanpa tumpang tindih
(3,3,1 lawan 7,10,8).

Arahnya masuk akal dan justru itu yang perlu diwaspadai: eq48 menghapus
sebagian keunggulan resolusi, dan resolusi adalah **bagian dari jalan pintas**
- lihat bagian 2, `ukuran_bbox` 0.9976 dan ketajaman yang berbeda hampir lima
kali lipat antar sumber. Jadi turunnya angka validation di sini **bukan**
tanda model menjadi lebih buruk dalam mengenali ayam mati; ia tanda salah satu
jalan pintasnya dipersempit. Apakah yang tersisa adalah kemampuan sebenarnya
hanya bisa dijawab tabel benchmark test, terhadap lantai 0.8831 di atas.

### supcon eq48: jenuh juga, persis seperti lengan asli

Seluruh sembilan run eq48 kini selesai, jadi kalimat ini ditulis dari angka
final, bukan dari dua seed pertama:

| metode (eq48) | epoch terpilih | val AUC | val bacc | crop salah dari 222 |
|---|---|---:|---:|---|
| `selfcon` | 32, 33, 23 | 0.9968 +/- 0.0008 | 0.9622 +/- 0.0073 | 7, 10, 8 |
| `supcon` | 1, 1, 1 | **1.0000 +/- 0.0000** | 0.9969 +/- 0.0027 | 1, 1, 0 |
| `ce` | 5, 14, 8 | **1.0000 +/- 0.0000** | **1.0000 +/- 0.0000** | **0, 0, 0** |

Ketiga seed `supcon` berhenti di **epoch 1** dengan val AUC 1.0000 - sama
persis dengan lengan asli. Jadi pelonggaran yang terlihat pada `selfcon` di
atas **tidak** terbawa ke `supcon` maupun `ce`: yang berubah karena eq48
hanyalah metode yang validation-nya memang sudah bergerak sejak lengan asli.

Satu hal yang **berubah** dari dugaan dua-seed: epoch terpilih `ce` eq48
adalah 5/14/8, bukan 1-2 seperti `ce` lengan asli. Tapi itu tidak berarti
validation-nya berguna - ketiganya tetap bacc 1.0000 tanpa satu pun crop
salah, jadi epoch mana pun setelah yang pertama sama sempurnanya dan yang
terpilih ditentukan `val_loss`, bukan kemampuan membedakan. Buktinya `tau`
ketiga seed itu **0.6013 / 0.7072 / 0.3966** - rentang 0.31 pada tiga model
yang sama-sama tidak pernah salah.

Ini bukti kedua untuk koreksi di bagian sebelumnya. Satu metode yang membaik
bukan perbaikan susunan data; ia perbedaan antar metode, dan pola itu kini
terlihat pada **dua** susunan crop yang berbeda, masing-masing lengkap tiga
seed.

### Palang untuk tabel benchmark: lantai test, bukan lantai validation

Bagian 2 mengukur lantai jalan pintas pada **validation** susunan dev2. Tabel
benchmark yang akan ditulis di bawah diukur pada **test chick**, yaitu 1215
crop dari 18 frame - susunan yang sama sekali berbeda, jadi lantainya juga
berbeda dan harus diambil dari sana.

Angka-angkanya sudah ada di `fixed_chick.json` babak 14 dan **tidak
bergantung pada model mana pun**: blok `nuisance` mengukur ciri gambar mentah
terhadap label test yang sama. Terarah (`max(AUC, 1-AUC)`), pooled:

| ciri (test chick) | pooled AUC | macro AUC |
|---|---:|---:|
| **saturasi** | **0.8831** | **0.9043** |
| terang | 0.8266 (mentah 0.1734) | 0.8267 |
| hue | 0.8125 (mentah 0.1875) | 0.7722 |
| bbox sisi pendek | 0.7860 | 0.7431 |
| bbox luas | 0.7240 | 0.6863 |
| bbox rasio / bantalan letterbox | 0.6994 (mentah 0.3006) | 0.7201 |
| conf detektor | 0.5398 | 0.5170 |
| ketajaman | 0.5363 | 0.5296 |

> **LANTAI test chick: 0.8831 pooled AUC (saturasi), 0.9043 macro.**

Tiga hal yang harus dibawa ke tabel hasil nanti:

1. **Lantainya beda dari lantai validation.** Di validation dev2 yang mengikat
   `ukuran_bbox` (0.9976); di test chick yang mengikat **saturasi**, dan
   ketajaman - ciri terkuat di susunan lama - di sini justru **paling lemah**
   (0.5363). Jangan bawa lantai satu susunan ke susunan lain.
2. **Lantai ini tidak bergeser karena lengan dev2.** Ia sifat data testnya,
   bukan sifat model, jadi palang untuk `fixed_chick_dev2` **sama persis**
   dengan palang babak 14. Itu justru yang membuat kedua babak bisa
   dibandingkan sama sekali.
3. **Babak 14 tidak melewatinya, tapi jaraknya tipis.** Rata-rata tiga seed
   terbaiknya (`eq48/supcon`, scorer `absolute`) adalah **0.792 +/- 0.073**;
   checkpoint tunggal terbaik dari seluruh 18 x 3 scorer adalah
   `eq48/supcon/s42` pada **0.8753**. Keduanya di bawah 0.8831, tapi yang
   kedua hanya **0.008** di bawahnya - dan itu satu seed, sementara simpangan
   antar-seed di lengan itu **+/-0.073**, sepuluh kali lipat jaraknya.

   Jadi pertanyaan untuk lengan dev2 bukan "apakah angkanya naik", melainkan
   "apakah ada konfigurasi yang **rata-rata tiga seednya** melewati 0.8831".
   Satu checkpoint yang kebetulan menyentuh 0.88 tidak menjawab apa pun pada
   sebaran selebar itu.

Satu catatan yang memperkuat butir 3: saturasi tetap 0.8849 pooled setelah
`acak16`, yaitu **naik sedikit** ketika susunan crop dihancurkan. Ciri yang
tidak peduli pada bentuk memang tidak akan terganggu oleh perusakan bentuk -
dan itulah persis yang membuatnya jalan pintas.

### Hasil benchmark test chick: kedua lengan gagal melewati lantai

Ini pembukaan **kedua** benchmark chick (yang pertama babak 14). Sesuai
peringatan protokol di bagian 6, hasilnya ditulis sebagai lengan eksperimen
terpisah dengan registry sendiri (`fixed_chick_dev2.*`), **bukan** sebagai
konfirmasi babak 14. Tidak ada angka test yang dipakai memilih epoch, ambang,
seed, checkpoint, atau scorer - seluruhnya sudah beku di registry sebelum
perintah benchmark dijalankan.

Scorer `absolute`, pooled AUC, intervensi `asli`, rata-rata tiga seed:

| susunan / metode | babak 14 (1 domain mati) | dev2 (2 domain mati) | selisih | lewat lantai 0.8831? |
|---|---:|---:|---:|:---:|
| `asli`/`selfcon` | 0.4929 +/- 0.1006 | 0.5279 +/- 0.0933 | +0.0350 | tidak |
| `asli`/`supcon` | 0.6756 +/- 0.0355 | 0.7042 +/- 0.0695 | +0.0286 | tidak |
| `asli`/`ce` | 0.5911 +/- 0.0366 | 0.5956 +/- 0.1638 | +0.0045 | tidak |
| `eq48`/`selfcon` | 0.5806 +/- 0.1201 | 0.4971 +/- 0.0636 | **-0.0835** | tidak |
| `eq48`/`supcon` | **0.7917 +/- 0.0726** | 0.7287 +/- 0.0671 | **-0.0630** | tidak |
| `eq48`/`ce` | 0.7443 +/- 0.0672 | 0.6897 +/- 0.0257 | **-0.0546** | tidak |

> **0 dari 6 konfigurasi melewati lantai 0.8831, pada ketiga scorer.**
> Rata-rata-3-seed terbaik **turun**: 0.7917 (babak 14) -> **0.7287** (dev2).
> Checkpoint tunggal terbaik juga turun: 0.8753 -> **0.7783**.

Jadi jawaban atas pertanyaan yang diajukan di atas - "apakah ada konfigurasi
yang rata-rata tiga seednya melewati 0.8831" - adalah **tidak**, dan arahnya
berlawanan dengan harapan: menambah domain ayam mati kedua **menurunkan** skor
benchmark, bukan menaikkannya.

Ketiga scorer sepakat, jadi ini bukan artefak pemilihan scorer:

| scorer | terbaik dev2 (rata-rata 3 seed) | konfigurasi |
|---|---:|---|
| `absolute` | 0.7287 | `eq48`/`supcon` |
| `relative_clean` | 0.6655 | `eq48`/`supcon` |
| `relative_operational` | 0.6661 | `eq48`/`supcon` |

**Kenapa turun, padahal validation-nya nyaris sempurna?** Karena yang
ditambahkan bukan ayam mati yang lebih mirip test, melainkan domain ketiga
dengan jalan pintasnya sendiri. Bagian 2 sudah mengukurnya: `ukuran_bbox`
justru naik ke 0.9976 di validation karena archive_4 adalah frame utuh sisi
pendek 480 px sementara crop PIO 71 px. Model yang belajar memisahkan
"crop besar" dari "crop kecil" mendapat validation 1.0000 dan **tidak punya
apa pun** untuk dibawa ke test chick, yang seluruh crop-nya berasal dari
detektor yang sama.

#### Gerbang kausal `acak16`: pola lama terulang, malah lebih kuat

| | babak 14 | dev2 |
|---|---|---|
| checkpoint yang skornya **turun** saat crop diacak | 0/18 tak terganggu | **6/18** |
| checkpoint yang skornya **naik** | naik di 3/6 kombinasi | **12/18** |
| delta rata-rata | ~0 | **+0.0277** |

Mengacak susunan 4x4 petak - yang menghancurkan pose ayam sepenuhnya -
**menaikkan** pooled AUC pada dua pertiga checkpoint. Kenaikan terbesar
`eq48/supcon/s44` dari 0.7555 ke **0.8271**, dan `eq48/selfcon/s42` dari
0.4304 ke 0.5676. Tidak ada model yang membaca pose akan berperilaku begitu.

Bandingkan dengan SDNET, di mana `acak16` menurunkan skor di **3/3** dan
`kabur` merusak paling parah - karena di sana yang dibaca memang retak. Vonis
[`uji_kewarasan_sdnet.md`](uji_kewarasan_sdnet.md) tidak berubah oleh lengan
dev2: pipelinenya berfungsi, datanya yang belum memuat sinyal yang dicari.

#### Keputusan @tau_val: ambang dari validation jenuh tidak bisa dipakai

Angka di atas semua tentang **urutan**. Keputusannya lebih buruk lagi:

| lengan / metode | bacc @tau_val (3 seed) | ayam mati tertangkap dari 22 | FP per frame |
|---|---|---|---|
| `asli`/`ce` | 0.5000, 0.4929, 0.4853 | 0, 0, 0 | 0.00, 0.72, 1.50 |
| `asli`/`selfcon` | 0.4691, 0.4701, 0.4967 | 0, 0, 0 | 3.17, 3.06, 0.33 |
| `asli`/`supcon` | 0.6395, 0.6347, 0.5119 | 7, 7, 1 | 2.00, 2.50, 1.11 |
| `eq48`/`ce` | 0.5573, 0.6211, 0.6639 | 3, 7, 8 | 1.11, 3.89, 1.83 |
| `eq48`/`selfcon` | 0.4859, 0.4837, 0.4864 | 0, 0, 0 | 1.44, 1.67, 1.39 |
| `eq48`/`supcon` | 0.4995, 0.6558, **0.7077** | 0, 7, **10** | 0.06, 0.33, 2.00 |

**Delapan belas checkpoint, tidak satu pun bacc 0.75.** Enam dari 18
menangkap **nol** ayam mati sambil tetap menghasilkan FP - artinya ambangnya
memotong di tempat yang tidak ada hubungannya dengan label. Ini konsekuensi
langsung dari validation yang jenuh: `tau` ketiga seed `ce` eq48 berjarak
0.31 satu sama lain padahal ketiganya bacc 1.0000 di validation.

Sebaran antar-seed di dalam satu konfigurasi (`eq48`/`supcon`: 0.4995 vs
0.7077, selisih 0.21) lebih besar daripada jarak antar konfigurasi mana pun.
Peringkat metode pada tabel ini **tidak terbaca** - dan itu temuan, bukan
kekurangan pelaporan.

### Waktu

Lengan asli: 9 run = 2840 detik GPU (~47 menit). Rata-rata per run: `ce` 55
detik, `selfcon` 425 detik, `supcon` 466 detik. Lengan eq48: 9 run = **2913
detik** (~49 menit); `ce` 71 detik, `selfcon` 434 detik, `supcon` 466 detik.
Total kedua lengan **5753 detik** (~1.6 jam). Jauh lebih murah daripada SDNET
(12005 detik) karena datanya 598 crop, bukan 8400 ubin.

Benchmark test chick di atas 18 checkpoint x 2 intervensi berjalan terpisah
dan jauh lebih murah lagi - hanya inferensi, tanpa pelatihan.

---

## 6. Reproduksi

```bash
# --- lengan asli ---
python src/build_crops.py --config configs/config_pio_dev2.yaml
python src/eval_shortcut_baseline.py --crops data/crops_pio_dev2 \
    --json outputs/predictions/pio_dev2_shortcut.json      # GERBANG WAJIB
python src/train.py --config configs/config_pio_dev2.yaml --method all \
    --seeds 42,43,44 --save-model all

# --- lengan eq48 ---
python src/build_crops.py --config configs/config_pio_dev2_eq.yaml
python src/eval_shortcut_baseline.py --crops data/crops_pio_dev2_eq \
    --json outputs/predictions/pio_dev2_eq_shortcut.json   # GERBANG WAJIB
python src/train.py --config configs/config_pio_dev2_eq.yaml --method all \
    --seeds 42,43,44 --save-model all

# --- benchmark: WAJIB pakai prefix sendiri, lihat bagian 4b ---
python src/eval_fixed_chick.py --output-prefix fixed_chick_dev2 \
    --registries outputs/predictions/pio_dev2_registry.json,outputs/predictions/pio_dev2_eq_registry.json
python src/report_fixed_chick.py \
    --json outputs/predictions/fixed_chick_dev2.json \
    --report outputs/reports/fixed_chick_dev2.md \
    --plot outputs/reports/fixed_chick_dev2.png
```
Keempat tambalan bagian 4 dijalankan **sesudah** kedua sweep membeku dan
**sebelum** benchmark, dalam urutan ini:

```bash
# (d) data_contract di registry; (e)(f)(g) keterangan susunan data
python outputs/tambalan/perbaiki_data_contract.py --tulis
python outputs/tambalan/perbaiki_keterangan_sumber.py --tulis

# catat berkas ber-hash yang berubah, TANPA menimpa hash bekunya (bagian 4j)
python outputs/tambalan/catat_tambalan_pasca_beku.py --tulis

# verifikasi ulang - harus LOLOS, bukan sekadar mencetak "BEDA" di layar
python outputs/tambalan/periksa_registry.py outputs/predictions/pio_dev2_registry.json
python outputs/tambalan/periksa_registry.py outputs/predictions/pio_dev2_eq_registry.json
python tests/test_protocol.py -v
```

Urutannya tidak bisa ditukar: `catat_tambalan_pasca_beku.py` merekam hash
berkas **sesudah** ditambal, jadi ia harus dijalankan sesudah kedua tambalan
dan sebelum registry diperiksa ulang. Kalau ada berkas sumber yang disentuh
lagi sesudah itu, `periksa_registry.py` akan `GAGAL` - dan itu memang
maksudnya.

> **Jangan pernah menggabung registry lama dan dev2 dalam satu perintah.**
> `--registries pio_dev_registry.json,pio_dev2_registry.json` akan terbit tanpa
> error, tapi merata-ratakan dua susunan data berbeda menjadi satu baris - lihat
> bagian 4e. Perintah di atas sengaja hanya memuat kedua registry dev2.

> **Perhatikan `--json` pada gerbang.** Nilainya harus sama dengan
> `protocol.shortcut_json` di config, bukan sembarang tempat. `train.py`
> membekukan hash berkas itu ke dalam registry (`shortcut_baseline_sha256`)
> dan **melempar `FileNotFoundError` kalau tidak ada** - tapi baru pada
> langkah terakhir, setelah kesembilan run selesai. Menaruhnya di tempat
> lain berarti kehilangan seluruh waktu GPU sweep. Ini sudah nyaris terjadi
> pada sesi ini: gerbang semula ditulis ke `outputs/reports/`, dan disalin
> ke tempat yang benar sebelum sweep mencapai langkah pembekuan.

Uji regresi: `python tests/test_protocol.py -v` -> **15/15 lolos**.

Tiga uji menjaga bahwa `allowed_provenance` tidak melumpuhkan penjaga
provenance lama: sumber tak terdaftar tetap ditolak, mendaftarkan archive_4
untuk kelas mati tidak membukanya untuk kelas hidup, dan kedua config beku lama
tetap tervalidasi pada 237/155.

Dua uji lagi ditambahkan pada sesi ini, mengunci susunan baru sebagaimana
susunan lama dikunci:

| uji | yang dijaga |
|---|---|
| `test_manifest_dev2_terkunci` | kedua config dev2 tervalidasi pada 376/222 |
| `test_dev2_punya_dua_sumber_mati` | kelas mati = 98 `coco_gt` + 200 `archive4_rgb`; hidup = 300 `pio_gt` |

Uji kedua diperlukan karena yang pertama saja tidak cukup: jumlah bisa
kebetulan tetap cocok sementara archive_4 hilang dari manifest dan digantikan
crop lain. Yang membedakan lengan ini bukan jumlahnya, melainkan **sumbernya**.

Keduanya diperiksa dengan mutasi - angka harapannya sengaja dirusak, dan
keduanya memang gagal (`FAILED (failures=2)`), lalu lolos lagi setelah
dipulihkan. Uji yang tidak pernah dilihat gagal belum tentu menguji apa pun.

`tests/test_protocol.py` **tidak** ikut di-hash `source_hashes()`, jadi
menambah uji di tengah sweep aman - berbeda dengan keempat tambalan di
bagian 4, yang harus menunggu sweep selesai lebih dulu.

## 7. Berkas terkait

| berkas | isi |
|---|---|
| [`shortcut_pio_dev2.json`](shortcut_pio_dev2.json) | gerbang jalan pintas lengan asli |
| [`shortcut_pio_dev2_eq.json`](shortcut_pio_dev2_eq.json) | gerbang jalan pintas lengan eq48 |
| [`../predictions/pio_dev2_registry.json`](../predictions/pio_dev2_registry.json) | registry beku lengan asli, 9 checkpoint |
| [`../tambalan/periksa_registry.py`](../tambalan/periksa_registry.py) | verifikasi satu registry beku: kebijakan, 9 checkpoint, seluruh hash |
| [`../tambalan/perbaiki_data_contract.py`](../tambalan/perbaiki_data_contract.py) | tambalan (d), dijalankan sesudah kedua sweep membeku |
| [`../tambalan/ringkas_lengan.py`](../tambalan/ringkas_lengan.py) | ringkasan 9 run satu lengan langsung dari `result.json`, tidak bergantung pada berkas bersama di bagian 4h |
| [`../tambalan/perbaiki_keterangan_sumber.py`](../tambalan/perbaiki_keterangan_sumber.py) | tambalan (e)(f)(g), dijalankan sesudah kedua sweep membeku |
| [`../tambalan/catat_tambalan_pasca_beku.py`](../tambalan/catat_tambalan_pasca_beku.py) | mencatat tambalan ke registry tanpa menimpa hash beku (bagian 4j) |
| [`../predictions/pio_dev2_eq_registry.json`](../predictions/pio_dev2_eq_registry.json) | registry beku lengan eq48, 9 checkpoint |
| [`fixed_chick_dev2.md`](fixed_chick_dev2.md) | hasil benchmark test chick kedua lengan, 18 checkpoint x 3 scorer |
| [`fixed_chick_dev2.png`](fixed_chick_dev2.png) | grafik pooled AP per keluarga, panel dibaca dari data (bagian 4i) |
| [`../predictions/fixed_chick_dev2.json`](../predictions/fixed_chick_dev2.json) | keluaran mentah benchmark, termasuk `bukan_audit` per rekaman |
| [`development_comparison_pio_dev2_eq.json`](development_comparison_pio_dev2_eq.json) | ringkasan validation lengan eq48 |
| [`development_comparison_pio_dev2.json`](development_comparison_pio_dev2.json) | ringkasan validation lengan asli, diselamatkan sebelum tertimpa (bagian 4h) |
| [`uji_kewarasan_sdnet.md`](uji_kewarasan_sdnet.md) | Bagian A - pipeline diuji di luar data ayam |
| [`kesimpulan_kronologi_2.md`](kesimpulan_kronologi_2.md) | langkah 4: "cari sumber ayam mati kedua" |

> **Dua salinan gerbang jalan pintas, jangan sunting yang keliru.** Berkas
> `outputs/reports/shortcut_pio_dev2.json` dan
> `outputs/predictions/pio_dev2_shortcut.json` saat ini **byte-identik**
> (sha256 `b08941d2...`), tapi hanya yang di `predictions/` yang dirujuk
> `protocol.shortcut_json` di config dan ikut di-hash ke
> `registry.shortcut_baseline_sha256`. Salinan di `reports/` hanya untuk
> dibaca manusia. Kalau gerbangnya dijalankan ulang, jalankan dengan `--json`
> yang menunjuk ke `predictions/` - lihat peringatan di bagian 6.
