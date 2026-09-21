# Perbandingan Metode x Augmentasi - Retak vs Utuh (SDNET2018 - uji kewarasan pipeline)

Data: train 1190 / val 408 / test 402 crop, ukuran input 224x224 (letterbox), backbone resnet18.

Kelas positif = ubin RETAK. Metrik utama = **balanced accuracy** (rata-rata recall kedua kelas), karena jumlah kelasnya timpang.

Tiap konfigurasi dijalankan beberapa seed; yang dilaporkan **mean ± simpangan baku**. `@0.5` = ambang bawaan, `@tau` = ambang yang dikalibrasi di **validation set** (tidak pernah di test set).

| Augmentasi | Metode | Bal.Acc @0.5 | Bal.Acc @tau | AUC | R.retak | R.utuh | n seed |
|---|---|---|---|---|---|---|---|
| `hier_addone` | ce **←** | **0.7994 ± 0.0120** | 0.8009 ± 0.0135 | 0.8403 ± 0.0079 | 0.7041 | 0.8947 | 3 |
| `simclr` | selfcon **←** | **0.7603 ± 0.0224** | 0.7563 ± 0.0191 | 0.8035 ± 0.0051 | 0.6939 | 0.8268 | 3 |
| `stacked_randaug` | supcon **←** | **0.7882 ± 0.0071** | 0.7844 ± 0.0089 | 0.8573 ± 0.0020 | 0.7211 | 0.8552 | 3 |
| _(lantai)_ | **STD_TERANG SAJA** | **0.5947** | - | 0.6430 | - | - | - |

Baris bertanda **←** adalah diagonal: pasangan metode-augmentasi yang menjadi rancangan utama (`selfcon`+`simclr`, `supcon`+`stacked_randaug`, `ce`+`hier_addone`).

## Lantai std_terang - baca ini sebelum memeringkat apa pun

Baris terakhir tabel bukan sebuah model. Itu simpangan baku terang (kanal V dari HSV) saja, dengan satu ambang yang dicocokkan di train dan diuji di test - **tanpa melihat isi gambar sama sekali**.

Angkanya **0.5947** (AUC 0.6430).

**Konsekuensinya: konfigurasi dengan mean di bawah 0.5947 belum membuktikan apa pun** - hasil yang sama bisa diperoleh tanpa belajar. Dari 3 konfigurasi, **3** berada di atas lantai, dan **3** di antaranya dijalankan lebih dari satu seed.

## Yang berhasil melewati lantai - tapi pada URUTAN, bukan keputusan

**3 dari 3** konfigurasi multi-seed melewati lantai 0.5947 pada bal.acc. Tapi bal.acc mengukur **keputusan** (setelah ambang), sedangkan AUC mengukur **urutan**. Keduanya bisa berbeda jauh di sini: dengan cuma 304 ubin utuh di test, ambang yang meleset satu crop saja sudah memotong 0.16 poin bal.acc walau urutannya sempurna.

Lantai std_terang punya AUC **0.6430** - setara salah mengurutkan **10637 dari 29792 pasangan** (retak x utuh). Konfigurasi berikut mengurutkan **lebih baik dari itu**, bahkan setelah dikurangi satu simpangan baku:

| Augmentasi | Metode | AUC | pasangan salah urut | n seed |
|---|---|---|---|---|
| `stacked_randaug` | supcon | **0.8573 ± 0.0020** | 4251.3 dari 29792 | 3 |
| `hier_addone` | ce | **0.8403 ± 0.0079** | 4756.8 dari 29792 | 3 |
| `simclr` | selfcon | **0.8035 ± 0.0051** | 5854.1 dari 29792 | 3 |

Syaratnya sengaja ketat: **minimal 2 seed** dan `mean - simpangan baku` masih di atas lantai. Konfigurasi 1 seed (baris `legacy`) tidak ikut walau AUC-nya tinggi - simpangan bakunya 0 cuma karena angkanya cuma satu, jadi tidak membuktikan kestabilan apa pun.

**Hasil di atas yang benar-benar mengalahkan 'tidak belajar apa pun'**, dan hanya pada urutan. Artinya representasinya memang memisahkan kedua kelas lebih baik daripada std_terang; yang belum beres adalah kalibrasi ambangnya. Validation di sini TIDAK jenuh, jadi ambangnya masih bisa dikalibrasi di sana.

Yang bisa diklaim: **pada test set ini**, urutannya mengalahkan lantai di seluruh 3 seed.

## Cara membaca grid ini

Sejak tiap metode punya augmentasinya sendiri, **dua hal berubah bersamaan** di diagonal (loss DAN augmentasi). Karena itu:

- **Diagonal** (baris ←) = rancangan yang diminta, tapi tidak bisa mengatribusikan sebab.
- **Bandingkan dalam satu nilai `aug` yang sama** (augmentasi ditahan, loss divariasikan) = perbandingan metode yang sah.
- **Bandingkan dalam satu `method` yang sama** (loss ditahan, augmentasi divariasikan) = menjawab "ini efek augmentasi atau efek loss?".

## Kebijakan augmentasi yang benar-benar dipakai tiap tahap

Kolom `Augmentasi` di tabel atas adalah augmentasi TAHAP UTAMA. Itu bukan keseluruhan cerita: `selfcon`/`supcon` punya dua tahap latih (kontrastif, lalu linear probe di atas encoder beku), dan tahap probe **sengaja dikunci ke `minimal`** (flip saja).

Alasannya: probe mengukur kualitas ENCODER. Kalau augmentasi probe ikut berbeda per metode, angkanya mencampur dua efek. `ce` tidak bisa ikut dikunci karena kepala *adalah* satu-satunya tahap latihnya - asimetri ini nyata dan tidak disembunyikan.

| Augmentasi | Metode | Tahap kontrastif | Tahap kepala/probe |
|---|---|---|---|
| `hier_addone` | ce | `-` | `hier_addone` |
| `simclr` | selfcon | `simclr` | `minimal` |
| `stacked_randaug` | supcon | `stacked_randaug` | `minimal` |

Nilai di atas dibaca dari `result.json` tiap run, bukan ditulis tangan - jadi tabel ini ikut berfungsi sebagai audit bahwa kunci probe benar-benar berlaku di seluruh grid.

## Keterangan metode

- **selfcon** - Self-Contrastive (NT-Xent/SimCLR) - tanpa label, lalu linear probe
- **supcon** - Supervised Contrastive - label utuh & retak digabung dalam loss
- **ce** - Klasifikasi biasa (Cross-Entropy) - baseline

## Catatan pembacaan hasil

- Terbaik pada diagonal: **ce** + `hier_addone` (bal.acc 0.7994 ± 0.0120). Pada **AUC** ceritanya berbeda: lihat bagian "Yang berhasil melewati lantai" di atas - ada konfigurasi yang mengurutkan lebih baik daripada lantai di seluruh 3 seed, dan yang belum beres di situ cuma ambangnya.
- Test set berisi 402 crop yang berasal dari hanya **7 foto asli**. Selisih kecil antar metode belum tentu bermakna; itulah sebabnya simpangan baku antar-seed ikut dilaporkan dan harus dibaca bersama rata-ratanya.
- Akurasi biasa menyesatkan di sini: menebak 'mati' untuk semua sampel sudah memberi akurasi ~79% tanpa model belajar apa pun. Karena itu balanced accuracy yang dipakai.
- `@tau` bukan angka yang lebih baik, melainkan angka yang menjawab pertanyaan berbeda: `@0.5` mengukur model apa adanya, `@tau` mengukur model setelah ambangnya dikalibrasi di validation set. Keduanya dilaporkan supaya tidak ada yang dipilih belakangan.
- `selfcon` tidak memakai label saat melatih encoder, jadi wajar kalau hasilnya paling lemah pada dataset sekecil ini - metode self-supervised umumnya baru unggul kalau data tak berlabelnya banyak (ribuan sampai jutaan).
- `ce` melatih seluruh jaringan, sedangkan `selfcon`/`supcon` hanya melatih linear probe di atas encoder beku. Perbedaan ini disengaja: yang diukur adalah kualitas representasi hasil contrastive.
- Alasan tiap augmentasi dipilih ada di [`augmentation_report.md`](augmentation_report.md).
