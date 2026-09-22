# Perbandingan Metode x Augmentasi - Ayam Mati vs Hidup (HEN v2 - semua frame beranotasi (pembanding))

> **Angka di laporan ini diukur di split VAL, bukan test.** Lengan ini memakai `protocol.development_only`, jadi manifestnya memang tidak punya split test - test ditahan supaya tidak terpakai selama pengembangan. Konsekuensinya mengikat cara membaca seluruh tabel: val dipakai memilih checkpoint DAN ambang, jadi angka val di sini optimistis dan **bukan** estimasi kemampuan pada data baru. Kolom `@tau` sengaja kosong - ambangnya dipilih di val, jadi 'val @tau' akan mengukur dirinya sendiri.

Data: train 1149 / val 851 (tanpa test) crop, ukuran input 224x224 (letterbox), backbone resnet18.

Kelas positif = ayam MATI. Metrik utama = **balanced accuracy** (rata-rata recall kedua kelas), karena jumlah kelasnya timpang.

Tiap konfigurasi dijalankan beberapa seed; yang dilaporkan **mean ± simpangan baku**. `@0.5` = ambang bawaan, `@tau` = ambang yang dikalibrasi di **validation set** (tidak pernah di test set). Di lengan ini `@tau` kosong karena val-lah yang memilih ambang.

Semua kolom di bawah diukur di split **VAL**.

| Augmentasi | Metode | Bal.Acc @0.5 | Bal.Acc @tau | AUC | R.mati | R.hidup | n seed |
|---|---|---|---|---|---|---|---|
| `hier_addone` | ce **←** | **0.9534 ± 0.0034** | - | 0.9854 ± 0.0021 | 0.9392 | 0.9676 | 3 |
| `simclr` | selfcon **←** | **0.9306 ± 0.0032** | - | 0.9715 ± 0.0013 | 0.9245 | 0.9367 | 3 |
| `stacked_randaug` | supcon **←** | **0.9421 ± 0.0081** | - | 0.9779 ± 0.0027 | 0.9064 | 0.9779 | 3 |
| _(lantai)_ | **KETAJAMAN SAJA** | **0.7179** | - | 0.7485 | - | - | - |

Baris bertanda **←** adalah diagonal: pasangan metode-augmentasi yang menjadi rancangan utama (`selfcon`+`simclr`, `supcon`+`stacked_randaug`, `ce`+`hier_addone`).

## Lantai ketajaman - baca ini sebelum memeringkat apa pun

Baris terakhir tabel bukan sebuah model. Itu ketajaman gambar (variance of Laplacian) saja, dengan satu ambang yang dicocokkan di train dan diuji di val - **tanpa melihat isi gambar sama sekali**. Lantainya diukur di split yang SAMA dengan skor model di atasnya (val); lantai dari split lain tidak bisa dibandingkan dengannya.

Angkanya **0.7179** (AUC 0.7485). ketajaman/resolusi

**Konsekuensinya: konfigurasi dengan mean di bawah 0.7179 belum membuktikan apa pun** - hasil yang sama bisa diperoleh tanpa belajar. Dari 3 konfigurasi, **3** berada di atas lantai, dan **3** di antaranya dijalankan lebih dari satu seed.

## Yang berhasil melewati lantai - tapi pada URUTAN, bukan keputusan

**3 dari 3** konfigurasi multi-seed melewati lantai 0.7179 pada bal.acc. Tapi bal.acc mengukur **keputusan** (setelah ambang), sedangkan AUC mengukur **urutan**. Keduanya bisa berbeda jauh di sini: dengan cuma 648 ayam hidup di val, ambang yang meleset satu crop saja sudah memotong 0.08 poin bal.acc walau urutannya sempurna.

Lantai ketajaman punya AUC **0.7485** - setara salah mengurutkan **33077 dari 131544 pasangan** (mati x hidup). Konfigurasi berikut mengurutkan **lebih baik dari itu**, bahkan setelah dikurangi satu simpangan baku:

| Augmentasi | Metode | AUC | pasangan salah urut | n seed |
|---|---|---|---|---|
| `hier_addone` | ce | **0.9854 ± 0.0021** | 1920.5 dari 131544 | 3 |
| `stacked_randaug` | supcon | **0.9779 ± 0.0027** | 2911.5 dari 131544 | 3 |
| `simclr` | selfcon | **0.9715 ± 0.0013** | 3749.0 dari 131544 | 3 |

Syaratnya sengaja ketat: **minimal 2 seed** dan `mean - simpangan baku` masih di atas lantai. Konfigurasi 1 seed (baris `legacy`) tidak ikut walau AUC-nya tinggi - simpangan bakunya 0 cuma karena angkanya cuma satu, jadi tidak membuktikan kestabilan apa pun.

**Hasil di atas yang benar-benar mengalahkan 'tidak belajar apa pun'**, dan hanya pada urutan. Artinya representasinya memang memisahkan kedua kelas lebih baik daripada ketajaman; yang belum beres adalah kalibrasi ambangnya. Validation di sini TIDAK jenuh, jadi ambangnya masih bisa dikalibrasi di sana.

Yang bisa diklaim: **pada val set ini**, urutannya mengalahkan lantai di seluruh 3 seed.

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
- **supcon** - Supervised Contrastive - label hidup & mati digabung dalam loss
- **ce** - Klasifikasi biasa (Cross-Entropy) - baseline

## Catatan pembacaan hasil

- Terbaik pada diagonal: **ce** + `hier_addone` (bal.acc 0.9534 ± 0.0034). Pada **AUC** ceritanya berbeda: lihat bagian "Yang berhasil melewati lantai" di atas - ada konfigurasi yang mengurutkan lebih baik daripada lantai di seluruh 3 seed, dan yang belum beres di situ cuma ambangnya.
- Split val berisi 851 crop (203 ayam mati, 648 ayam hidup). Selisih kecil antar metode belum tentu bermakna; itulah sebabnya simpangan baku antar-seed ikut dilaporkan dan harus dibaca bersama rata-ratanya.
- Akurasi biasa menyesatkan di sini: menebak 'mati' untuk semua sampel sudah memberi akurasi ~79% tanpa model belajar apa pun. Karena itu balanced accuracy yang dipakai.
- `@tau` bukan angka yang lebih baik, melainkan angka yang menjawab pertanyaan berbeda: `@0.5` mengukur model apa adanya, `@tau` mengukur model setelah ambangnya dikalibrasi di validation set. Keduanya dilaporkan supaya tidak ada yang dipilih belakangan.
- `selfcon` tidak memakai label saat melatih encoder, jadi wajar kalau hasilnya paling lemah pada dataset sekecil ini - metode self-supervised umumnya baru unggul kalau data tak berlabelnya banyak (ribuan sampai jutaan).
- `ce` melatih seluruh jaringan, sedangkan `selfcon`/`supcon` hanya melatih linear probe di atas encoder beku. Perbedaan ini disengaja: yang diukur adalah kualitas representasi hasil contrastive.
- Alasan tiap augmentasi dipilih ada di [`augmentation_report.md`](augmentation_report.md).
