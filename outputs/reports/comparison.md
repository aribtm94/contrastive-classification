# Perbandingan Metode x Augmentasi - Ayam Mati vs Hidup

Data: train 75 / val 22 / test 33 crop, ukuran input 224x224 (letterbox), backbone resnet18.

Kelas positif = ayam MATI. Metrik utama = **balanced accuracy** (rata-rata recall kedua kelas), karena jumlah kelasnya timpang.

Tiap konfigurasi dijalankan beberapa seed; yang dilaporkan **mean ± simpangan baku**. `@0.5` = ambang bawaan, `@tau` = ambang yang dikalibrasi di **validation set** (tidak pernah di test set).

| Augmentasi | Metode | Bal.Acc @0.5 | Bal.Acc @tau | AUC | R.mati | R.hidup | n seed |
|---|---|---|---|---|---|---|---|
| `hier_addone` | selfcon | **0.7786 ± 0.0986** | 0.7703 ± 0.1058 | 0.9363 ± 0.0462 | 0.9000 | 0.6571 | 5 |
| `hier_addone` | supcon | **0.7511 ± 0.0627** | 0.8066 ± 0.0657 | 0.9121 ± 0.0488 | 0.9308 | 0.5714 | 5 |
| `hier_addone` | ce **←** | **0.8599 ± 0.0940** | 0.8846 ± 0.0767 | 0.9605 ± 0.0283 | 0.9769 | 0.7428 | 5 |
| `legacy` | selfcon | **0.7857** | - | 1.0000 | 1.0000 | 0.5714 | 1 |
| `legacy` | supcon | **0.8516** | - | 0.9176 | 0.8462 | 0.8571 | 1 |
| `legacy` | ce | **0.9615** | - | 0.9725 | 0.9231 | 1.0000 | 1 |
| `simclr` | selfcon **←** | **0.7863 ± 0.1049** | 0.8082 ± 0.1237 | 0.9132 ± 0.0565 | 0.9154 | 0.6571 | 5 |
| `simclr` | supcon | **0.8066 ± 0.1585** | 0.8478 ± 0.0812 | 0.9945 ± 0.0085 | 0.9846 | 0.6286 | 5 |
| `simclr` | ce | **0.7094 ± 0.1077** | 0.7302 ± 0.0931 | 0.9330 ± 0.0509 | 0.9615 | 0.4571 | 5 |
| `stacked_randaug` | selfcon | **0.7170 ± 0.0997** | 0.7692 ± 0.1163 | 0.9286 ± 0.0649 | 0.9769 | 0.4571 | 5 |
| `stacked_randaug` | supcon **←** | **0.7028 ± 0.0863** | 0.8121 ± 0.0539 | 0.8703 ± 0.0700 | 0.9769 | 0.4286 | 5 |
| `stacked_randaug` | ce | **0.7428 ± 0.1160** | 0.8000 ± 0.1143 | 0.9505 ± 0.0510 | 1.0000 | 0.4857 | 5 |
| _(lantai)_ | **KETAJAMAN SAJA** | **0.9231** | - | 0.9670 | - | - | - |

Baris bertanda **←** adalah diagonal: pasangan metode-augmentasi yang menjadi rancangan utama (`selfcon`+`simclr`, `supcon`+`stacked_randaug`, `ce`+`hier_addone`).

## Lantai ketajaman - baca ini sebelum memeringkat apa pun

Baris terakhir tabel bukan sebuah model. Itu ketajaman gambar (variance of Laplacian) saja, dengan satu ambang yang dicocokkan di train dan diuji di test - **tanpa melihat isi gambar sama sekali**.

Angkanya **0.9231** (AUC 0.9670). Penyebabnya cara data terbentuk: crop ayam hidup median sisi pendek 83 px (semuanya diperbesar ke 224), crop ayam mati 212 px. Jadi ketajaman ikut menandai kelas.

**Konsekuensinya: konfigurasi dengan mean di bawah 0.9231 belum membuktikan apa pun** - hasil yang sama bisa diperoleh tanpa belajar. Dari 12 konfigurasi, **1** berada di atas lantai - dan semuanya cuma satu seed, jadi tidak ada simpangan baku yang bisa menyanggah keberuntungan satu undian. **Di antara konfigurasi yang dijalankan 5 seed, tidak ada yang di atas lantai pada bal.acc.**

## Yang berhasil melewati lantai - tapi pada URUTAN, bukan keputusan

Tidak ada konfigurasi 5-seed yang melewati lantai 0.9231 jika diukur dengan bal.acc (yang melewatinya hanya 1 baris 1 seed, yang tidak membuktikan kestabilan apa pun). Tapi bal.acc mengukur **keputusan** (setelah ambang), sedangkan AUC mengukur **urutan**. Keduanya bisa berbeda jauh di sini: dengan cuma 7 ayam hidup di test, ambang yang meleset satu crop saja sudah memotong 7.14 poin bal.acc walau urutannya sempurna.

Lantai ketajaman punya AUC **0.9670** - setara salah mengurutkan **6 dari 182 pasangan** (mati x hidup). Konfigurasi berikut mengurutkan **lebih baik dari itu**, bahkan setelah dikurangi satu simpangan baku:

| Augmentasi | Metode | AUC | pasangan salah urut | n seed |
|---|---|---|---|---|
| `simclr` | supcon | **0.9945 ± 0.0085** | 1.0 dari 182 | 5 |

Syaratnya sengaja ketat: **minimal 2 seed** dan `mean - simpangan baku` masih di atas lantai. Konfigurasi 1 seed (baris `legacy`) tidak ikut walau AUC-nya tinggi - simpangan bakunya 0 cuma karena angkanya cuma satu, jadi tidak membuktikan kestabilan apa pun.

**Ini satu-satunya hasil di seluruh grid yang benar-benar mengalahkan 'tidak belajar apa pun'**, dan hanya pada urutan. Artinya representasinya memang memisahkan kedua kelas lebih baik daripada ketajaman; yang belum beres adalah kalibrasi ambangnya - dan itu tidak bisa diperbaiki lewat validation set di sini, karena val-nya jenuh (lihat catatan di [`augmentation_report.md`](augmentation_report.md)).

Tetap perlu hati-hati: 7 ayam hidup itu sedikit sekali, jadi AUC setinggi ini lebih mudah terjadi kebetulan daripada kelihatannya. Yang bisa diklaim: **pada test set ini**, urutannya mengalahkan lantai di seluruh 5 seed.

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
| `hier_addone` | selfcon | `hier_addone` | `minimal` |
| `hier_addone` | supcon | `hier_addone` | `minimal` |
| `hier_addone` | ce | `-` | `hier_addone` |
| `simclr` | selfcon | `simclr` | `minimal` |
| `simclr` | supcon | `simclr` | `minimal` |
| `simclr` | ce | `-` | `simclr` |
| `stacked_randaug` | selfcon | `stacked_randaug` | `minimal` |
| `stacked_randaug` | supcon | `stacked_randaug` | `minimal` |
| `stacked_randaug` | ce | `-` | `stacked_randaug` |

Nilai di atas dibaca dari `result.json` tiap run, bukan ditulis tangan - jadi tabel ini ikut berfungsi sebagai audit bahwa kunci probe benar-benar berlaku di seluruh grid.

## Keterangan metode

- **selfcon** - Self-Contrastive (NT-Xent/SimCLR) - tanpa label, lalu linear probe
- **supcon** - Supervised Contrastive - label hidup & mati digabung dalam loss
- **ce** - Klasifikasi biasa (Cross-Entropy) - baseline

## Catatan pembacaan hasil

- Terbaik pada diagonal: **ce** + `hier_addone` (bal.acc 0.8599 ± 0.0940) - tapi masih **di bawah lantai 0.9231**, jadi belum bisa disebut berhasil pada bal.acc. Pada **AUC** ceritanya berbeda: lihat bagian "Yang berhasil melewati lantai" di atas - ada konfigurasi yang mengurutkan lebih baik daripada lantai di seluruh 5 seed, dan yang belum beres di situ cuma ambangnya.
- Test set berisi 33 crop yang berasal dari hanya **7 foto asli**. Selisih kecil antar metode belum tentu bermakna; itulah sebabnya simpangan baku antar-seed ikut dilaporkan dan harus dibaca bersama rata-ratanya.
- Akurasi biasa menyesatkan di sini: menebak 'mati' untuk semua sampel sudah memberi akurasi ~79% tanpa model belajar apa pun. Karena itu balanced accuracy yang dipakai.
- `@tau` bukan angka yang lebih baik, melainkan angka yang menjawab pertanyaan berbeda: `@0.5` mengukur model apa adanya, `@tau` mengukur model setelah ambangnya dikalibrasi di validation set. Keduanya dilaporkan supaya tidak ada yang dipilih belakangan.
- `selfcon` tidak memakai label saat melatih encoder, jadi wajar kalau hasilnya paling lemah pada dataset sekecil ini - metode self-supervised umumnya baru unggul kalau data tak berlabelnya banyak (ribuan sampai jutaan).
- `ce` melatih seluruh jaringan, sedangkan `selfcon`/`supcon` hanya melatih linear probe di atas encoder beku. Perbedaan ini disengaja: yang diukur adalah kualitas representasi hasil contrastive.
- Alasan tiap augmentasi dipilih ada di [`augmentation_report.md`](augmentation_report.md).

---

Cerita utuh seluruh eksperimen - berurutan, dengan gambar tiap
percobaan dan alasan di balik setiap perubahan - ada di
[`kronologi_lengkap.md`](kronologi_lengkap.md).
