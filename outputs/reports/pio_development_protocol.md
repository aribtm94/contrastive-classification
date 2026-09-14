# Protokol Development PIO + Roboflow

## Batas data

Sesuai arahan dosen, eksperimen baru memisahkan data secara keras:

- **Train/validation:** ayam hidup dari PIO (`pio_gt`, CCTV) dan ayam mati dari
  Roboflow `dead-chikens` (`coco_gt`, close-up).
- **Test:** seluruh 18 frame pada `dataset/chick`, termasuk 11 berkas bernama
  `ayam (...)` dan 7 berkas bernama `chick (...)`.
- Label, crop, mask, atau metrik chick tidak dipakai untuk training, validation,
  early stopping, kalibrasi ambang, atau pemilihan checkpoint.

Dataset chick sudah pernah diukur dalam eksperimen historis. Karena itu ia
adalah **benchmark test tetap retrospektif**, bukan holdout prospektif yang
benar-benar belum pernah dilihat. Hasil konfirmatori memerlukan benchmark baru.

## Development split yang dibekukan

| split | hidup PIO | mati Roboflow | total |
|---|---:|---:|---:|
| train | 179 | 58 | 237 |
| validation | 115 | 40 | 155 |
| total | 294 | 98 | 392 |

Split dilakukan per `base_image`. Varian augmentasi Roboflow dari foto asli
yang sama dan seluruh crop satu frame PIO tidak boleh bocor ke dua split.
`dev_split_lock.json` mengikat mapping base-image dan hash manifest.

Pembagian internal lama bernama `test` tidak lagi digunakan sebagai test;
barisnya digabung ke validation. Satu-satunya test adalah benchmark ayam/chick.

## Pipeline yang dijalankan

Tiga pipeline diagonal, masing-masing seed 42, 43, dan 44:

1. SelfCon + SimCLR, dilanjutkan linear probe;
2. SupCon + Stacked RandAugment, dilanjutkan linear probe;
3. CE + Hierarchical Add-One.

Dijalankan pada PIO asli dan PIO dengan equalization 48 px. Ini perbandingan
**pipeline**, bukan atribusi kausal fungsi loss, karena loss dan augmentasi
berubah bersama.

## Loss per epoch

Setiap epoch mencatat objective training dan validation:

- SelfCon contrastive: NT-Xent train/validation;
- SupCon contrastive: SupCon train/validation;
- linear probe dan CE: weighted cross-entropy train/validation;
- learning rate yang dipakai dan learning rate sesudah scheduler;
- metrik validation untuk tahap classifier.

Validation contrastive memakai dua view deterministik yang tetap antar-epoch,
sehingga perubahan validation loss berasal dari perubahan bobot model. Model
classifier tetap dipilih berdasarkan balanced accuracy validation dengan AUC
sebagai tie-break sesuai protokol historis. `tau_val` dihitung hanya dari skor
validation checkpoint terpilih.

## Keterbatasan yang tidak disembunyikan

Pada development, **label identik dengan domain**: semua hidup adalah PIO/CCTV
dan semua mati adalah Roboflow/close-up. Karena itu validation yang tinggi belum
membuktikan model mengenali kondisi ayam. Baseline tanpa model tetap wajib
dibaca bersama hasil:

- PIO asli: hue validation AUC-terarah 0.914 dan ketajaman 0.878;
- PIO eq48: hue validation AUC-terarah 0.922; ketajaman turun ke 0.565;
- ukuran bbox metadata mencapai AUC-terarah 0.985 pada keduanya.

Equalization mengurangi ketajaman, tetapi tidak menghilangkan jalan pintas warna
atau ukuran. Benchmark chick menguji apakah kemampuan itu berpindah ke domain
yang dituju.

## Aturan pembukaan test

Training penuh menyimpan sembilan checkpoint per varian beserta hash, epoch,
threshold validation, config, manifest, split lock, baseline shortcut, versi
runtime, dan commit Git ke registry. Evaluator test hanya menerima checkpoint
dari registry dan menolak hash yang berubah. Ia tidak mencari atau memilih
checkpoint terbaik dari hasil chick.
