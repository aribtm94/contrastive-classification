# Benchmark Test Ayam Tetap

## Kontrak

Train dan validation hanya memakai ayam hidup pio_gt/cctv dan ayam mati archive4_rgb/closeup_kaggle + coco_gt/closeup. Seluruh 18 frame ayam/chick dipakai sebagai test. Tidak ada angka test yang dipakai memilih epoch, threshold, seed, checkpoint, atau scorer.

Benchmark ini **retrospektif**, karena dataset chick sudah pernah dibaca dalam eksperimen historis. Validation development juga memiliki `label = domain`; hasil tinggi belum otomatis berarti model mengenali ayam mati.

Kohort clean berisi 943 ayam valid (22 mati, 921 hidup). Operational ranking memasukkan 272 crop `bukan` sebagai nuisance, bukan sebagai ayam hidup.

## Hasil asli, rata-rata tiga seed

| varian | metode | scorer | AP pooled | AUC pooled | AP macro | AUC macro | Recall@3 | MRR |
|---|---|---|---:|---:|---:|---:|---:|---:|
| pio_dev2/asli | ce | absolute | 0.037 +/- 0.018 | 0.596 +/- 0.164 | 0.104 +/- 0.054 | 0.585 +/- 0.176 | 0.074 +/- 0.085 | 0.105 +/- 0.058 |
| pio_dev2/asli | ce | relative_clean | 0.049 +/- 0.028 | 0.593 +/- 0.044 | 0.151 +/- 0.064 | 0.610 +/- 0.037 | 0.111 +/- 0.111 | 0.152 +/- 0.066 |
| pio_dev2/asli | ce | relative_operational | 0.051 +/- 0.028 | 0.610 +/- 0.040 | 0.156 +/- 0.064 | 0.625 +/- 0.036 | 0.148 +/- 0.140 | 0.157 +/- 0.066 |
| pio_dev2/asli | selfcon | absolute | 0.026 +/- 0.007 | 0.528 +/- 0.093 | 0.073 +/- 0.018 | 0.507 +/- 0.098 | 0.028 +/- 0.028 | 0.082 +/- 0.024 |
| pio_dev2/asli | selfcon | relative_clean | 0.025 +/- 0.005 | 0.485 +/- 0.035 | 0.078 +/- 0.037 | 0.485 +/- 0.049 | 0.019 +/- 0.032 | 0.078 +/- 0.037 |
| pio_dev2/asli | selfcon | relative_operational | 0.027 +/- 0.008 | 0.500 +/- 0.029 | 0.082 +/- 0.039 | 0.496 +/- 0.036 | 0.019 +/- 0.032 | 0.083 +/- 0.039 |
| pio_dev2/asli | supcon | absolute | 0.075 +/- 0.029 | 0.704 +/- 0.069 | 0.232 +/- 0.068 | 0.699 +/- 0.067 | 0.269 +/- 0.143 | 0.264 +/- 0.085 |
| pio_dev2/asli | supcon | relative_clean | 0.059 +/- 0.033 | 0.575 +/- 0.106 | 0.184 +/- 0.071 | 0.607 +/- 0.120 | 0.213 +/- 0.143 | 0.200 +/- 0.082 |
| pio_dev2/asli | supcon | relative_operational | 0.063 +/- 0.034 | 0.587 +/- 0.083 | 0.192 +/- 0.068 | 0.622 +/- 0.097 | 0.213 +/- 0.143 | 0.216 +/- 0.082 |
| pio_dev2_eq/eq48 | ce | absolute | 0.132 +/- 0.094 | 0.690 +/- 0.026 | 0.320 +/- 0.061 | 0.664 +/- 0.007 | 0.389 +/- 0.048 | 0.334 +/- 0.072 |
| pio_dev2_eq/eq48 | ce | relative_clean | 0.118 +/- 0.039 | 0.632 +/- 0.041 | 0.266 +/- 0.049 | 0.638 +/- 0.036 | 0.296 +/- 0.125 | 0.269 +/- 0.052 |
| pio_dev2_eq/eq48 | ce | relative_operational | 0.127 +/- 0.034 | 0.660 +/- 0.054 | 0.293 +/- 0.053 | 0.664 +/- 0.042 | 0.315 +/- 0.098 | 0.296 +/- 0.057 |
| pio_dev2_eq/eq48 | selfcon | absolute | 0.023 +/- 0.003 | 0.497 +/- 0.064 | 0.105 +/- 0.018 | 0.503 +/- 0.067 | 0.037 +/- 0.032 | 0.106 +/- 0.019 |
| pio_dev2_eq/eq48 | selfcon | relative_clean | 0.026 +/- 0.002 | 0.536 +/- 0.025 | 0.073 +/- 0.008 | 0.566 +/- 0.015 | 0.019 +/- 0.032 | 0.073 +/- 0.008 |
| pio_dev2_eq/eq48 | selfcon | relative_operational | 0.026 +/- 0.003 | 0.532 +/- 0.035 | 0.086 +/- 0.012 | 0.560 +/- 0.027 | 0.065 +/- 0.016 | 0.088 +/- 0.016 |
| pio_dev2_eq/eq48 | supcon | absolute | 0.187 +/- 0.052 | 0.729 +/- 0.067 | 0.443 +/- 0.075 | 0.710 +/- 0.101 | 0.491 +/- 0.085 | 0.471 +/- 0.084 |
| pio_dev2_eq/eq48 | supcon | relative_clean | 0.236 +/- 0.041 | 0.666 +/- 0.057 | 0.418 +/- 0.061 | 0.712 +/- 0.058 | 0.454 +/- 0.032 | 0.440 +/- 0.067 |
| pio_dev2_eq/eq48 | supcon | relative_operational | 0.245 +/- 0.044 | 0.666 +/- 0.059 | 0.420 +/- 0.060 | 0.709 +/- 0.070 | 0.454 +/- 0.042 | 0.443 +/- 0.070 |

## Audit crop bukan

Jumlah berikut dirata-ratakan atas tiga seed; tiap baris menunjukkan berapa dari 272 crop bukan ayam memicu alarm/ranking operational.

| varian | metode | >0.5 | >tau validation | top-1 | top-3 | top-5 |
|---|---|---:|---:|---:|---:|---:|
| pio_dev2/asli | ce | 25.7 | 21.3 | 16.7 | 45.7 | 72.3 |
| pio_dev2/asli | selfcon | 10.3 | 11.7 | 16.0 | 46.7 | 72.7 |
| pio_dev2/asli | supcon | 9.3 | 3.7 | 3.3 | 8.0 | 15.7 |
| pio_dev2_eq/eq48 | ce | 62.7 | 57.7 | 14.7 | 43.0 | 69.3 |
| pio_dev2_eq/eq48 | selfcon | 31.3 | 10.3 | 15.3 | 44.7 | 69.7 |
| pio_dev2_eq/eq48 | supcon | 21.3 | 4.3 | 2.3 | 13.7 | 27.0 |

## Sensitivitas acak16

Acak16 versi baru deterministik, tanpa petak tetap, dan mencakup seluruh piksel. Hasil ini hanya mengukur sensitivitas terhadap susunan global; ia tidak membuktikan penurunan berasal eksklusif dari pose.

Kolom delta adalah acak16 dikurangi asli. Delta mendekati nol berarti skor bertahan walau susunan petak dirusak, jadi model tidak membaca susunan global crop.

| varian | metode | scorer | AP asli | AP acak16 | delta AP | AUC asli | AUC acak16 | delta AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| pio_dev2/asli | ce | absolute | 0.037 | 0.032 | -0.004 | 0.596 | 0.602 | +0.006 |
| pio_dev2/asli | ce | relative_clean | 0.049 | 0.030 | -0.019 | 0.593 | 0.524 | -0.069 |
| pio_dev2/asli | ce | relative_operational | 0.051 | 0.034 | -0.017 | 0.610 | 0.548 | -0.062 |
| pio_dev2/asli | selfcon | absolute | 0.026 | 0.027 | +0.002 | 0.528 | 0.553 | +0.026 |
| pio_dev2/asli | selfcon | relative_clean | 0.025 | 0.025 | -0.001 | 0.485 | 0.429 | -0.056 |
| pio_dev2/asli | selfcon | relative_operational | 0.027 | 0.027 | -0.000 | 0.500 | 0.454 | -0.046 |
| pio_dev2/asli | supcon | absolute | 0.075 | 0.054 | -0.021 | 0.704 | 0.703 | -0.001 |
| pio_dev2/asli | supcon | relative_clean | 0.059 | 0.044 | -0.015 | 0.575 | 0.468 | -0.107 |
| pio_dev2/asli | supcon | relative_operational | 0.063 | 0.054 | -0.009 | 0.587 | 0.539 | -0.048 |
| pio_dev2_eq/eq48 | ce | absolute | 0.132 | 0.112 | -0.020 | 0.690 | 0.715 | +0.026 |
| pio_dev2_eq/eq48 | ce | relative_clean | 0.118 | 0.055 | -0.063 | 0.632 | 0.617 | -0.015 |
| pio_dev2_eq/eq48 | ce | relative_operational | 0.127 | 0.059 | -0.068 | 0.660 | 0.653 | -0.007 |
| pio_dev2_eq/eq48 | selfcon | absolute | 0.023 | 0.029 | +0.006 | 0.497 | 0.567 | +0.070 |
| pio_dev2_eq/eq48 | selfcon | relative_clean | 0.026 | 0.021 | -0.004 | 0.536 | 0.426 | -0.110 |
| pio_dev2_eq/eq48 | selfcon | relative_operational | 0.026 | 0.022 | -0.005 | 0.532 | 0.449 | -0.083 |
| pio_dev2_eq/eq48 | supcon | absolute | 0.187 | 0.184 | -0.003 | 0.729 | 0.769 | +0.041 |
| pio_dev2_eq/eq48 | supcon | relative_clean | 0.236 | 0.161 | -0.075 | 0.666 | 0.615 | -0.051 |
| pio_dev2_eq/eq48 | supcon | relative_operational | 0.245 | 0.170 | -0.075 | 0.666 | 0.630 | -0.036 |
