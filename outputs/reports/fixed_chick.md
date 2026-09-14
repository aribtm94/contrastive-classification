# Benchmark Test Ayam Tetap

## Kontrak

Train dan validation hanya memakai ayam hidup PIO + ayam mati Roboflow. Seluruh 18 frame ayam/chick dipakai sebagai test. Tidak ada angka test yang dipakai memilih epoch, threshold, seed, checkpoint, atau scorer.

Benchmark ini **retrospektif**, karena dataset chick sudah pernah dibaca dalam eksperimen historis. Validation development juga memiliki `label = domain`; hasil tinggi belum otomatis berarti model mengenali ayam mati.

Kohort clean berisi 943 ayam valid (22 mati, 921 hidup). Operational ranking memasukkan 272 crop `bukan` sebagai nuisance, bukan sebagai ayam hidup.

## Hasil asli, rata-rata tiga seed

| varian | metode | scorer | AP pooled | AUC pooled | AP macro | AUC macro | Recall@3 | MRR |
|---|---|---|---:|---:|---:|---:|---:|---:|
| asli | ce | absolute | 0.039 +/- 0.005 | 0.591 +/- 0.037 | 0.140 +/- 0.057 | 0.600 +/- 0.034 | 0.093 +/- 0.085 | 0.149 +/- 0.060 |
| asli | ce | relative_clean | 0.030 +/- 0.015 | 0.484 +/- 0.072 | 0.076 +/- 0.033 | 0.504 +/- 0.064 | 0.037 +/- 0.064 | 0.075 +/- 0.034 |
| asli | ce | relative_operational | 0.040 +/- 0.026 | 0.538 +/- 0.072 | 0.099 +/- 0.041 | 0.567 +/- 0.074 | 0.093 +/- 0.116 | 0.099 +/- 0.043 |
| asli | selfcon | absolute | 0.024 +/- 0.006 | 0.493 +/- 0.101 | 0.081 +/- 0.041 | 0.477 +/- 0.109 | 0.037 +/- 0.032 | 0.080 +/- 0.041 |
| asli | selfcon | relative_clean | 0.058 +/- 0.056 | 0.563 +/- 0.056 | 0.118 +/- 0.077 | 0.597 +/- 0.086 | 0.074 +/- 0.085 | 0.119 +/- 0.077 |
| asli | selfcon | relative_operational | 0.064 +/- 0.062 | 0.585 +/- 0.053 | 0.139 +/- 0.098 | 0.614 +/- 0.066 | 0.102 +/- 0.112 | 0.142 +/- 0.097 |
| asli | supcon | absolute | 0.091 +/- 0.026 | 0.676 +/- 0.036 | 0.267 +/- 0.076 | 0.687 +/- 0.044 | 0.343 +/- 0.112 | 0.312 +/- 0.118 |
| asli | supcon | relative_clean | 0.046 +/- 0.028 | 0.601 +/- 0.144 | 0.130 +/- 0.080 | 0.639 +/- 0.118 | 0.056 +/- 0.056 | 0.130 +/- 0.081 |
| asli | supcon | relative_operational | 0.050 +/- 0.024 | 0.658 +/- 0.137 | 0.148 +/- 0.077 | 0.686 +/- 0.126 | 0.093 +/- 0.089 | 0.154 +/- 0.079 |
| eq48 | ce | absolute | 0.165 +/- 0.086 | 0.744 +/- 0.067 | 0.335 +/- 0.136 | 0.710 +/- 0.053 | 0.361 +/- 0.139 | 0.346 +/- 0.142 |
| eq48 | ce | relative_clean | 0.091 +/- 0.033 | 0.664 +/- 0.040 | 0.239 +/- 0.054 | 0.678 +/- 0.028 | 0.231 +/- 0.089 | 0.241 +/- 0.053 |
| eq48 | ce | relative_operational | 0.107 +/- 0.043 | 0.684 +/- 0.032 | 0.255 +/- 0.046 | 0.700 +/- 0.030 | 0.269 +/- 0.058 | 0.258 +/- 0.044 |
| eq48 | selfcon | absolute | 0.033 +/- 0.011 | 0.581 +/- 0.120 | 0.149 +/- 0.059 | 0.606 +/- 0.104 | 0.093 +/- 0.064 | 0.147 +/- 0.058 |
| eq48 | selfcon | relative_clean | 0.033 +/- 0.014 | 0.576 +/- 0.060 | 0.102 +/- 0.060 | 0.611 +/- 0.060 | 0.056 +/- 0.096 | 0.102 +/- 0.062 |
| eq48 | selfcon | relative_operational | 0.037 +/- 0.019 | 0.589 +/- 0.055 | 0.102 +/- 0.054 | 0.618 +/- 0.057 | 0.056 +/- 0.096 | 0.102 +/- 0.055 |
| eq48 | supcon | absolute | 0.386 +/- 0.063 | 0.792 +/- 0.073 | 0.503 +/- 0.059 | 0.762 +/- 0.077 | 0.519 +/- 0.080 | 0.536 +/- 0.101 |
| eq48 | supcon | relative_clean | 0.216 +/- 0.014 | 0.661 +/- 0.104 | 0.366 +/- 0.017 | 0.710 +/- 0.093 | 0.407 +/- 0.032 | 0.376 +/- 0.017 |
| eq48 | supcon | relative_operational | 0.223 +/- 0.017 | 0.687 +/- 0.093 | 0.369 +/- 0.017 | 0.740 +/- 0.088 | 0.417 +/- 0.028 | 0.384 +/- 0.026 |

## Audit crop bukan

Jumlah berikut dirata-ratakan atas tiga seed; tiap baris menunjukkan berapa dari 272 crop bukan ayam memicu alarm/ranking operational.

| varian | metode | >0.5 | >tau validation | top-1 | top-3 | top-5 |
|---|---|---:|---:|---:|---:|---:|
| asli | ce | 11.0 | 8.3 | 17.3 | 47.3 | 74.0 |
| asli | selfcon | 13.0 | 2.3 | 16.0 | 46.0 | 73.3 |
| asli | supcon | 6.0 | 3.7 | 4.3 | 15.7 | 25.7 |
| eq48 | ce | 53.0 | 44.3 | 17.0 | 49.7 | 76.7 |
| eq48 | selfcon | 27.7 | 68.7 | 17.0 | 48.0 | 74.3 |
| eq48 | supcon | 22.7 | 49.0 | 5.7 | 23.3 | 38.3 |

## Sensitivitas acak16

Acak16 versi baru deterministik, tanpa petak tetap, dan mencakup seluruh piksel. Hasil ini hanya mengukur sensitivitas terhadap susunan global; ia tidak membuktikan penurunan berasal eksklusif dari pose.

Kolom delta adalah acak16 dikurangi asli. Delta mendekati nol berarti skor bertahan walau susunan petak dirusak, jadi model tidak membaca susunan global crop.

| varian | metode | scorer | AP asli | AP acak16 | delta AP | AUC asli | AUC acak16 | delta AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| asli | ce | absolute | 0.039 | 0.042 | +0.003 | 0.591 | 0.645 | +0.054 |
| asli | ce | relative_clean | 0.030 | 0.029 | -0.001 | 0.484 | 0.507 | +0.023 |
| asli | ce | relative_operational | 0.040 | 0.037 | -0.003 | 0.538 | 0.552 | +0.013 |
| asli | selfcon | absolute | 0.024 | 0.057 | +0.032 | 0.493 | 0.598 | +0.105 |
| asli | selfcon | relative_clean | 0.058 | 0.034 | -0.024 | 0.563 | 0.485 | -0.078 |
| asli | selfcon | relative_operational | 0.064 | 0.038 | -0.025 | 0.585 | 0.515 | -0.070 |
| asli | supcon | absolute | 0.091 | 0.050 | -0.041 | 0.676 | 0.661 | -0.014 |
| asli | supcon | relative_clean | 0.046 | 0.026 | -0.019 | 0.601 | 0.522 | -0.079 |
| asli | supcon | relative_operational | 0.050 | 0.032 | -0.018 | 0.658 | 0.593 | -0.065 |
| eq48 | ce | absolute | 0.165 | 0.073 | -0.091 | 0.744 | 0.721 | -0.024 |
| eq48 | ce | relative_clean | 0.091 | 0.040 | -0.050 | 0.664 | 0.589 | -0.075 |
| eq48 | ce | relative_operational | 0.107 | 0.043 | -0.065 | 0.684 | 0.623 | -0.062 |
| eq48 | selfcon | absolute | 0.033 | 0.089 | +0.056 | 0.581 | 0.673 | +0.093 |
| eq48 | selfcon | relative_clean | 0.033 | 0.025 | -0.008 | 0.576 | 0.452 | -0.125 |
| eq48 | selfcon | relative_operational | 0.037 | 0.026 | -0.011 | 0.589 | 0.483 | -0.106 |
| eq48 | supcon | absolute | 0.386 | 0.223 | -0.163 | 0.792 | 0.769 | -0.022 |
| eq48 | supcon | relative_clean | 0.216 | 0.046 | -0.170 | 0.661 | 0.592 | -0.069 |
| eq48 | supcon | relative_operational | 0.223 | 0.048 | -0.175 | 0.687 | 0.629 | -0.057 |
