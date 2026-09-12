# Tahap 1 Same-Frame (acak16)

## Protokol

Evaluasi memakai 943 crop valid (22 mati, 921 hidup) dari 18 frame. Sebanyak 272 crop `bukan` dikeluarkan. Skor relatif utama ditetapkan sebelum hasil dilihat: jarak kosinus fitur backbone 512-d terhadap median koordinat-wise frame secara leave-one-out.

Ambang keputusan dipilih dari 17 frame lain untuk setiap frame uji. CI 95% me-resample 18 frame sebagai klaster (10000 replikasi).

## Hasil Tiga Seed

| domain | metode | scorer | AUC pooled | AP pooled | AUC macro | Recall@3 | MRR |
|---|---|---|---:|---:|---:|---:|---:|
| closeup | ce | absolute | 0.587 +/- 0.016 | 0.048 +/- 0.010 | 0.628 +/- 0.052 | 0.111 +/- 0.056 | 0.169 +/- 0.047 |
| closeup | ce | feature512_median_loo | 0.532 +/- 0.044 | 0.124 +/- 0.016 | 0.534 +/- 0.021 | 0.148 +/- 0.032 | 0.186 +/- 0.030 |
| closeup | selfcon | absolute | 0.690 +/- 0.044 | 0.111 +/- 0.041 | 0.750 +/- 0.043 | 0.250 +/- 0.073 | 0.261 +/- 0.046 |
| closeup | selfcon | feature512_median_loo | 0.484 +/- 0.024 | 0.071 +/- 0.041 | 0.506 +/- 0.032 | 0.111 +/- 0.000 | 0.134 +/- 0.044 |
| closeup | supcon | absolute | 0.595 +/- 0.081 | 0.103 +/- 0.119 | 0.588 +/- 0.112 | 0.157 +/- 0.112 | 0.191 +/- 0.122 |
| closeup | supcon | feature512_median_loo | 0.527 +/- 0.088 | 0.032 +/- 0.003 | 0.554 +/- 0.100 | 0.037 +/- 0.064 | 0.099 +/- 0.021 |
| pio | ce | absolute | 0.590 +/- 0.071 | 0.037 +/- 0.012 | 0.562 +/- 0.095 | 0.056 +/- 0.056 | 0.090 +/- 0.037 |
| pio | ce | feature512_median_loo | 0.523 +/- 0.045 | 0.039 +/- 0.009 | 0.537 +/- 0.018 | 0.074 +/- 0.032 | 0.109 +/- 0.013 |
| pio | selfcon | absolute | 0.622 +/- 0.038 | 0.093 +/- 0.073 | 0.625 +/- 0.031 | 0.185 +/- 0.140 | 0.195 +/- 0.087 |
| pio | selfcon | feature512_median_loo | 0.447 +/- 0.103 | 0.023 +/- 0.006 | 0.447 +/- 0.106 | 0.019 +/- 0.032 | 0.063 +/- 0.018 |
| pio | supcon | absolute | 0.678 +/- 0.066 | 0.086 +/- 0.032 | 0.696 +/- 0.060 | 0.269 +/- 0.153 | 0.271 +/- 0.094 |
| pio | supcon | feature512_median_loo | 0.524 +/- 0.063 | 0.030 +/- 0.008 | 0.543 +/- 0.079 | 0.028 +/- 0.028 | 0.090 +/- 0.031 |
| pio_eq48 | ce | absolute | 0.695 +/- 0.070 | 0.079 +/- 0.043 | 0.701 +/- 0.085 | 0.296 +/- 0.236 | 0.232 +/- 0.124 |
| pio_eq48 | ce | feature512_median_loo | 0.509 +/- 0.064 | 0.025 +/- 0.005 | 0.504 +/- 0.067 | 0.019 +/- 0.032 | 0.070 +/- 0.016 |
| pio_eq48 | selfcon | absolute | 0.635 +/- 0.068 | 0.083 +/- 0.064 | 0.666 +/- 0.086 | 0.194 +/- 0.121 | 0.194 +/- 0.094 |
| pio_eq48 | selfcon | feature512_median_loo | 0.454 +/- 0.022 | 0.027 +/- 0.008 | 0.483 +/- 0.034 | 0.074 +/- 0.085 | 0.091 +/- 0.044 |
| pio_eq48 | supcon | absolute | 0.691 +/- 0.062 | 0.278 +/- 0.053 | 0.698 +/- 0.074 | 0.454 +/- 0.105 | 0.446 +/- 0.080 |
| pio_eq48 | supcon | feature512_median_loo | 0.618 +/- 0.033 | 0.059 +/- 0.026 | 0.647 +/- 0.009 | 0.139 +/- 0.028 | 0.155 +/- 0.062 |

## Ringkasan

- Semua kombinasi dilaporkan secara deskriptif; tidak ada metode yang dipilih dari data uji ini.
- Baseline ketajaman tanpa equalization: AUC 0.544, AP 0.026, Recall@3 0.000.
- Baseline ketajaman setelah eq48: AUC 0.176, AP 0.014, Recall@3 0.000.
- Angka ini adalah evaluasi clean-crop; pusat frame tidak memasukkan 272 crop `bukan`.

## Sensitivitas

Scorer medoid dan median 5-NN dihitung sebagai analisis sensitivitas, bukan untuk mengganti scorer utama setelah hasil dilihat. Detail lengkap per checkpoint, CI, ambang fold, dan skor mentah ada di JSON/CSV.
