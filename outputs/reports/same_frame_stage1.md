# Tahap 1 Same-Frame (asli)

## Protokol

Evaluasi memakai 943 crop valid (22 mati, 921 hidup) dari 18 frame. Sebanyak 272 crop `bukan` dikeluarkan. Skor relatif utama ditetapkan sebelum hasil dilihat: jarak kosinus fitur backbone 512-d terhadap median koordinat-wise frame secara leave-one-out.

Ambang keputusan dipilih dari 17 frame lain untuk setiap frame uji. CI 95% me-resample 18 frame sebagai klaster (10000 replikasi).

## Hasil Tiga Seed

| domain | metode | scorer | AUC pooled | AP pooled | AUC macro | Recall@3 | MRR |
|---|---|---|---:|---:|---:|---:|---:|
| closeup | ce | absolute | 0.617 +/- 0.073 | 0.067 +/- 0.033 | 0.640 +/- 0.091 | 0.213 +/- 0.153 | 0.199 +/- 0.114 |
| closeup | ce | feature512_median_loo | 0.627 +/- 0.014 | 0.126 +/- 0.017 | 0.662 +/- 0.010 | 0.278 +/- 0.111 | 0.241 +/- 0.036 |
| closeup | selfcon | absolute | 0.668 +/- 0.056 | 0.059 +/- 0.026 | 0.687 +/- 0.067 | 0.167 +/- 0.096 | 0.187 +/- 0.095 |
| closeup | selfcon | feature512_median_loo | 0.502 +/- 0.052 | 0.027 +/- 0.004 | 0.513 +/- 0.049 | 0.056 +/- 0.056 | 0.085 +/- 0.037 |
| closeup | supcon | absolute | 0.668 +/- 0.047 | 0.078 +/- 0.048 | 0.688 +/- 0.058 | 0.167 +/- 0.127 | 0.243 +/- 0.121 |
| closeup | supcon | feature512_median_loo | 0.613 +/- 0.045 | 0.043 +/- 0.007 | 0.607 +/- 0.058 | 0.139 +/- 0.048 | 0.158 +/- 0.054 |
| pio | ce | absolute | 0.550 +/- 0.074 | 0.030 +/- 0.007 | 0.529 +/- 0.099 | 0.056 +/- 0.056 | 0.088 +/- 0.038 |
| pio | ce | feature512_median_loo | 0.594 +/- 0.083 | 0.051 +/- 0.016 | 0.612 +/- 0.068 | 0.204 +/- 0.064 | 0.170 +/- 0.028 |
| pio | selfcon | absolute | 0.621 +/- 0.056 | 0.035 +/- 0.011 | 0.603 +/- 0.067 | 0.083 +/- 0.048 | 0.113 +/- 0.059 |
| pio | selfcon | feature512_median_loo | 0.521 +/- 0.052 | 0.027 +/- 0.007 | 0.552 +/- 0.050 | 0.037 +/- 0.064 | 0.077 +/- 0.021 |
| pio | supcon | absolute | 0.691 +/- 0.097 | 0.102 +/- 0.051 | 0.710 +/- 0.093 | 0.333 +/- 0.242 | 0.318 +/- 0.139 |
| pio | supcon | feature512_median_loo | 0.541 +/- 0.127 | 0.029 +/- 0.011 | 0.594 +/- 0.150 | 0.037 +/- 0.032 | 0.092 +/- 0.035 |
| pio_eq48 | ce | absolute | 0.732 +/- 0.067 | 0.143 +/- 0.100 | 0.708 +/- 0.070 | 0.315 +/- 0.179 | 0.318 +/- 0.134 |
| pio_eq48 | ce | feature512_median_loo | 0.614 +/- 0.055 | 0.076 +/- 0.064 | 0.622 +/- 0.046 | 0.130 +/- 0.085 | 0.171 +/- 0.056 |
| pio_eq48 | selfcon | absolute | 0.629 +/- 0.082 | 0.041 +/- 0.011 | 0.644 +/- 0.085 | 0.111 +/- 0.056 | 0.174 +/- 0.086 |
| pio_eq48 | selfcon | feature512_median_loo | 0.524 +/- 0.118 | 0.027 +/- 0.009 | 0.555 +/- 0.107 | 0.074 +/- 0.085 | 0.098 +/- 0.048 |
| pio_eq48 | supcon | absolute | 0.741 +/- 0.063 | 0.324 +/- 0.021 | 0.713 +/- 0.084 | 0.472 +/- 0.028 | 0.469 +/- 0.039 |
| pio_eq48 | supcon | feature512_median_loo | 0.711 +/- 0.077 | 0.187 +/- 0.047 | 0.759 +/- 0.094 | 0.435 +/- 0.112 | 0.396 +/- 0.039 |

## Ringkasan

- Semua kombinasi dilaporkan secara deskriptif; tidak ada metode yang dipilih dari data uji ini.
- Baseline ketajaman tanpa equalization: AUC 0.536, AP 0.032, Recall@3 0.111.
- Baseline ketajaman setelah eq48: AUC 0.349, AP 0.017, Recall@3 0.000.
- Angka ini adalah evaluasi clean-crop; pusat frame tidak memasukkan 272 crop `bukan`.

## Sensitivitas

Scorer medoid dan median 5-NN dihitung sebagai analisis sensitivitas, bukan untuk mengganti scorer utama setelah hasil dilihat. Detail lengkap per checkpoint, CI, ambang fold, dan skor mentah ada di JSON/CSV.
