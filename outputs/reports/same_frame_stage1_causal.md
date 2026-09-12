# Gerbang Kausal Tahap 1: Asli vs Acak16

Delta di bawah adalah `acak16 - asli`. Nilai AUC yang tidak turun setelah susunan bentuk dihancurkan berarti scorer belum terbukti membaca pose.

| domain | metode | scorer | delta AUC macro | CI 95% klaster | delta Recall@3 |
|---|---|---|---:|---:|---:|
| closeup | ce | absolute | -0.011 +/- 0.041 | [-0.081, +0.061] | -0.102 |
| closeup | ce | feature512_median_loo | -0.128 +/- 0.020 | [-0.229, -0.025] | -0.130 |
| closeup | selfcon | absolute | +0.063 +/- 0.032 | [+0.018, +0.107] | +0.083 |
| closeup | selfcon | feature512_median_loo | -0.008 +/- 0.078 | [-0.089, +0.070] | +0.056 |
| closeup | supcon | absolute | -0.100 +/- 0.124 | [-0.163, -0.041] | -0.009 |
| closeup | supcon | feature512_median_loo | -0.053 +/- 0.049 | [-0.116, +0.006] | -0.102 |
| pio | ce | absolute | +0.033 +/- 0.143 | [-0.037, +0.096] | +0.000 |
| pio | ce | feature512_median_loo | -0.075 +/- 0.051 | [-0.160, +0.003] | -0.130 |
| pio | selfcon | absolute | +0.022 +/- 0.043 | [-0.077, +0.130] | +0.102 |
| pio | selfcon | feature512_median_loo | -0.105 +/- 0.105 | [-0.159, -0.044] | -0.019 |
| pio | supcon | absolute | -0.014 +/- 0.034 | [-0.051, +0.027] | -0.065 |
| pio | supcon | feature512_median_loo | -0.051 +/- 0.090 | [-0.122, +0.012] | -0.009 |
| pio_eq48 | ce | absolute | -0.007 +/- 0.075 | [-0.065, +0.057] | -0.019 |
| pio_eq48 | ce | feature512_median_loo | -0.118 +/- 0.090 | [-0.188, -0.049] | -0.111 |
| pio_eq48 | selfcon | absolute | +0.022 +/- 0.004 | [-0.042, +0.084] | +0.083 |
| pio_eq48 | selfcon | feature512_median_loo | -0.072 +/- 0.116 | [-0.148, -0.008] | +0.000 |
| pio_eq48 | supcon | absolute | -0.014 +/- 0.062 | [-0.058, +0.035] | -0.019 |
| pio_eq48 | supcon | feature512_median_loo | -0.111 +/- 0.102 | [-0.201, -0.023] | -0.296 |

## Keputusan

Gerbang kausal **lolos secara arah**: 9 dari 9 kombinasi domain/metode mengalami penurunan AUC macro scorer relatif utama setelah bentuk dihancurkan, dan 5 di antaranya memiliki CI 95% seluruhnya di bawah nol. Ini menunjukkan scorer relatif membaca informasi susunan/bentuk, meski belum membuktikan bahwa seluruh sinyalnya adalah pose ayam.

Intervensi ini tidak dipakai memilih metode. Manfaat skor relatif terhadap classifier absolut tetap harus dinilai bersama metrik pooled, macro, dan Recall@K pada hasil asli.
