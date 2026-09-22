# Lengan HEN v2 — laporan SEMENTARA (tiga dari empat lengan selesai)

> **Status berkas ini: SEMENTARA, ditulis di tengah jalan.** Lengan 4 (`semua_eq`)
> **masih dilatih** saat berkas ini ditulis — nol dari 9 run-nya punya `result.json`,
> dan `outputs/reports_hen_semua_eq/` belum ada. Semua angka classifier di bawah
> berasal dari lengan 1–3 saja. Lantai jalan pintas lengan 4 **sudah** terukur
> (gerbang dijalankan sebelum latih, sesuai urutan di rencana), jadi lengan 4
> muncul di tabel lantai tapi **tidak** di tabel model. Bedanya harus dibaca apa
> adanya, jangan dikaburkan: "lantai sudah ada" ≠ "lengan sudah jalan".
>
> Taksiran selesai lengan 4: **~16:45–17:00** (dasar taksirannya di §6).

Tanggal: 22 September 2026.

---

## 1. Apa yang diuji lengan ini, dan mengapa ada empat

Susunan HEN v2 dipilih untuk satu alasan tunggal: **`domain` identik antar kelas.**
Kelas mati dan kelas hidup dua-duanya `origin = hen_gt`, `domain = hen_kandang`,
dua-duanya anotasi manusia dari zip yang sama. Itu membuat kegagalan yang sudah
tercatat di lengan PIO — label yang sebetulnya menandai *sumber rekaman*, bukan
kondisi ayam — mustahil secara konstruksi, bukan cuma mustahil menurut harapan.

Empat lengan = dua susunan × dua perlakuan resolusi:

| lengan | susunan | equalize (short side 96) | status |
|---|---|---|---|
| 1 | `campur` — hanya frame yang memuat mati **dan** hidup | tidak | **selesai** |
| 2 | `campur_eq` | ya | **selesai** |
| 3 | `semua` — seluruh HEN v2 | tidak | **selesai** |
| 4 | `semua_eq` | ya | **masih latih** |

Susunan `campur` yang paling ketat: karena satu frame memuat kedua kelas, tidak ada
ciri global tingkat-frame (pencahayaan kandang, kamera, resolusi ekspor) yang bisa
memisahkan label — ciri itu dibagi rata oleh kedua kelas dalam satu foto yang sama.
Susunan `semua` sengaja **lebih longgar** dan fungsinya pembanding: selisih lantai
antara keduanya *adalah* pengukuran sumbangan jalan pintas tingkat-frame.

---

## 2. Sensus manifest — `max_alive_per_dead` memang aktif

Dibaca langsung dari `manifest.csv` tiap lengan (label `1` = mati, `0` = hidup):

| lengan | crop | foto unik | train hidup | train mati | val hidup | val mati |
|---|---|---|---|---|---|---|
| `campur` / `campur_eq` | 203 | 36 | 76 | 44 | 50 | 33 |
| `semua` / `semua_eq` | 2000 | 1034 | 852 | 297 | 648 | 203 |

Pasangan asli/eq48 **identik crop-per-crop** (203 = 203, 2000 = 2000, foto 36 = 36,
1034 = 1034). Itu yang diinginkan: satu-satunya perbedaan antar pasangan adalah
`crops.equalize_resolution.enabled`, bukan komposisi datanya.

**Tiga hal yang harus disebut, bukan didiamkan.**

1. Lengan `semua` berakhir **500 mati / 1500 hidup**, bukan 719/4378 seperti angka
   proyeksi di rencana. `max_alive_per_dead: 3.0` **aktif** dan memangkas kelas hidup
   ke tepat 3× kelas mati. Rasio 1:3 itu keputusan yang dipasang sendiri, jadi
   angka bacc di lengan ini tidak boleh dibandingkan dengan lengan lain yang rasio
   kelasnya beda.
2. Kelas hidup dipangkas dua kali, dan hanya sisi hidup yang tercatat angkanya:
   `max_alive_per_dead` membuang **2216** crop hidup, dan `min_box_size` membuang
   **245** crop hidup lagi (terukur pada build ini, dicatat di kepala
   `configs/config_hen_semua.yaml:29-31`). Angka 12 dipilih karena pada 16 filternya
   membuang 8 crop **hidup** dan nol mati — filter yang memangkas satu kelas saja
   adalah jalan pintas yang kita pasang dengan tangan sendiri.
3. Kelas mati berakhir **500**, sementara proyeksi rencana 719 pada 1399 foto unik;
   build ini menemukan **1034** foto unik. Ke mana 219 crop mati itu pergi **belum
   diukur** — kandidatnya `min_box_size` dan de-dup ekspor-ulang per foto, tapi
   hitungan buangan yang tercetak hanya untuk sisi hidup. Ini dicatat sebagai lubang
   yang masih terbuka, bukan dijelaskan dengan dugaan.

Lengan `campur` kecil: val 83 crop (50 hidup, 33 mati). **Satu crop ≈ 1,0 poin bacc.**
Itu batas resolusi yang mengikat seluruh pembacaan lengan 1 dan 2.

---

## 3. Lantai jalan pintas — empat lengan, DUA aturan berdampingan

Ini blok terpenting di laporan sementara ini, dan dilaporkan dengan dua kolom karena
satu angka akan menyesatkan.

Lantai = satu ciri, satu ambang dicocokkan di **train**, diuji di **val**, tanpa
melihat isi gambar. AUC-nya berarah: `max(auc, 1−auc)`, positif = mati.
Sumber: `outputs/predictions_hen/hen_<lengan>_shortcut.json`.

| lengan | ciri terpilih oleh `train_bacc` | val_bacc-nya | ciri terpilih oleh AUC berarah | AUC | val_bacc-nya |
|---|---|---|---|---|---|
| `campur` | `ukuran_bbox` | 0.5900 | **`saturasi`** | **0.7436** | **0.7136** |
| `campur_eq` | `ukuran_bbox` | 0.5900 | **`saturasi`** | **0.7430** | **0.7136** |
| `semua` | `ketajaman` | 0.7179 | `ketajaman` | **0.7485** | **0.7179** |
| `semua_eq` | `ukuran_bbox` | 0.6929 | `ukuran_bbox` | **0.7301** | **0.6929** |

Rincian penuh tiap lengan (urut AUC menurun):

**`campur`** — saturasi 0.7436 · hue 0.6321 · std_terang 0.6006 · ukuran_bbox 0.5852 ·
rasio_bbox 0.5342 · ketajaman 0.5327 · terang 0.5012
**`campur_eq`** — saturasi 0.7430 · hue 0.6315 · std_terang 0.6055 · ukuran_bbox 0.5852 ·
ketajaman 0.5358 · rasio_bbox 0.5342 · terang 0.5012
**`semua`** — ketajaman 0.7485 · ukuran_bbox 0.7301 · saturasi 0.6806 · rasio_bbox 0.5468 ·
std_terang 0.5269 · hue 0.5187 · terang 0.5088
**`semua_eq`** — ukuran_bbox 0.7301 · ketajaman 0.7234 · saturasi 0.6711 · rasio_bbox 0.5468 ·
hue 0.5233 · std_terang 0.5206 · terang 0.5082

### 3a. Dua aturan tidak sepakat di lengan 1 dan 2

Di `campur` dan `campur_eq` aturan `train_bacc` menunjuk `ukuran_bbox` sementara
aturan AUC menunjuk `saturasi`, dan palangnya beda jauh: **0.5900 vs 0.7136, selisih
12,4 poin bacc.** Ini pengulangan cacat yang sudah tercatat: dua aturan untuk satu
besaran menghasilkan dua angka lantai yang diam-diam berbeda.

Aturan yang dipakai di laporan ini dan di `src/report.py` adalah **AUC berarah
tertinggi**, sama dengan gerbang `eval_shortcut_baseline.py`. Alasannya: palang wajib
mewakili **kebocoran terbesar yang ada**, dan `train_bacc` bisa tinggi hanya karena
ambangnya kebetulan pas di train (di `campur`, `ukuran_bbox` train_bacc 0.7033 jatuh
ke val_bacc 0.5900 — itu overfit ambang, bukan kebocoran). Memilih angka yang lebih
rendah = menurunkan palang sendiri.

### 3b. Proyeksi rencana TIDAK berlaku — dan itu melemahkan klaim, bukan memperkuat

Rencana memproyeksikan lantai lengan `semua` di `sisi_pendek` **0.8182**. Setelah
split ulang per foto, ciri itu (bernama `ukuran_bbox` di skrip) mengukur hanya
**0.7301**, dan yang tertinggi justru `ketajaman` 0.7485.

Arah koreksinya harus dinyatakan terang: palangnya **turun ~7 poin AUC**, jadi
lolosnya model di lengan 3 adalah klaim yang **lebih lemah** dari yang direncanakan,
bukan lebih kuat. Palang rendah membuat lolos lebih mudah.

### 3c. eq48 hampir tidak bisa apa-apa di lengan ini

| yang diukur | asli | eq48 | selisih |
|---|---|---|---|
| `campur`, saturasi (pengikat) | 0.7436 | 0.7430 | −0.0006 |
| `semua`, ketajaman | 0.7485 | 0.7234 | −0.0251 |
| `semua`, ukuran_bbox | 0.7301 | **0.7301** | **0.0000** |

`ukuran_bbox` **persis nol** selisihnya, dan itu bukan kebetulan: ciri itu dibaca
dari **manifest**, yaitu sisi pendek bbox pada foto ASLI, bukan dari piksel crop.
Equalize bekerja pada piksel, jadi secara konstruksi ia tidak bisa menyentuhnya. Di
`semua_eq` ciri itu naik jadi pengikat justru karena `ketajaman` yang berhasil
diturunkan (0.7485 → 0.7234), sementara `ukuran_bbox` tidak bergerak — jalan pintas
tidak mati, hanya **bertukar**.

Konsekuensi praktis: di lengan `semua_eq`, eq48 hanya memindahkan palang dari
0.7485 ke 0.7301, turun 1,8 poin. eq48 bukan penawar untuk susunan ini.

---

## 4. Hasil classifier, lengan 1–3 (semua di split VAL)

Ketiga lengan memakai `protocol.development_only: true`, jadi manifestnya **tidak
punya split test** — test ditahan supaya tidak terpakai selama pengembangan.
Konsekuensinya mengikat: val dipakai memilih checkpoint **dan** ambang, jadi angka
di bawah optimistis dan **bukan** estimasi kemampuan pada data baru. Kolom `@tau`
sengaja kosong di seluruh laporan HEN.

### Lengan 1 — `campur` (val 83 crop; lantai bacc 0.7136 / AUC 0.7436)

| Augmentasi | Metode | Bal.Acc | AUC | R.mati | R.hidup |
|---|---|---|---|---|---|
| `hier_addone` | ce | **0.8323 ± 0.0194** | 0.8590 ± 0.0088 | 0.7980 | 0.8667 |
| `stacked_randaug` | supcon | 0.8205 ± 0.0232 | 0.8463 ± 0.0028 | 0.7677 | 0.8733 |
| `simclr` | selfcon | 0.7390 ± 0.0255 | 0.8050 ± 0.0267 | 0.7980 | 0.6800 |

3/3 di atas lantai bacc. selfcon paling tipis: 0.7390 − 0.0255 = 0.7135, **menyentuh
lantai 0.7136 dari bawah** setelah dikurangi satu simpangan baku. Dengan 1 crop ≈ 1 poin,
selfcon di lengan ini tidak membuktikan apa pun di luar galat pengukuran.

### Lengan 2 — `campur_eq` (val 83 crop; lantai bacc 0.7136 / AUC 0.7430)

| Augmentasi | Metode | Bal.Acc | AUC | R.mati | R.hidup |
|---|---|---|---|---|---|
| `hier_addone` | ce | **0.8290 ± 0.0148** | **0.8748 ± 0.0164** | 0.7980 | 0.8600 |
| `stacked_randaug` | supcon | 0.8190 ± 0.0284 | 0.8626 ± 0.0154 | 0.7980 | 0.8400 |
| `simclr` | selfcon | 0.7541 ± 0.0255 | 0.7952 ± 0.0479 | 0.8283 | 0.6800 |

Bandingkan berpasangan dengan lengan 1 — lantainya hampir sama (0.7430 vs 0.7436),
jadi pasangan ini perbandingan yang sah:

| metode | bacc asli → eq48 | AUC asli → eq48 |
|---|---|---|
| ce | 0.8323 → 0.8290 (−0.0033) | 0.8590 → **0.8748** (+0.0158) |
| supcon | 0.8205 → 0.8190 (−0.0015) | 0.8463 → **0.8626** (+0.0163) |
| selfcon | 0.7390 → **0.7541** (+0.0151) | 0.8050 → 0.7952 (−0.0098) |

Arah bacc dan AUC **berlawanan** untuk ketiga metode. Semua selisih ini di bawah satu
simpangan baku antar-seed, jadi bacaan yang benar: **eq48 tidak mengubah apa pun yang
bisa dibedakan dari derau di lengan campur.** Itu konsisten dengan §3c — lantainya
juga tak bergerak (−0.0006), jadi tidak ada jalan pintas yang tersedia untuk dimatikan.

### Lengan 3 — `semua` (val 851 crop; lantai bacc 0.7179 / AUC 0.7485)

| Augmentasi | Metode | Bal.Acc | AUC | R.mati | R.hidup |
|---|---|---|---|---|---|
| `hier_addone` | ce | **0.9534 ± 0.0034** | **0.9854 ± 0.0021** | 0.9392 | 0.9676 |
| `stacked_randaug` | supcon | 0.9421 ± 0.0081 | 0.9779 ± 0.0027 | 0.9064 | 0.9779 |
| `simclr` | selfcon | 0.9306 ± 0.0032 | 0.9715 ± 0.0013 | 0.9245 | 0.9367 |

3/3 di atas lantai dengan jarak lebar (terkecil: selfcon 0.9306 − 0.0032 = 0.9274,
yaitu **21,0 poin** di atas 0.7179). Val di sini **851 crop**, sepuluh kali lengan 1,
jadi simpangan baku antar-seed yang ≤0.008 memang berarti sesuatu.

Tapi jangan dibaca sebagai "lengan 3 lebih baik". Lengan 3 **lebih longgar**: lantainya
0.7485 vs 0.7436, dan ciri pengikatnya `ketajaman` — ciri tingkat-frame, yang di lengan
`campur` mustahil bekerja (0.5327, praktis kebetulan) justru karena kedua kelas
berbagi frame. Sebagian jarak 21 poin itu ruang yang dibuka oleh keberadaan jalan
pintas tingkat-frame, bukan seluruhnya kemampuan mengenali ayam mati.

### 4a. Urutan metode konsisten di ketiga lengan

ce > supcon > selfcon pada bacc **dan** AUC, di lengan 1, 2, dan 3. Tapi urutan itu
tidak bisa mengatribusikan sebab: tiap metode terikat augmentasinya sendiri
(`ce`+`hier_addone`, `supcon`+`stacked_randaug`, `selfcon`+`simclr`), jadi diagonal
mengubah **loss DAN augmentasi bersamaan**. Yang bisa dinyatakan: ce dengan
`hier_addone` unggul secara stabil; **apakah karena loss-nya atau augmentasinya masih
belum terjawab** dan hanya bisa dijawab dengan menahan salah satunya tetap.

Kebijakan probe sudah dikunci dan terbukti di seluruh grid: `selfcon`/`supcon` memakai
`minimal` (flip saja) di tahap probe, dibaca dari `result.json` tiap run, bukan
ditulis tangan. `ce` tidak bisa ikut dikunci karena kepalanya *adalah* satu-satunya
tahap latihnya — asimetri ini nyata dan tidak disembunyikan.

---

## 5. Yang BELUM dikerjakan

| pekerjaan | status |
|---|---|
| lengan 4 `semua_eq`, 9 run | **sedang latih**, 0/9 `result.json` |
| `report.py --config configs/config_hen_semua_eq.yaml` | menunggu lengan 4 |
| `eval_fixed_chick.py` atas 4 registry HEN (`--interventions asli,acak16`) | belum |
| `report_fixed_chick.py` | belum |
| Bagian B — KolektorSDD2 dua tahap (B1–B6) | belum |
| `eval_intervensi.py` | Kolektor saja; **tidak bisa** jalan di lengan HEN karena `eval_intervensi.py:94` hanya membaca split literal `test`, dan `development_only` tidak punya test — konsekuensi protokol, bukan bug |

Registry lengan 4 juga belum ada di `outputs/predictions_hen/` (hanya
`hen_semua_eq_shortcut.json` yang ada — gerbang jalan sebelum latih).

---

## 6. Dasar taksiran waktu lengan 4

Durasi per-seed terukur dari lengan yang sudah selesai: selfcon ~25 menit,
supcon ~29 menit, ce ~6 menit. Lengan 3 memakan **180 menit bersih** setelah
mengeluarkan satu jeda nyata 33 menit (12:01–12:34, proses di-*suspend* lewat
`_jeda.ps1`, bukan hang) yang mengembungkan `supcon__s42` ke 58 menit dari ~29 menit
saudara seed-nya.

Rasio eq/asli **0.91** diukur dari lengan 1 vs lengan 2 yang sudah lengkap →
lengan 4 ≈ **164 menit**, selesai ~16:45. Skenario sama-dengan-lengan-3: 17:01.
Pesimis: 17:28.

Catatan metode yang perlu diingat: **mtime log bukan penanda kemajuan.** Sempat
tampak macet 3,5 jam (`_run36.log` mtime 10:28 sementara jam 14:00) — ternyata log
ter-*buffer* dan menyusul saat run selesai; prosesnya sendiri punya 60.745 detik CPU
dan 8 thread Running. Penanda yang bisa dipercaya adalah `result.json` per run.

---

## 7. Yang sudah bisa disimpulkan sekarang

1. **Susunan `domain` identik berhasil dibangun.** Kedua kelas satu origin, satu
   domain, satu zip. Kegagalan `label = domain` dari lengan PIO tidak bisa terjadi
   di sini secara konstruksi.
2. **Lantai turun dari proyeksi, jadi palangnya lebih rendah dari rencana** —
   0.7485 bukan 0.8182. Lolos di lengan 3 adalah klaim lebih lemah dari yang
   dirancang, dan itu harus dinyatakan, bukan dinikmati.
3. **eq48 tidak menjadi penawar di susunan ini.** Di `campur` ia tak bergerak
   (−0.0006 di lantai, semua selisih model di bawah 1 SD). Di `semua` ia menurunkan
   `ketajaman` tapi `ukuran_bbox` naik menggantikannya — jalan pintas **bertukar**,
   tidak mati, karena ciri itu dibaca dari manifest dan tak tersentuh operasi piksel.
4. **Dua aturan pemilihan lantai masih tidak sepakat** di lengan 1 dan 2, dengan
   jarak 12,4 poin bacc. Laporan ini memakai AUC berarah dan menyebut angka aturan
   lain berdampingan, supaya palangnya tidak bisa dipilih belakangan.
5. **Seluruh angka model di sini val, bukan test.** Tidak ada satu pun lengan HEN
   yang punya split test. Klaim generalisasi menunggu benchmark `eval_fixed_chick.py`,
   yang belum dijalankan.

---

## 8. Reproduksi

```sh
PY="C:/Arib/MASSA AYAM/generalisasi-ayam-skripsi/.venv-yolo/Scripts/python.exe"

# gerbang lantai (sudah jalan untuk keempat lengan)
for c in campur campur_eq semua semua_eq; do
  "$PY" src/eval_shortcut_baseline.py --crops data/crops_hen_$c \
        --json outputs/predictions_hen/hen_${c}_shortcut.json
done

# latih (lengan 1-3 selesai; semua_eq sedang jalan)
for c in campur campur_eq semua semua_eq; do
  "$PY" src/train.py --config configs/config_hen_$c.yaml --method all \
        --seeds 42,43,44 --save-model all
done

# laporan per lengan (semua_eq menunggu)
for c in campur campur_eq semua semua_eq; do
  "$PY" src/report.py --config configs/config_hen_$c.yaml
done
```

Berkas sumber angka: `outputs/reports_hen_{campur,campur_eq,semua}/comparison.{csv,md}`,
`outputs/predictions_hen/hen_*_shortcut.json`, `data/crops_hen_*/manifest.csv`.
