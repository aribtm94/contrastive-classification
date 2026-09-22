# Hitung Manual — Deteksi sampai Klasifikasi Kontrastif

Rantai penuh **I → X**: dari gambar mentah masuk YOLOv8m (I–II), lewat pembentukan crop (III–V), augmentasi dan encoder kontrastif (VI–VII), loss tiga metode (VIII), linear probe (IX), sampai ambang dan metrik beserta lantai ketajamannya (X).

<aside>
🧮

**Dari mana angkanya.** Dua sumber, dan keduanya disebut terang-terangan:

| Bagian | Sumber angka |
| --- | --- |
| I–II | `scripts/extract_manual_deteksi.py` → `results/hitung_manual_deteksi.json` |
| III–X | `scripts/extract_manual_klasifikasi.py` → `results/hitung_manual_klasifikasi.json` |

Kecuali dua contoh bertanda 📄 **pinjaman** di II(a)–II(b), yang diukur pada proyek lain (PIO + SAFECount) dan dipakai hanya untuk memperlihatkan bentuk aritmetikanya. Bagian yang bukan data asli diberi tanda ⚠️ asumsi. Tidak ada satu pun angka contoh yang dikarang.

</aside>

- **Crop contoh (mati):** `data/crops/dead/dead_train_image--10-_16_00.jpg`, berasal dari `image--10-_jpg.rf.0fc7ca9f...jpg` (416×416), bbox anotasi $(36,\ 124,\ 416,\ 336.5)$, `origin = coco_gt`, split **test**.
- **Crop contoh (hidup):** `data/crops/alive/alive_train_image--10-_18_001.jpg`, foto asal yang **sama**, bbox $(226.3,\ 0.2,\ 356.3,\ 49.2)$, conf 0.4403, `origin = bird_det`, split **test**.
- **Model:** ResNet-18 pretrained ImageNet + projection head 512→512→128 + kepala linear 512→2. Checkpoint `outputs/runs/selfcon/model.pt` (metode `selfcon`, augmentasi `simclr`, seed 42).
- **Sumber metrik:** `outputs/runs/{selfcon,supcon,ce}/result.json` beserta `val_scores.npz` dan `test_scores.npz`.
- **Konvensi:** box ditulis $(x_1,y_1,x_2,y_2)$ dalam piksel ruang gambar asal; tensor ditulis $(C,H,W)$; kelas **0 = hidup**, **1 = mati**; kelas positif = **mati**; angka dibulatkan 4–6 desimal.

Semua angka dapat dihasilkan ulang dengan:

```bash
PY="C:/Arib/MASSA AYAM/generalisasi-ayam-skripsi/.venv-yolo/Scripts/python.exe"
"$PY" scripts/extract_manual_deteksi.py       # bagian I-II
"$PY" scripts/extract_manual_klasifikasi.py   # bagian III-X
```

Tiap bagian memakai pola yang sama: **alur** operasi, **fungsi krusial** beserta arti simbolnya, lalu **contoh hitung memakai data asli**.

## Peta alur besar

```
gambar mentah (432x432 / 400x400)
        |
        v
I. Pra-pemrosesan
   - letterbox ke 640x640, r > 1, bantalan 0 px
   - BGR -> RGB, /255
        |
        v
II. Forward YOLOv8m
   - backbone -> neck -> Detect (8 400 anchor)
   - DFL decode -> conf >= 0.25 -> NMS IoU 0.7
   - HANYA untuk kelas hidup; kelas mati dari bbox anotasi
        |
        v
III. Ekstraksi crop
   - lebarkan bbox 10% (pad_box)
   - potong, klip ke tepi gambar
   - buang bbox < 16 px
        |
        v
IV. Letterbox ke 224x224
   - r = min(224/h, 224/w), INTER_AREA / INTER_LINEAR
   - kanvas abu-abu (114,114,114), tempel di tengah
        |
        v
V. Normalisasi
   - BGR -> RGB, /255
   - (x - mu) / sigma  (ImageNet)      -> tensor 3x224x224
        |
        v
VI. Augmentasi (hanya saat latih)
   - satu kebijakan per metode: simclr | stacked_randaug | hier_addone
   - RNG turunan per-op -> dua view dari crop yang sama
        |
        +-------------------+
        v                   v
VII. Forward encoder   (view kedua)
   - ResNet18 -> f (512-d)
   - projector -> g (128-d) -> L2-normalisasi -> z
   - classifier -> logit (2-d)
        |
        v
VIII. Loss
   - selfcon: NT-Xent(z1, z2)          [tanpa label]
   - supcon : SupCon(z1, z2, y)        [pakai label]
   - ce     : Cross-Entropy berbobot(logit, y)
        |
        v
IX. Linear probe (hanya selfcon & supcon)
   - backbone DIBEKUKAN, hanya classifier dilatih
   - logit -> softmax -> s = P(mati)
        |
        v
X. Ambang & metrik
   - tau dipilih di VALIDATION (dataran bacc terlebar)
   - bacc, AUC (Mann-Whitney), matriks konfusi
   - dibandingkan terhadap LANTAI KETAJAMAN 0.9231
```

Dua hal yang perlu dipegang sejak awal:

- **Titik pisah metode ada di bagian VIII, bukan di arsitektur.** Ketiga metode memakai backbone, head, ukuran input, dan optimizer yang sama persis. Yang berbeda hanya fungsi loss-nya — dan, sayangnya, juga kebijakan augmentasinya (lihat catatan di bagian VI).
- **Bagian X bukan pelengkap.** Pada dataset ini ketajaman gambar saja sudah memberi balanced accuracy 0.9231 tanpa melihat isi gambar. Angka model apa pun harus dibaca relatif terhadap lantai itu.

---

## I. Pra-pemrosesan

<aside>
⚠️

**Provenans bagian I–II.** Halaman aslinya ditulis untuk proyek **lain** — penghitungan ayam PIO + SAFECount — dan hampir setiap konstantanya berbeda dari repo ini. Yang ditulis di bawah adalah bagian I–II **versi repo ini**, diukur ulang lewat `scripts/extract_manual_deteksi.py` → `results/hitung_manual_deteksi.json`.

| | PIO / SAFECount (halaman lama) | Repo ini |
| --- | --- | --- |
| Ukuran gambar asal | 1080×1920 | 432×432 dan 400×400 |
| `imgsz` deteksi | 960 | **640** |
| Padding letterbox | 210 px atas-bawah | **0 px** (gambar sudah persegi) |
| Lantai confidence | 0.05 | **0.25** |
| Ambang IoU NMS | 0.45 | **0.7** |
| Kelas yang lewat detektor | semua objek | **hanya kelas hidup**; kelas mati dari bbox anotasi COCO |

Satu-satunya bagian yang **dipinjam apa adanya** dari halaman lama adalah contoh hitung conv/BN/DFL per-elemen di II(a)–II(b). Angka itu hanya bisa diambil dengan hook forward pass pada bobot deteksi, dan tidak pernah dijalankan pada repo ini. Contoh itu diberi tanda 📄 **pinjaman** dan tidak boleh dikutip sebagai hasil repo ini — yang berlaku darinya adalah **bentuk rumusnya**, bukan nilainya.

</aside>

### Alur

```
gambar asli (432x432 atau 400x400, BGR)
  -> r = min(640/H, 640/W)
  -> resize ke 640x640 (INTER_LINEAR, karena r > 1)
  -> padding abu-abu: 0 px, sebab gambar sudah persegi
  -> BGR->RGB, /255                     -> input YOLOv8m
```

### Fungsi krusial

**Letterbox**

$$
r=\min\left(\frac{S}{H_0},\frac{S}{W_0}\right),\qquad S=640
$$

$$
H_n=\operatorname{round}(H_0r),\qquad W_n=\operatorname{round}(W_0r)
$$

$$
d_w=\frac{S-W_n}{2},\qquad d_h=\frac{S-H_n}{2}
$$

**Normalisasi masuk YOLO**

$$
x_{YOLO}=\frac{p}{255}
$$

**Keterangan:**

- $H_0,W_0$: tinggi dan lebar gambar asli.
- $S$: sisi kanvas letterbox, di sini 640 — nilai `detection.imgsz` pada `configs/config.yaml`.
- $r$: faktor skala yang menjaga aspect ratio.
- $H_n,W_n$: ukuran setelah resize, sebelum padding.
- $d_w,d_h$: padding pada satu sisi horizontal dan vertikal.
- $p$: nilai piksel 8-bit 0–255.

### Contoh hitung (data asli)

Untuk gambar 432×432:

$$
r=\min(640/432,\ 640/432)=1.481481
$$

$$
H_n=W_n=\operatorname{round}(432\times1.481481)=640,\qquad d_w=d_h=0
$$

Untuk gambar 400×400: $r=1.6$, $H_n=W_n=640$, $d_w=d_h=0$.

Dua hal yang berbeda dari kasus PIO dan penting dibaca sampai bagian X:

- **Tidak ada bantalan abu-abu sama sekali di tahap deteksi.** Gambar sumbernya sudah persegi, jadi $d_w=d_h=0$. Bantalan baru muncul di bagian IV, saat *crop* yang tidak persegi di-letterbox ke 224×224 — dan di sana bantalan itu justru menjadi **confound**: fraksi bantalan saja sudah memberi AUC 0.9038 (lihat Lampiran B).
- **$r>1$ di kedua ukuran**, artinya gambar **diperbesar**, bukan diperkecil, dengan `INTER_LINEAR`. Tidak ada detail baru yang dibuat — hanya piksel yang diregangkan.

Keluaran nyata detektor pada 18 gambar uji (`outputs/predictions/detections.json`, 1215 box):

| Besaran | Nilai |
| --- | --- |
| Box per gambar | min 37, median 68.5, maks 114 |
| Confidence | min 0.2511, median 0.8215, maks 0.9085 |
| Sisi pendek box (ruang gambar asal) | min 4.35, median 38.65, maks 79.61 px |
| Sisi pendek box (ruang 640) | median 57.26 px |

Confidence minimum 0.2511 bukan kebetulan: `detection.conf = 0.25` memotong apa pun di bawahnya sebelum keluaran ditulis.

---

## II. Forward YOLOv8m

### Alur

```
input 3x640x640
 L0  Conv(3->48, k3, s2) + BN + SiLU      48x320x320
 L1  Conv(48->96, k3, s2)                 96x160x160
 L2  C2f(96, n=2)                         96x160x160
 L3-4  Conv s2 + C2f(192, n=4)           192x80x80
 L5-8  ... (384 @40, 576 @20)
 L9  SPPF(576)                           576x20x20      <- akhir backbone
 L10-15 neck: upsample + concat + C2f    192x80x80   = P3 (stride 8)
 L16-18                                  384x40x40   = P4 (stride 16)
 L19-21                                  576x20x20   = P5 (stride 32)
 L22 Detect: tiap sel -> 64 logit DFL (4 sisi x 16 bin) + logit kelas
        total anchor = 80^2 + 40^2 + 20^2 = 8 400
 -> decode box -> filter conf >= 0.25 -> NMS IoU 0.7 -> box akhir
```

Jumlah anchor di atas adalah konsekuensi aritmetika dari $S=640$, bukan angka yang disalin: $(640/8)^2+(640/16)^2+(640/32)^2=6400+1600+400=8400$. Pada $S=960$ (halaman lama) angkanya 18 900. Arsitektur YOLOv8m-nya identik; yang berubah hanya ukuran kanvas.

### Fungsi krusial

**Konvolusi** (tanpa bias karena diikuti BN):

$$
z=\sum_{c=1}^{3}\sum_{i,j}X_c(i,j)\,K_c(i,j)
$$

$$
H_{out}=\left\lfloor\frac{H+2p-k}{s}\right\rfloor+1
$$

**BatchNorm mode inferensi** (memakai running stats) dan **SiLU**:

$$
\hat z=\frac{z-\mu_{run}}{\sqrt{\sigma^2_{run}+\epsilon}},\qquad y_{BN}=\gamma\hat z+\beta
$$

$$
\operatorname{SiLU}(y)=y\,\sigma(y)=\frac{y}{1+e^{-y}}
$$

**DFL decode** (per sisi kiri, atas, kanan, bawah):

$$
p_k=\operatorname{softmax}(\ell)_k,\qquad d=\sum_{k=0}^{15}k\,p_k
$$

$$
\text{box}=\big((a_x-d_l)s,\ (a_y-d_t)s,\ (a_x+d_r)s,\ (a_y+d_b)s\big)
$$

**Confidence kelas**:

$$
c=\sigma(\ell_{cls})=\frac{1}{1+e^{-\ell_{cls}}}
$$

SiLU **tidak** dipakai pada logit kelas; logit langsung masuk sigmoid.

**IoU untuk NMS**:

$$
IoU=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}
$$

Box dengan confidence lebih rendah dibuang jika IoU-nya melebihi **0.7** terhadap box yang sudah dipilih.

**Keterangan:**

- $X$: patch input yang tertutup kernel; $K$: kernel convolution; $z$: hasil sebelum BN.
- $H,p,k,s$: ukuran input, padding, ukuran kernel, dan stride; $\lfloor\cdot\rfloor$: pembulatan ke bawah.
- $\mu_{run},\sigma^2_{run}$: mean dan variance hasil training yang dipakai saat inferensi.
- $\epsilon$: konstanta kecil penstabil pembagian; $\gamma,\beta$: parameter skala dan geser BN.
- $\hat z$: nilai setelah distandardisasi; $y_{BN}$: keluaran BN; $\sigma(\cdot)$: sigmoid.
- $\ell$: 16 logit DFL untuk satu sisi; $\ell_{cls}$: logit kelas.
- $p_k$: probabilitas bin ke-$k$; $d$: jarak kontinu hasil ekspektasi.
- $d_l,d_t,d_r,d_b$: jarak anchor ke sisi kiri, atas, kanan, bawah, dalam satuan sel.
- $(a_x,a_y)$: titik anchor, yaitu pusat sel (kolom + 0.5, baris + 0.5); $s$: stride level.
- $c$: confidence kelas setelah sigmoid.

### Contoh hitung

**(a) 📄 pinjaman — Conv L0 + BN + SiLU.** Contoh berikut **diukur pada proyek PIO** (kanvas 960, filter ke-0, output baris 202 kolom 226). Nilainya **tidak berlaku** untuk repo ini; yang dipinjam adalah bentuk aritmetikanya.

| Kanal | $\sum X_cK_c$ |
| --- | --- |
| R | 0.161126 |
| G | −0.411236 |
| B | 0.010924 |

$$
z=0.161126-0.411236+0.010924=-0.239186
$$

BN dengan $\mu_{run}=-0.035797$, $\sigma^2_{run}=0.014130$, $\epsilon=0.001$, $\gamma=3.140625$, $\beta=3.931641$:

$$
\hat z=\frac{-0.239186-(-0.035797)}{\sqrt{0.014130+0.001}}=\frac{-0.203389}{0.123004}=-1.653531
$$

$$
y_{BN}=3.140625(-1.653531)+3.931641=-1.261481,\qquad \operatorname{SiLU}(-1.261481)=-0.278433
$$

Ukuran output L0 untuk repo ini (bukan pinjaman), $S=640$:

$$
\left\lfloor\frac{640+2-3}{2}\right\rfloor+1=320
$$

**(b) 📄 pinjaman — Detect head, decode DFL.** Juga dari proyek PIO. Anchor ke-6993 di level P3 (stride 8), sel (baris 58, kolom 33), jadi $(a_x,a_y)=(33.5,58.5)$. Setelah softmax, massa probabilitas sisi kiri terkumpul di bin 4 ($p=0.4228$) dan bin 5 ($p=0.5700$):

$$
d_l\approx3(0.0016)+4(0.4228)+5(0.5700)+6(0.0054)=4.5789
$$

Dengan $d_t=3.0688$, $d_r=3.2105$, $d_b=3.0607$:

$$
x_1=(33.5-4.5789)8=231.369,\qquad y_1=(58.5-3.0688)8=443.450
$$

$$
x_2=(33.5+3.2105)8=293.684,\qquad y_2=(58.5+3.0607)8=492.486
$$

Logit kelas $\ell_{cls}=2.32133$ memberi $c=1/(1+e^{-2.32133})=0.910628$.

Pada repo ini indeks anchor tertinggi adalah 8399, bukan 18899, sedangkan stride P3 tetap 8 — rumusnya sama, populasi anchor-nya yang berbeda.

**(c) NMS — data repo ini.** Halaman lama mencontohkan penekanan dengan IoU 0.9976 > 0.45. Repo ini memakai ambang 0.7, dan keluaran yang tersimpan hanya berisi box yang **selamat**, sehingga contoh box yang ditekan tidak bisa diambil darinya. Yang bisa ditunjukkan adalah batas bawahnya: dua box bertetangga yang **lolos berdua** pada `ayam (1).jpg`,

| | box | luas |
| --- | --- | --- |
| A | (0.82, 198.34, 83.04, 253.23) | 4513.5 |
| B | (7.38, 211.22, 82.18, 253.36) | 3152.1 |

$$
IoU(A,B)=0.694782<0.7
$$

Selisihnya 0.0052 dari ambang. Dengan ambang PIO 0.45, box B akan dibuang. Ini contoh konkret bahwa ambang NMS bukan detail kosmetik: pada kepadatan ayam seperti ini, 0.45 versus 0.7 mengubah berapa ekor yang masuk ke tahap klasifikasi.

**(d) Ringkasan keluaran.** 18 gambar, 1215 box lolos, conf minimum 0.2511 (tepat di atas lantai 0.25), maksimum 0.9085.

<aside>
⚠️

Record tidak menyimpan pencocokan TP/FP per box, jadi precision dan recall tidak dihitung di sini. Angka P 0.9595 / R 0.8881 / mAP50 0.9353 / mAP50-95 0.7628 yang beredar berasal dari **evaluasi PIO**, bukan dari repo ini, dan tidak boleh ditulis sebagai performa deteksi repo ini.

</aside>

### Jembatan ke bagian III

Titik yang paling mudah salah dibaca: **tidak semua crop datang dari detektor.**

| Kelas | Asal box | Jumlah crop | conf tercatat |
| --- | --- | --- | --- |
| hidup (0) | detektor `bird_det` | 32 | 0.2603 – 0.8890 |
| mati (1) | bbox anotasi COCO | 98 | selalu 1.0 |

Kelas mati **tidak pernah melewati detektor sama sekali** — bbox-nya dibaca dari `_annotations.coco.json`, dan `conf = 1.0` dicatat apa adanya sebagai penanda "ini anotasi manusia", bukan sebagai kepercayaan model. Akibatnya seluruh bagian I–II hanya berlaku untuk **separuh** populasi crop. Ini salah satu sumber asimetri yang membuat lantai ketajaman di bagian X setinggi itu.

Dari 130 crop yang jadi: train 75 (20 hidup / 55 mati), val 22 (5 / 17), test 33 (7 / 26). Yang dibuang saat pembentukan: 56 box hidup karena bersinggungan dengan bbox ayam mati (status ambigu), dan 5 box hidup karena terlalu kecil.

Median sisi pendek bbox sebelum crop: **83.4 px** untuk hidup, **212.25 px** untuk mati. Selisih 2.5× inilah yang menentukan berapa banyak bantalan abu-abu muncul di bagian IV — dan bantalan itu confound-nya.

---

## III. Ekstraksi crop

### Alur

```
box hasil NMS (atau bbox anotasi untuk kelas mati)
  -> lebarkan 10% ke empat arah        (pad_box)
  -> klip ke [0,W] x [0,H]
  -> bulatkan ke bilangan bulat
  -> potong img[y1:y2, x1:x2]
  -> tolak kalau lebar atau tinggi < 2 px
```

Kelas **mati** memakai bbox anotasi COCO (`origin = coco_gt`, conf = 1.0). Kelas **hidup** memakai deteksi YOLOv8m-COCO kelas `bird` pada **foto yang sama**, dengan syarat conf ≥ 0.25, sisi terpendek ≥ 48 px, dan IoU terhadap bbox ayam mati mana pun < 0.01.

### Fungsi krusial

**Pelebaran bbox:**

$$
d_w=(x_2-x_1)\,\rho,\qquad d_h=(y_2-y_1)\,\rho
$$

$$
x_1'=\max(0,\ x_1-d_w),\quad y_1'=\max(0,\ y_1-d_h)
$$

$$
x_2'=\min(W,\ x_2+d_w),\quad y_2'=\min(H,\ y_2+d_h)
$$

Keempatnya lalu dibulatkan: $\operatorname{round}(\cdot)$.

**Penyaring ayam hidup:**

$$
\text{terima} \iff c\ge 0.25 \ \wedge\ \min(w,h)\ge 48 \ \wedge\ \max_k \operatorname{IoU}(b,\ b^{\text{mati}}_k) < 0.01
$$

**Keterangan:**

- $(x_1,y_1,x_2,y_2)$: bbox masukan dalam piksel ruang gambar asal.
- $\rho$: rasio pelebaran, `bbox_padding: 0.10` di `configs/config.yaml`.
- $d_w,d_h$: tambahan piksel pada satu sisi horizontal dan vertikal. Karena ditambahkan di kiri **dan** kanan, lebar total bertambah $2d_w$, yaitu 20%, bukan 10%.
- $W,H$: lebar dan tinggi gambar asal; klip ini yang membuat crop bisa lebih sempit dari yang diminta bila bbox menempel tepi.
- $c$: confidence detektor `bird`; ayam mati tidak lewat jalur ini sehingga conf-nya 1.0.
- $\operatorname{IoU}$: irisan dibagi gabungan, ambangnya sengaja 0.01 dan bukan 0.10 karena banyak ayam mati dalam satu foto tidak dianotasi — semakin ketat, semakin kecil risiko ayam mati salah masuk ke kelas "hidup".

### Contoh hitung (data asli)

Crop contoh mati, dari foto 416×416, bbox $(36,\ 124,\ 416,\ 336.5)$ sehingga $w=380$ dan $h=212.5$:

$$
d_w=380(0.10)=38.0,\qquad d_h=212.5(0.10)=21.25
$$

$$
x_1'=\max(0,\ 36-38)=0,\qquad y_1'=\max(0,\ 124-21.25)=102.75
$$

$$
x_2'=\min(416,\ 416+38)=416,\qquad y_2'=\min(416,\ 336.5+21.25)=357.75
$$

Setelah pembulatan: $(0,\ 103,\ 416,\ 358)$, jadi crop berukuran **416×255** (lebar × tinggi), atau dalam bentuk array `(255, 416, 3)`.

Perhatikan dua klip yang aktif di sini. Ke kiri, $36-38=-2$ dipotong menjadi 0, jadi hanya 36 px konteks yang benar-benar didapat, bukan 38. Ke kanan, bbox sudah menyentuh tepi $x=416$ sehingga tidak ada tambahan sama sekali. Pelebaran 10% adalah permintaan, bukan jaminan — dan untuk bbox yang menempel tepi, konteks yang diterima kedua kelas jadi tidak simetris.

---

## IV. Letterbox ke 224×224

### Alur

```
crop (255x416, BGR)
  -> r = min(224/h, 224/w)
  -> (nh, nw) = round(h*r), round(w*r)
  -> resize: INTER_AREA kalau mengecil, INTER_LINEAR kalau membesar
  -> kanvas 224x224 diisi abu-abu (114,114,114)
  -> tempel di tengah pada (top, left) = ((224-nh)//2, (224-nw)//2)
```

Ini letterbox yang sama idenya dengan bagian I, tapi targetnya 224×224 dan bukan 640×640, dan dijalankan pada crop, bukan pada gambar penuh. Bedanya menentukan: di bagian I bantalannya 0 px karena gambar sumber sudah persegi, sedangkan di sini crop hampir tidak pernah persegi, jadi bantalan selalu muncul.

### Fungsi krusial

$$
r=\min\!\left(\frac{S}{h},\ \frac{S}{w}\right),\qquad S=224
$$

$$
n_h=\max\big(1,\operatorname{round}(hr)\big),\qquad n_w=\max\big(1,\operatorname{round}(wr)\big)
$$

$$
t=\left\lfloor\frac{S-n_h}{2}\right\rfloor,\qquad l=\left\lfloor\frac{S-n_w}{2}\right\rfloor
$$

$$
I'[\,t:t+n_h,\ l:l+n_w\,]=\operatorname{resize}(I,\ (n_w,n_h)),\qquad
\text{sisanya}=(114,114,114)
$$

Bagian kanvas yang berupa bantalan:

$$
\phi=1-\frac{n_h\,n_w}{S^2}
$$

**Keterangan:**

- $h,w$: tinggi dan lebar crop masukan.
- $S$: sisi keluaran, 224, ditetapkan `image_size` di config dan merupakan **batas yang diberikan**, bukan hasil tuning.
- $r$: faktor skala; $r<1$ berarti crop dikecilkan, $r>1$ berarti diperbesar.
- $n_h,n_w$: ukuran setelah resize, sebelum ditempel.
- $t,l$: offset tempel dari atas dan dari kiri. Karena pembagiannya dibulatkan ke bawah, bantalan bawah bisa 1 px lebih tebal daripada bantalan atas.
- $\phi$: bagian kanvas yang bukan gambar. Besaran ini penting karena ia **ikut berkorelasi dengan kelas** (lihat contoh hitung).
- Interpolasi: `INTER_AREA` saat mengecil (merata-ratakan area, tidak menimbulkan aliasing), `INTER_LINEAR` saat membesar. Pilihan ini yang membuat crop kecil jadi kabur dan crop besar tetap tajam — akar dari kebocoran ketajaman di bagian X.

### Contoh hitung (data asli)

Crop mati berukuran $h=255$, $w=416$:

$$
r=\min\!\left(\frac{224}{255},\ \frac{224}{416}\right)=\min(0.878431,\ 0.538462)=0.538462
$$

$$
n_h=\operatorname{round}(255\times0.538462)=137,\qquad n_w=\operatorname{round}(416\times0.538462)=224
$$

$$
t=\left\lfloor\frac{224-137}{2}\right\rfloor=43,\qquad l=\left\lfloor\frac{224-224}{2}\right\rfloor=0
$$

Bantalan atas 43 px, bantalan bawah $224-137-43=44$ px, bantalan kiri-kanan nol. Karena $r<1$, interpolasinya `INTER_AREA`. Bagian kanvas yang bantalan:

$$
\phi=1-\frac{137\times224}{224\times224}=1-\frac{137}{224}=0.388393
$$

Verifikasi: piksel pojok $(0,0)$ pada hasil bernilai $(114,114,114)$ — memang bantalan.

<aside>
⚠️

**Bantalan ini ikut terpanggang ke berkas.** `build_crops.py` menyimpan crop yang **sudah** 224×224, jadi bilah abu-abu sudah ada di dalam `.jpg` sebelum augmentasi apa pun dijalankan. Akibatnya tidak satu pun kebijakan augmentasi bisa menghapusnya. Diukur di split train: median $\phi$ crop hidup 0.174 versus crop mati 0.138, dan 23 dari 55 crop mati sama sekali tidak punya bilah sementara crop hidup **selalu** punya. Bagian bantalan saja sudah punya AUC 0.9038 di test. Ia tidak dipakai sebagai lantai resmi hanya karena ambangnya tidak berpindah dari train ke test (bacc jatuh dari 0.7114 ke 0.6538), tapi sebagai kebocoran ia nyata. Rinciannya di `outputs/reports/augmentation_report.md`.

</aside>

---

## V. Normalisasi

### Alur

```
crop 224x224 BGR uint8
  -> cv2.cvtColor BGR->RGB
  -> float32 / 255                    -> rentang [0, 1]
  -> (x - mu_c) / sigma_c             -> ImageNet
  -> transpose HWC -> CHW             -> tensor 3x224x224
```

### Fungsi krusial

$$
x^{01}=\frac{p}{255},\qquad x_c=\frac{x^{01}_c-\mu_c}{\sigma_c}
$$

$$
\mu=(0.485,\ 0.456,\ 0.406),\qquad \sigma=(0.229,\ 0.224,\ 0.225)
$$

**Keterangan:**

- $p$: nilai piksel 8-bit, 0–255, per kanal.
- $x^{01}$: piksel setelah dibagi 255.
- $\mu_c,\sigma_c$: mean dan standard deviation ImageNet untuk kanal $c$, pada skala 0–1.
- $x_c$: nilai akhir yang masuk ke jaringan.
- Urutan kanal berubah dari BGR (bawaan OpenCV) ke RGB, sebab $\mu$ dan $\sigma$ didefinisikan untuk RGB. Menukar urutannya tidak membuat program gagal, cuma membuat semua angkanya salah diam-diam.
- Angka $\mu$ dan $\sigma$ **bukan** dihitung dari dataset ayam. Nilainya mengikuti bobot ResNet-18 pretrained ImageNet yang dipakai sebagai titik awal, jadi harus identik dengan yang dipakai saat bobot itu dilatih. Konstanta ini tidak boleh di-tuning.

### Contoh hitung (data asli)

Satu piksel di pusat crop mati, posisi (baris 112, kolom 112). Nilai mentahnya BGR $(126,\ 136,\ 136)$, jadi setelah ditukar ke RGB menjadi $(136,\ 136,\ 126)$:

| Kanal | $p$ | $x^{01}=p/255$ | Normalisasi | $x_c$ |
| --- | --- | --- | --- | --- |
| R | 136 | 136/255 = 0.533333 | (0.533333 − 0.485) / 0.229 | **0.211063** |
| G | 136 | 136/255 = 0.533333 | (0.533333 − 0.456) / 0.224 | **0.345238** |
| B | 126 | 126/255 = 0.494118 | (0.494118 − 0.406) / 0.225 | **0.391634** |

Nilai ini sama persis dengan `to_tensor(...)[:, 112, 112]` dari PyTorch.

Piksel bantalan abu-abu juga punya nilai tetap yang bisa dihitung sekali untuk selamanya, $p=114$ pada ketiga kanal:

$$
x^{01}=\frac{114}{255}=0.447059
$$

$$
x=\left(\frac{0.447059-0.485}{0.229},\ \frac{0.447059-0.456}{0.224},\ \frac{0.447059-0.406}{0.225}\right)=(-0.165682,\ -0.039916,\ 0.182484)
$$

Angka ini layak dicatat: **bantalan bukan nol setelah normalisasi.** Ia adalah tiga konstanta yang tersebar rata di bilah atas-bawah atau kiri-kanan, dan bagi jaringan itu tekstur datar yang sangat mudah dikenali. Pada crop contoh, 38.8% kanvasnya berisi persis tiga angka ini.

Rentang seluruh tensor crop mati: $[-2.117904,\ 1.646536]$. Batas bawahnya adalah $p=0$ pada kanal R, yaitu $(0-0.485)/0.229=-2.117904$ — cocok, berarti crop ini memang punya piksel hitam penuh.

---

## VI. Augmentasi — dua view dari satu crop

### Alur

```
crop 224x224 (yang tersimpan di disk)
  -> tentukan benih item:  s(j) = (seed * 1000003 + j) mod 2^32
       view 1: j1 = i + epoch * 7919
       view 2: j2 = j1 + 104729
  -> untuk tiap op di kebijakan, slot ke-k:
       rng_op = Random( s XOR (crc32("nama#k") * 2654435761) mod 2^32 )
       jalankan op kalau rng_op.random() < p
  -> dua gambar berbeda dari crop yang SAMA -> dua tensor
```

Kebijakan yang dipakai tiap metode:

| metode | tahap kontrastif | tahap kepala |
| --- | --- | --- |
| `selfcon` | `simclr` | `minimal` (flip saja) |
| `supcon` | `stacked_randaug` | `minimal` |
| `ce` | — (tidak ada) | `hier_addone` |

Isi kebijakan `simclr`, persis urutan Chen dkk. (ICML 2020, Appendix A):

| slot | op | p | parameter |
| --- | --- | --- | --- |
| 0 | `rrc` | 1.0 | scale 0.20–1.00, ratio 0.75–1.3333 |
| 1 | `hflip` | 0.5 | — |
| 2 | `color_jitter` | 0.8 | strength 1.0 |
| 3 | `gray` | 0.2 | — |
| 4 | `blur` | 0.5 | kernel_frac 0.10, sigma 0.1–2.0 |

### Fungsi krusial

**Benih per item dan per view:**

$$
s(j)=(\text{seed}\cdot 1\,000\,003 + j)\ \bmod\ 2^{32}
$$

$$
j_1=i+\text{epoch}\cdot 7919,\qquad j_2=j_1+104\,729
$$

**Benih per op** (inilah yang membuat kebijakan bisa dibandingkan):

$$
s_{op}=\Big(s \ \oplus\ \big(\operatorname{crc32}(\text{nama}\Vert\text{"\#"}\Vert k)\cdot 2654435761\big)\Big)\ \bmod\ 2^{32}
$$

**Random resized crop:**

$$
a=hw\cdot U(0.20,\ 1.00),\qquad r=\exp\big(U(\ln 0.75,\ \ln 1.3333)\big)
$$

$$
n_w=\operatorname{round}\!\sqrt{ar},\qquad n_h=\operatorname{round}\!\sqrt{a/r}
$$

lalu potongan $n_h\times n_w$ di posisi acak dikembalikan ke $h\times w$ dengan `INTER_LINEAR`.

**Gaussian blur SimCLR:**

$$
k=\operatorname{odd}\big(\lfloor 0.10\cdot\min(h,w)\rfloor\big),\qquad \sigma\sim U(0.1,\ 2.0)
$$

**Keterangan:**

- $i$: indeks item dalam dataset; `epoch`: nomor epoch, jadi tiap epoch memberi pasangan view yang berbeda untuk item yang sama.
- $j_1,j_2$: indeks turunan untuk view pertama dan kedua. Jarak 104 729 adalah bilangan prima yang cukup besar supaya kedua aliran tidak pernah bertabrakan dalam satu run.
- 1 000 003, 7919, 2654435761: konstanta pencampur (prima besar dan konstanta Knuth) — nilainya tidak penting selain bahwa ia menyebar benih secara merata.
- $\oplus$: XOR bitwise; $k$: nomor slot op dalam daftar kebijakan.
- **Kenapa RNG diturunkan per-op dan bukan satu aliran bersama:** dengan aliran bersama, menambah satu op akan menggeser semua undian sesudahnya, sehingga "seed sama, kebijakan beda" menghasilkan potongan crop yang sama sekali lain. Dengan RNG turunan, `rrc` menarik kotak yang sama di setiap kebijakan yang memuatnya, jadi perbandingan antar-kebijakan benar-benar mengisolasi op yang berbeda.
- $a$: luas target potongan; $r$: rasio aspek, diundi dalam ruang logaritma supaya $r$ dan $1/r$ sama mungkinnya.
- $k$: ukuran kernel blur, dipaksa ganjil dan minimal 3. Pada gambar 224 px, $k = 23$ — jauh lebih kuat daripada $k=3/5$ versi lama, dan itulah yang menyerang jalan pintas ketajaman.
- **Tidak ada rotasi di kebijakan mana pun**, walaupun ketiga paper memakainya. Orientasi (tergeletak versus berdiri) adalah sinyal nyata yang membedakan mati dan hidup di dataset ini; memutar gambar menghapusnya.

### Contoh hitung (data asli)

Untuk item $i=0$ pada epoch 0, dengan seed 42:

$$
j_1=0,\qquad j_2=104\,729
$$

$$
s_1=(42\cdot 1\,000\,003+0)\bmod 2^{32}=42\,000\,126
$$

$$
s_2=(42\cdot 1\,000\,003+104\,729)\bmod 2^{32}=42\,104\,855
$$

Dijalankan pada crop mati contoh dengan kebijakan `simclr`, op yang benar-benar menyala:

| | op yang menyala | vLap hasil |
| --- | --- | --- |
| crop asli | — | 1036.45 |
| view 1 ($s_1$) | `rrc0.83` | 374.00 |
| view 2 ($s_2$) | `rrc0.48`, `hflip`, `color1.0`, `blur23` | 17.47 |

View 1 hanya kena crop 83% luas; `hflip`, `color_jitter`, `gray`, dan `blur` semuanya gagal undian. View 2 kena empat op sekaligus, termasuk blur $k=23$, dan ketajamannya runtuh dari 1036 ke 17 — **turun 59×**. Inilah mekanisme yang diharapkan mematikan kebocoran ketajaman: kalau ketajaman dihancurkan secara acak, jaringan tidak bisa mengandalkannya.

Terukur di seluruh dataset, memang turun — tapi tidak sampai nol:

| kebijakan | AUC ketajaman | median vLap hidup | median vLap mati |
| --- | --- | --- | --- |
| *(tanpa augmentasi)* | **0.9945** | 133.9 | 926.0 |
| `legacy` | **0.8468** ± 0.0108 | 45.2 | 229.5 |
| `simclr` | **0.7181** ± 0.0167 | 10.6 | 49.5 |
| `stacked_randaug` | **0.7513** ± 0.0141 | 175.8 | 529.9 |
| `hier_addone` | **0.7421** ± 0.0178 | 17.0 | 84.5 |

<aside>
⚠️

**Konsekuensi rancangan yang harus ditulis di laporan.** Karena tiap metode memakai kebijakan augmentasi dari paper yang berbeda, membandingkan ketiganya **pada diagonal** (`selfcon`+`simclr` vs `supcon`+`stacked_randaug` vs `ce`+`hier_addone`) mengubah **dua** hal sekaligus: fungsi loss *dan* augmentasi. Perbandingan diagonal karena itu **tidak bisa** mengatribusikan perbedaan skor ke fungsi loss. Yang memulihkan perbandingan yang sah adalah membaca **kolom** grid penuh — augmentasi ditahan tetap, loss divariasikan (`python src/train.py --grid full`). Semua contoh hitung di bawah memakai satu sel diagonal sebagai ilustrasi mekanisme, bukan sebagai bukti superioritas metode.

</aside>

---

## VII. Forward encoder

### Alur

```
input 3x224x224
 conv1  Conv(3->64, k7, s2, p3) + BN + ReLU      64x112x112
 maxpool k3 s2 p1                                64x56x56
 layer1  2x BasicBlock(64)                       64x56x56
 layer2  2x BasicBlock(128, s2)                 128x28x28
 layer3  2x BasicBlock(256, s2)                 256x14x14
 layer4  2x BasicBlock(512, s2)                 512x7x7
 avgpool global                                 512x1x1
 flatten                                        f  (512-d)   <- FITUR
        |
        +--------------------------+
        v                          v
 projector:                   classifier:
   Linear(512->512)             Linear(512->2)
   BatchNorm1d(512)                  |
   ReLU                              v
   Linear(512->128)              logit (2-d)
        |
        v
   L2-normalisasi -> z (128-d, ||z|| = 1)   <- EMBEDDING
```

`net.fc` bawaan ResNet-18 (512→1000 kelas ImageNet) dibuang dan diganti `nn.Identity()`, jadi `features()` mengembalikan vektor 512-d hasil global average pooling.

Jumlah parameter: backbone 11 176 512, projector 329 344, classifier 1 026 — total **11 506 882**.

### Fungsi krusial

**Konvolusi** (tanpa bias karena diikuti BN):

$$
z=\sum_{c=1}^{3}\sum_{i,j}X_c(i,j)\,K_c(i,j),\qquad
H_{out}=\left\lfloor\frac{H+2p-k}{s}\right\rfloor+1
$$

**BatchNorm mode inferensi** dan **ReLU**:

$$
\hat z=\frac{z-\mu_{run}}{\sqrt{\sigma^2_{run}+\epsilon}},\qquad y=\gamma\hat z+\beta,\qquad
\operatorname{ReLU}(y)=\max(0,\ y)
$$

**BasicBlock** (jalur residual):

$$
\text{out}=\operatorname{ReLU}\big(\mathcal{F}(x)+x\big)
$$

**Global average pooling:**

$$
f_c=\frac{1}{7\times 7}\sum_{i=1}^{7}\sum_{j=1}^{7}A_c(i,j)
$$

**Projection head dan normalisasi L2:**

$$
g=W_2\,\operatorname{ReLU}\!\big(\operatorname{BN}(W_1f+b_1)\big)+b_2,\qquad
z=\frac{g}{\lVert g\rVert_2}
$$

**Kepala klasifikasi:**

$$
\ell = W_{cls}f+b_{cls}\ \in\ \mathbb{R}^2
$$

**Keterangan:**

- $X$: patch input yang tertutup kernel; $K$: kernel; $z$: hasil konvolusi sebelum BN.
- $H,p,k,s$: ukuran input, padding, ukuran kernel, stride. Untuk conv1: $k=7$, $s=2$, $p=3$.
- $\mu_{run},\sigma^2_{run}$: statistik hasil training yang dibekukan saat inferensi; $\epsilon$: penstabil pembagian; $\gamma,\beta$: skala dan geser yang dipelajari.
- $\mathcal{F}$: dua conv+BN dalam satu blok; penjumlahan $+x$ adalah koneksi residual.
- $A_c$: peta fitur kanal $c$ berukuran 7×7 di akhir `layer4`; $f_c$: rata-ratanya.
- $f$: vektor fitur 512-d. **Ini yang dipakai kepala klasifikasi**, bukan $z$.
- $W_1\in\mathbb{R}^{512\times512}$, $W_2\in\mathbb{R}^{128\times512}$; $g$: keluaran projector sebelum dinormalisasi; $z$: embedding pada bola satuan.
- $\ell$: dua logit, indeks 0 = hidup, indeks 1 = mati.
- **Kenapa loss kontrastif dihitung di $z$ dan bukan di $f$:** projection head menyerap distorsi yang dipaksakan loss kontrastif, sehingga $f$ tidak ikut terdistorsi dan tetap berguna untuk klasifikasi. Ini temuan Chen dkk.: fitur sebelum projector secara konsisten lebih baik untuk tugas hilir daripada fitur sesudahnya.
- **Kenapa $z$ dinormalisasi:** setelah $\lVert z\rVert=1$, hasil kali titik $z_i^\top z_j$ persis sama dengan cosine similarity, yang rentangnya terbatas $[-1,1]$. Tanpa itu, loss bisa diturunkan hanya dengan memperbesar norma embedding, bukan dengan memperbaiki arahnya.

### Contoh hitung (data asli)

**(a) conv1 + BN + ReLU**, filter ke-0, pada posisi output (baris 56, kolom 56). Dengan $s=2$ dan $p=3$, jendela sumbernya mulai di baris $56\cdot2-3=109$ dan kolom 109, jadi patch 7×7 diambil dari baris 109–115 dan kolom 109–115.

Baris pertama patch, kanal R: $[0.4166,\ 0.3994,\ 0.3481,\ 0.4508,\ 0.3823,\ 0.3823,\ 0.2967]$

Baris pertama kernel, kanal R: $[-0.0102,\ -0.0067,\ -0.0037,\ 0.0727,\ 0.0544,\ 0.0155,\ -0.0140]$

Jumlah per kanal atas seluruh 7×7:

$$
\sum X_R K_R=0.164148,\qquad \sum X_G K_G=0.193869,\qquad \sum X_B K_B=0.058953
$$

$$
z=0.164148+0.193869+0.058953=0.416969
$$

PyTorch memberi 0.416970 pada posisi yang sama — selisihnya murni pembulatan float.

BN dengan $\mu_{run}=0.009958$, $\sigma^2_{run}=0.304421$, $\epsilon=10^{-5}$, $\gamma=0.238157$, $\beta=0.223719$:

$$
\hat z=\frac{0.416969-0.009958}{\sqrt{0.304421+0.00001}}=\frac{0.407011}{0.551748}=0.737670
$$

$$
y=0.238157(0.737670)+0.223719=0.399400
$$

$$
\operatorname{ReLU}(0.399400)=0.399400
$$

Ukuran output:

$$
\left\lfloor\frac{224+6-7}{2}\right\rfloor+1=112
$$

**(b) Dari fitur ke embedding.** Lima elemen pertama $f$ untuk kedua crop uji:

| | $f_0$ | $f_1$ | $f_2$ | $f_3$ | $f_4$ | $\lVert f\rVert$ |
| --- | --- | --- | --- | --- | --- | --- |
| crop mati | 0.613887 | 0.000000 | 1.338272 | 1.262139 | 1.520847 | 25.5166 |
| crop hidup | 0.740360 | 0.029555 | 2.014408 | 1.818386 | 0.408593 | — |

Hanya 0.98% elemen $f$ yang tepat nol, jadi ReLU di akhir backbone hampir tidak mematikan apa pun — representasinya padat, bukan jarang.

Neuron pertama projector, dihitung tangan dari $f$ crop mati:

$$
h_0=\sum_{d=1}^{512}W_1[0,d]\,f_d+b_1[0]=-0.311772
$$

PyTorch: −0.311772. Lalu BN-nya, dengan $\mu_{run}=-0.202979$, $\sigma^2_{run}=0.323459$, $\gamma=1.000749$, $\beta=-0.001091$:

$$
\hat h_0=\frac{-0.311772-(-0.202979)}{\sqrt{0.323459+0.00001}}=\frac{-0.108793}{0.568739}=-0.191287
$$

$$
y_0=1.000749(-0.191287)+(-0.001091)=-0.192522
\ \ \longrightarrow\ \ \operatorname{ReLU}(y_0)=0
$$

Neuron ini mati untuk crop ini. Itu normal — yang penting bukan satu neuron, tapi arah vektor akhirnya.

Setelah lapisan kedua, sebelum normalisasi:

$$
g_{0:5}=(0.014798,\ -0.209008,\ 0.231027,\ -0.938677,\ 0.465558),\qquad \lVert g\rVert=5.746755
$$

Dibagi normanya:

$$
z_0=\frac{0.014798}{5.746755}=0.002575
$$

Verifikasi lima elemen pertama $z$ crop mati: $(0.002575,\ -0.036370,\ 0.040201,\ -0.163340,\ 0.081012)$ — cocok. Dan $\lVert z\rVert=1.0$ persis untuk kedua crop.

Cosine similarity antara crop mati dan crop hidup:

$$
z^{\text{mati}\top} z^{\text{hidup}}=0.640474
$$

Angka 0.64 untuk dua crop dari **kelas berbeda** cukup tinggi. Itu masuk akal: keduanya diambil dari foto yang sama, kandang yang sama, pencahayaan yang sama. Yang harus dipisahkan encoder hanyalah kondisi ayamnya — dan itu memang bagian yang sulit.

---

## VIII. Loss — tiga metode

### Alur

```
satu batch B crop, label y
  -> view 1 dan view 2 -> z1, z2   (masing-masing Bx128, ternormalisasi)
  -> gabung: z = [z1; z2]          (2B x 128)
  -> matriks similarity S = z z^T / T   (2B x 2B)
        |
        +-- selfcon: NT-Xent   positif = HANYA pasangan view (i, i+B)
        +-- supcon : SupCon    positif = SEMUA anchor berlabel sama
        +-- ce     : tidak memakai z sama sekali, langsung logit vs y
```

### Fungsi krusial

**Matriks similarity** (berlaku untuk kedua loss kontrastif):

$$
S_{ij}=\frac{z_i^\top z_j}{T}
$$

**NT-Xent** (Chen dkk. 2020) — cross-entropy atas satu positif:

$$
\mathcal{L}_{\text{NT-Xent}}=\frac{1}{2B}\sum_{i=1}^{2B}
-\log\frac{\exp(S_{i,\pi(i)})}{\sum_{k\neq i}\exp(S_{ik})}
$$

Dalam kode ini dihitung sebagai `F.cross_entropy(S, target)` setelah diagonalnya diisi $-\infty$, yang secara aljabar sama dengan:

$$
\mathcal{L}_i=\operatorname{logsumexp}_{k\neq i}(S_{ik})-S_{i,\pi(i)}
$$

**SupCon** (Khosla dkk. 2020), varian rata-rata **di luar** logaritma:

$$
\mathcal{L}_{\text{SupCon}}=\frac{1}{2B}\sum_{i=1}^{2B}
\frac{-1}{|P(i)|}\sum_{q\in P(i)}
\log\frac{\exp(S_{iq})}{\sum_{k\neq i}\exp(S_{ik})}
$$

$$
P(i)=\{\,q\neq i\ :\ y_q=y_i\,\}
$$

Sebelum eksponensiasi, tiap baris dikurangi maksimumnya, $S_{ij}\leftarrow S_{ij}-\max_j S_{ij}$ — trik penstabil yang tidak mengubah nilai loss karena faktor konstan lenyap dari pembilang dan penyebut.

**Cross-Entropy berbobot:**

$$
\mathcal{L}_{\text{CE}}=\frac{\sum_{i}w_{y_i}\big(-\log p_{i,y_i}\big)}{\sum_i w_{y_i}},
\qquad p_{i,c}=\frac{e^{\ell_{ic}}}{\sum_{k}e^{\ell_{ik}}}
$$

$$
w_0=\frac{n}{2n_0},\qquad w_1=\frac{n}{2n_1}
$$

**Keterangan:**

- $B$: jumlah crop dalam batch; $2B$: jumlah anchor, karena tiap crop menyumbang dua view.
- $T$: temperature, 0.07. Makin kecil $T$, makin tajam distribusinya dan makin besar hukuman bagi negatif yang mirip.
- $\pi(i)$: pasangan view dari anchor $i$, yaitu $i+B$ kalau $i<B$ dan $i-B$ kalau tidak.
- $P(i)$: himpunan positif SupCon — **semua** anchor berlabel sama, termasuk view dari crop lain; $|P(i)|$ jumlahnya.
- $k\neq i$: penyebut memuat semua anchor lain, positif maupun negatif, tapi tidak dirinya sendiri.
- $\ell_{ic}$: logit kelas $c$ untuk crop $i$; $p_{i,c}$ softmax-nya.
- $w_0,w_1$: bobot kelas; $n_0,n_1$ jumlah crop hidup dan mati di train, $n=n_0+n_1$.
- **Kenapa penyebut CE-nya $\sum_i w_{y_i}$ dan bukan $B$:** `F.cross_entropy` dengan argumen `weight` memakai rata-rata berbobot, bukan rata-rata biasa. Kalau loss per epoch dijumlah dengan penyebut $B$, batch dengan komposisi kelas berbeda jadi tidak sebanding. Itu sebabnya `_ce_denominator` di `src/train.py` memakai `weight[y].sum()`.
- **Beda pokok NT-Xent dan SupCon** ada di pembilang: NT-Xent hanya mengakui satu positif (view lain dari crop yang sama), sedangkan SupCon mengakui semua crop sekelas. Konsekuensinya, dua ayam mati yang berbeda adalah **negatif** bagi NT-Xent — ia justru didorong memisahkan keduanya, padahal kelasnya sama.

### Contoh hitung (data asli)

Batch contoh $B=4$: dua crop mati dan dua crop hidup dari split train, dua view masing-masing dibuat dengan kebijakan `simclr` dan benih turunan yang sama seperti bagian VI.

| idx | crop | label | op view 1 | op view 2 |
| --- | --- | --- | --- | --- |
| 0 | `dead_test_image--1-_3_00` | 1 | `rrc0.83` | `rrc0.48`, `hflip`, `color1.0`, `blur23` |
| 1 | `dead_test_image--17-_1_00` | 1 | `rrc0.72`, `color1.0` | `rrc0.48`, `color1.0` |
| 2 | `alive_train_image--42-_19_000` | 0 | `rrc0.80`, `blur23` | `rrc0.62`, `color1.0`, `gray`, `blur23` |
| 3 | `alive_train_image--42-_19_001` | 0 | `rrc0.77`, `hflip`, `blur23` | `rrc0.44`, `hflip`, `color1.0`, `blur23` |

Label 2B setelah view digabung: $[1,1,0,0,\ 1,1,0,0]$.

Matriks cosine $z_i^\top z_j$ (4 desimal):

```
        0       1       2       3       4       5       6       7
0   1.0000  0.1091  0.0307 -0.0655  0.9551  0.0547  0.0673 -0.0101
1   0.1091  1.0000 -0.0789  0.0458  0.0533  0.9544 -0.0936 -0.0890
2   0.0307 -0.0789  1.0000  0.3374  0.1299 -0.0450  0.9703  0.2959
3  -0.0655  0.0458  0.3374  1.0000 -0.0080  0.0049  0.2638  0.8601
4   0.9551  0.0533  0.1299 -0.0080  1.0000  0.0079  0.1649  0.0609
5   0.0547  0.9544 -0.0450  0.0049  0.0079  1.0000 -0.0713 -0.0904
6   0.0673 -0.0936  0.9703  0.2638  0.1649 -0.0713  1.0000  0.2634
7  -0.0101 -0.0890  0.2959  0.8601  0.0609 -0.0904  0.2634  1.0000
```

Baca diagonal sekundernya: $S_{0,4}=0.9551$, $S_{1,5}=0.9544$, $S_{2,6}=0.9703$, $S_{3,7}=0.8601$. Itu pasangan view dari crop yang sama, dan keempatnya di atas 0.86 — encoder ini memang sudah terlatih. Bandingkan dengan $S_{2,3}=0.3374$: dua crop **hidup berbeda** hanya semirip 0.34, jauh di bawah pasangan view.

Setelah dibagi $T=0.07$, baris 0 menjadi:

$$
S_{0,\cdot}=[14.2857,\ 1.5589,\ 0.4379,\ -0.9362,\ \mathbf{13.6449},\ 0.7817,\ 0.9616,\ -0.1448]
$$

**(a) NT-Xent, baris 0.** Positifnya indeks $\pi(0)=0+4=4$, jadi $S_{0,4}=13.644885$. Diagonal $S_{0,0}$ dibuang (diisi $-\infty$), menyisakan 6 negatif.

Logsumexp-nya, dengan mengurangi maksimum 13.6449 lebih dulu:

$$
\sum_{k\neq 0}e^{S_{0k}-13.6449}=1.0+6{\times}10^{-6}+2{\times}10^{-6}+0+3{\times}10^{-6}+3{\times}10^{-6}+1{\times}10^{-6}=1.000015
$$

$$
\operatorname{logsumexp}=13.6449+\log(1.000015)=13.644899
$$

$$
\mathcal{L}_0=13.644899-13.644885=0.000014
$$

Loss kedelapan baris, dalam satuan $10^{-5}$: $[1.4,\ 1.1,\ 19.3,\ 79.1,\ 2.8,\ 0.7,\ 9.5,\ 53.2]$. Rata-ratanya:

$$
\mathcal{L}_{\text{NT-Xent}}=0.000209
$$

Hampir nol — dan itu **bukan kabar baik**. NT-Xent nyaris nol berarti tiap anchor sudah mengenali pasangan view-nya sendiri di antara 6 pengecoh. Dengan $B=4$ soalnya memang mudah; pada batch asli $B=32$ ada 62 pengecoh per anchor, jadi angkanya tidak sekecil ini. Yang perlu diingat: **besar-kecilnya loss kontrastif bukan ukuran mutu representasi** — ia hanya mengukur seberapa mudah soal yang sedang disajikan.

**(b) SupCon, baris 0.** Anchor 0 berlabel 1 (mati). Positifnya semua anchor lain berlabel 1, yaitu indeks $\{1,\ 4,\ 5\}$, jadi $|P(0)|=3$ — tiga, bukan satu seperti NT-Xent.

Log-probabilitas ketiga positif itu:

$$
\log p_{0,1}=-12.085989,\quad \log p_{0,4}=-0.000015,\quad \log p_{0,5}=-12.863218
$$

$$
\mathcal{L}_0=\frac{12.085989+0.000015+12.863218}{3}=\frac{24.949222}{3}=8.316407
$$

Loss kedelapan baris: $[8.3164,\ 8.3162,\ 6.2253,\ 5.3294,\ 8.8052,\ 8.7917,\ 6.7306,\ 5.5288]$. Rata-ratanya:

$$
\mathcal{L}_{\text{SupCon}}=7.255452
$$

Bandingkan kedua angka: 0.000209 versus 7.255452, pada batch yang **sama persis**. Selisih empat orde ini bukan tanda satu metode lebih baik — ia menunjukkan keduanya menjawab soal yang berbeda. NT-Xent cukup menemukan satu pasangan view; SupCon harus menarik crop mati yang **berbeda** (indeks 1 dan 5) mendekat ke anchor 0, dan menurut matriks di atas keduanya masih di similarity 0.1091 dan 0.0547 — praktis ortogonal. Itulah yang membuat $\log p$-nya sekitar $-12$.

<aside>
⚠️

**Jangan bandingkan nilai loss lintas metode.** Angka 0.000209 dan 7.255452 berasal dari batch, encoder, dan seed yang identik, tapi mengukur hal yang tidak sama. Perbandingan yang sah hanya lewat metrik hilirnya di bagian X. Hal yang sama berlaku untuk kurva loss: kurva SupCon yang nyaris datar sepanjang 60 epoch bukan tanda pelatihan gagal.

</aside>

**(c) Cross-Entropy berbobot.** Split train punya 20 crop hidup dan 55 crop mati, $n=75$:

$$
w_0=\frac{75}{2\cdot 20}=1.875,\qquad w_1=\frac{75}{2\cdot 55}=0.681818
$$

Logit keempat crop batch (tanpa augmentasi, crop apa adanya):

| idx | $y$ | $\ell_0$ (hidup) | $\ell_1$ (mati) | $-\log p_{y}$ | $w_y$ |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | −1.667866 | 1.308713 | 0.049711 | 0.681818 |
| 1 | 1 | −0.530303 | 0.819705 | 0.230507 | 0.681818 |
| 2 | 0 | 1.349790 | −1.100927 | 0.082715 | 1.875 |
| 3 | 0 | 1.278947 | −1.402608 | 0.066215 | 1.875 |

$$
\text{pembilang}=0.049711(0.681818)+0.230507(0.681818)+0.082715(1.875)+0.066215(1.875)=0.470301
$$

$$
\text{penyebut}=0.681818+0.681818+1.875+1.875=5.113636
$$

$$
\mathcal{L}_{\text{CE}}=\frac{0.470301}{5.113636}=0.091970
$$

Tanpa bobot, rata-rata biasa keempat sukunya memberi 0.107287. Bobot menurunkannya di sini karena dua crop yang paling mudah (indeks 0 dan 1) kebetulan dari kelas mayoritas. Ini juga ilustrasi kenapa penyebutnya harus $\sum_i w_{y_i}$: kalau dibagi $B=4$, hasilnya $0.470301/4=0.117575$ — bukan rata-rata berbobot apa pun, dan tidak sebanding antar batch dengan komposisi kelas berbeda.

---

## IX. Linear probe → logit → softmax → keputusan

### Alur

```
selfcon:  backbone DIBEKUKAN -> f (512) -> Linear(512,2) -> logit
supcon :  backbone DIBEKUKAN -> f (512) -> Linear(512,2) -> logit
ce     :  backbone IKUT DILATIH -> f (512) -> Linear(512,2) -> logit
                     |
                     v
              softmax 2 kelas
                     |
                     v
        skor = p(mati) = p[:,1]  -> dibandingkan dengan ambang
```

Projector dan vektor $z$ **tidak dipakai lagi** di tahap ini. Keduanya hanya alat untuk membentuk $f$ selama pelatihan kontrastif; begitu probe dilatih, jalurnya $f \rightarrow$ `logits` saja. Itu sebabnya `model.pt` tetap menyimpan bobot projector, tapi inferensi tidak menyentuhnya.

### Fungsi krusial

**Lapisan linear:**

$$
\ell_c=\sum_{d=1}^{512}W_{cd}f_d+b_c,\qquad c\in\{0,1\}
$$

**Softmax dua kelas:**

$$
p_c=\frac{e^{\ell_c}}{e^{\ell_0}+e^{\ell_1}}
$$

Untuk dua kelas, softmax setara sigmoid atas selisih logit:

$$
p_1=\frac{e^{\ell_1}}{e^{\ell_0}+e^{\ell_1}}=\frac{1}{1+e^{-(\ell_1-\ell_0)}}=\sigma(\ell_1-\ell_0)
$$

**Skor dan keputusan:**

$$
s=p_1,\qquad \hat{y}=\mathbb{1}[\,s\ge\tau\,]
$$

**Keterangan:**

- $f\in\mathbb{R}^{512}$: fitur keluaran backbone setelah global average pooling, semuanya $\ge 0$ karena melewati ReLU.
- $W\in\mathbb{R}^{2\times512}$, $b\in\mathbb{R}^{2}$: bobot dan bias `net.fc` pengganti — 1026 parameter, satu-satunya yang dilatih pada tahap probe untuk `selfcon` dan `supcon`.
- $\ell_c$: logit, skala bebas, boleh negatif.
- $p_1$: probabilitas kelas **mati**. Inilah yang disimpan sebagai `y_score` dan dipakai di seluruh bagian X.
- $\tau$: ambang keputusan, dipilih di bagian X dari split **validation saja**.
- **Kenapa sigmoid selisih itu penting:** ia menunjukkan bahwa yang menentukan keputusan hanyalah **selisih** $\ell_1-\ell_0$, bukan nilai mutlak keduanya. Menggeser kedua logit sebesar konstanta yang sama tidak mengubah apa pun.
- **Kenapa backbone dibekukan pada linear probe:** supaya yang diuji benar-benar kualitas representasi kontrastif, bukan kemampuan backbone untuk belajar ulang dari label. Kalau backbone ikut dilatih, `selfcon` berubah jadi sekadar pra-pelatihan untuk CE, dan perbandingan tiga metodenya kehilangan makna.

### Contoh hitung (data asli)

Dua crop dari bagian VII, lewat `logits()` checkpoint `outputs/runs/selfcon/model.pt`:

| crop | $\ell_0$ (hidup) | $\ell_1$ (mati) | $p_0$ | $p_1$ = skor |
| --- | --- | --- | --- | --- |
| mati | 0.127201 | 0.846548 | 0.327537 | **0.672463** |
| hidup | 0.861968 | −0.303586 | 0.762340 | **0.237660** |

Cek manual untuk crop mati, lewat jalur sigmoid:

$$
\ell_1-\ell_0=0.846548-0.127201=0.719347
$$

$$
p_1=\sigma(0.719347)=\frac{1}{1+e^{-0.719347}}=\frac{1}{1+0.487073}=0.672463
$$

Cocok dengan softmax penuh sampai 6 desimal. Untuk crop hidup selisihnya $-1.165554$, dan $\sigma(-1.165554)=0.237660$.

Dengan ambang biasa $\tau=0.5$ keduanya sudah benar: 0.672 → mati, 0.238 → hidup. Tapi perhatikan jaraknya dari ambang. Crop mati hanya 0.172 di atas 0.5 — jauh dari yakin, padahal ini crop yang bagi mata manusia jelas. Itu gejala yang akan muncul lagi di bagian X: skor kelas mati menumpuk di 0.53–0.88 dan skor kelas hidup di 0.24–0.67, dengan pita tumpang tindih yang tipis tapi nyata.

<aside>
🧮

**Cek cepat konsistensi:** $p_0+p_1=0.327537+0.672463=1.0$ untuk crop mati dan $0.762340+0.237660=1.0$ untuk crop hidup. Kalau jumlahnya bukan 1, berarti softmax diterapkan pada sumbu yang salah.

</aside>

---

## X. Ambang, metrik, dan lantai ketajaman

### Alur

```
skor val  -> cari ambang tau (HANYA val, test tidak boleh dilihat)
                |
                v
skor test -> keputusan y_hat = 1[s >= tau]
                |
                +-- bacc      : rata-rata recall dua kelas    (KEPUTUSAN)
                +-- ROC-AUC   : proporsi pasangan urut benar  (URUTAN)
                |
                v
        bandingkan dengan LANTAI ketajaman
        (bacc 0.9231 / AUC 0.9670, tanpa melihat isi gambar)
```

### Fungsi krusial

**Balanced accuracy:**

$$
\text{bacc}=\frac{1}{2}\left(\frac{TP}{TP+FN}+\frac{TN}{TN+FP}\right)
$$

**ROC-AUC lewat statistik Mann–Whitney U** — proporsi pasangan (mati, hidup) yang terurut benar:

$$
\text{AUC}=\frac{1}{n_1n_0}\sum_{i:\,y_i=1}\ \sum_{j:\,y_j=0}
\Big(\mathbb{1}[s_i>s_j]+\tfrac{1}{2}\mathbb{1}[s_i=s_j]\Big)
$$

**Pemilihan ambang** (`pick_threshold`, `src/train.py`) — titik tengah dataran terlebar:

$$
C=\Big\{u_1-\varepsilon\Big\}\cup\Big\{\tfrac{u_k+u_{k+1}}{2}\Big\}_{k=1}^{m-1}\cup\Big\{u_m+\varepsilon\Big\}
$$

$$
G^\star=\arg\max_{G\ \text{gugus kontigu},\ \text{bacc}(t)\ \ge\ \max_t \text{bacc}(t)-10^{-12}}
\big(\,c_{\max(G)}-c_{\min(G)},\ |G|\,\big)
$$

$$
\tau=\frac{c_{\min(G^\star)}+c_{\max(G^\star)}}{2}
$$

**Keterangan:**

- $TP$: crop mati yang diputuskan mati; $FN$: crop mati yang lolos sebagai hidup; $TN$, $FP$ sebaliknya. Kelas positif = **mati**.
- $n_1,n_0$: jumlah crop mati dan hidup di split yang diukur.
- $u_1<\dots<u_m$: skor unik di validation; $C$: himpunan kandidat ambang, $m+1$ buah.
- $\varepsilon=10^{-6}$: dua titik ujung supaya kandidat "semua diputuskan mati" dan "semua diputuskan hidup" ikut dievaluasi.
- $G$: gugus indeks kandidat yang **berurutan** dan semuanya mencapai bacc maksimum; dipilih yang **terlebar** dalam satuan skor, seri dipecah oleh jumlah anggota.
- **Kenapa titik tengah dataran terlebar, bukan sembarang argmax:** kalau ada banyak ambang yang sama bagusnya di val, memilih yang pertama akan menempel di skor satu crop tertentu, dan sedikit pergeseran skor di test langsung membalik keputusannya. Titik tengah dataran terlebar memberi jarak aman terbesar ke kedua sisi.
- **Kenapa bacc, bukan accuracy:** test set berisi 26 crop mati dan 7 hidup. Menebak "mati" untuk semuanya memberi accuracy 0.788 dengan bacc 0.5. Accuracy di sini menyesatkan; bacc tidak.
- **Kenapa AUC dilaporkan berdampingan dengan bacc:** bacc mengukur **keputusan** setelah ambang, AUC mengukur **urutan** sebelum ambang. Dengan hanya 7 crop hidup, satu crop yang jatuh di sisi salah ambang memotong $1/(2\cdot7)=7.14$ poin bacc walaupun urutannya sempurna. Dua angka itu bisa bercerita sangat berbeda tentang model yang sama.

### Contoh hitung (data asli)

**(a) Ambang dari validation.** Split val checkpoint `selfcon`: 5 crop hidup, 17 crop mati, 22 skor semuanya unik.

Skor kelas hidup: $[0.2200,\ 0.2358,\ 0.4847,\ 0.4888,\ 0.4961]$
Skor kelas mati (5 terendah): $[0.5490,\ 0.6041,\ 0.6045,\ 0.6113,\ 0.6534]$, tertinggi 0.8755.

Kedua kelas **terpisah sempurna**: skor hidup tertinggi 0.49611 masih di bawah skor mati terendah 0.54904. Jadi bacc maksimum di val adalah 1.0, dan tiap kandidat ambang di celah itu mencapainya. Dari 23 kandidat, hanya **satu** yang jatuh di dalam celah — titik tengah antara dua skor unik yang berdampingan:

$$
\tau=\frac{0.49611047+0.54904056}{2}=0.52257550
$$

Dataran terlebar di sini hanya memuat satu kandidat, lebar 0.0. Itu bukan karena algoritmanya gagal, melainkan karena resolusi kandidatnya kasar: hanya ada satu titik tengah di celah 0.053 itu. Konsekuensinya penting — **val yang terpisah sempurna tidak memberi informasi apa pun untuk mengkalibrasi ambang.** Setiap nilai antara 0.49611 dan 0.54904 sama-sama sempurna di val, dan val tidak bisa memberi tahu mana yang lebih baik di test.

<aside>
⚠️

**Val jenuh pada bacc 1.0.** Kalau bacc val sudah 1.0, val kehilangan fungsinya sebagai alat pemilih — baik untuk memilih ambang maupun untuk memilih checkpoint terbaik. Pada grid ini `selfcon` dan `ce` sama-sama mencapai val bacc 1.0000 / val AUC 1.0000, sehingga urutan mereka di test sepenuhnya ditentukan hal-hal yang tidak pernah terlihat di val.

</aside>

**(b) Terapkan ke test.** 26 crop mati, 7 hidup.

Skor kelas hidup: $[0.2377,\ 0.2689,\ 0.3689,\ 0.4339,\ 0.4562,\ 0.4811,\ \mathbf{0.6724}]$
Skor kelas mati, terendah: $[0.5334,\ 0.5410,\ 0.5573,\ 0.5972,\ 0.6319,\ 0.6722,\ \dots]$, tertinggi 0.8776.

Dengan $\tau=0.522576$:

- Semua 26 crop mati $\ge\tau$ → $TP=26$, $FN=0$
- 6 dari 7 crop hidup $<\tau$ → $TN=6$
- Satu crop hidup berskor 0.672439 → $FP=1$

$$
\text{bacc}=\frac{1}{2}\left(\frac{26}{26}+\frac{6}{7}\right)=\frac{1+0.857143}{2}=0.928571
$$

$$
\text{accuracy}=\frac{26+6}{33}=0.969697
$$

Dengan ambang biasa $\tau=0.5$ hasilnya **identik** (26/6/1/0, bacc 0.928571), karena tidak ada satu pun skor yang jatuh di antara 0.5 dan 0.5226. Ambang hasil tuning tidak memperbaiki apa-apa di sini — ia hanya tidak merusak.

**(c) AUC dihitung tangan.** Ada $26\times7=182$ pasangan (mati, hidup). Satu-satunya crop hidup yang bermasalah adalah yang berskor **0.672439** — ia lebih tinggi daripada 6 crop mati:

$$
0.5334,\quad 0.5410,\quad 0.5573,\quad 0.5972,\quad 0.6319,\quad 0.6722
$$

Enam crop hidup lainnya berada di bawah **semua** crop mati, jadi tidak menyumbang kesalahan. Tidak ada skor yang sama persis, jadi suku $\tfrac12\mathbb{1}[s_i=s_j]$ nol.

$$
\text{AUC}=1-\frac{6}{182}=1-0.032967=0.967033
$$

Angka yang sama keluar dari `roc_auc()` di `src/train.py` lewat jalur peringkat Mann–Whitney: 0.967033. Dua jalan, satu hasil — itu cek yang layak dipakai setiap kali implementasi AUC diragukan.

**(d) Bandingkan dengan lantai ketajaman.** Lantai adalah pengklasifikasi yang **tidak melihat isi gambar sama sekali**: ia hanya mengukur variance of Laplacian tiap crop, lalu memakai satu ambang yang dicocokkan di train.

| | bacc test | AUC test | pasangan salah urut |
| --- | --- | --- | --- |
| Lantai ketajaman saja | **0.9231** | **0.9670** | 6 dari 182 |
| `selfcon` + `simclr` (contoh ini) | 0.9286 | 0.9670 | 6 dari 182 |

Model ini **tidak mengalahkan lantai** — AUC-nya sama persis, dan bacc-nya cuma unggul 0.0055, setara kurang dari satu crop. Semua kerja di bagian III–IX menghasilkan urutan yang tidak lebih baik daripada mengukur ketajaman gambar. Penyebabnya ada di data, bukan di model: crop ayam hidup punya median sisi pendek 83 px dan crop mati 212 px, keduanya diperbesar ke 224, sehingga ketajaman ikut menandai kelas.

<aside>
⚠️

**Baca lantai sebelum memeringkat apa pun.** Angka bacc 0.9286 terdengar bagus sampai ditaruh bersebelahan dengan 0.9231 yang diperoleh tanpa belajar. Perbandingan antar metode di bawah lantai tidak bisa ditafsirkan — perbedaannya bisa berasal seluruhnya dari seberapa banyak tiap metode tanpa sengaja membaca ketajaman.

</aside>

**(e) Diagonal tiga metode.** Ketiganya seed 42, masing-masing dengan augmentasi bawaan paper-nya:

| Metode | Augmentasi | val bacc | val AUC | test bacc | test AUC | TP/TN/FP/FN | $\tau$ | detik |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `selfcon` | `simclr` | 1.0000 | 1.0000 | 0.9286 | 0.9670 | 26/6/1/0 | 0.5226 | 61.0 |
| `supcon` | `stacked_randaug` | 0.8529 | 0.9294 | 0.7857 | 0.8901 | 26/4/3/0 | 0.4691 | 65.5 |
| `ce` | `hier_addone` | 1.0000 | 1.0000 | 0.9093 | 0.9341 | 25/6/1/1 | 0.3044 | 13.8 |

Tiga catatan atas tabel ini:

1. **Ketiganya di sekitar atau di bawah lantai.** Tidak satu pun membuktikan sudah membaca kondisi ayam alih-alih ketajaman crop.
2. **Diagonal ini bukan perbandingan metode.** Tiap baris berbeda dalam **dua** hal sekaligus — loss dan augmentasi. `supcon` yang terendah di sini tidak berarti SupCon lebih buruk; pada kolom `simclr` justru SupCon yang paling tinggi AUC-nya (0.9945 ± 0.0085 atas 5 seed, 1 dari 182 pasangan salah urut) dan itu satu-satunya konfigurasi yang benar-benar melewati lantai. Yang bisa mengatribusikan sebab hanyalah **kolom** grid penuh, bukan diagonalnya.
3. **Satu seed tidak membuktikan kestabilan.** Semua angka di tabel ini seed 42 saja. Dengan 7 crop hidup di test, satu keputusan berbeda sudah menggeser bacc 7.14 poin.

---

## Lampiran A: hal-hal yang mudah salah dicatat

| Yang mudah diasumsikan | Kenyataannya di repo ini | Bagian |
| --- | --- | --- |
| Bantalan letterbox bernilai nol setelah normalisasi | Bernilai $(-0.1657,\ -0.0399,\ 0.1825)$ — tiga konstanta datar yang mudah dikenali jaringan | IV, V |
| Bantalan itu netral terhadap label | Bantalan saja memberi AUC 0.9038; fraksinya berkorelasi dengan kelas karena rasio aspek crop berbeda | IV |
| Augmentasi memakai satu RNG global | RNG diturunkan per-operasi: `crc32(op + "#" + slot) * 2654435761 XOR seed` | VI |
| Kedua view dibuat dari benih yang berdekatan | Offset 104729 dipakai justru supaya kedua view tidak berkorelasi | VI |
| Blur kernel-nya tetap | $k$ = bilangan ganjil dari $0.1\cdot\min(h,w)$, jadi 23 px pada kanvas 224 | VI |
| Loss kecil = representasi bagus | NT-Xent 0.000209 dan SupCon 7.255452 pada batch yang sama; keduanya mengukur soal berbeda | VIII |
| CE berbobot dibagi jumlah sampel | Dibagi $\sum_i w_{y_i}$, bukan $B$ — 5.113636, bukan 4 | VIII |
| $z$ dipakai saat inferensi | Projector berhenti dipakai setelah pelatihan kontrastif; inferensi lewat $f\rightarrow$ `logits` | IX |
| Ambang val hasil tuning selalu membantu | Di contoh ini $\tau=0.5226$ memberi hasil identik dengan $\tau=0.5$ | X |
| bacc dan AUC bercerita sama | Lantai ketajaman: bacc 0.9231 tapi AUC 0.9670 — selalu sebut metrik mana yang dibandingkan | X |
| Diagonal tabel = perbandingan metode | Tiap baris diagonal berbeda dalam loss **dan** augmentasi; hanya kolom grid penuh yang bisa mengatribusikan sebab | VI, X |

## Lampiran B: batas yang jujur

- **Ukuran data.** 130 crop total: 75 train, 22 val, 33 test. Di test hanya ada **7 crop hidup**, sehingga satu keputusan berbeda menggeser bacc 7.14 poin. Selang kepercayaan angka mana pun di halaman ini lebar.
- **Val tidak bisa memilih.** Val terpisah sempurna (bacc 1.0000, AUC 1.0000) untuk `selfcon` dan `ce`. Val karena itu tidak bisa dipakai untuk memilih ambang, checkpoint, maupun metode.
- **Lantai belum terlewati pada bacc.** Dari seluruh grid, tidak ada konfigurasi dengan ≥2 seed yang melewati lantai 0.9231 pada bacc. Yang melewati lantai pada **AUC** hanya satu: `supcon` + `simclr`, 0.9945 ± 0.0085 atas 5 seed (1 dari 182 pasangan salah urut).
- **Confound yang sudah teridentifikasi.** Ketajaman (lantai bacc 0.9231 / AUC 0.9670) dan fraksi bantalan letterbox (AUC 0.9038). Keduanya lahir dari cara crop terbentuk — ayam hidup terdeteksi kecil (median sisi pendek 83 px), ayam mati datang dari anotasi besar (212 px) — bukan dari kondisi ayamnya.
- **Angka di halaman ini satu seed.** Semua contoh memakai seed 42 dan checkpoint `outputs/runs/selfcon/model.pt`. Angka multi-seed ada di `outputs/reports/comparison.md`.

<aside>
✅

Jalur satu data, utuh: foto asal → box NMS (bagian II) → lebarkan 10% dan potong (III) → letterbox 224×224 dengan bantalan 114 (IV) → BGR→RGB, /255, normalisasi ImageNet (V) → dua view lewat kebijakan augmentasi metode (VI) → ResNet-18 → $f$ 512-d → projector → $z$ 128-d (VII) → NT-Xent / SupCon / CE berbobot (VIII) → linear probe → softmax → skor $P(\text{mati})$ (IX) → ambang dari val, bacc dan AUC, dibandingkan lantai ketajaman (X).

</aside>
