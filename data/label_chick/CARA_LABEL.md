# Cara melabeli 1215 crop chick

Tujuannya: menyiapkan crop latih **dari domain CCTV yang sama**, supaya label
tidak lagi identik dengan domain. Ini satu-satunya cara menutup jalan pintas
ketajaman - lihat `outputs/reports/kronologi_lengkap.md` babak 10 dan 11.

## Isi folder

| berkas | isi |
|---|---|
| `crops/` | 1215 crop, nama `0001.jpg` sampai `1215.jpg` |
| `lembar_kontak/` | 18 halaman, `01_...` sampai `18_...` - **mulai dari sini** |
| `lembar_label.csv` | daftar 1215 baris, kolom `label` menunggu diisi |

Nomornya **berjalan terus 1 sampai 1215** menembus batas gambar sumber, jadi
satu nomor cukup untuk menunjuk satu crop - tidak perlu menyebut gambarnya.
Nomor yang tercetak di petak = nama berkas crop = kolom `nomor` di CSV.

## Langkahnya

1. Buka `lembar_kontak/` berurutan dari `01_` sampai `18_`; halaman 1 memuat
   nomor 0001-0080, halaman 2 lanjut 0081-0143, begitu seterusnya sampai 1215.
   Tiap petak ada **nomornya**, dan petak berdasar abu menunjukkan batas crop -
   kalau ayamnya tidak memenuhi petak, crop-nya memang terpotong.

2. **Yang berbingkai merah bertulis MATI sudah diketahui** - 22 crop yang
   cocok dengan mask acuan (IoU >= 0.5). Tidak perlu dicek ulang, tapi
   **boleh dikoreksi** kalau menurut Anda keliru.

3. Catat nomor yang **mati** dan yang **bukan ayam** saja - cukup angkanya,
   misal `1060, 1069, 1071`. Sisanya otomatis dianggap hidup, sesuai
   keputusan Anda.

4. Isi kolom `label` di `lembar_label.csv` dengan salah satu:

   | isi | artinya |
   |---|---|
   | `mati` | ayam mati |
   | `bukan` | bukan ayam utuh: terpotong, tumpang-tindih, alat, lantai |
   | *(kosong)* | ayam hidup - tidak perlu diisi |

   Barisnya sudah urut nomor, jadi baris ke-N di CSV = crop nomor N.

   Kolom `dugaan_awal` biarkan apa adanya; itu catatan asal-usul 22 crop acuan,
   berguna untuk memeriksa apakah mask acuan sendiri ada yang salah. Kolom
   `gambar_sumber` dan `det_id` juga jangan diubah - itu tali yang menyambungkan
   tiap crop ke `detections.json` dan ke bbox yang dinilai `eval_on_chick.py`.

## Yang penting diperhatikan

**`bukan` bukan kelas buangan.** Crop terpotong dan tumpang-tindih adalah
kasus yang benar-benar muncul saat sistem dipakai. Memisahkannya berarti
angka nanti bisa dilaporkan dua kali: pada crop bersih saja, dan pada semua
crop apa adanya. Keduanya jujur, asal disebut yang mana.

**Jangan melabeli sambil melihat skor model.** Kalau ragu, tandai `bukan`.

**18 gambar ini tetap khusus untuk test.** Sesuai arahan dosen, tidak satu pun
crop, mask, atau label ayam/chick masuk train maupun validation. Development
memakai ayam hidup PIO dan ayam mati Roboflow, dibagi per gambar sumber. Karena
benchmark ini sudah pernah dipakai pada eksperimen historis, hasil berikutnya
disebut evaluasi retrospektif, bukan holdout prospektif yang benar-benar buta.

## Sesudah selesai

Simpan CSV-nya, lalu bilang ke saya. Yang berikutnya:
build crop latih dari label ini -> latih ulang -> **uji acak-petak**
(`src/eval_intervensi.py`). Kalau AUC tidak jatuh saat crop diacak, model
masih belum membaca bentuk - berapa pun angkanya.
