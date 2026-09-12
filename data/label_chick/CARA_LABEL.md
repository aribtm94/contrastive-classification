# Cara melabeli 1215 crop chick

Tujuannya: menyiapkan crop latih **dari domain CCTV yang sama**, supaya label
tidak lagi identik dengan domain. Ini satu-satunya cara menutup jalan pintas
ketajaman - lihat `outputs/reports/kronologi_lengkap.md` babak 10 dan 11.

## Isi folder

| berkas | isi |
|---|---|
| `crops/` | 1215 crop, nama `<gambar>_det<NNN>.jpg` |
| `lembar_kontak/` | 18 halaman, satu per gambar sumber - **mulai dari sini** |
| `lembar_label.csv` | daftar 1215 baris, kolom `label` menunggu diisi |

## Langkahnya

1. Buka `lembar_kontak/`, satu halaman per gambar. Tiap petak ada **nomornya**
   (itu `det_id`), dan petak berdasar abu menunjukkan batas crop - kalau
   ayamnya tidak memenuhi petak, crop-nya memang terpotong.

2. **Yang berbingkai merah bertulis MATI sudah diketahui** - 22 crop yang
   cocok dengan mask acuan (IoU >= 0.5). Tidak perlu dicek ulang, tapi
   **boleh dikoreksi** kalau menurut Anda keliru.

3. Catat nomor yang **mati** dan yang **bukan ayam** saja. Sisanya otomatis
   dianggap hidup, sesuai keputusan Anda.

4. Isi kolom `label` di `lembar_label.csv` dengan salah satu:

   | isi | artinya |
   |---|---|
   | `mati` | ayam mati |
   | `bukan` | bukan ayam utuh: terpotong, tumpang-tindih, alat, lantai |
   | *(kosong)* | ayam hidup - tidak perlu diisi |

   Kolom `dugaan_awal` biarkan apa adanya; itu catatan asal-usul 22 crop acuan,
   berguna untuk memeriksa apakah mask acuan sendiri ada yang salah.

## Yang penting diperhatikan

**`bukan` bukan kelas buangan.** Crop terpotong dan tumpang-tindih adalah
kasus yang benar-benar muncul saat sistem dipakai. Memisahkannya berarti
angka nanti bisa dilaporkan dua kali: pada crop bersih saja, dan pada semua
crop apa adanya. Keduanya jujur, asal disebut yang mana.

**Jangan melabeli sambil melihat skor model.** Kalau ragu, tandai `bukan`.

**18 gambar ini juga satu-satunya alat ukur yang ada.** Kalau semuanya dipakai
melatih, tidak tersisa apa pun untuk menguji. Rencana pembagiannya: sebagian
gambar untuk latih, sisanya **tidak disentuh sama sekali** sampai akhir -
dibagi per **gambar sumber**, bukan per crop, supaya ayam yang sama tidak
muncul di dua sisi.

## Sesudah selesai

Simpan CSV-nya, lalu bilang ke saya. Yang berikutnya:
build crop latih dari label ini -> latih ulang -> **uji acak-petak**
(`src/eval_intervensi.py`). Kalau AUC tidak jatuh saat crop diacak, model
masih belum membaca bentuk - berapa pun angkanya.
