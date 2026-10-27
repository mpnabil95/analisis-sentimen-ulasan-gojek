# Dokumentasi Dataset Ulasan Gojek

## Sumber dan cakupan data

Dataset berasal dari ulasan aplikasi **Gojek** (`com.gojek.app`) pada Google Play Store. Skrip [`scraping_gojek.py`](../src/scraping_gojek.py) menggunakan `google-play-scraper` dengan bahasa `id`, negara `id`, dan pengurutan berdasarkan ulasan terbaru.

Snapshot `data/raw/ulasan_gojek_raw.csv` terdiri atas **15.000 ulasan** dan empat kolom berikut.

| Kolom | Deskripsi |
|---|---|
| `content` | Isi teks ulasan |
| `score` | Rating bintang (1–5) |
| `at` | Waktu ulasan ditulis |
| `thumbsUpCount` | Jumlah tanda suka |

Rentang tanggal pada kolom `at` adalah **10 Mei–18 Oktober 2026**. Rentang tersebut merujuk pada tanggal ulasan, bukan waktu pengambilan data.

## Persiapan dataset berlabel

Pengolahan dataset mengikuti urutan:

1. Menghapus konten kosong atau nilai hilang.
2. Menghapus duplikat berdasarkan teks mentah.
3. Melakukan cleaning, case folding, normalisasi slang, stopword filtering, dan stemming berbasis aturan ringan.
4. Menghapus teks kosong dan duplikat `text_stem` setelah preprocessing.
5. Menghitung skor polaritas berbasis leksikon dengan penanganan negasi sederhana, kemudian menghasilkan label `negatif`, `netral`, atau `positif`.

Jumlah data akhir untuk implementasi ini adalah **10.863 teks unik**, dengan rincian:

| Kelas | Jumlah |
|---|---:|
| Negatif | 5.110 |
| Positif | 4.074 |
| Netral | 1.679 |
| **Total** | **10.863** |

Label `polarity` merupakan **pseudo-label leksikon** dan belum diverifikasi terhadap anotasi manusia independen. Oleh karena itu, evaluasi model tidak boleh diinterpretasikan sebagai pengukuran langsung ketepatan sentimen menurut manusia.

## Konsistensi antarversi

Implementasi sebelumnya memakai NLTK dan PySastrawi, sedangkan versi repository ini menggunakan preprocessing offline dengan stopword tetap dan stemming ringan. Perbedaan metode menghasilkan jumlah sampel yang tidak sama:

| Tahap eksperimen | Jumlah sampel | Keterangan |
|---|---:|---|
| Notebook awal | 11.665 | TensorFlow/Keras, NLTK, PySastrawi |
| Laporan audit antara | 10.569 | Deduplikasi dari dataset berlabel pada tahap audit |
| **Versi repository ini** | **10.863** | Pipeline offline, deduplikasi hasil preprocessing |

**10.863** merupakan angka yang digunakan untuk hasil evaluasi yang disertakan dalam [`results/metrics/hasil_evaluasi_final.json`](../results/metrics/hasil_evaluasi_final.json). Akurasi dari implementasi berbeda tidak digabungkan dalam satu hasil eksperimen.

## Reproduksibilitas, privasi, dan batasan

Pembagian dataset menggunakan seed `42`. Reproduksi eksperimen sebaiknya memakai snapshot `data/raw/ulasan_gojek_raw.csv` yang tersedia karena pengambilan ulang melalui Google Play Store dapat menghasilkan kumpulan ulasan berbeda.

Dataset terdiri dari konten publik yang dapat mengandung data pribadi serta tidak selalu mewakili keseluruhan pengguna aplikasi. Distribusi dan publikasinya perlu memperhatikan privasi dan ketentuan sumber data.
