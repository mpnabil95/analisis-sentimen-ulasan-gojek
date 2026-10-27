# Catatan Evaluasi Kriteria Submission Dicoding

Evaluasi ini menggunakan halaman **Penilaian** pada dokumen *Submission 1: Proyek Analisis Sentimen* sebagai acuan. Hasil pemeriksaan bersifat **mandiri**, bukan keputusan atau persetujuan reviewer Dicoding.

| Kriteria atau saran penilaian | Bukti pada repository | Status |
|---|---|---|
| Implementasi deep learning | Bidirectional LSTM menggunakan PyTorch | Terpenuhi secara teknis |
| Akurasi training dan testing >92% pada tiga skema | LinearSVC 99,25%/92,50%; Logistic Regression 99,78%/90,76%; BiLSTM 94,23%/89,09% | **Belum sepenuhnya terpenuhi** |
| Minimal tiga kelas sentimen | Negatif, netral, positif | Terpenuhi |
| Minimal 10.000 sampel | 10.863 teks unik setelah preprocessing | Terpenuhi |
| Tiga variasi skema pelatihan | Tiga algoritma/representasi dan pembagian data berbeda | Terpenuhi |
| Inferensi dengan keluaran label sentimen | Enam contoh inferensi tersimpan pada notebook dan file CSV | Terpenuhi |

Seluruh skema mencapai akurasi pengujian **di atas 85%**, tetapi tidak seluruhnya melebihi 92%. Angka tersebut tidak boleh ditafsirkan sebagai bukti pemenuhan seluruh saran penilaian.

Pemeriksaan ini belum mencakup seluruh isi bagian **Kriteria Utama** dan **Ketentuan Berkas Submission** pada platform Dicoding. Persyaratan resmi perlu diperiksa sebelum melakukan pengumpulan tugas.

Implementasi saat ini memakai PyTorch dan preprocessing offline, berbeda dari eksperimen TensorFlow/Keras dengan NLTK serta PySastrawi yang dijalankan sebelumnya. Rincian perubahan tercatat pada [`METODOLOGI.md`](METODOLOGI.md).

**Status:** artefak eksperimen dan hasil evaluasi tersedia untuk ditinjau. Kelayakan submission akhir tetap mengikuti ketentuan yang berlaku serta keputusan reviewer Dicoding.
