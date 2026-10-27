# Metodologi Analisis Sentimen

Dokumen ini menjabarkan alur preprocessing, strategi pelabelan, pembagian data, pelatihan model, evaluasi, dan keterbatasan pada implementasi repository.

## 1. Implementasi dan cakupan eksperimen

Pipeline yang tersedia menggunakan **PyTorch** untuk Bidirectional LSTM serta preprocessing yang dirancang agar dapat dijalankan secara offline. Eksperimen terdahulu menggunakan **TensorFlow/Keras, NLTK, dan PySastrawi**.

Perbedaan tersebut memengaruhi tokenisasi, stemming, jumlah data hasil pembersihan, serta hasil pemodelan. Dengan demikian, angka evaluasi di repository ini merupakan hasil **eksperimen tersendiri**, bukan reproduksi identik dari notebook sebelumnya.

## 2. Preprocessing teks

Tahapan preprocessing terdiri atas:

1. **Cleaning:** menghapus mention, hashtag, URL, angka, tanda baca, dan karakter non-ASCII.
2. **Case folding:** menyamakan kapitalisasi ke huruf kecil.
3. **Normalisasi slang:** memetakan sejumlah kata tidak baku dan singkatan ke bentuk yang lebih konsisten.
4. **Stopword filtering:** menerapkan daftar stopword tetap dengan pengecualian kosakata yang berpengaruh terhadap sentimen.
5. **Stemming ringan:** mereduksi sejumlah bentuk imbuhan memakai aturan dan pengecualian kata tertentu.
6. **Deduplikasi:** menghapus baris dengan `text_stem` identik sebelum proses pemisahan training dan testing.

Stemming yang digunakan merupakan pendekatan berbasis aturan sederhana, **bukan implementasi PySastrawi**. Kode lengkapnya terdapat di [`pipeline_gojek.py`](../src/pipeline_gojek.py).

## 3. Pelabelan otomatis

Label dibentuk dari leksikon positif dan negatif yang masing-masing memiliki bobot numerik. Polaritas ditentukan dari akumulasi skor kata, termasuk pembalikan skor untuk pola negasi sederhana.

- Skor lebih dari nol → `positif`
- Skor sama dengan nol → `netral`
- Skor kurang dari nol → `negatif`

Label ini bersifat **pseudo-label**, sehingga kesalahan pada kamus atau aturan negasi dapat terbawa ke proses pelatihan. Kesesuaian dengan anotasi manusia belum diuji secara independen.

## 4. Konfigurasi pelatihan

| Skema | Klasifier | Fitur | Rasio data |
|---|---|---|---|
| 1 | LinearSVC (L1) | TF-IDF unigram/bigram | 80:20 |
| 2 | Logistic Regression (L2) | Bag-of-Words unigram/bigram | 70:30 |
| 3 | Bidirectional LSTM (PyTorch) | Vocabulary + word embedding | 80:20, kemudian 10% bagian latih sebagai validation |

Proses pembagian data dilakukan secara **stratified** dengan seed `42`. Vectorizer dan vocabulary hanya dipelajari dari subset training untuk mengurangi risiko kebocoran informasi. Pada BiLSTM, pemilihan bobot didasarkan pada validation accuracy, sedangkan test set digunakan setelah pemilihan bobot.

## 5. Evaluasi dan inferensi

Metrik utama meliputi akurasi training, akurasi testing, **macro F1-score**, classification report per kelas, dan confusion matrix. Kurva pelatihan BiLSTM digunakan untuk memantau perkembangan performa training dan validation.

Seluruh nilai yang digunakan dalam dokumentasi mengacu pada [`results/metrics/hasil_evaluasi_final.json`](../results/metrics/hasil_evaluasi_final.json), sementara visualisasi tersedia pada direktori [`results/figures/`](../results/figures/). Contoh inferensi terdiri atas enam ulasan baru, dengan keluaran kategorikal dari ketiga model dan voting mayoritas.

## 6. Batasan penelitian

- Pseudo-label otomatis belum dibandingkan dengan anotasi manusia.
- Deduplikasi hanya menjamin keunikan `text_stem`, bukan seluruh kemiripan semantik.
- Sarkasme, negasi yang tidak berdekatan, dan sentimen campuran tetap menjadi tantangan bagi pendekatan berbasis aturan.
- Akurasi dapat berubah jika data, preprocessing, hyperparameter, versi pustaka, atau perangkat diubah.
- Pembacaan artefak `joblib` hanya dilakukan dari sumber tepercaya untuk menghindari risiko deserialisasi berbahaya.

Hasil pelatihan dilaporkan sesuai keluaran eksperimen, termasuk skema yang belum mencapai semua ambang penilaian.
