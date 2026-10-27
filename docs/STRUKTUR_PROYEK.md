# Panduan Struktur Repository

Seluruh path berikut bersifat relatif terhadap root repository.

| Folder | Isi | Keterangan |
|---|---|---|
| `notebooks/` | `notebook.ipynb` | Notebook utama dan output eksekusi tersimpan |
| `data/raw/` | `ulasan_gojek_raw.csv` | Snapshot 15.000 ulasan; jangan ditimpa bila ingin mereproduksi hasil |
| `data/processed/` | `ulasan_gojek_labeled.csv` | Dataset berlabel dari pipeline offline |
| `data/lexicons/` | `lexicon_positive.csv`, `lexicon_negative.csv` | Kamus untuk pseudo-label |
| `src/` | Modul preprocessing, training, scraping, inferensi, audit | Semua dapat dijalankan dengan `python -m src.<nama_modul>` dari root |
| `models/` | Model `joblib`, checkpoint PyTorch, vocabulary | Digunakan oleh `src.inference` |
| `results/figures/` | Visualisasi PNG | Direferensikan oleh README dan notebook |
| `results/metrics/` | Laporan JSON dan CSV | Hasil eksekusi dan audit |
| `docs/` | Dokumen metodologi, dataset, catatan submission, dan panduan GitHub | Referensi tambahan |
| `tests/` | Unit test dan pemeriksaan integritas dataset | Jalankan `python -m pytest tests -q` |

## Perintah utama

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
python -m src.audit_dataset_gojek
python -m src.inference "Aplikasi Gojek sangat membantu"
python -m src.train_models
```

Notebook `notebooks/notebook.ipynb` dapat dijalankan melalui Jupyter dari root atau dari direktori `notebooks/`.

> `src.train_models` akan memperbarui dataset hasil preprocessing, artefak model, grafik, dan metrik. `src.scraping_gojek` akan memperbarui snapshot data mentah. Gunakan dengan sengaja agar hasil lama tidak tertimpa.
