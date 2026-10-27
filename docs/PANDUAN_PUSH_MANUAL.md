# Panduan Publikasi Repository

Dokumen ini berisi checklist teknis untuk meninjau proyek sebelum memperbarui branch `main`. Semua perubahan GitHub dilakukan secara manual dari lingkungan lokal setelah berkas diperiksa.

## Checklist pra-publikasi

- [ ] Pastikan `README.md`, `docs/DATASET.md`, `docs/METODOLOGI.md`, dan `docs/STATUS_SUBMISSION.md` sesuai dengan hasil eksperimen.
- [ ] Tinjau `notebooks/notebook.ipynb`, termasuk seluruh output dan metrik evaluasi.
- [ ] Pastikan penggunaan PyTorch dan preprocessing offline sudah sesuai dengan versi yang akan dipublikasikan.
- [ ] Verifikasi kesesuaian dataset, model, gambar, dan laporan hasil.
- [ ] Jalankan pengujian yang relevan serta periksa perubahan lokal dengan Git.

## Pemeriksaan lokal

Setelah isi paket disalin ke direktori checkout repository lokal, jalankan:

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
python -m src.inference "Gojek sangat membantu"

git status
git diff --stat
```

## Publikasi

Jika semua perubahan sudah sesuai, commit dan push dapat dilakukan secara manual:

```bash
git add -A
git commit -m "Document and finalize Gojek sentiment analysis experiments"
git push origin main
```

Hindari penggunaan `git push --force` tanpa alasan khusus untuk menjaga riwayat perubahan. Simpan cadangan apabila terdapat pekerjaan lokal yang belum di-commit.
