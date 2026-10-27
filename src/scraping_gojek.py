"""Pengumpulan ulasan aplikasi Gojek dari Google Play Store.

Menimpa data/raw/ulasan_gojek_raw.csv saat dijalankan. Cadangkan file asli
jika ingin mereproduksi eksperimen dengan snapshot data yang sama.
"""
from google_play_scraper import reviews, Sort
import pandas as pd
from pathlib import Path
APP_ID = 'com.gojek.app'
TARGET = 15000
all_reviews = []
continuation_token = None
print('Memulai scraping ulasan Gojek dari Google Play Store...')
while len(all_reviews) < TARGET:
    batch, continuation_token = reviews(
        APP_ID,lang='id',country='id',sort=Sort.NEWEST,count=200,
        continuation_token=continuation_token
    )
    if not batch:
        break
    all_reviews.extend(batch)
    print(f'Ulasan terkumpul: {len(all_reviews):,}',end='\r')
print(f'\nTotal ulasan: {len(all_reviews):,}')
df = pd.DataFrame(all_reviews)
df = df[['content','score','at','thumbsUpCount']]
output_path = Path(__file__).resolve().parents[1] / 'data/raw/ulasan_gojek_raw.csv'
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path,index=False,encoding='utf-8')
print(f'Dataset disimpan -> {output_path}')
