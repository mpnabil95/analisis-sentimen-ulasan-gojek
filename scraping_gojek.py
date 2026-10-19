from google_play_scraper import reviews, Sort
import pandas as pd

APP_ID = 'com.gojek.app'
TARGET  = 15000

all_reviews = []
continuation_token = None

print("Memulai scraping ulasan Gojek dari Google Play Store...")
print(f"Target: {TARGET:,} ulasan\n")

while len(all_reviews) < TARGET:
    batch, continuation_token = reviews(
        APP_ID,
        lang='id',
        country='id',
        sort=Sort.NEWEST,
        count=200,
        continuation_token=continuation_token
    )
    if not batch:
        break
    all_reviews.extend(batch)
    print(f"Ulasan terkumpul: {len(all_reviews):,}", end='\r')

print(f"\nTotal ulasan: {len(all_reviews):,}")

df = pd.DataFrame(all_reviews)
df = df[['content', 'score', 'at', 'thumbsUpCount']]
df.to_csv('ulasan_gojek_raw.csv', index=False, encoding='utf-8')
print("Dataset disimpan --> ulasan_gojek_raw.csv")