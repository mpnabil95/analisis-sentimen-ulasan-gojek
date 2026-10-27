"""Pipeline preprocessing, pelabelan leksikon, dan pemeriksaan integritas dataset.

Modul dapat diimpor tanpa mengunduh data atau melatih model.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
CLASSES = ("negatif", "netral", "positif")
NEGATION_WORDS = {"tidak", "bukan", "jangan", "belum", "kurang", "tanpa"}

SLANGWORDS = {
    "gak":"tidak", "ga":"tidak", "ngga":"tidak", "nggak":"tidak", "gk":"tidak", "enggak":"tidak", "ndak":"tidak", "kagak":"tidak", "g":"tidak",
    "gabisa":"tidak bisa", "gbs":"tidak bisa", "gakbisa":"tidak bisa", "bgt":"banget", "bgtt":"banget", "bgttt":"banget",
    "yg":"yang", "dg":"dengan", "dgn":"dengan", "trs":"terus", "udah":"sudah", "udh":"sudah", "sdh":"sudah", "dah":"sudah",
    "blm":"belum", "blom":"belum", "belom":"belum", "mo":"mau", "pengen":"ingin", "kepengen":"ingin",
    "gmn":"bagaimana", "gmana":"bagaimana", "gimana":"bagaimana", "knp":"kenapa", "knpa":"kenapa",
    "klo":"kalau", "kl":"kalau", "kal":"kalau", "kalo":"kalau", "jg":"juga", "jga":"juga", "jgn":"jangan",
    "tp":"tapi", "tpi":"tapi", "sm":"sama", "ama":"sama", "emg":"memang", "emang":"memang", "mmg":"memang",
    "makasih":"terima kasih", "makasi":"terima kasih", "terimakasih":"terima kasih", "mks":"terima kasih", "tq":"terima kasih", "thx":"terima kasih",
    "mantep":"mantap", "mantul":"mantap", "oke":"ok", "oce":"ok", "okay":"ok", "josss":"jos", "joss":"jos",
    "apk":"aplikasi", "app":"aplikasi", "aplikasinya":"aplikasi", "drivernya":"driver", "drv":"driver",
    "orderan":"pesanan", "pesen":"pesan", "mesen":"pesan", "pesenan":"pesanan", "batalin":"batal", "dibatalin":"batal", "dibatalkan":"batal", "dicancel":"batal", "cancel":"batal",
    "nunggu":"tunggu", "menunggu":"tunggu", "dapet":"dapat", "dpt":"dapat", "sampe":"sampai", "nyampe":"sampai",
    "pake":"pakai", "pk":"pakai", "dipake":"pakai", "dipakai":"pakai", "cepet":"cepat", "lelet":"lambat", "lemot":"lambat", "eror":"error",
    "ongkirnya":"ongkir", "kesel":"kesal", "nyesel":"menyesal", "diperbaiki":"perbaiki", "memperbaiki":"perbaiki",
    "anjg":"anjing", "anjgghh":"anjing", "anjir":"anjing", "bgst":"bangsat", "sy":"saya", "aku":"saya", "gue":"saya", "gw":"saya", "gua":"saya",
    "lo":"kamu", "lu":"kamu", "elu":"kamu", "bener":"benar", "bner":"benar", "baek":"baik", "gampang":"mudah",
    "adain":"ada", "mnta":"minta", "moga":"semoga", "sprti":"seperti", "krn":"karena", "karna":"karena", "pdhl":"padahal", "pdhal":"padahal",
    "bbrp":"beberapa", "aja":"saja", "aj":"saja", "doang":"saja",
}
SENTIMENT_WHITELIST = {
    'tidak','bukan','jangan','belum','kurang','tanpa', 'baik','bagus','sangat','lebih','paling','cukup','lumayan','biasa',
    'bisa','mau','ingin','perlu','harus','banyak','sedikit', 'lama','baru','benar','salah','jelas','tepat','cocok','pas',
    'naik','turun','tinggi','rendah','besar','kecil','jauh','dekat','cepat','lambat','mudah','sulit','murah','mahal','aman','nyaman',
    'puas','kecewa','bantu','membantu','tolong','kasih','terima','sukses','gagal','lancar','macet','ramah','kasar','parah','buruk',
    'jelek','keren','mantap','enak','hemat','boros','untung','rugi','masalah','kendala','batal','tunggu','perbaiki','penting','luar',
}
CUSTOM_STOPWORDS = {'iya','yaa','ya','na','ku','nya','si','kah','tah','pun','mah','nah','deh','dong','sih','loh','lho','woi','woy',
                    'gojek','aplikasi','saja','kok','kan','tuh','gitu','gini'}


def cleaning_text(text: object) -> str:
    """Membersihkan teks sebelum tahap normalisasi lanjutan."""
    if not isinstance(text, str):
        return ''
    text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)
    text = re.sub(r'#[A-Za-z0-9_]+', ' ', text)
    text = re.sub(r'RT[\s]', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[0-9]+', ' ', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def normalize_slang(text: str) -> str:
    return ' '.join(SLANGWORDS.get(token, token) for token in text.split())


# Eksperimen sebelumnya menggunakan NLTK dan PySastrawi.
# Versi offline menggunakan stopwords tetap dan stemming berbasis aturan ringan.
# Perbedaan implementasi menyebabkan hasil dievaluasi sebagai eksperimen terpisah.
PORTABLE_STOPWORDS = set("""ada adalah agar akan akibat aku anda apa apabila atas atau bahwa bagi bahkan bagaimana baik bukan cuma dari daripada dalam dan dengan dia di dimana ini itu jadi jika juga kalau kami kamu karena ke kepada ketika kini lagi lalu maka mungkin namun oleh pada para paling pula pun saja sampai sangat satu sejak sebagai sebuah sedang sehingga setiap sudah suatu supaya tanpa tapi telah tentang terhadap tetapi tidak untuk walau yang yakni yaitu bisa cuma hanya begitu begitu dirinya kita mereka nya ku lah kah danlah per serta jadi kepada lewat setelah sebelum saat bukan hanya malah serta agar masih merasa terasa""".split())

def offline_stopwords() -> set[str]:
    return (PORTABLE_STOPWORDS | CUSTOM_STOPWORDS) - SENTIMENT_WHITELIST

# Pengecualian stemming diterapkan pada beberapa kata yang memengaruhi sentimen,
# terutama agar 'perbaiki' tidak keliru berubah menjadi 'baik'.
STEM_EXCEPTIONS = {'perbaiki':'perbaiki','diperbaiki':'perbaiki','memperbaiki':'perbaiki',
                   'membantu':'bantu','terbantu':'bantu','memudahkan':'mudah',
                   'mengecewakan':'kecewa','menyesal':'nyesal','menunggu':'tunggu',
                   'dibatalkan':'batal','membatalkan':'batal','diblokir':'blokir',
                   'terlambat':'lambat','mencekik':'cekik','berhasil':'hasil',
                   'menyenangkan':'senang','memuaskan':'puas','mengganggu':'ganggu',
                   'terganggu':'ganggu','digunakan':'guna','pengguna':'guna'}

def stem_light(word: str) -> str:
    if word in STEM_EXCEPTIONS:
        return STEM_EXCEPTIONS[word]
    # Stemming berbasis aturan sederhana; bukan implementasi PySastrawi.
    if len(word) > 7 and word.endswith('nya'):
        word = word[:-3]
    if len(word) > 7 and word.endswith('kan'):
        word = word[:-3]
    elif len(word) > 6 and word.endswith('an'):
        word = word[:-2]
    if len(word) > 7:
        for prefix in ('diper','meng','meny','ber','ter','di','ke'):
            if word.startswith(prefix) and len(word)-len(prefix) >= 4:
                word = word[len(prefix):]
                break
    return word

class TextPreprocessor:
    def __init__(self, stopwords_set=None, stemmer=None):
        self.stopwords = offline_stopwords() if stopwords_set is None else stopwords_set
        self.stem_cache = dict(STEM_EXCEPTIONS)
        self.stemmer = stemmer

    def stem_token(self, word: str) -> str:
        if word not in self.stem_cache:
            self.stem_cache[word] = stem_light(word)
        return self.stem_cache[word]

    def transform(self, text: str) -> str:
        """Cleaning -> lowercase -> slang -> tokenize -> stops -> light stemming."""
        s = normalize_slang(cleaning_text(text).lower())
        filtered = [w for w in s.split() if w not in self.stopwords and len(w) > 1]
        return ' '.join(self.stem_token(w) for w in filtered)

def load_lexicon(path: str | Path, stemmer: TextPreprocessor | None = None) -> dict[str,int]:
    lexicon: dict[str,int] = {}
    with open(path, encoding='utf-8', newline='') as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                try:
                    lexicon[row[0].strip()] = int(row[1])
                except (ValueError, IndexError):
                    pass
    if stemmer:
        for w, val in list(lexicon.items()):
            root = stemmer.stem_token(w)
            if root not in lexicon:
                lexicon[root] = val
    return lexicon


def load_both_lexicons(positive: str | Path, negative: str | Path, processor: TextPreprocessor):
    """Tambahkan root tanpa menabrak entri dari polaritas lawan."""
    from_positive = load_lexicon(positive)
    from_negative = load_lexicon(negative)
    for w, v in list(from_positive.items()):
        root = processor.stem_token(w)
        if root not in from_positive and root not in from_negative:
            from_positive[root] = v
    for w, v in list(from_negative.items()):
        root = processor.stem_token(w)
        if root not in from_negative and root not in from_positive:
            from_negative[root] = v
    return from_positive, from_negative


def sentiment_lexicon(text_stemmed: str, positive: dict[str,int], negative: dict[str,int]) -> tuple[int,str]:
    score, negate_next = 0, False
    for word in text_stemmed.split():
        if word in NEGATION_WORDS:
            negate_next = True
            continue
        word_score = positive.get(word, negative.get(word, 0))
        if word_score != 0:
            score += -word_score if negate_next else word_score
            negate_next = False
        elif negate_next:
            score -= 1
            negate_next = False
    if negate_next:
        score -= 1
    label = 'positif' if score > 0 else ('negatif' if score < 0 else 'netral')
    return score, label


def preprocess_reviews(raw: pd.DataFrame, processor: TextPreprocessor) -> tuple[pd.DataFrame,dict]:
    if 'content' not in raw:
        raise ValueError("Dataset wajib memiliki kolom 'content'")
    out = raw.dropna(subset=['content']).copy()
    out = out[out['content'].astype(str).str.strip().ne('')]
    counts = {'raw_rows':len(raw),'after_remove_missing':len(out)}
    out = out.drop_duplicates(subset='content').reset_index(drop=True)
    counts['after_dedup_raw_content'] = len(out)
    out['text_stem'] = out['content'].map(processor.transform)
    out = out[out['text_stem'].str.strip().ne('')].copy()
    counts['after_drop_empty_stems'] = len(out)
    # Duplikat input model dihapus sebelum pembagian training dan testing.
    out = out.drop_duplicates(subset='text_stem').reset_index(drop=True)
    counts['after_dedup_text_stem'] = len(out)
    counts['stem_duplicates_removed'] = counts['after_drop_empty_stems'] - len(out)
    return out, counts


def label_reviews(df: pd.DataFrame, positive: dict[str,int], negative: dict[str,int]) -> pd.DataFrame:
    result = df.copy()
    labels = result['text_stem'].map(lambda x: sentiment_lexicon(x,positive,negative))
    result['polarity_score'] = labels.map(lambda x:x[0])
    result['polarity'] = labels.map(lambda x:x[1])
    return result


def validate_dataset(df: pd.DataFrame, minimum_rows: int = 10000) -> dict:
    if len(df) < minimum_rows:
        raise ValueError(f"Sampel = {len(df)}, lebih kecil dari minimum {minimum_rows}")
    if df['text_stem'].isna().any() or (df['text_stem'].str.strip() == '').any():
        raise ValueError('Ada teks kosong dalam text_stem')
    if df['text_stem'].duplicated().any():
        raise ValueError('Ada text_stem duplikat')
    if set(df['polarity']) != set(CLASSES):
        raise ValueError(f"Kelas tidak tepat tiga: {set(df['polarity'])}")
    return {'rows':int(len(df)), 'class_counts':{str(k):int(v) for k,v in df['polarity'].value_counts().items()}, 'unique_stems':True}


def make_split(X, y, test_size:float, seed:int=SEED):
    a,b,ya,yb = train_test_split(X,y,test_size=test_size, random_state=seed, stratify=y)
    if set(map(str,a)) & set(map(str,b)):
        raise AssertionError('Data leakage: terdapat overlap text_stem train-test')
    return a,b,ya,yb


def assert_no_overlap(*arrays: Iterable[str]):
    sets = [set(map(str,x)) for x in arrays]
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            if sets[i] & sets[j]:
                raise AssertionError(f'Overlap antar split {i} dan {j}')
    return True
