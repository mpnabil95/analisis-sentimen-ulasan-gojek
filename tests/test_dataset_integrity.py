"""Pengujian integritas dataset untuk eksperimen sentimen Gojek."""
from pathlib import Path
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]

def test_snapshot_schema():
    df=pd.read_csv(ROOT/'data/raw/ulasan_gojek_raw.csv')
    assert len(df)==15000
    assert {'content','score','at','thumbsUpCount'}<=set(df.columns)

def test_training_data_unique_if_generated():
    path=ROOT/'data/processed/ulasan_gojek_labeled.csv'
    if not path.exists():pytest.skip('training notebook has not generated labeled CSV')
    df=pd.read_csv(path)
    assert len(df)>=10000
    assert df['text_stem'].notna().all()
    assert not df['text_stem'].duplicated().any()
    assert set(df['polarity'])=={'negatif','positif','netral'}
