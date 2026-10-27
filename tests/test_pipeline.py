"""Pengujian unit preprocessing dan pelabelan, tanpa melatih model."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.pipeline_gojek import (cleaning_text,normalize_slang,TextPreprocessor,
                            sentiment_lexicon,preprocess_reviews,validate_dataset)

def test_cleaning_text():
    assert cleaning_text('Bagus sekali!!! 🤗')=='Bagus sekali'
    assert normalize_slang('gak bagus')=='tidak bagus'

def test_negation_lexicon():
    assert sentiment_lexicon('bagus',{'bagus':4},{'buruk':-4})[1]=='positif'
    assert sentiment_lexicon('tidak bagus',{'bagus':4},{'buruk':-4})[1]=='negatif'
    assert sentiment_lexicon('tidak buruk',{'bagus':4},{'buruk':-4})[1]=='positif'

def test_deduplication_before_split():
    df=pd.DataFrame({'content':['Mantap!','mantap','buruk','Sangat bagus']})
    pre=TextPreprocessor()
    out,summary=preprocess_reviews(df,pre)
    assert out['text_stem'].is_unique
    assert len(out)==3
