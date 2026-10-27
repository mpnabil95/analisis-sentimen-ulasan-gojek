"""Antarmuka CLI untuk inferensi sentimen menggunakan model tersimpan.

  python -m src.inference "Aplikasi Gojek sangat membantu"

Hanya memuat file model yang berasal dari repo sendiri/terpercaya.
"""
import argparse
import os
import json
from pathlib import Path
from collections import Counter
import joblib
import torch
torch.set_num_threads(min(2, os.cpu_count() or 2))
from .pipeline_gojek import TextPreprocessor
from .train_models import BiLSTMClassifier, MAX_LEN
ROOT=Path(__file__).resolve().parents[1]


def predict(text):
    pre=TextPreprocessor()
    clean=pre.transform(text)
    # Memuat joblib atau checkpoint dari sumber yang tidak dipercaya tidak aman.
    m1=joblib.load(ROOT/'models/skema_1_linear_svc_tfidf.joblib')
    m2=joblib.load(ROOT/'models/skema_2_logreg_bow.joblib')
    vocab=json.loads((ROOT/'models/vocab_bilstm.json').read_text(encoding='utf-8'))
    ckpt=torch.load(ROOT/'models/skema_3_bilstm.pt',map_location='cpu',weights_only=True)
    net=BiLSTMClassifier(ckpt['vocab_size'],ckpt['embed_dim'],ckpt['hidden_size'],len(ckpt['label_order']))
    net.load_state_dict(ckpt['state_dict']);net.eval()
    token_ids=[vocab.get(w,1) for w in clean.split()][:ckpt['max_len']] or [1]
    with torch.no_grad():
        logits=net(torch.tensor([token_ids]),torch.tensor([len(token_ids)]))
        probs=torch.softmax(logits,dim=1).squeeze(0).tolist()
    preds=[str(m1.predict([clean])[0]),str(m2.predict([clean])[0]),ckpt['label_order'][max(range(len(probs)),key=lambda i:probs[i])]]
    return {'input':text,'processed':clean,'linear_svc':preds[0],
            'logistic_regression':preds[1],'bilstm':preds[2],
            'bilstm_confidence':round(max(probs),4),
            'majority_vote':Counter(preds).most_common(1)[0][0]}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Inferensi 3 skema sentimen Gojek')
    parser.add_argument('text',help='Ulasan yang akan diklasifikasikan')
    args=parser.parse_args()
    print(json.dumps(predict(args.text),ensure_ascii=False,indent=2))
