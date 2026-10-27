"""Pelatihan tiga skema klasifikasi sentimen dengan dukungan eksekusi CPU.

Dijalankan via notebooks/notebook.ipynb atau `python -m src.train_models`.
"""
from __future__ import annotations
import json, os, random, time, sys, math, platform
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .pipeline_gojek import (SEED, TextPreprocessor, preprocess_reviews, load_both_lexicons,
                             label_reviews, validate_dataset, make_split, assert_no_overlap)

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/raw/ulasan_gojek_raw.csv'
RESULTS=ROOT/'results/metrics'
MODELS=ROOT/'models'
ASSETS=ROOT/'results/figures'
MAX_LEN=28
VOCAB_MAX=10000
EMBED_DIM=32
HIDDEN_SIZE=16
BATCH_SIZE=256
EPOCHS=12
PATIENCE=3

def setup():
    for p in (RESULTS,MODELS,ASSETS,ROOT/'data/processed'): p.mkdir(parents=True,exist_ok=True)
    random.seed(SEED); np.random.seed(SEED);torch.manual_seed(SEED)
    torch.set_num_threads(min(2,os.cpu_count() or 2))
    try:torch.use_deterministic_algorithms(True)
    except Exception:pass
    return 'cpu'


def tokenize(text):return text.split()

class SentenceVocabulary:
    def __init__(self,max_words=VOCAB_MAX):
        self.words={'<PAD>':0,'<UNK>':1}
        self.max_words=max_words
    def fit(self,training_texts):
        count=Counter(word for text in training_texts for word in tokenize(str(text)))
        for word,_ in sorted(count.items(),key=lambda x:(-x[1],x[0])):
            if len(self.words)>=self.max_words:break
            self.words[word]=len(self.words)
        return self
    def encode(self,text):
        ids=[self.words.get(w,1) for w in tokenize(str(text))][:MAX_LEN]
        return ids if ids else [1]

class ReviewDataset(Dataset):
    def __init__(self,texts,y,vocab):
        self.texts=[vocab.encode(s) for s in texts]
        self.y=np.asarray(y,dtype=np.int64)
    def __len__(self):return len(self.y)
    def __getitem__(self,i):return self.texts[i],int(self.y[i])

def collate(items):
    sz=max(len(s) for s,_ in items)
    ids=torch.zeros((len(items),sz),dtype=torch.long)
    lens=torch.empty(len(items),dtype=torch.long)
    ys=torch.empty(len(items),dtype=torch.long)
    for i,(s,y) in enumerate(items):
        ids[i,:len(s)]=torch.tensor(s,dtype=torch.long)
        lens[i]=len(s);ys[i]=y
    return ids,lens,ys

class BiLSTMClassifier(nn.Module):
    def __init__(self,vocab_size,embed_dim=EMBED_DIM,hidden_size=HIDDEN_SIZE,classes=3):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.lstm=nn.LSTM(embed_dim,hidden_size,batch_first=True,bidirectional=True)
        self.dropout=nn.Dropout(0.30)
        self.linear1=nn.Linear(2*hidden_size,64)
        self.linear2=nn.Linear(64,classes)
    def forward(self,ids,lengths):
        x=self.embedding(ids)
        p=pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
        p,_=self.lstm(p)
        x,_=pad_packed_sequence(p,batch_first=True,total_length=ids.shape[1])
        mask=torch.arange(ids.shape[1],device=ids.device)[None,:] < lengths.to(ids.device)[:,None]
        pooled=x.masked_fill(~mask[:,:,None],torch.finfo(x.dtype).min).max(dim=1).values
        return self.linear2(self.dropout(torch.relu(self.linear1(self.dropout(pooled)))))

def prediction(model,loader,criterion=None,optimizer=None):
    training=optimizer is not None
    model.train(training)
    n,correct,total_loss=0,0,0.
    outs=[]; gold=[]
    for ids,lengths,ys in loader:
        with torch.set_grad_enabled(training):
            logits=model(ids,lengths)
            loss=(criterion or nn.CrossEntropyLoss())(logits,ys)
            if training:
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        guessed=logits.argmax(1)
        correct+=(guessed==ys).sum().item(); n+=len(ys)
        total_loss+=loss.item()*len(ys)
        outs.extend(guessed.detach().cpu().tolist());gold.extend(ys.tolist())
    return {'loss':float(total_loss/max(n,1)),'accuracy':float(correct/max(n,1)),'pred':outs,'true':gold}


def build_evaluation(y_train,y_pred_train,y_test,y_pred_test,order=('negatif','netral','positif')):
    return {
        'training_accuracy':round(100*accuracy_score(y_train,y_pred_train),4),
        'testing_accuracy':round(100*accuracy_score(y_test,y_pred_test),4),
        'testing_macro_f1':round(f1_score(y_test,y_pred_test,average='macro',zero_division=0),4),
        'report':classification_report(y_test,y_pred_test,labels=list(order),output_dict=True,zero_division=0),
        'confusion_matrix':confusion_matrix(y_test,y_pred_test,labels=list(order)).tolist(),
        'train_size':int(len(y_train)), 'test_size':int(len(y_test)),
    }

def plot_cm(e, title, filename):
    plt.figure(figsize=(6.5,5.1))
    sns.heatmap(e['confusion_matrix'],annot=True,fmt='d',cmap='Blues',xticklabels=['Negatif','Netral','Positif'],yticklabels=['Negatif','Netral','Positif'])
    plt.title(title);plt.xlabel('Prediksi');plt.ylabel('Aktual');plt.tight_layout()
    plt.savefig(ASSETS/filename,dpi=160);plt.close()

def run():
    setup()
    start=time.time()
    print('Memuat data dan membersihkan teks...')
    raw=pd.read_csv(DATA)
    preprocessor=TextPreprocessor()
    normalized,counts=preprocess_reviews(raw,preprocessor)
    pos,neg=load_both_lexicons(ROOT/'data/lexicons/lexicon_positive.csv',ROOT/'data/lexicons/lexicon_negative.csv',preprocessor)
    df=label_reviews(normalized,pos,neg)
    integrity=validate_dataset(df)
    df.to_csv(ROOT/'data/processed/ulasan_gojek_labeled.csv',index=False,encoding='utf-8')
    print('Audit:',counts,'kelas:',integrity['class_counts'])
    X=df['text_stem'].to_numpy(); y=df['polarity'].to_numpy()
    results={'metadata':{'project':'Analisis sentimen ulasan aplikasi Gojek','seed':SEED,
           'framework_deep_learning':'PyTorch (offline implementation; berbeda dari eksperimen TensorFlow/Keras sebelumnya)',
           'processing':'normalisasi slang + fixed stopwords + light stemmer offline; berbeda dari eksperimen PySastrawi/NLTK sebelumnya',
           'data_source':'data/raw/ulasan_gojek_raw.csv','timestamp_utc':pd.Timestamp.now(tz='UTC').isoformat(),
           'python':sys.version.split()[0],'pandas':pd.__version__,'scikit_learn':__import__('sklearn').__version__,
           'torch':torch.__version__,'execution_device':'cpu'},
           'dataset':{**counts,**integrity},'schemes':{}}

    # Skema 1: data dibagi sebelum proses fitting TF-IDF dan LinearSVC.
    print('Melatih skema 1 (LinearSVC + TF-IDF)...')
    tr,te,ytr,yte=make_split(X,y,0.20)
    m1=Pipeline([('tfidf',TfidfVectorizer(max_features=10000,ngram_range=(1,2),max_df=.95,sublinear_tf=True)),
                 ('svm',LinearSVC(penalty='l1',dual=False,C=1.5,max_iter=3000,random_state=SEED))])
    m1.fit(tr,ytr);ev=build_evaluation(ytr,m1.predict(tr),yte,m1.predict(te))
    results['schemes']['skema_1_linear_svc_tfidf']={'algorithm':'LinearSVC (L1)','features':'TF-IDF 1-2 gram','split':'80/20',**ev}
    joblib.dump(m1,MODELS/'skema_1_linear_svc_tfidf.joblib',compress=3)
    plot_cm(ev,'Skema 1 — LinearSVC + TF-IDF','confusion_matrix_skema_1.png')
    print('Skema 1 akurasi:',ev['training_accuracy'],ev['testing_accuracy'])

    # Skema 2: Bag-of-Words dan Logistic Regression dengan rasio 70/30.
    print('Melatih skema 2 (Logistic Regression + BoW)...')
    tr2,te2,ytr2,yte2=make_split(X,y,0.30)
    m2=Pipeline([('bow',CountVectorizer(max_features=10000,ngram_range=(1,2),max_df=.95)),
                 ('lr',LogisticRegression(C=2,max_iter=1000,solver='lbfgs',random_state=SEED))])
    m2.fit(tr2,ytr2); ev=build_evaluation(ytr2,m2.predict(tr2),yte2,m2.predict(te2))
    results['schemes']['skema_2_logreg_bow']={'algorithm':'Logistic Regression (L2)','features':'Bag of Words 1-2 gram','split':'70/30',**ev}
    joblib.dump(m2,MODELS/'skema_2_logreg_bow.joblib',compress=3)
    plot_cm(ev,'Skema 2 — Logistic Regression + BoW','confusion_matrix_skema_2.png')
    print('Skema 2 akurasi:',ev['training_accuracy'],ev['testing_accuracy'])

    # Skema 3: pemisahan training, validation, dan testing untuk BiLSTM.
    print('Melatih skema 3 (PyTorch Bidirectional LSTM)...')
    tr_full,te3,y_full,yt3=make_split(X,y,0.20)
    tr3,va3,ytr3,yva3=make_split(tr_full,y_full,0.10)
    assert_no_overlap(tr3,va3,te3)
    labels=['negatif','netral','positif']
    map_labels={k:i for i,k in enumerate(labels)}
    vocab=SentenceVocabulary().fit(tr3)
    def loader(xx,yy,shuffle=False):
        ids=[map_labels[z] for z in yy]
        return DataLoader(ReviewDataset(xx,ids,vocab),batch_size=BATCH_SIZE,shuffle=shuffle,collate_fn=collate,num_workers=0)
    train_loader=loader(tr3,ytr3,True);train_eval_loader=loader(tr3,ytr3)
    val_loader=loader(va3,yva3);test_loader=loader(te3,yt3)
    net=BiLSTMClassifier(len(vocab.words))
    weights=torch.tensor([len(ytr3)/(3*np.sum(ytr3==x)) for x in labels],dtype=torch.float32)
    loss=nn.CrossEntropyLoss(weight=weights)
    optim=torch.optim.Adam(net.parameters(),lr=.002,weight_decay=1e-4)
    history=[];best=-1.;bad=0;best_epoch=0;best_state=None
    for epoch in range(1,EPOCHS+1):
        st=time.time()
        train_stats=prediction(net,train_loader,loss,optim)
        val_stats=prediction(net,val_loader,loss)
        history.append({'epoch':epoch,'training_accuracy':round(100*train_stats['accuracy'],3),
                        'validation_accuracy':round(100*val_stats['accuracy'],3),
                        'training_loss':round(train_stats['loss'],5),'validation_loss':round(val_stats['loss'],5)})
        print(f'  Epoch {epoch}: train={train_stats["accuracy"]:.2%} val={val_stats["accuracy"]:.2%} ({time.time()-st:.1f}s)',flush=True)
        if val_stats['accuracy'] > best+0.0001:
            best=val_stats['accuracy'];best_epoch=epoch;bad=0
            best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        else:bad+=1
        if bad>=PATIENCE:
            print(f'  Early stopping; best epoch={best_epoch}')
            break
    net.load_state_dict(best_state)
    tstat=prediction(net,train_eval_loader,loss);vstat=prediction(net,val_loader,loss);teststat=prediction(net,test_loader,loss)
    yp_train=[labels[j] for j in tstat['pred']]; yp_test=[labels[j] for j in teststat['pred']]
    ev=build_evaluation(ytr3,yp_train,yt3,yp_test)
    results['schemes']['skema_3_bilstm']={'algorithm':'PyTorch Bidirectional LSTM','features':'Train-only learned word embedding',
        'split':'80/20 with 10% validation of train','validation_accuracy':round(100*vstat['accuracy'],4),
        'validation_size':len(va3),'best_epoch':best_epoch,'history':history,'vocabulary_size':len(vocab.words),**ev}
    (MODELS/'vocab_bilstm.json').write_text(json.dumps(vocab.words,ensure_ascii=False),encoding='utf-8')
    torch.save({'state_dict':net.state_dict(),'vocab_size':len(vocab.words),'embed_dim':EMBED_DIM,
                'hidden_size':HIDDEN_SIZE,'label_order':labels,'max_len':MAX_LEN},MODELS/'skema_3_bilstm.pt')
    plot_cm(ev,'Skema 3 — PyTorch Bidirectional LSTM','confusion_matrix_skema_3.png')
    print('Skema 3 akurasi:',ev['training_accuracy'],ev['testing_accuracy'])

    # Riwayat pelatihan dan visualisasi evaluasi disimpan sebagai artefak.
    fig,ax=plt.subplots(1,1,figsize=(7,4.5))
    ax.plot([h['epoch'] for h in history],[h['training_accuracy'] for h in history],label='Training')
    ax.plot([h['epoch'] for h in history],[h['validation_accuracy'] for h in history],label='Validation')
    ax.set(xlabel='Epoch',ylabel='Accuracy (%)',title='BiLSTM training history');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(ASSETS/'bilstm_training_history.png',dpi=160);plt.close(fig)
    plt.figure(figsize=(8,4.7));names=['LinearSVC','LogReg','BiLSTM'];rs=list(results['schemes'].values());ii=np.arange(3)
    plt.bar(ii-.18,[r['training_accuracy'] for r in rs],width=.36,label='Training')
    plt.bar(ii+.18,[r['testing_accuracy'] for r in rs],width=.36,label='Testing')
    plt.xticks(ii,names);plt.ylim(0,105);plt.ylabel('Accuracy (%)');plt.legend();plt.title('Perbandingan 3 Skema')
    plt.tight_layout();plt.savefig(ASSETS/'perbandingan_akurasi.png',dpi=160);plt.close()
    plt.figure(figsize=(7,4));plt.bar(labels,[int((df['polarity']==z).sum()) for z in labels]);plt.title('Distribusi Kelas Berlabel');plt.tight_layout()
    plt.savefig(ASSETS/'distribusi_kelas.png',dpi=160);plt.close()

    # Inferensi menggunakan bobot model terlatih, bukan hasil prediksi hard-coded.
    samples=[
        'Aplikasi Gojek sangat bagus, mudah digunakan dan driver ramah',
        'GoFood murah, pelayanan sangat memuaskan',
        'Aplikasi sering error dan sangat mengecewakan',
        'Driver terlalu lama, pesanan saya batal terus',
        'Saya menggunakan aplikasi untuk bepergian setiap hari',
        'Saya memesan makanan dari restoran dekat kantor'
    ]
    preds=[];net.eval()
    for t in samples:
        cleaned=preprocessor.transform(t)
        p1=str(m1.predict([cleaned])[0]);p2=str(m2.predict([cleaned])[0])
        toks=vocab.encode(cleaned)
        with torch.no_grad():
            logits=net(torch.tensor(toks,dtype=torch.long).unsqueeze(0),torch.tensor([len(toks)]))
            prob=torch.softmax(logits,dim=1).squeeze(0).numpy()
        p3=labels[int(prob.argmax())];vote=Counter([p1,p2,p3]).most_common(1)[0][0]
        preds.append({'input':t,'processed':cleaned,'linear_svc':p1,'logreg':p2,'bilstm':p3,
                      'bilstm_confidence':round(float(prob.max()),4),'majority_vote':vote})
    results['inference']=preds
    results['execution_seconds']=round(time.time()-start,2)
    (RESULTS/'hasil_evaluasi_final.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    pd.DataFrame([{'scheme':k,'algorithm':v['algorithm'],'train_accuracy':v['training_accuracy'],
                   'test_accuracy':v['testing_accuracy'],'macro_f1':v['testing_macro_f1']} for k,v in results['schemes'].items()]).to_csv(RESULTS/'perbandingan_akurasi_final.csv',index=False)
    pd.DataFrame(preds).to_csv(RESULTS/'contoh_inference.csv',index=False)
    print('Hasil berhasil disimpan. Durasi:',results['execution_seconds'],'detik')
    return results

if __name__=='__main__':run()
