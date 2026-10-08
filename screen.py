import pandas as pd
from transformers import *
import torch
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('./raw.csv',sep=';')
print(df.head())
df = df[:10]

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

model.eval()

embeddings = []
for idx, row in df.iterrows():
    text = 'TITLE:' + str(row['Article Title']) + ' ABSTRACT:' + str(row['Abstract'])
    texts = [text]
    batch = tokenizer(
            texts, 
            truncation=True, 
            return_tensors="pt",
            padding='max_length', 
            max_length=512
            )

    with torch.no_grad():
        out = model(**batch)
        print(out)
        emb = out.last_hidden_state
        print(emb.shape)
    embeddings.append(emb.squeeze())


print(embeddings)
print(len(embeddings))
